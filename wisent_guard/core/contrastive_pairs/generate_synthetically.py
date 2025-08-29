import json
import random
import re
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple
import concurrent.futures
from functools import lru_cache
import threading
import time
from datetime import datetime
import torch

import numpy as np
from sentence_transformers import SentenceTransformer

from wisent_guard.core.model import Model

from ..response import NegativeResponse, PositiveResponse
from .contrastive_database import ContrastivePairDatabase
from .contrastive_pair import ContrastivePair
from .contrastive_pair_set import ContrastivePairSet
from .contrastive_generation_conf import CONTRASTIVE_GEN, QUESTION_GEN, QUESTION_PARSE
from .quality_check import quality_check_synthetic_pairs
from .question_bank import QuestionBank


class TimingContext:
    """Context manager for timing operations."""
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"⏱️  [{datetime.now().strftime('%H:%M:%S')}] Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.verbose:
            print(f"⏱️  [{datetime.now().strftime('%H:%M:%S')}] Completed: {self.name} (took {self.duration:.2f}s)")


class SyntheticContrastivePairGenerator:
    """Generate contrastive pairs synthetically from natural language trait descriptions."""

    def __init__(
        self,
        model: Model,
        similarity_threshold: float = 0.8,
        db_path: Optional[str] = None,
        db_similarity_threshold: float = 0.95,
        question_bank_path: Optional[str] = None,
        max_workers: int = 4,
        cache_size: int = 1000,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the synthetic pair generator.

        Args:
            model: The language model to use for generation
            similarity_threshold: Threshold for deduplication of questions/pairs (0-1, higher = more strict)
            db_path: Optional path to the contrastive pair database. If None, caching is disabled.
            db_similarity_threshold: Threshold for retrieving a set from the database (0-1, higher = more strict).
            question_bank_path: Optional path to the question bank. If None, uses default location.
            max_workers: Maximum number of parallel workers for generation
            cache_size: Size of the LRU cache for model responses
            generation_kwargs: Optional dict of generation parameters to pass to the model (e.g., enable_thinking=False)
        """
        self.model: Model = model
        self.similarity_threshold: float = similarity_threshold
        self.db_similarity_threshold: float = db_similarity_threshold
        self.max_workers: int = max_workers
        self.generation_kwargs: Dict[str, Any] = generation_kwargs or {}
        
        # Handle enable_thinking parameter
        self.suppress_thinking_tokens = False
        if 'enable_thinking' in self.generation_kwargs and self.generation_kwargs['enable_thinking'] is False:
            self.suppress_thinking_tokens = True
            # Set enable_thinking=False on the model if it has the attribute
            if hasattr(self.model, 'hf_model') and self.model.hf_model is not None:
                if hasattr(self.model.hf_model, 'generation_config'):
                    self.model.hf_model.generation_config.enable_thinking = False
                if hasattr(self.model.hf_model, 'config'):
                    self.model.hf_model.config.enable_thinking = False

        self.similarity_model: Optional[SentenceTransformer] = (
            self._initialize_similarity_model()
        )
        self.database: Optional[ContrastivePairDatabase] = self._initialize_database(
            db_path
        )
        self.question_bank: QuestionBank = QuestionBank(question_bank_path)
        
        # Thread-safe cache for model responses
        self._cache_lock = threading.Lock()
        self._response_cache: Dict[str, str] = {}
        self._cache_size = cache_size
        
        # Initialize thread pool executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Timing statistics
        self.timing_stats: Dict[str, List[float]] = {}
        self.enable_timing = True  # Can be disabled if needed

    def _initialize_similarity_model(self) -> Optional[SentenceTransformer]:
        """Loads the SentenceTransformer model for similarity calculations."""
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Similarity model loaded successfully.")
            return model
        except Exception as e:
            print(
                f"⚠️ WARNING: Could not load similarity model ('all-MiniLM-L6-v2'). VDB caching and diversity selection will be disabled. Error: {e}"
            )
            return None

    def _initialize_database(
        self, db_path: Optional[str]
    ) -> Optional[ContrastivePairDatabase]:
        """Initializes the contrastive pair database if a path is provided.
        Args:
            db_path: The path to the database directory. If None, caching is disabled.
        Returns:
            An optional ContrastivePairDatabase instance
        """
        if not db_path:
            print("ℹ️ No database path provided. Contrastive pair caching is disabled.")
            return None

        try:
            database = ContrastivePairDatabase(db_path)
            print(
                f"🗂️  Initialized contrastive pair database at: {Path(db_path).resolve()}"
            )
            return database
        except (IOError, OSError, json.JSONDecodeError, RuntimeError) as e:
            print(
                f"⚠️ WARNING: Failed to initialize contrastive pair database at '{db_path}'. Caching is disabled. Error: {e}"
            )
            return None

    def generate_questions(
        self, trait_description: str, num_questions: int, force_new: bool = False
    ) -> list[str]:
        """
        Get questions from the bank or generate new ones if needed.

        Args:
            trait_description: Natural language description of desired trait
            num_questions: Number of questions to get
            force_new: If True, always generate new questions instead of using bank

        Returns:
            list of question descriptions
        """
        print(f"🔍 [generate_questions] Starting with trait: {trait_description}, num_questions: {num_questions}")
        print(f"🔍 [generate_questions] self.model type: {type(self.model)}")
        print(f"🔍 [generate_questions] self.model.tokenizer type: {type(self.model.tokenizer)}")
        
        if self.model is None and force_new:
            raise ValueError("No model loaded. Cannot generate new questions.")
            
        print(f"📚 Checking question bank for trait: '{trait_description}'")
        
        # Check how many questions are available in the bank
        available_count = self.question_bank.get_available_count()
        unused_count = self.question_bank.get_unused_count(trait_description)
        
        print(f"📊 Question bank status: {available_count} total, {unused_count} unused for this trait")
        
        # Determine if we need to generate new questions
        need_to_generate = force_new or available_count < num_questions
        
        if not need_to_generate:
            # Try to get questions from the bank
            questions = self.question_bank.get_questions(num_questions, trait_description, prefer_unused=True)
            if len(questions) == num_questions:
                print(f"✅ Retrieved {num_questions} questions from bank")
                for i, question in enumerate(questions):
                    print(f"   Question {i+1}: {question}")
                return questions
            else:
                need_to_generate = True
        
        if need_to_generate:
            # Generate new questions
            print(f"🎯 Generating new questions...")
            
            # Calculate how many to generate
            if force_new:
                target_new = num_questions * QUESTION_GEN.OVERGENERATION_FACTOR
            else:
                # Generate enough to fill the gap plus some extra
                gap = max(num_questions - available_count, 50)  # At least 50 new questions
                target_new = gap * 2  # Generate 2x what we need
            
            print(f"🔍 [generate_questions] About to call _generate_new_questions_parallel")
            try:
                generated_questions = self._generate_new_questions_parallel(target_new)
                print(f"🔍 [generate_questions] _generate_new_questions_parallel returned {len(generated_questions)} questions")
            except Exception as e:
                print(f"🔍 [generate_questions] _generate_new_questions_parallel failed with: {e}")
                import traceback
                print(f"🔍 [generate_questions] Traceback:\n{traceback.format_exc()}")
                raise
            
            # Add to bank
            added_count = self.question_bank.add_questions(generated_questions)
            
            # Now get the questions we need
            questions = self.question_bank.get_questions(num_questions, trait_description, prefer_unused=True)
            
            print(f"✅ Using {len(questions)} questions (added {added_count} new to bank)")
            for i, question in enumerate(questions):
                print(f"   Question {i+1}: {question}")
            
            return questions
    
    def _generate_new_questions_parallel(self, target_count: int) -> list[str]:
        """
        Generate new questions using the LLM.
        
        Args:
            target_count: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        print(f"🔍 [_generate_new_questions_parallel] Starting with target_count: {target_count}")
        print(f"🔍 [_generate_new_questions_parallel] self.model type: {type(self.model)}")
        print(f"🔍 [_generate_new_questions_parallel] self.model.tokenizer type: {type(self.model.tokenizer)}")
        
        if self.model is None:
            raise ValueError("No model loaded. Cannot generate questions.")
            
        all_questions: list[str] = []
        
        num_prompts_per_template: int = max(1, target_count // len(QUESTION_GEN.PROMPT_TEMPLATES))
        
        with TimingContext(f"Parallel question generation ({len(QUESTION_GEN.PROMPT_TEMPLATES)} templates)", self.enable_timing):
            # Create futures for parallel generation
            futures = []
            for i, template in enumerate(QUESTION_GEN.PROMPT_TEMPLATES):
                prompt: str = template.format(num_prompts=num_prompts_per_template)
                print(f"   Submitting template {i+1}/{len(QUESTION_GEN.PROMPT_TEMPLATES)} for parallel generation")
                
                future = self.executor.submit(
                    self._cached_generate_response, 
                    prompt, 
                    QUESTION_GEN.CONFIG
                )
                futures.append((i, future))
            
            # Collect results
            for i, future in futures:
                response = future.result()
                questions = self._parse_questions_from_response(response)
                all_questions.extend(questions)
                print(f"   ✅ Template {i+1} generated {len(questions)} questions")
        
        # Deduplicate
        with TimingContext(f"Deduplicating {len(all_questions)} questions", self.enable_timing):
            unique_questions: list[str] = self._deduplicate_questions(all_questions)
        
        # Select diverse questions up to target count
        if len(unique_questions) > target_count:
            with TimingContext(f"Selecting {target_count} diverse questions from {len(unique_questions)}", self.enable_timing):
                selected_questions = self._select_diverse_questions(unique_questions, target_count)
        else:
            selected_questions = unique_questions
        
        return selected_questions

    def _parse_questions_from_response(self, response: str) -> list[str]:
        """Parse individual questions from model response using regex and filters.
        Args:
            response: The raw response text from the model
        Returns:
            A list of parsed question strings
        """
        questions: list[str] = []

        # Regex to remove common list prefixes (e.g., "1.", "-", "* ") and markdown
        prefix_re: re.Pattern = re.compile(r"^\s*(?:\d+\.|\-|\*|•|[a-e]\))\s*")
        markdown_re: re.Pattern = re.compile(r"(\*\*|\*)")

        lines: list[str] = response.split("\n")

        for line in lines:
            # 1. Clean the line
            cleaned: str = markdown_re.sub("", prefix_re.sub("", line.strip()))

            # 2. Basic filtering
            if not cleaned or not (
                QUESTION_PARSE.MIN_QUESTION_LENGTH
                < len(cleaned)
                < QUESTION_PARSE.MAX_QUESTION_LENGTH
            ):
                continue

            cleaned_lower: str = cleaned.lower()

            # 3. Filter based on content
            if any(
                phrase in cleaned_lower for phrase in QUESTION_PARSE.SKIP_PHRASES
            ) or any(
                phrase in cleaned_lower for phrase in QUESTION_PARSE.REFUSAL_PHRASES
            ):
                continue

            # 4. Validate based on structure (question or imperative)
            word_count: int = len(cleaned.split())
            is_question: bool = (
                "?" in cleaned and word_count <= QUESTION_PARSE.MAX_QUESTION_WORDS
            )
            is_imperative: bool = (
                any(kw in cleaned_lower for kw in QUESTION_PARSE.IMPERATIVE_KEYWORDS)
                and word_count <= QUESTION_PARSE.MAX_IMPERATIVE_WORDS
            )

            if is_question or is_imperative:
                questions.append(cleaned)

        return questions[: QUESTION_PARSE.CANDIDATE_LIMIT]

    def _deduplicate_questions(self, questions: list[str]) -> list[str]:
        """Remove duplicate or very similar questions using batch processing.
        Args:
            questions: The list of question strings to deduplicate
        Returns:
            A list of unique question strings
        """
        print(f"🔍 [_deduplicate_questions] Starting with {len(questions)} questions")
        print(f"🔍 [_deduplicate_questions] self.similarity_model: {self.similarity_model}")
        
        if not self.similarity_model or len(questions) <= 1:
            # Fallback to simple text-based deduplication
            print(f"🔍 [_deduplicate_questions] Using text-based deduplication")
            return list(set(questions))

        # Batch encode all questions at once
        all_embeddings = self.similarity_model.encode(questions, batch_size=32)
        
        unique_indices = [0]  # Start with first question
        unique_embeddings = [all_embeddings[0]]

        for i in range(1, len(questions)):
            # Calculate similarities with all unique questions so far
            similarities = np.dot(all_embeddings[i], np.array(unique_embeddings).T)
            
            if np.max(similarities) <= self.similarity_threshold:
                unique_indices.append(i)
                unique_embeddings.append(all_embeddings[i])

        return [questions[i] for i in unique_indices]

    def _select_diverse_questions(
        self, questions: list[str], target_count: int
    ) -> list[str]:
        """Select the most diverse questions using optimized computation.
        Args:
            questions: The list of question strings to select from
            target_count: The number of diverse questions to select
        Returns:
            A list of diverse question strings
        """
        if len(questions) <= target_count:
            return questions

        if not self.similarity_model:
            # Random selection fallback
            return random.sample(questions, target_count)

        # Batch encode all questions
        embeddings = self.similarity_model.encode(questions, batch_size=32)
        
        # Use greedy selection with vectorized operations
        selected_indices = [0]
        selected_embeddings = embeddings[0:1]  # Start with first embedding

        for _ in range(target_count - 1):
            # Calculate distances from all remaining to all selected
            remaining_mask = np.ones(len(questions), dtype=bool)
            remaining_mask[selected_indices] = False
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_indices) == 0:
                break
            
            # Vectorized distance calculation
            remaining_embeddings = embeddings[remaining_indices]
            similarities = np.dot(remaining_embeddings, selected_embeddings.T)
            min_similarities = np.min(similarities, axis=1)
            
            # Select the one with maximum minimum distance (least similar to any selected)
            best_idx_in_remaining = np.argmin(min_similarities)
            best_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_idx)
            selected_embeddings = np.vstack([selected_embeddings, embeddings[best_idx]])

        return [questions[i] for i in selected_indices]

    def _generate_contrastive_pairs_parallel(
        self, questions: List[str], trait_description: str, current_count: int, target_count: int
    ) -> List[ContrastivePair]:
        """Generate multiple contrastive pairs in parallel."""
        print(f"🔄 Generating contrastive pairs from {len(questions)} questions...")
        
        # Limit questions to what we need
        remaining_needed = target_count - current_count
        questions_to_process = questions[:remaining_needed + 5]  # Small buffer
        
        with TimingContext(f"Parallel pair generation ({len(questions_to_process)} pairs)", self.enable_timing):
            # Create futures for parallel generation
            futures = []
            for i, question in enumerate(questions_to_process):
                future = self.executor.submit(
                    self._generate_single_contrastive_pair,
                    question,
                    trait_description,
                    i + 1,
                    len(questions_to_process)
                )
                futures.append((question, future))
            
            # Collect results
            successful_pairs = []
            for question, future in futures:
                pair = future.result()
                if pair is not None:
                    successful_pairs.append(pair)
                    if current_count + len(successful_pairs) >= target_count:
                        print(f"✅ Reached target of {target_count} pairs for selection")
                        break
        
        return successful_pairs
    
    def _generate_single_contrastive_pair(
        self, question: str, trait_description: str, index: int, total: int
    ) -> Optional[ContrastivePair]:
        """Generate a single contrastive pair with error handling."""
        try:
            print(f"   Generating pair {index}/{total}: {question[:50]}...")
            pair = self.generate_contrastive_pair(question, trait_description)
            return pair
        except Exception as e:
            print(f"   ⚠️ Error generating pair: {str(e)[:100]}")
            return None
    
    def _batch_compute_pair_embeddings(self, pairs: List[ContrastivePair]) -> np.ndarray:
        """Compute embeddings for all pairs in batch."""
        if not self.similarity_model:
            return np.array([])
        
        with TimingContext(f"Batch embedding computation for {len(pairs)} pairs", self.enable_timing):
            # Prepare all texts for batch encoding
            all_texts = []
            for pair in pairs:
                all_texts.extend([
                    pair.prompt,
                    pair.positive_response.text,
                    pair.negative_response.text
                ])
            
            # Batch encode
            all_embeddings = self.similarity_model.encode(all_texts, batch_size=32)
            
            # Reshape and combine embeddings for each pair
            pair_embeddings = []
            for i in range(0, len(all_embeddings), 3):
                combined = np.concatenate([
                    all_embeddings[i],      # prompt
                    all_embeddings[i+1],    # positive
                    all_embeddings[i+2]     # negative
                ])
                norm = np.linalg.norm(combined)
                pair_embeddings.append(combined / norm if norm != 0 else combined)
        
        return np.array(pair_embeddings)
    
    def _get_pair_embedding(self, pair: ContrastivePair) -> np.ndarray:
        """Computes a single embedding for a contrastive pair (for backward compatibility)."""
        embeddings = self._batch_compute_pair_embeddings([pair])
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def _select_diverse_pairs(
        self, pairs: list[ContrastivePair], target_count: int
    ) -> list[ContrastivePair]:
        """Selects a diverse subset of contrastive pairs using their embeddings.
        Args:
            pairs: The list of contrastive pairs to select from
            target_count: The number of diverse pairs to select
        Returns:
            A list of diverse contrastive pairs
        """
        if len(pairs) <= target_count:
            return pairs

        if not self.similarity_model:
            print("⚠️ DEBUG: No similarity model found. Selecting pairs randomly.")
            return random.sample(pairs, target_count)

        print(
            f"🔎 Selecting {target_count} diverse pairs from {len(pairs)} candidates..."
        )

        # Batch compute embeddings
        embeddings = self._batch_compute_pair_embeddings(pairs)

        selected_indices: list[int] = [0]
        for _ in range(target_count - 1):
            remaining_indices: list[int] = [
                i for i in range(len(pairs)) if i not in selected_indices
            ]
            if not remaining_indices:
                break

            # Use greedy selection with vectorized operations
            selected_embeddings = embeddings[selected_indices]
            
            # Vectorized distance calculation
            remaining_embeddings = embeddings[remaining_indices]
            similarities = np.dot(remaining_embeddings, selected_embeddings.T)
            min_similarities = np.min(similarities, axis=1)
            
            # Select the one with maximum minimum distance (least similar to any selected)
            best_idx_in_remaining = np.argmin(min_similarities)
            best_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_idx)

        return [pairs[i] for i in selected_indices]

    def _cached_generate_response(self, prompt: str, config: dict[str, Any]) -> str:
        """Generate response with caching.
        Args:
            prompt: The input prompt for the model
            config: Configuration parameters for the model generation
        Returns
            response: The generated response text
        """
        # Create cache key from prompt and config
        cache_key = f"{prompt}:{json.dumps(config, sort_keys=True)}"
        
        # Check cache
        with self._cache_lock:
            if cache_key in self._response_cache:
                return self._response_cache[cache_key]
        
        # Generate response
        import time
        start_time = time.time()
        print(f"🟡 [Generation] Starting generation for prompt (first 100 chars): {prompt[:100]}...")
        
        # Merge generation_kwargs with config, giving precedence to generation_kwargs
        merged_config = {**config, **self.generation_kwargs}
        
        # Extract layer_index from config (required parameter for Model.generate)
        layer_index = merged_config.pop('layer_index', 15)  # Default to layer 15
        
        # Extract enable_thinking to pass to format_prompt
        enable_thinking = merged_config.pop('enable_thinking', None)
        
        # Format the prompt with model-specific parameters
        if enable_thinking is not None:
            formatted_prompt = self.model.format_prompt(prompt, enable_thinking=enable_thinking)
        else:
            formatted_prompt = self.model.format_prompt(prompt)
        
        print(f"🟡 [Generation] Formatted prompt, tokenizing...")
        # Generate using the formatted prompt directly
        # Tokenize
        inputs = self.model.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
        print(f"🟡 [Generation] Tokenized, input shape: {inputs['input_ids'].shape}")
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        print(f"🟡 [Generation] Starting model.generate() with max_new_tokens: {merged_config.get('max_new_tokens', 150)}")
        gen_start = time.time()
        
        # Generate
        with torch.no_grad():
            # Import GenerationConfig from transformers
            from transformers import GenerationConfig
            
            # Create generation config with proper parameters
            gen_config_dict = {
                'max_new_tokens': merged_config.get('max_new_tokens', 150),
                'temperature': merged_config.get('temperature', 0.7),
                'top_p': merged_config.get('top_p', 0.9),
                'do_sample': True,
                'pad_token_id': self.model.tokenizer.pad_token_id,
                'eos_token_id': self.model.tokenizer.eos_token_id,
            }
            
            # Remove None values to avoid issues
            gen_config_dict = {k: v for k, v in gen_config_dict.items() if v is not None}
            
            # Create GenerationConfig object
            try:
                generation_config = GenerationConfig(**gen_config_dict)
                outputs = self.model.hf_model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            except:
                # Fallback for older transformers versions - use dict directly
                outputs = self.model.hf_model.generate(
                    **inputs,
                    **gen_config_dict
                )
        
        gen_time = time.time() - gen_start
        print(f"🟡 [Generation] Generation completed in {gen_time:.2f} seconds")
        
        # Decode
        response = self.model.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Basic post-processing - only remove explicit think tags if they somehow appear
        import re
        if '<think>' in response:
            # This shouldn't happen with enable_thinking=False, but just in case
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = response.strip()
        
        # No aggressive cleaning needed anymore since we're properly using enable_thinking=False
        
        # Update cache
        with self._cache_lock:
            # Implement simple LRU by removing oldest entries if cache is full
            if len(self._response_cache) >= self._cache_size:
                # Remove ~10% of oldest entries
                num_to_remove = max(1, self._cache_size // 10)
                for _ in range(num_to_remove):
                    self._response_cache.pop(next(iter(self._response_cache)))
            
            self._response_cache[cache_key] = response
        
        return response
    
    def _generate_response(self, prompt: str, config: dict[str, Any]) -> str:
        """Wrapper for backward compatibility."""
        return self._cached_generate_response(prompt, config)

    def generate_contrastive_pair(
        self, question: str, trait_description: str
    ) -> ContrastivePair:
        """
        Generate a contrastive pair for a specific question.

        Args:
            question: The question to generate responses for
            trait_description: The trait description for context

        Returns:
            ContrastivePair object
        """
        print(
            f"🔄 DEBUG: Generating contrastive pair for question: {question[:100]}..."
        )
        print(f"🔄 DEBUG: Trait: {trait_description}")

        # Generate positive response (demonstrates the trait)
        positive_prompt: str = CONTRASTIVE_GEN.POSITIVE_PROMPT_TEMPLATE.format(
            question=question, trait_description=trait_description
        )
        print(f"🔄 DEBUG: Positive prompt: {positive_prompt}")
        positive_response: str = self._generate_response(
            positive_prompt, CONTRASTIVE_GEN.CONFIG
        )
        print(f"🔄 DEBUG: Positive response: {positive_response[:100]}...")

        # Generate negative response (without the trait)
        negative_prompt: str = CONTRASTIVE_GEN.NEGATIVE_PROMPT_TEMPLATE.format(
            question=question, trait_description=trait_description
        )
        print(f"🔄 DEBUG: Negative prompt: {negative_prompt}")
        negative_response: str = self._generate_response(
            negative_prompt, CONTRASTIVE_GEN.CONFIG
        )
        print(f"🔄 DEBUG: Negative response: {negative_response[:100]}...")

        # Create contrastive pair - always use the question directly
        prompt: str = question.strip()
        print(f"🔄 DEBUG: Using question as direct prompt: {prompt}")

        pair: ContrastivePair = ContrastivePair(
            prompt=prompt,
            positive_response=PositiveResponse(text=positive_response),
            negative_response=NegativeResponse(text=negative_response),
        )

        # Store metadata
        pair.question = question
        pair.trait_description = trait_description

        print(f"🔄 DEBUG: Created contrastive pair successfully")

        return pair

    def generate_pairs_until_target(
        self, trait_description: str, target_pairs: int
    ) -> list[ContrastivePair]:
        """
        Generate exactly the target number of contrastive pairs.
        Automatically generates more questions as needed until target is reached.
        
        Args:
            trait_description: Natural language description of desired trait
            target_pairs: Exact number of pairs to generate
            
        Returns:
            List of ContrastivePair objects
        """
        print(f"🎯 Generating exactly {target_pairs} contrastive pairs for trait: {trait_description}")
        
        successful_pairs: list[ContrastivePair] = []
        questions_used = 0
        attempts = 0
        max_attempts = 20  # Prevent infinite loops
        
        # Start with a conservative batch size
        batch_size = max(5, target_pairs * 2)
        
        while len(successful_pairs) < target_pairs and attempts < max_attempts:
            remaining_pairs = target_pairs - len(successful_pairs)
            
            # Adjust batch size based on success rate
            if len(successful_pairs) > 0 and questions_used > 0:
                success_rate = len(successful_pairs) / questions_used
                # Estimate how many questions we need with some buffer
                batch_size = int(remaining_pairs / success_rate * 1.3) if success_rate > 0 else remaining_pairs * 3
            
            batch_size = max(5, min(batch_size, 50))  # Keep batch size reasonable
            
            print(f"\n📝 Getting batch of {batch_size} questions (attempt {attempts + 1})...")
            questions = self.generate_questions(trait_description, batch_size, force_new=False)
            questions_used += len(questions)
            
            print(f"✅ Generated {len(questions)} questions")
            print(f"🔄 Creating contrastive pairs...")
            
            for i, question in enumerate(questions):
                if len(successful_pairs) >= target_pairs:
                    break
                    
                try:
                    pair = self.generate_contrastive_pair(question, trait_description)
                    
                    # Check for safety filter refusals
                    safety_phrases = [
                        "I can't assist", "I cannot assist", "I'm not able", "I cannot provide", 
                        "I can't help", "I cannot create", "I can't fulfill", "I cannot fulfill",
                        "I'm unable to", "I cannot generate", "explicit content", "I apologize"
                    ]
                    pos_text = pair.positive_response.text.lower()
                    neg_text = pair.negative_response.text.lower()
                    
                    if any(phrase.lower() in pos_text for phrase in safety_phrases):
                        print(f"   🚫 Safety filter triggered on positive response: {pair.positive_response.text[:50]}...")
                        continue
                    if any(phrase.lower() in neg_text for phrase in safety_phrases):
                        print(f"   🚫 Safety filter triggered on negative response: {pair.negative_response.text[:50]}...")
                        continue
                    
                    successful_pairs.append(pair)
                    print(f"   ✅ Pair {len(successful_pairs)}/{target_pairs} created successfully")
                except Exception as e:
                    print(f"   ⚠️ Failed to create pair: {str(e)[:100]}")
                    continue
            
            attempts += 1
            
            if len(successful_pairs) < target_pairs:
                print(f"📊 Progress: {len(successful_pairs)}/{target_pairs} pairs created")
        
        if len(successful_pairs) < target_pairs:
            print(f"⚠️ Warning: Only created {len(successful_pairs)}/{target_pairs} pairs after {attempts} attempts")
        else:
            print(f"\n✅ Successfully created all {target_pairs} pairs using {questions_used} questions")
            if questions_used > 0:
                print(f"📊 Overall success rate: {len(successful_pairs)/questions_used*100:.1f}%")
            else:
                print(f"📊 No questions were used (this shouldn't happen)")
        
        return successful_pairs

    def generate_contrastive_pair_set(
        self,
        trait_description: str,
        num_pairs: int = 30,
        pair_overgeneration_factor: float = 1.5,
        force_regenerate: bool = False,
        verbose_timing: bool = False,
        name: Optional[str] = None,  # Added to fix bug where multiple callers pass this
    ) -> ContrastivePairSet:
        """
        Generate a complete contrastive pair set from a trait description.
        Checks a vector database first to avoid re-generation if available.

        Args:
            trait_description: Natural language description of desired trait
            num_pairs: Number of contrastive pairs to generate
            pair_overgeneration_factor: Factor to overgenerate pairs for diversity selection
            force_regenerate: If True, bypasses the cache and generates a new set.
            verbose_timing: If True, shows detailed timing for each step
            name: Optional name for the pair set (used by various callers)

        Returns:
            ContrastivePairSet with generated pairs
        """
        # Override instance timing setting if verbose_timing is specified
        original_timing = self.enable_timing
        if verbose_timing:
            self.enable_timing = True
        
        overall_start = time.time()
        use_database: bool = self.database is not None and not force_regenerate

        if not self.similarity_model and use_database:
            print("⚠️ WARNING: Similarity model not loaded. Cannot use database cache.")
            use_database = False

        if use_database:
            print(f"🔍 Checking database for similar trait: '{trait_description}'")
            trait_embedding: np.ndarray = self.similarity_model.encode(
                [trait_description]
            )

            found_set_info: Optional[dict] = self.database.search_for_trait(
                trait_embedding, self.db_similarity_threshold
            )

            if found_set_info is not None:
                cached_pair_count: int = found_set_info.get("pair_count", 0)
                print(
                    f"✅ Found sufficiently similar pair set in cache (ID: {found_set_info['set_id']}) with {cached_pair_count} pairs."
                )

                if num_pairs <= cached_pair_count:
                    print("Sufficient pairs in cache. Retrieving...")
                    return self.database.get_set_by_id(found_set_info["set_id"])
                else:
                    print(
                        f"⚠️ Cache has {cached_pair_count} pairs, but {num_pairs} are requested. Augmenting the set..."
                    )
                    # Load existing set and generate the missing pairs
                    existing_set: ContrastivePairSet = self.database.get_set_by_id(
                        found_set_info["set_id"]
                    )
                    additional_pairs_needed: int = num_pairs - cached_pair_count

                    # We need to generate more questions and pairs, avoiding duplicates from the existing set
                    # For simplicity, we'll generate a new batch and combine, then re-filter for diversity.
                    # A more advanced implementation could try to find questions dissimilar to existing ones.

                    print(
                        f"🔄 Generating {additional_pairs_needed} additional pairs..."
                    )
                    newly_generated_set: ContrastivePairSet = self._generate_new_pairs(
                        trait_description,
                        additional_pairs_needed,
                        pair_overgeneration_factor,
                    )

                    # Combine, ensure diversity, and update the cache
                    combined_set = self._combine_and_update_set(
                        existing_set,
                        newly_generated_set,
                        num_pairs,
                        trait_description,
                        trait_embedding,
                    )
                    return combined_set

        print(
            f"ℹ️ No suitable pair set found in cache or caching is disabled. Generating a new one for trait: '{trait_description}'"
        )

        # If we are generating and have a database, we need the embedding for caching later.
        trait_embedding: Optional[np.ndarray] = None
        if use_database:
            trait_embedding = self.similarity_model.encode([trait_description])

        # Generate the full set of new pairs
        new_pair_set = self._generate_new_pairs(
            trait_description, num_pairs, pair_overgeneration_factor
        )

        # Add the newly generated set to the database
        if use_database and trait_embedding is not None:
            print(f"💾 Caching new pair set to database...")
            self.database.add_set(new_pair_set, trait_embedding)
            print(f"✅ Cached new set.")

        overall_end = time.time()
        print(f"\n📊 Total generation time: {overall_end - overall_start:.2f}s")
        
        # Restore original timing setting
        self.enable_timing = original_timing
        
        return new_pair_set

    def _generate_new_pairs(
        self, trait_description: str, num_pairs: int, pair_overgeneration_factor: float
    ) -> ContrastivePairSet:
        """Generate a new set of contrastive pairs.
        Args:
            trait_description: Natural language description of desired trait
            num_pairs: Number of contrastive pairs to generate
            pair_overgeneration_factor: Factor to overgenerate pairs for diversity selection
        Returns:
            ContrastivePairSet with generated pairs
        """
        print(f"🎯 Target: {num_pairs} high-quality contrastive pairs")
        
        all_pairs: list[ContrastivePair] = []
        questions_generated = 0
        attempts = 0
        max_attempts = 10  # Prevent infinite loops
        
        # Calculate initial batch size based on expected success rate
        initial_batch_size = max(10, int(num_pairs * 1.5))
        
        while len(all_pairs) < int(num_pairs * pair_overgeneration_factor) and attempts < max_attempts:
            # Determine how many more pairs we need
            pairs_needed = int(num_pairs * pair_overgeneration_factor) - len(all_pairs)
            
            # Estimate questions needed based on success rate so far
            if len(all_pairs) > 0 and questions_generated > 0:
                success_rate = len(all_pairs) / questions_generated
                # Add 50% buffer to account for variance
                batch_size = int(pairs_needed / success_rate * 1.5) if success_rate > 0 else pairs_needed * 3
            else:
                batch_size = initial_batch_size
            
            # Generate a batch of questions
            print(f"\n📝 Generating batch of {batch_size} questions (attempt {attempts + 1})...")
            questions: list[str] = self.generate_questions(trait_description, batch_size)
            questions_generated += len(questions)
            
            if not questions:
                print(f"⚠️ No questions generated in attempt {attempts + 1}")
                attempts += 1
                continue
                
            print(f"✅ Generated {len(questions)} unique questions")
            
            # Generate pairs in parallel
            batch_pairs = self._generate_contrastive_pairs_parallel(
                questions, trait_description, len(all_pairs), int(num_pairs * pair_overgeneration_factor)
            )
            all_pairs.extend(batch_pairs)
            
            if len(questions) > 0:
                success_rate_pct = (len(batch_pairs)/len(questions)*100) if len(questions) > 0 else 0
                print(f"📊 Batch results: {len(batch_pairs)}/{len(questions)} successful ({success_rate_pct:.1f}% success rate)")
            else:
                print(f"📊 Batch results: No questions to process")
            print(f"📊 Total progress: {len(all_pairs)}/{int(num_pairs * pair_overgeneration_factor)} pairs")
            
            attempts += 1
        
        if len(all_pairs) < num_pairs:
            print(f"⚠️ Warning: Only generated {len(all_pairs)} pairs, target was {num_pairs}")
        
        print(f"\n✅ Successfully generated {len(all_pairs)} raw contrastive pairs from {questions_generated} questions")
        if questions_generated > 0:
            print(f"📊 Overall success rate: {len(all_pairs)/questions_generated*100:.1f}%")
        else:
            print(f"📊 Overall success rate: No questions were generated")
        
        # Select diverse pairs
        diverse_pairs: list[ContrastivePair] = self._select_diverse_pairs(
            all_pairs, num_pairs
        )
        print(f"✅ Selected {len(diverse_pairs)} diverse pairs")

        # Generate name from trait description
        pair_set_name = f"synthetic_{trait_description[:30]}"
        pair_set: ContrastivePairSet = ContrastivePairSet(
            name=pair_set_name,
            task_type="synthetic",
            pairs=diverse_pairs,
        )

        print("🔍 Applying quality check to filter pairs...")
        return quality_check_synthetic_pairs(
            pair_set, trait_description, strict_mode=True
        )

    def _combine_and_update_set(
        self,
        existing_set: ContrastivePairSet,
        new_set: ContrastivePairSet,
        target_pair_count: int,
        trait_embedding: np.ndarray,
    ) -> ContrastivePairSet:
        """Combines an existing set with a new one, ensures diversity, and updates the database.
        Args:
            existing_set: The existing ContrastivePairSet from the database
            new_set: The newly generated ContrastivePairSet
            target_pair_count: The desired number of pairs in the final set
            trait_embedding: The embedding of the trait description for caching purposes
        Returns:
            ContrastivePairSet with combined and deduplicated pairs"""

        # Combine pairs and deduplicate based on prompt
        combined_pairs_map: dict[str, ContrastivePair] = {
            p.prompt: p for p in existing_set.pairs
        }
        combined_pairs_map.update({p.prompt: p for p in new_set.pairs})
        all_unique_pairs: list[ContrastivePair] = list(combined_pairs_map.values())

        print(f"🤝 Combined sets, resulting in {len(all_unique_pairs)} unique pairs.")

        # Select the most diverse subset
        final_diverse_pairs: list[ContrastivePair] = self._select_diverse_pairs(
            all_unique_pairs, target_pair_count
        )

        # Create the final, augmented pair set
        augmented_set = ContrastivePairSet(
            name=existing_set.name,
            task_type=existing_set.task_type,
            pairs=final_diverse_pairs,
        )

        # Add the new, larger set to the database. This creates a new entry.
        print(
            f"💾 Caching augmented pair set with {len(final_diverse_pairs)} pairs to database..."
        )
        self.database.add_set(augmented_set, trait_embedding)
        print(f"✅ Cached augmented set.")

        return augmented_set

    def save_to_json(self, pair_set: ContrastivePairSet, filepath: str) -> None:
        """Save contrastive pair set to JSON file.
        Args:
            pair_set: ContrastivePairSet to save
            filepath: Path to save the JSON file
        Returns:
            None
        """
        data: dict[str, Any] = {
            "name": pair_set.name,
            "task_type": pair_set.task_type,
            "pairs": [],
        }

        for pair in pair_set.pairs:
            pair_data: dict[str, Any] = {
                "prompt": pair.prompt,
                "positive_response": pair.positive_response.text,
                "negative_response": pair.negative_response.text,
            }
            data["pairs"].append(pair_data)

        filepath: Path = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(pair_set.pairs)} pairs to {filepath}")

    def load_from_json(self, filepath: str) -> ContrastivePairSet:
        """Load contrastive pair set from JSON file.
        Args:
            filepath: Path to JSON file

        Returns:
            ContrastivePairSet with loaded pairs
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        pair_set: ContrastivePairSet = ContrastivePairSet(
            name=data.get("name", "loaded_synthetic"),
            task_type=data.get("task_type", "synthetic"),
        )

        for pair_data in data["pairs"]:
            pair = ContrastivePair(
                prompt=pair_data["prompt"],
                positive_response=PositiveResponse(text=pair_data["positive_response"]),
                negative_response=NegativeResponse(text=pair_data["negative_response"]),
            )

            pair_set.pairs.append(pair)

        print(f"📂 Loaded {len(pair_set.pairs)} pairs from {filepath}")

        return pair_set

    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def generate_synthetic_pairs_cli(
    trait_description: str,
    num_pairs: int = 30,
    output_file: Optional[str] = None,
    model=None,
    force_regenerate: bool = False,
    verbose_timing: bool = False,
    max_workers: int = 4,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> ContrastivePairSet:
    """
    CLI function to generate synthetic contrastive pairs.

    Args:
        trait_description: Natural language description of desired trait
        num_pairs: Number of pairs to generate
        output_file: Optional file to save pairs to
        model: Model instance to use
        force_regenerate: If True, bypasses the cache and generates a new set.
        verbose_timing: If True, shows detailed timing for each step
        max_workers: Number of parallel workers for generation
        generation_kwargs: Optional dict of generation parameters to pass to the model (e.g., enable_thinking=False)

    Returns:
        Generated ContrastivePairSet
    """
    print("DEBUG: In generate_synthetic_pairs_cli")
    print(f"  trait_description: {trait_description}")
    print(f"  num_pairs: {num_pairs}")
    
    if model is None:
        raise ValueError("Model must be provided")

    generator: SyntheticContrastivePairGenerator = SyntheticContrastivePairGenerator(
        model, max_workers=max_workers, generation_kwargs=generation_kwargs
    )
    
    pair_set: ContrastivePairSet = generator.generate_contrastive_pair_set(
        trait_description=trait_description,
        num_pairs=num_pairs,
        force_regenerate=force_regenerate,
        verbose_timing=verbose_timing,
    )

    if output_file:
        generator.save_to_json(pair_set, output_file)

    return pair_set


def load_synthetic_pairs_cli(filepath: str, model=None) -> ContrastivePairSet:
    """
    CLI function to load synthetic contrastive pairs from JSON.

    Args:
        filepath: Path to JSON file
        model: Model instance (for compatibility)

    Returns:
        Loaded ContrastivePairSet
    """
    generator: SyntheticContrastivePairGenerator = SyntheticContrastivePairGenerator(
        model
    )
    return generator.load_from_json(filepath)
