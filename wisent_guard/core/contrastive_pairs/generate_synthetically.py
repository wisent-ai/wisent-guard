import json
import random
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from wisent_guard.core.model import Model

from ..response import NegativeResponse, PositiveResponse
from .contrastive_database import ContrastivePairDatabase
from .contrastive_pair import ContrastivePair
from .contrastive_pair_set import ContrastivePairSet
from .contrastive_generation_conf import CONTRASTIVE_GEN, QUESTION_GEN, QUESTION_PARSE
from .quality_check import quality_check_synthetic_pairs


class SyntheticContrastivePairGenerator:
    """Generate contrastive pairs synthetically from natural language trait descriptions."""

    def __init__(
        self,
        model: Model,
        similarity_threshold: float = 0.8,
        db_path: Optional[str] = None,
        db_similarity_threshold: float = 0.95,
    ):
        """
        Initialize the synthetic pair generator.

        Args:
            model: The language model to use for generation
            similarity_threshold: Threshold for deduplication of questions/pairs (0-1, higher = more strict)
            db_path: Optional path to the contrastive pair database. If None, caching is disabled.
            db_similarity_threshold: Threshold for retrieving a set from the database (0-1, higher = more strict).
        """
        self.model: Model = model
        self.similarity_threshold: float = similarity_threshold
        self.db_similarity_threshold: float = db_similarity_threshold

        self.similarity_model: Optional[SentenceTransformer] = (
            self._initialize_similarity_model()
        )
        self.database: Optional[ContrastivePairDatabase] = self._initialize_database(
            db_path
        )

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
        self, trait_description: str, num_questions: int
    ) -> list[str]:
        """
        Generate diverse questions where the trait would be relevant.

        Args:
            trait_description: Natural language description of desired trait
            num_questions: Number of questions to generate

        Returns:
            list of question descriptions
        """
        print(f"🎯 DEBUG: Generating questions for trait: '{trait_description}'")
        print(f"🎯 DEBUG: Target number of questions: {num_questions}")

        target_questions: int = num_questions * QUESTION_GEN.OVERGENERATION_FACTOR
        all_questions: list[str] = []

        print(
            f"🎯 DEBUG: Will generate {target_questions} total questions to select {num_questions} best ones"
        )

        num_prompts_per_template: int = target_questions // len(
            QUESTION_GEN.PROMPT_TEMPLATES
        )

        for i, template in enumerate(QUESTION_GEN.PROMPT_TEMPLATES):
            prompt: str = template.format(num_prompts=num_prompts_per_template)
            print(
                f"🎯 DEBUG: Using prompt template {i+1}/{len(QUESTION_GEN.PROMPT_TEMPLATES)}"
            )
            print(f"🎯 DEBUG: Template: {prompt[:100]}...")
            try:
                response: str
                response, _ = self.model.generate(prompt, **QUESTION_GEN.CONFIG)

                print(f"🎯 DEBUG: Generated response length: {len(response)} chars")
                print(f"🎯 DEBUG: Response preview: {response[:200]}...")

                # Parse questions from response
                questions: list[str] = self._parse_questions_from_response(response)
                print(f"🎯 DEBUG: Parsed {len(questions)} questions from this template")
                for j, question in enumerate(questions):
                    print(f"🎯 DEBUG:   Question {j+1}: {question[:100]}...")
                all_questions.extend(questions)

            except Exception as e:
                print(f"🎯 DEBUG: Error generating questions with template: {e}")
                continue

        print(f"🎯 DEBUG: Total questions before deduplication: {len(all_questions)}")

        # Deduplicate and select most diverse questions
        unique_questions: list[str] = self._deduplicate_questions(all_questions)
        print(
            f"🎯 DEBUG: Unique questions after deduplication: {len(unique_questions)}"
        )

        # Select the best diverse questions
        selected_questions: list[str] = self._select_diverse_questions(
            unique_questions, num_questions
        )
        print(f"🎯 DEBUG: Final selected questions: {len(selected_questions)}")

        for i, question in enumerate(selected_questions):
            print(f"🎯 DEBUG: Final question {i+1}: {question}")

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
        """Remove duplicate or very similar questions.
        Args:
            questions: The list of question strings to deduplicate
        Returns:
            A list of unique question strings
        """
        if not self.similarity_model or len(questions) <= 1:
            # Fallback to simple text-based deduplication
            return list(set(questions))

        unique_questions: list[str] = []

        for question in questions:
            is_duplicate: bool = False

            if unique_questions:
                # Check similarity with existing questions
                question_embedding: np.ndarray = self.similarity_model.encode(
                    [question]
                )
                existing_embeddings: np.ndarray = self.similarity_model.encode(
                    unique_questions
                )

                # Calculate cosine similarities
                similarities: np.ndarray = np.dot(
                    question_embedding, existing_embeddings.T
                )[0]

                if np.max(similarities) > self.similarity_threshold:
                    is_duplicate = True

            if not is_duplicate:
                unique_questions.append(question)

        return unique_questions

    def _select_diverse_questions(
        self, questions: list[str], target_count: int
    ) -> list[str]:
        """Select the most diverse questions up to target count.
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

        # Use embeddings to select diverse questions
        embeddings: np.ndarray = self.similarity_model.encode(questions)

        selected_indices: list[int] = [0]  # Start with first question

        for _ in range(target_count - 1):
            remaining_indices: list[int] = [
                i for i in range(len(questions)) if i not in selected_indices
            ]

            if not remaining_indices:
                break

            # Find question most different from already selected ones
            max_min_distance: float = -1.0
            best_idx: int = remaining_indices[0]

            for idx in remaining_indices:
                # Calculate minimum distance to any selected question
                distances: list[float] = []
                for selected_idx in selected_indices:
                    distance: float = 1 - np.dot(
                        embeddings[idx], embeddings[selected_idx]
                    )
                    distances.append(distance)

                min_distance: float = min(distances)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx

            selected_indices.append(best_idx)

        return [questions[i] for i in selected_indices]

    def _get_pair_embedding(self, pair: ContrastivePair) -> np.ndarray:
        """Computes a single embedding for a contrastive pair.
        Args:
            pair: The ContrastivePair object to compute embedding for
        Returns:
            A numpy array representing the embedding of the contrastive pair
        """
        if not self.similarity_model:
            return np.array([])

        # Embed all components of the pair
        prompt_emb: np.ndarray = self.similarity_model.encode([pair.prompt])[0]
        pos_emb: np.ndarray = self.similarity_model.encode(
            [pair.positive_response.text]
        )[0]
        neg_emb: np.ndarray = self.similarity_model.encode(
            [pair.negative_response.text]
        )[0]

        # Concatenate and normalize to get a single representative vector
        combined_emb: np.ndarray = np.concatenate([prompt_emb, pos_emb, neg_emb])
        norm = np.linalg.norm(combined_emb)
        return combined_emb / norm if norm != 0 else combined_emb

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

        embeddings: np.ndarray = np.array([self._get_pair_embedding(p) for p in pairs])

        selected_indices: list[int] = [0]
        for _ in range(target_count - 1):
            remaining_indices: list[int] = [
                i for i in range(len(pairs)) if i not in selected_indices
            ]
            if not remaining_indices:
                break

            # Use vectorized operations for faster distance calculation
            selected_embeddings: np.ndarray = embeddings[selected_indices]
            remaining_embeddings: np.ndarray = embeddings[remaining_indices]

            # Calculate cosine similarity, then convert to distance
            dist_matrix: np.ndarray = 1 - np.dot(
                remaining_embeddings, selected_embeddings.T
            )

            # Find the minimum distance from each remaining point to any selected point
            min_distances: np.ndarray = np.min(dist_matrix, axis=1)

            # Find the point that has the maximum minimum-distance
            max_dist_idx: int = np.argmax(min_distances)

            selected_indices.append(remaining_indices[max_dist_idx])

        return [pairs[i] for i in selected_indices]

    def _generate_response(self, prompt: str, config: dict[str, Any]) -> str:
        """Generates a response from the model using the provided configuration.
        Args:
            prompt: The input prompt for the model
            config: Configuration parameters for the model generation
        Returns
            response: The generated response text
        """
        response: str
        response, _ = self.model.generate(prompt, **config)
        return response.strip()

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
            
            print(f"\n📝 Generating batch of {batch_size} questions (attempt {attempts + 1})...")
            questions = self.generate_questions(trait_description, batch_size)
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
            print(f"📊 Overall success rate: {len(successful_pairs)/questions_used*100:.1f}%")
        
        return successful_pairs

    def generate_contrastive_pair_set(
        self,
        trait_description: str,
        num_pairs: int = 30,
        pair_overgeneration_factor: float = 1.5,
        force_regenerate: bool = False,
    ) -> ContrastivePairSet:
        """
        Generate a complete contrastive pair set from a trait description.
        Checks a vector database first to avoid re-generation if available.

        Args:
            trait_description: Natural language description of desired trait
            num_pairs: Number of contrastive pairs to generate
            pair_overgeneration_factor: Factor to overgenerate pairs for diversity selection
            force_regenerate: If True, bypasses the cache and generates a new set.

        Returns:
            ContrastivePairSet with generated pairs
        """
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
            print(f"✅ Generated {len(questions)} unique questions")
            
            # Try to create pairs from the questions
            print("🔄 Generating contrastive pairs from questions...")
            batch_pairs = []
            for i, question in enumerate(questions):
                try:
                    print(f"   Generating pair {i+1}/{len(questions)}: {question[:50]}...")
                    pair: ContrastivePair = self.generate_contrastive_pair(
                        question, trait_description
                    )
                    batch_pairs.append(pair)
                    all_pairs.append(pair)
                    
                    # Check if we have enough pairs
                    if len(all_pairs) >= int(num_pairs * pair_overgeneration_factor):
                        print(f"✅ Reached target of {int(num_pairs * pair_overgeneration_factor)} pairs for selection")
                        break
                except Exception as e:
                    print(f"   ⚠️ Error generating pair for question '{question[:50]}': {e}")
                    continue
            
            print(f"📊 Batch results: {len(batch_pairs)}/{len(questions)} successful ({len(batch_pairs)/len(questions)*100:.1f}% success rate)")
            print(f"📊 Total progress: {len(all_pairs)}/{int(num_pairs * pair_overgeneration_factor)} pairs")
            
            attempts += 1
        
        if len(all_pairs) < num_pairs:
            print(f"⚠️ Warning: Only generated {len(all_pairs)} pairs, target was {num_pairs}")
        
        print(f"\n✅ Successfully generated {len(all_pairs)} raw contrastive pairs from {questions_generated} questions")
        print(f"📊 Overall success rate: {len(all_pairs)/questions_generated*100:.1f}%")
        
        # Select diverse pairs
        diverse_pairs: list[ContrastivePair] = self._select_diverse_pairs(
            all_pairs, num_pairs
        )
        print(f"✅ Selected {len(diverse_pairs)} diverse pairs")

        pair_set: ContrastivePairSet = ContrastivePairSet(
            name=f"synthetic_{trait_description[:30]}",
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


def generate_synthetic_pairs_cli(
    trait_description: str,
    num_pairs: int = 30,
    output_file: Optional[str] = None,
    model=None,
    force_regenerate: bool = False,
) -> ContrastivePairSet:
    """
    CLI function to generate synthetic contrastive pairs.

    Args:
        trait_description: Natural language description of desired trait
        num_pairs: Number of pairs to generate
        output_file: Optional file to save pairs to
        model: Model instance to use
        force_regenerate: If True, bypasses the cache and generates a new set.

    Returns:
        Generated ContrastivePairSet
    """
    if model is None:
        raise ValueError("Model must be provided")

    generator: SyntheticContrastivePairGenerator = SyntheticContrastivePairGenerator(
        model
    )

    pair_set: ContrastivePairSet = generator.generate_contrastive_pair_set(
        trait_description=trait_description,
        num_pairs=num_pairs,
        name=f"synthetic_{trait_description.replace(' ', '_')[:20]}",
        force_regenerate=force_regenerate,
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
