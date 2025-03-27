"""
Main ActivationGuard class for the wisent-guard package
"""

import os
import torch
from typing import List, Dict, Tuple, Any, Optional, Union, Set
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .vectors import ContrastiveVectors
from .monitor import ActivationMonitor
from .inference import SafeInference
from .utils.helpers import ensure_dir
from .utils.logger import get_logger

class ActivationGuard:
    """
    Main class for monitoring and guarding against harmful content using activation vectors.
    
    This class provides a high-level API for creating contrastive vectors from
    harmful/harmless phrase pairs, monitoring model activations, and blocking harmful
    content during generation.
    """
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        layers: Optional[List[int]] = None,
        threshold: float = 0.7,
        save_dir: str = "./wisent_guard_data",
        device: Optional[str] = None,
        token_strategy: str = "target_token",  # Default to target_token for A/B tokens
        log_level: str = "info",
        log_file: Optional[str] = None,
    ):
        """
        Initialize the ActivationGuard.
        
        Args:
            model: Model to guard or model name to load
            tokenizer: Tokenizer for the model or tokenizer name to load
            layers: Specific layers to monitor. If None, will use all available layers.
            threshold: Similarity threshold for identifying harmful content
            save_dir: Directory for saving and loading vectors
            device: Device to run the model on (e.g., 'cuda', 'cpu')
            token_strategy: Strategy for token selection:
                           - "target_token": Look for specific tokens like "A" or "B" (default)
                           - "last": Last token in sequence (legacy)
                           - "all": Store all tokens for later selection
            log_level: Logging level ('debug', 'info', 'warning', 'error')
            log_file: Optional file to write logs to
        """
        # Set up logger
        self.logger = get_logger(level=log_level, log_file=log_file)
        self.logger.info("Initializing ActivationGuard")
        
        # Set up directories
        self.save_dir = save_dir
        ensure_dir(self.save_dir)
        self.logger.info(f"Using save directory: {self.save_dir}")
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info("Using CUDA GPU")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                self.logger.info("Using Apple Silicon GPU via MPS")
            else:
                self.device = "cpu"
                self.logger.info("Using CPU")
        else:
            self.device = device
            self.logger.info(f"Using user-specified device: {self.device}")
        
        # Set token strategy - default to target_token for multiple-choice
        self.token_strategy = token_strategy
        self.logger.info(f"Using token strategy: {self.token_strategy}")
        
        # Load model and tokenizer if strings are provided
        if isinstance(model, str):
            self.logger.info(f"Loading model from {model}...")
            self.model = AutoModelForCausalLM.from_pretrained(model)
            self.model.to(self.device)
        else:
            self.model = model
            self.logger.info("Using provided model")
            
        if tokenizer is None and isinstance(model, str):
            self.logger.info(f"Loading tokenizer from {model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        elif isinstance(tokenizer, str):
            self.logger.info(f"Loading tokenizer from {tokenizer}...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
            self.logger.info("Using provided tokenizer")
            
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.logger.info("Setting padding token to EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set up layers to monitor
        self.layers = layers if layers is not None else list(range(min(12, self.model.config.num_hidden_layers)))
        self.logger.info(f"Monitoring layers: {self.layers}")
        
        # Initialize vectors, monitor, and inference
        self.vectors = ContrastiveVectors(save_dir=self.save_dir)
        self.monitor = None
        self.inference = None
        self.threshold = threshold
        self.logger.info(f"Similarity threshold set to {self.threshold}")
        
        # Try to load existing vectors
        if self.vectors.load_vectors():
            self.logger.info(f"Loaded existing vectors from {self.save_dir}")
            # Initialize monitor and inference with loaded vectors
            self._initialize_monitor_and_inference()
        else:
            self.logger.info("No existing vectors found")
        
        # Set target tokens for multiple-choice format
        self._setup_target_tokens()
    
    def _setup_target_tokens(self):
        """Set up target tokens for multiple-choice format (A and B)."""
        if self.monitor is None:
            # We'll set this up when the monitor is initialized
            return
        
        target_tokens = ["A", "B"]
        self.monitor.hooks.set_target_tokens(self.tokenizer, target_tokens)
        self.logger.info(f"Multiple-choice tokens set: {target_tokens}")
    
    def _initialize_monitor_and_inference(self):
        """Initialize monitor and inference components with current vectors."""
        self.logger.debug("Initializing monitor and inference components")
        
        # Always use target_token strategy internally for token selection
        actual_token_strategy = "target_token"
        if self.token_strategy != "target_token":
            self.logger.info(f"Note: Using 'target_token' strategy internally instead of '{self.token_strategy}' for more consistent results")
        
        self.monitor = ActivationMonitor(
            model=self.model,
            vectors=self.vectors,
            layers=self.layers,
            threshold=self.threshold,
            token_strategy=actual_token_strategy,
        )
        
        self.inference = SafeInference(
            model=self.model,
            tokenizer=self.tokenizer,
            monitor=self.monitor
        )
        
        # Set up target tokens for multiple-choice format
        self._setup_target_tokens()
        
        self.logger.debug("Monitor and inference components initialized")
    
    def _convert_to_multiple_choice(self, harmful_phrase: str, harmless_phrase: str) -> Tuple[str, str]:
        """
        Convert a harmful/harmless phrase pair to multiple-choice format.
        
        Args:
            harmful_phrase: The harmful phrase
            harmless_phrase: The harmless phrase
            
        Returns:
            Tuple of (harmful_mc, harmless_mc) in multiple-choice format
        """
        self.logger.debug("Converting to multiple-choice format")
        
        # Create question from the phrases
        question = "Which of the following statements is true?"
        
        # Format as multiple-choice with A as harmless and B as harmful
        harmless_mc = self._format_multiple_choice(
            question,
            harmless_phrase,  # A is harmless/truthful
            harmful_phrase,   # B is harmful/false
            "A"  # Correct answer is A
        )
        
        harmful_mc = self._format_multiple_choice(
            question,
            harmless_phrase,  # A is harmless/truthful
            harmful_phrase,   # B is harmful/false
            "B"  # Harmful answer is B
        )
        
        self.logger.debug("Multiple-choice conversion complete")
        return harmful_mc, harmless_mc
    
    def train_on_phrase_pairs(self, phrase_pairs: List[Dict[str, str]], category: str = "harmful_content") -> None:
        """
        Train the guard on pairs of harmful and harmless phrases.
        Internally converts all phrases to multiple-choice format for more consistent activation collection.
        
        Args:
            phrase_pairs: List of dictionaries with 'harmful' and 'harmless' keys
            category: Category label for the vector pairs
        """
        self.logger.info(f"Training on {len(phrase_pairs)} phrase pairs for category '{category}'...")
        self.logger.info("Converting phrase pairs to multiple-choice format for consistent activation collection...")
        
        # Make sure we have a monitor initialized
        if self.monitor is None:
            self.logger.debug("Initializing monitor for the first time")
            self._initialize_monitor_and_inference()
        
        # Convert phrase pairs to multiple-choice and process each pair
        for i, pair in enumerate(tqdm(phrase_pairs, desc="Processing phrase pairs")):
            harmful_phrase = pair["harmful"]
            harmless_phrase = pair["harmless"]
            
            self.logger.debug(f"Processing pair {i+1}/{len(phrase_pairs)}")
            self.logger.debug(f"Harmful: {harmful_phrase[:50]}..." if len(harmful_phrase) > 50 else f"Harmful: {harmful_phrase}")
            self.logger.debug(f"Harmless: {harmless_phrase[:50]}..." if len(harmless_phrase) > 50 else f"Harmless: {harmless_phrase}")
            
            # Convert to multiple-choice format
            harmful_mc, harmless_mc = self._convert_to_multiple_choice(harmful_phrase, harmless_phrase)
            
            # Get activations for harmful phrase in multiple-choice format
            self.logger.debug("Collecting activations for harmful phrase")
            self.monitor.reset()
            harmful_input_ids = self.tokenizer.encode(harmful_mc, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(harmful_input_ids)
            harmful_activations = self.monitor.hooks.get_activations()
            self.logger.debug(f"Collected activations from {len(harmful_activations)} layers for harmful phrase")
            
            # Get activations for harmless phrase in multiple-choice format
            self.logger.debug("Collecting activations for harmless phrase")
            self.monitor.reset()
            harmless_input_ids = self.tokenizer.encode(harmless_mc, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(harmless_input_ids)
            harmless_activations = self.monitor.hooks.get_activations()
            self.logger.debug(f"Collected activations from {len(harmless_activations)} layers for harmless phrase")
            
            # Move activations to CPU for consistent storage (especially important for MPS)
            self.logger.debug("Moving activations to CPU for consistent storage")
            cpu_harmful_activations = {}
            cpu_harmless_activations = {}
            
            for layer, tensor in harmful_activations.items():
                cpu_harmful_activations[layer] = tensor.detach().cpu()
            
            for layer, tensor in harmless_activations.items():
                cpu_harmless_activations[layer] = tensor.detach().cpu()
            
            # Store activations for each layer
            self.logger.debug("Storing activations for each layer")
            for layer in self.layers:
                if layer in cpu_harmful_activations and layer in cpu_harmless_activations:
                    self.vectors.add_vector_pair(
                        category=category,
                        layer=layer,
                        harmful_vector=cpu_harmful_activations[layer],
                        harmless_vector=cpu_harmless_activations[layer]
                    )
                    self.logger.debug(f"Stored vector pair for layer {layer}")
                else:
                    self.logger.warning(f"Missing activations for layer {layer}")
            
            self.logger.debug(f"Completed processing pair {i+1}/{len(phrase_pairs)}")
        
        # Compute and save contrastive vectors
        self.logger.info("Computing contrastive vectors...")
        self.vectors.compute_contrastive_vectors()
        self.logger.info("Saving vectors to disk...")
        self.vectors.save_vectors()
        
        # Re-initialize monitor with new vectors
        self.logger.info("Reinitializing monitor with new vectors")
        self._initialize_monitor_and_inference()
        
        self.logger.info(f"Successfully trained on {len(phrase_pairs)} phrase pairs")
    
    def train_on_multiple_choice_pairs(self, questions: List[Dict[str, Any]], category: str = "hallucination") -> None:
        """
        Train the guard on multiple-choice pairs where A is correct and B is incorrect.
        
        Args:
            questions: List of dictionaries with question data
            category: Category label for the vector pairs
        
        Example question format:
        {
            "question": "What is the capital of France?",
            "choice_a": "Paris is the capital of France.",
            "choice_b": "London is the capital of France."
        }
        """
        print(f"Training on {len(questions)} multiple-choice questions for category '{category}'...")
        
        # Prepare phrase pairs in multiple-choice format
        phrase_pairs = []
        for q in questions:
            # Create A (correct) and B (incorrect) response phrases
            a_phrase = self._format_multiple_choice(
                q["question"], 
                q["choice_a"], 
                q["choice_b"], 
                "A"
            )
            
            b_phrase = self._format_multiple_choice(
                q["question"], 
                q["choice_a"], 
                q["choice_b"], 
                "B"
            )
            
            phrase_pairs.append({
                "harmless": a_phrase,  # A is correct/harmless
                "harmful": b_phrase    # B is incorrect/harmful
            })
        
        # Use our internal method to train
        self._train_on_formatted_pairs(phrase_pairs, category=category)
    
    def _train_on_formatted_pairs(self, formatted_pairs: List[Dict[str, str]], category: str) -> None:
        """
        Train on already formatted phrase pairs (internal method).
        
        Args:
            formatted_pairs: List of dictionaries with 'harmful' and 'harmless' keys already formatted
            category: Category label for the vector pairs
        """
        # Make sure we have a monitor initialized
        if self.monitor is None:
            self._initialize_monitor_and_inference()
        
        # Process each phrase pair
        for pair in tqdm(formatted_pairs, desc="Processing formatted pairs"):
            harmful_phrase = pair["harmful"]
            harmless_phrase = pair["harmless"]
            
            # Get activations for harmful phrase
            self.monitor.reset()
            harmful_input_ids = self.tokenizer.encode(harmful_phrase, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(harmful_input_ids)
            harmful_activations = self.monitor.hooks.get_activations()
            
            # Get activations for harmless phrase
            self.monitor.reset()
            harmless_input_ids = self.tokenizer.encode(harmless_phrase, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(harmless_input_ids)
            harmless_activations = self.monitor.hooks.get_activations()
            
            # Store activations for each layer
            for layer in self.layers:
                if layer in harmful_activations and layer in harmless_activations:
                    self.vectors.add_vector_pair(
                        category=category,
                        layer=layer,
                        harmful_vector=harmful_activations[layer],
                        harmless_vector=harmless_activations[layer]
                    )
        
        # Compute and save contrastive vectors
        self.vectors.compute_contrastive_vectors()
        self.vectors.save_vectors()
        
        # Re-initialize monitor with new vectors
        self._initialize_monitor_and_inference()
    
    def _format_multiple_choice(self, question: str, choice_a: str, choice_b: str, answer: str) -> str:
        """
        Format a multiple-choice question with the given answer.
        
        Args:
            question: The question text
            choice_a: The text for option A
            choice_b: The text for option B
            answer: The answer ("A" or "B")
            
        Returns:
            Formatted text with the answer
        """
        return f"<|user|>\n{question}\nA. {choice_a}\nB. {choice_b}\n<|assistant|>\n{answer}"
    
    def _format_input_as_multiple_choice(self, text: str) -> str:
        """
        If input is not already in multiple-choice format, try to convert it.
        
        Args:
            text: Input text
            
        Returns:
            Multiple-choice formatted text if possible, or original text
        """
        # Check if text already appears to be in multiple-choice format
        if "<|assistant|>\nA" in text or "<|assistant|>\nB" in text:
            return text
        
        # For detection, add a temporary A answer
        if "<|user|>" in text and "<|assistant|>" not in text:
            return f"{text}\n<|assistant|>\nA"
        
        # Can't easily convert, return as is
        return text
    
    def is_harmful(self, text: str, categories: Optional[List[str]] = None) -> bool:
        """
        Check if text contains harmful content.
        
        Args:
            text: Text to check
            categories: Specific categories to check against
            
        Returns:
            True if harmful content is detected, False otherwise
        """
        if self.monitor is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
        
        # Format as multiple-choice if needed
        text = self._format_input_as_multiple_choice(text)
        
        # Reset monitor
        self.monitor.reset()
        
        # Encode text
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Run forward pass to get activations
        with torch.no_grad():
            self.model(input_ids)
        
        # Check if harmful
        return self.monitor.is_harmful(categories=categories)
    
    def generate_safe_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        skip_prompt_check: bool = False,
        use_multiple_choice: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response while monitoring for harmful content.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            skip_prompt_check: Whether to skip the initial prompt safety check
            use_multiple_choice: Whether to use multiple-choice format internally
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated text and safety information
        """
        if self.inference is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
        
        if use_multiple_choice:
            # Create a simple multiple-choice format with the prompt
            choice_a = "I'll provide a helpful and accurate response."
            choice_b = "I'll provide a response with inaccurate or harmful information."
            question = f"{prompt}\nSelect one:"
            
            result = self.generate_multiple_choice_response(
                question=question,
                choice_a=choice_a,
                choice_b=choice_b,
                max_new_tokens=1,  # Just need A or B
                **kwargs
            )
            
            # If not blocked and response contains A, generate a real response
            if not result.get("blocked", False) and "A" in result.get("response", ""):
                # Generate the actual response with standard function
                return self.inference.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    skip_prompt_check=True,  # Already checked in multiple-choice
                    **kwargs
                )
            elif not result.get("blocked", False):
                # Model chose B or something else, block it
                return {
                    "response": "I cannot provide that response as it might contain inaccurate information.",
                    "blocked": True,
                    "reason": "Model showed intent to provide problematic content"
                }
            else:
                # Already blocked
                return result
        else:
            # Use standard generation
            return self.inference.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                skip_prompt_check=skip_prompt_check,
                **kwargs
            )
    
    def generate_multiple_choice_response(
        self,
        question: str,
        choice_a: str,
        choice_b: str,
        max_new_tokens: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a multiple-choice response (A or B) while monitoring for harmful content.
        
        Args:
            question: The question text
            choice_a: The text for option A
            choice_b: The text for option B
            max_new_tokens: Maximum number of tokens to generate (default 1 for A/B)
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated response (A or B) and safety information
        """
        if self.inference is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
        
        # Format as multiple-choice
        prompt = f"{question}\nA. {choice_a}\nB. {choice_b}"
        
        # Generate with limited tokens (just enough for A or B)
        result = self.inference.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        return result
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available categories.
        
        Returns:
            List of category names
        """
        return self.vectors.get_available_categories()
    
    def get_similarity(self, text: str, categories: Optional[List[str]] = None) -> float:
        """
        Get the similarity of text to harmful patterns.
        
        Args:
            text: Text to check
            categories: Specific categories to check against
            
        Returns:
            Highest similarity score
        """
        if self.monitor is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
        
        # Format as multiple-choice if needed
        text = self._format_input_as_multiple_choice(text)
        
        # Reset monitor
        self.monitor.reset()
        
        # Encode text
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Run forward pass to get activations
        with torch.no_grad():
            self.model(input_ids)
        
        # Get similarity scores
        results = self.monitor.check_activations(categories=categories)
        
        # Find the highest similarity score
        max_similarity = 0.0
        for category_result in results.values():
            max_similarity = max(max_similarity, category_result["max_similarity"])
        
        return max_similarity
    
    def get_triggered_category(self, text: str) -> Optional[str]:
        """
        Get the category that was triggered by the text.
        
        Args:
            text: Text to check
            
        Returns:
            Category name or None if no category was triggered
        """
        if self.monitor is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
        
        # Format as multiple-choice if needed
        text = self._format_input_as_multiple_choice(text)
        
        # Reset monitor
        self.monitor.reset()
        
        # Encode text
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Run forward pass to get activations
        with torch.no_grad():
            self.model(input_ids)
        
        # Get the most harmful category
        category_info = self.monitor.get_most_harmful_category()
        if category_info is not None:
            return category_info[0]
        
        return None
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set the similarity threshold for harmful content detection.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.threshold = threshold
        
        if self.monitor is not None:
            self.monitor.threshold = threshold
    
    def get_contrastive_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the contrastive vectors and their properties.
        
        This method provides an overview of all vectors, training data, and settings
        to help understand the complete contrastive vector flow.
        
        Returns:
            Dictionary with summary information
        """
        self.logger.info("Generating contrastive vector summary")
        
        summary = {
            "settings": {
                "layers": self.layers,
                "device": self.device,
                "threshold": self.threshold,
                "token_strategy": self.token_strategy,
                "save_dir": self.save_dir
            },
            "vectors": {},
            "categories": self.vectors.get_available_categories()
        }
        
        # Get category statistics
        for category in summary["categories"]:
            self.logger.debug(f"Collecting summary for category: {category}")
            category_info = {
                "num_pairs": self.vectors.metadata["num_pairs"].get(category, 0),
                "layers": [],
                "sample_similarities": {}
            }
            
            # Collect layer information
            available_layers = set()
            for layer in self.layers:
                vector = self.vectors.get_contrastive_vector(category, layer)
                if vector is not None:
                    self.logger.debug(f"Found vector for layer {layer}")
                    available_layers.add(layer)
                    category_info["layers"].append({
                        "layer": layer,
                        "shape": list(vector.shape),
                        "norm": float(torch.norm(vector).item())
                    })
            
            # Calculate some sample similarities if we have a monitor
            if self.monitor is not None and available_layers:
                self.logger.debug("Calculating sample similarities")
                
                # Generate a few test cases (positive and negative)
                positive_case = f"I'll provide information on how to engage in {category}"
                negative_case = f"I'll provide information on how to prevent {category}"
                
                # Get similarities
                self.monitor.reset()
                positive_ids = self.tokenizer.encode(positive_case, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    self.model(positive_ids)
                pos_results = self.monitor.check_activations([category])
                
                self.monitor.reset()
                negative_ids = self.tokenizer.encode(negative_case, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    self.model(negative_ids)
                neg_results = self.monitor.check_activations([category])
                
                # Add to summary
                category_info["sample_similarities"]["positive_case"] = {
                    "text": positive_case,
                    "max_similarity": float(pos_results[category]["max_similarity"]),
                    "is_harmful": pos_results[category]["is_harmful"],
                    "layer_similarities": {k: float(v) for k, v in pos_results[category]["layer_similarities"].items()}
                }
                
                category_info["sample_similarities"]["negative_case"] = {
                    "text": negative_case,
                    "max_similarity": float(neg_results[category]["max_similarity"]),
                    "is_harmful": neg_results[category]["is_harmful"],
                    "layer_similarities": {k: float(v) for k, v in neg_results[category]["layer_similarities"].items()}
                }
            
            summary["vectors"][category] = category_info
        
        self.logger.info("Summary generation complete")
        return summary
    
    def print_contrastive_summary(self) -> None:
        """
        Print a formatted summary of the contrastive vectors to the console.
        
        This method generates a human-readable report of all vectors and their properties.
        """
        summary = self.get_contrastive_summary()
        
        print("\n" + "="*80)
        print(" "*30 + "CONTRASTIVE VECTOR SUMMARY")
        print("="*80)
        
        # Print settings
        print("\nSETTINGS:")
        print(f"  Device: {summary['settings']['device']}")
        print(f"  Layers: {summary['settings']['layers']}")
        print(f"  Threshold: {summary['settings']['threshold']}")
        print(f"  Token Strategy: {summary['settings']['token_strategy']}")
        print(f"  Save Directory: {summary['settings']['save_dir']}")
        
        # Print categories
        print(f"\nCATEGORIES ({len(summary['categories'])}):")
        for category in summary["categories"]:
            cat_info = summary["vectors"][category]
            print(f"\n  • {category.upper()} ({cat_info['num_pairs']} pairs)")
            
            # Print layers
            if cat_info["layers"]:
                print("    Layers:")
                for layer_info in cat_info["layers"]:
                    print(f"      - Layer {layer_info['layer']}: shape={layer_info['shape']}, norm={layer_info['norm']:.4f}")
            else:
                print("    No layer vectors found")
            
            # Print sample similarities
            if "sample_similarities" in cat_info and cat_info["sample_similarities"]:
                print("    Sample Similarities:")
                
                pos_info = cat_info["sample_similarities"]["positive_case"]
                pos_max = pos_info["max_similarity"]
                is_harmful = "YES" if pos_info["is_harmful"] else "NO"
                print(f"      - Positive: '{pos_info['text']}'")
                print(f"        Max Similarity: {pos_max:.4f} (Harmful: {is_harmful})")
                
                neg_info = cat_info["sample_similarities"]["negative_case"]
                neg_max = neg_info["max_similarity"]
                is_harmful = "YES" if neg_info["is_harmful"] else "NO"
                print(f"      - Negative: '{neg_info['text']}'")
                print(f"        Max Similarity: {neg_max:.4f} (Harmful: {is_harmful})")
        
        print("\n" + "="*80) 