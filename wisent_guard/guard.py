"""
Main ActivationGuard class for the wisent-guard package
"""

import os
import torch
import re
from typing import List, Dict, Tuple, Any, Optional, Union, Set
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .vectors import ContrastiveVectors
from .monitor import ActivationMonitor
from .inference import SafeInference
from .utils.helpers import ensure_dir
from .utils.logger import get_logger

# Default tokens that can be overridden via environment variables
DEFAULT_USER_TOKEN = "<|user|>"
DEFAULT_ASSISTANT_TOKEN = "<|assistant|>"

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
        auto_load_vectors: bool = False,  # Add parameter to control automatic vector loading
        user_token: Optional[str] = None,
        assistant_token: Optional[str] = None,
        force_format: Optional[str] = None,
        force_llama_format: Optional[bool] = None  # For backward compatibility
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
            auto_load_vectors: Whether to automatically load all vectors from save_dir (default: False)
            user_token: Custom user token override (default: from WISENT_USER_TOKEN env var or "<|user|>")
            assistant_token: Custom assistant token override (default: from WISENT_ASSISTANT_TOKEN env var or "<|assistant|>")
            force_format: Force specific format: "llama31", "mistral", "legacy", or None for auto-detect
            force_llama_format: (Deprecated) For backward compatibility
        """
        # Set up logger
        self.logger = get_logger(level=log_level, log_file=log_file)
        self.logger.info("Initializing ActivationGuard")
        
        # Get user/assistant tokens from parameters, env vars, or defaults
        self.user_token = user_token or os.environ.get("WISENT_USER_TOKEN", DEFAULT_USER_TOKEN)
        self.assistant_token = assistant_token or os.environ.get("WISENT_ASSISTANT_TOKEN", DEFAULT_ASSISTANT_TOKEN)
        
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
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
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
        
        # Handle force_llama_format for backward compatibility
        if force_llama_format is not None:
            if force_llama_format is True:
                force_format = "llama31"
            elif force_llama_format is False:
                force_format = "legacy"
            self.logger.warning("force_llama_format is deprecated, use force_format instead")
        
        # Convert boolean force_format to string for backward compatibility
        if force_format is True:
            self.force_format = "llama31"
        elif force_format is False:
            self.force_format = "legacy"
        else:
            self.force_format = force_format
        self.logger.info(f"Similarity threshold set to {self.threshold}")
        
        # Check if model is likely a Llama 3.1 model and set format accordingly
        model_name = getattr(self.model.config, "_name_or_path", "").lower()
        is_llama_3 = bool(re.search(r"llama-?3", model_name, re.IGNORECASE))
        is_mistral = bool(re.search(r"mistral", model_name, re.IGNORECASE))
        
        self.logger.debug(f"Model name: {model_name}")
        self.logger.debug(f"Llama 3 detection: {is_llama_3}")
        self.logger.debug(f"Mistral detection: {is_mistral}")
        
        if is_llama_3:
            if self.force_format == "legacy":
                self.logger.warning("Detected Llama 3.1 model but legacy format is forced. This may cause issues.")
            elif self.force_format is None:
                self.force_format = "llama31"
                self.logger.info("Detected Llama 3.1 model. Automatically enabling Llama 3.1 prompt format.")
            elif self.force_format == "llama31":
                self.logger.info("Detected Llama 3.1 model. Will use Llama 3.1 prompt format.")
                
            if self.force_format == "llama31":
                self.logger.info("Llama 3.1 format will use special tokens:")
                self.logger.info("  <|begin_of_text|><|start_header_id|>user<|end_header_id|>...")
                self.logger.info("  Note: User/assistant token settings don't affect Llama 3.1 format")
        elif is_mistral:
            if self.force_format == "legacy":
                self.logger.warning("Detected Mistral model but legacy format is forced. This may cause issues.")
            elif self.force_format is None:
                self.force_format = "mistral"
                self.logger.info("Detected Mistral model. Automatically enabling Mistral prompt format.")
            elif self.force_format == "mistral":
                self.logger.info("Detected Mistral model. Will use Mistral prompt format.")
                
            if self.force_format == "mistral":
                self.logger.info("Mistral format will use special tokens:")
                self.logger.info("  [INST] instruction [/INST] response")
                self.logger.info("  Note: User/assistant token settings don't affect Mistral format")
        else:
            self.logger.info(f"Using legacy format with user token: {self.user_token}")
            self.logger.info(f"Using legacy format with assistant token: {self.assistant_token}")
        
        # Only load vectors if auto_load_vectors is True
        if auto_load_vectors:
            self.logger.info("Auto-loading vectors from save directory")
            self.load_vectors()
        else:
            self.logger.info("Skipping auto-loading of vectors (use load_vectors() to load explicitly)")
        
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
            monitor=self.monitor,
            user_token=self.user_token,
            assistant_token=self.assistant_token,
            force_format=self.force_format
        )
        
        # Set up target tokens for multiple-choice format
        self._setup_target_tokens()
        
        self.logger.debug("Monitor and inference components initialized")
    
    def _format_prompt(self, instruction: str, response: str = None) -> str:
        """
        Format a prompt using the appropriate format (Llama 3.1 or legacy)
        """
        # Check which format to use
        if self.force_format == "llama31":
            # Llama 3.1 special tokens
            BEGIN_TEXT = "<|begin_of_text|>"
            START_HEADER = "<|start_header_id|>"
            END_HEADER = "<|end_header_id|>"
            EOT = "<|eot_id|>"
            
            if response is not None:
                # Format with both instruction and response
                return f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{instruction}{EOT}{START_HEADER}assistant{END_HEADER}\n{response}"
            else:
                # Format with just instruction
                return f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{instruction}{EOT}{START_HEADER}assistant{END_HEADER}"
        elif self.force_format == "mistral":
            # Mistral format tokens
            INST_START = "[INST]"
            INST_END = "[/INST]"
            
            if response is not None:
                # Format with both instruction and response
                return f"{INST_START} {instruction} {INST_END} {response}"
            else:
                # Format with just instruction
                return f"{INST_START} {instruction} {INST_END}"
        else:
            # Legacy format
            if response is not None:
                return f"{self.user_token}\n{instruction}\n{self.assistant_token}\n{response}"
            else:
                return f"{self.user_token}\n{instruction}\n{self.assistant_token}"
    
    def _format_multiple_choice(self, question: str, choice_a: str, choice_b: str, answer: str) -> str:
        """Format a multiple-choice question with the given answer."""
        instruction = f"{question}\nA. {choice_a}\nB. {choice_b}"
        return self._format_prompt(instruction, answer)
    
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
        question = "Which of the following statements is better?"
        
        # Format as multiple-choice with A as harmless and B as harmful
        # Option A = harmless, Option B = harmful
        harmful_mc = self._format_multiple_choice(
            question=question,
            choice_a=harmless_phrase,
            choice_b=harmful_phrase,
            answer="B"  # Harmful choice (B) is correct for harmful_mc
        )
        
        # Option A = harmless, Option B = harmful
        harmless_mc = self._format_multiple_choice(
            question=question,
            choice_a=harmless_phrase,
            choice_b=harmful_phrase,
            answer="A"  # Harmless choice (A) is correct for harmless_mc
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
            
            # Add example logging to show what we're processing
            if i == 0:  # Only log the first example to avoid spam
                print(f"\n==== EXAMPLE TRAINING PAIR ====")
                print(f"HARMFUL: {harmful_phrase[:100]}..." if len(harmful_phrase) > 100 else f"HARMFUL: {harmful_phrase}")
                print(f"HARMLESS: {harmless_phrase[:100]}..." if len(harmless_phrase) > 100 else f"HARMLESS: {harmless_phrase}")
            
            self.logger.debug(f"Processing pair {i+1}/{len(phrase_pairs)}")
            self.logger.debug(f"Harmful: {harmful_phrase[:50]}..." if len(harmful_phrase) > 50 else f"Harmful: {harmful_phrase}")
            self.logger.debug(f"Harmless: {harmless_phrase[:50]}..." if len(harmless_phrase) > 50 else f"Harmless: {harmless_phrase}")
            
            # Convert to multiple-choice format
            harmful_mc, harmless_mc = self._convert_to_multiple_choice(harmful_phrase, harmless_phrase)
            
            # Log the converted format
            if i == 0:  # Only log the first example
                print(f"\nFORMATTED HARMFUL MC: {harmful_mc}")
            
            # Get activations for harmful phrase in multiple-choice format
            self.logger.debug("Collecting activations for harmful phrase")
            self.monitor.reset()
            harmful_input_ids = self.tokenizer.encode(harmful_mc, return_tensors="pt").to(self.device)
            
            # Log token IDs and decoded tokens
            if i == 0:  # Only log the first example
                print("\n==== ACTIVATION COLLECTION ====")
                print(f"Input shape: {harmful_input_ids.shape}")
                print(f"Last token ID: {harmful_input_ids[0][-1].item()} ('{self.tokenizer.decode([harmful_input_ids[0][-1].item()])}')")
            
            with torch.no_grad():
                self.model(harmful_input_ids)
            harmful_activations = self.monitor.hooks.get_activations()
            
            # Show sample of collected activations
            if i == 0:
                print("\nCOLLECTED ACTIVATIONS:")
                for layer, activation in harmful_activations.items():
                    print(f"Layer {layer}: Shape={activation.shape}")
                    # Show a small sample of activation values
                    if activation.dim() >= 2:
                        # Handle different tensor dimensions
                        if activation.dim() == 2:
                            # For 2D tensors [batch_size, hidden_size]
                            sample = activation[0, :5].cpu().numpy()
                        else:
                            # For 3D tensors [batch_size, seq_len, hidden_size]
                            sample = activation[0, 0, :5].cpu().numpy()
                        print(f"  Sample values: {sample}")
                    
                    # Only show one layer as example
                    if layer == list(harmful_activations.keys())[0]:
                        break
            
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
    
    def _format_input_as_multiple_choice(self, text: str) -> str:
        """
        Format a single input text as a multiple-choice question.
        
        Args:
            text: The input text
            
        Returns:
            Formatted multiple-choice question
        """
        self.logger.debug(f"Formatting as multiple-choice: {text[:50]}...")
        
        # Create a simple true/false question for the input
        question = "Which of the following statements is true?"
        
        # Option A: The provided statement is true
        # Option B: The provided statement is false
        instruction = f"{question}\nA. {text}\nB. The statement in option A is false."
        
        # We don't include an answer, just the question
        return self._format_prompt(instruction)
    
    def is_harmful(self, text: str, categories: Optional[List[str]] = None) -> bool:
        """
        Check if text would lead to a harmful response by evaluating the first token of the response.
        
        Args:
            text: Input prompt to check
            categories: Specific categories to check against
            
        Returns:
            True if harmful content is detected in the response, False otherwise
        """
        if self.monitor is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
        
        # Reset monitor
        self.monitor.reset()
        
        # Format the prompt for generation
        formatted_prompt = f"{self.user_token}\n{text}\n{self.assistant_token}"
        
        # Encode the prompt
        prompt_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate just the first token of the response
        with torch.no_grad():
            # Get the first token of what would be generated
            outputs = self.model.generate(
                prompt_ids,
                max_new_tokens=1,  # Just generate the first token
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Get the full sequence including the prompt and the first generated token
            full_sequence = torch.cat([prompt_ids, outputs.sequences[:, prompt_ids.shape[1]:]], dim=1)
            
            # Process the full sequence to get activations
            self.model(full_sequence)
        
        # Check if harmful
        return self.monitor.is_harmful(categories=categories, is_response_token=True)
    
    def get_similarity(self, text: str, categories: Optional[List[str]] = None) -> float:
        """
        Get the similarity of the first token of response to harmful patterns.
        
        Args:
            text: Input prompt to check
            categories: Specific categories to check against
            
        Returns:
            Highest similarity score
        """
        if self.monitor is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
        
        # Reset monitor
        self.monitor.reset()
        
        # Format the prompt for generation
        formatted_prompt = f"{self.user_token}\n{text}\n{self.assistant_token}"
        
        # Encode the prompt
        prompt_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate just the first token of the response
        with torch.no_grad():
            # Get the first token of what would be generated
            outputs = self.model.generate(
                prompt_ids,
                max_new_tokens=1,  # Just generate the first token
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Get the full sequence including the prompt and the first generated token
            full_sequence = torch.cat([prompt_ids, outputs.sequences[:, prompt_ids.shape[1]:]], dim=1)
            
            # Process the full sequence to get activations
            self.model(full_sequence)
        
        # Get similarity scores
        results = self.monitor.check_activations(categories=categories, is_response_token=True)
        
        # Find the highest similarity score
        max_similarity = 0.0
        for category_result in results.values():
            max_similarity = max(max_similarity, category_result["max_similarity"])
        
        return max_similarity
    
    def generate_safe_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        skip_prompt_check: bool = False,
        use_multiple_choice: bool = True,
        token_by_token: bool = True,
        return_token_scores: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response while monitoring for harmful content.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            skip_prompt_check: Whether to skip the initial prompt safety check
            use_multiple_choice: Whether to use multiple-choice format internally
            token_by_token: Whether to generate token-by-token and check each token (default: True)
            return_token_scores: Whether to include similarity scores for each token (default: True)
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated text and safety information.
            Also includes token-level similarity scores by default.
        """
        if self.inference is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
        
        # Generate response using simplified inference API
        result = self.inference.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        # Process token scores to determine if any token is harmful
        blocked = False
        block_reason = None
        harmful_token = None
        
        # Check if any token exceeds the threshold
        if 'token_scores' in result:
            for token in result['token_scores']:
                if token.get('similarity', 0.0) >= self.threshold:
                    blocked = True
                    harmful_category = token.get('category', 'unknown')
                    token_text = token.get('token_text', '')
                    token_position = token.get('position', 0)
                    similarity = token.get('similarity', 0.0)
                    
                    harmful_token = token
                    block_reason = f"Token at position {token_position} ('{token_text}') exceeded threshold: {similarity:.4f} for category '{harmful_category}'"
                    break
        
        # Add safety information to the result
        enhanced_result = {
            "response": result["response"],
            "token_scores": result.get("token_scores", []) if return_token_scores else [],
            "blocked": blocked,
            "reason": block_reason,
            "harmful_token": harmful_token
        }
        
        return enhanced_result
    
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
        
        # Process token scores to determine if any token is harmful
        blocked = False
        block_reason = None
        
        # Check if any token exceeds the threshold
        if 'token_scores' in result:
            for token in result['token_scores']:
                if token.get('similarity', 0.0) >= self.threshold:
                    blocked = True
                    harmful_category = token.get('category', 'unknown')
                    similarity = token.get('similarity', 0.0)
                    block_reason = f"Response token exceeded threshold: {similarity:.4f} for category '{harmful_category}'"
                    break
        
        # Add safety information to the result
        enhanced_result = {
            "response": result["response"],
            "token_scores": result.get("token_scores", []),
            "blocked": blocked,
            "reason": block_reason
        }
        
        return enhanced_result
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available categories.
        
        Returns:
            List of category names
        """
        return self.vectors.get_available_categories()
    
    def get_triggered_category(self, text: str) -> Optional[str]:
        """
        Get the category that was triggered by the model's first response token.
        
        Args:
            text: Input prompt to check
            
        Returns:
            Category name or None if no category was triggered
        """
        if self.monitor is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
        
        # Reset monitor
        self.monitor.reset()
        
        # Format the prompt for generation
        formatted_prompt = f"{self.user_token}\n{text}\n{self.assistant_token}"
        
        # Encode the prompt
        prompt_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate just the first token of the response
        with torch.no_grad():
            # Get the first token of what would be generated
            outputs = self.model.generate(
                prompt_ids,
                max_new_tokens=1,  # Just generate the first token
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Get the full sequence including the prompt and the first generated token
            full_sequence = torch.cat([prompt_ids, outputs.sequences[:, prompt_ids.shape[1]:]], dim=1)
            
            # Process the full sequence to get activations
            self.model(full_sequence)
        
        # Get the most harmful category
        most_harmful = self.monitor.get_most_harmful_category()
        if most_harmful is not None:
            category, similarity = most_harmful
            if similarity >= self.threshold:
                return category
        
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
    
    def analyze_response_tokens(
        self,
        prompt: str,
        max_tokens: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response token-by-token and analyze each token for harmful content.
        
        This method provides detailed token-level analysis, including similarity scores
        for each token generated in the response. It's useful for understanding why 
        specific responses might be flagged as harmful.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the response, token scores, and blocking information
        """
        if self.inference is None:
            raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
            
        # Use the simplified inference API
        result = self.inference.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            **kwargs
        )
        
        # Process token scores to determine if any token is harmful
        blocked = False
        block_reason = None
        harmful_token = None
        max_similarity = 0.0
        max_similarity_category = None
        
        # Check each token's similarity score
        if 'token_scores' in result:
            for token in result['token_scores']:
                similarity = token.get('similarity', 0.0)
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similarity_category = token.get('category')
                
                if similarity >= self.threshold:
                    blocked = True
                    harmful_category = token.get('category', 'unknown')
                    token_text = token.get('token_text', '')
                    token_position = token.get('position', 0)
                    
                    harmful_token = token
                    block_reason = f"Token at position {token_position} ('{token_text}') exceeded threshold: {similarity:.4f} for category '{harmful_category}'"
                    break
        
        # Add analysis information to the result
        enhanced_result = {
            "response": result["response"],
            "token_scores": result.get("token_scores", []),
            "blocked": blocked,
            "reason": block_reason,
            "harmful_token": harmful_token,
            "max_similarity": max_similarity,
            "max_similarity_category": max_similarity_category
        }
        
        return enhanced_result
    
    def load_vectors(self, categories: Optional[List[str]] = None, layers: Optional[List[int]] = None) -> bool:
        """
        Explicitly load contrastive vectors from disk with fine-grained control.
        
        Args:
            categories: Specific categories to load. If None, all available categories are loaded.
            layers: Specific layers to load. If None, all available layers are loaded.
            
        Returns:
            True if vectors were loaded successfully, False otherwise
        """
        self.logger.info(f"Explicitly loading vectors from {self.save_dir}")
        
        if categories:
            self.logger.info(f"Loading specific categories: {categories}")
        else:
            self.logger.info("No categories specified, will load all available categories")
            
        if layers:
            self.logger.info(f"Loading specific layers: {layers}")
        else:
            self.logger.info("No layers specified, will load all available layers")
        
        # Load vectors with specified filters
        success = self.vectors.load_vectors(categories=categories, layers=layers)
        
        if success:
            self.logger.info(f"Successfully loaded vectors from {self.save_dir}")
            # Initialize monitor and inference with loaded vectors
            self._initialize_monitor_and_inference()
        else:
            self.logger.warning("Failed to load vectors")
            
        return success
        
    def clear_vectors(self, categories: Optional[List[str]] = None) -> None:
        """
        Clear loaded vectors to start fresh.
        
        Args:
            categories: Specific categories to clear. If None, all categories are cleared.
        """
        self.logger.info("Clearing loaded vectors")
        
        if categories:
            self.logger.info(f"Clearing specific categories: {categories}")
            for category in categories:
                self.vectors.clear_vectors(category=category)
        else:
            self.logger.info("Clearing all categories")
            self.vectors.clear_vectors()
            
        # Reinitialize monitor with empty vectors
        if self.monitor is not None:
            self._initialize_monitor_and_inference() 