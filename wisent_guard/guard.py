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
from .classifier import ActivationClassifier

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
        token_strategy: str = "last",  # Keep parameter but ignore other values
        log_level: str = "info",
        log_file: Optional[str] = None,
        auto_load_vectors: bool = False,
        user_token: Optional[str] = None,
        assistant_token: Optional[str] = None,
        force_format: Optional[str] = None,
        force_llama_format: Optional[bool] = None,  # For backward compatibility
        # New parameters for classifier support
        use_classifier: bool = False,
        classifier_path: Optional[str] = None,
        classifier_threshold: float = 0.5,
        # New parameters for early termination
        early_termination: bool = False,
        placeholder_message: str = "Sorry, this response was blocked due to potentially harmful content.",
        # New parameters for response logging
        enable_logging: bool = False,
        log_file_path: str = "./harmful_responses.json"
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
            token_strategy: Parameter kept for backward compatibility, but "last" token is always used
            log_level: Logging level ('debug', 'info', 'warning', 'error')
            log_file: Optional file to write logs to
            auto_load_vectors: Whether to automatically load all vectors from save_dir (default: False)
            user_token: Custom user token override (default: from WISENT_USER_TOKEN env var or "<|user|>")
            assistant_token: Custom assistant token override (default: from WISENT_ASSISTANT_TOKEN env var or "<|assistant|>")
            force_format: Force specific format: "llama31", "mistral", "legacy", or None for auto-detect
            force_llama_format: (Deprecated) For backward compatibility
            use_classifier: Whether to use ML-based classifier instead of threshold-based detection
            classifier_path: Path to trained classifier model (required if use_classifier is True)
            classifier_threshold: Classification threshold for the ML model (default: 0.5)
            early_termination: Whether to terminate generation early when harmful content is detected (default: False)
            placeholder_message: Message to return when generation is terminated early (default: "Sorry, this response was blocked due to potentially harmful content.")
            enable_logging: Whether to enable logging of detected harmful responses
            log_file_path: Path to the log file for detected harmful responses
        """
        # Set up logger
        self.logger = get_logger(level=log_level, log_file=log_file)
        self.logger.info("Initializing ActivationGuard")
        
        # Early termination settings
        self.early_termination = early_termination
        self.placeholder_message = placeholder_message
        self.logger.info(f"Early termination: {early_termination}, Message: {placeholder_message}")
        
        # Response logging settings
        self.enable_logging = enable_logging
        self.log_file_path = log_file_path
        if self.enable_logging:
            self.logger.info(f"Response logging enabled. Log file: {log_file_path}")
            # Initialize log file if it doesn't exist
            if not os.path.exists(os.path.dirname(log_file_path)):
                try:
                    os.makedirs(os.path.dirname(log_file_path))
                    self.logger.info(f"Created directory for log file: {os.path.dirname(log_file_path)}")
                except Exception as e:
                    self.logger.warning(f"Could not create directory for log file: {e}")
            
            # Initialize log file with empty array if it doesn't exist
            if not os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'w') as f:
                        import json
                        json.dump([], f)
                    self.logger.info(f"Initialized empty log file: {log_file_path}")
                except Exception as e:
                    self.logger.warning(f"Could not initialize log file: {e}")
        
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
        
        # Force using "last" token strategy regardless of what was provided
        self.token_strategy = "last"
        if token_strategy != "last":
            self.logger.warning(f"Ignoring provided token_strategy='{token_strategy}'. Always using 'last' token strategy.")
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
        
        # Initialize monitor and inference objects
        self.monitor = None
        self.inference = None
        self.threshold = threshold
        
        # Set up approach: classifier or vector
        self.use_classifier = use_classifier
        self.classifier = None
        self.vectors = None  # Initialize vectors attribute to None
        
        # Set default force_format 
        self.force_format = force_format
        
        # Initialize classifier if requested
        if use_classifier:
            if classifier_path is None:
                self.logger.warning("use_classifier=True but no classifier_path provided. Cannot use classifier-based detection.")
                self.use_classifier = False
            else:
                try:
                    self.classifier = ActivationClassifier(
                        model_path=classifier_path,
                        threshold=classifier_threshold,
                        positive_class_label="harmful"
                    )
                    self.logger.info(f"Using ML-based classifier from {classifier_path} with threshold {classifier_threshold}")
                except Exception as e:
                    self.logger.error(f"Failed to load classifier: {e}")
                    self.logger.warning("Cannot use classifier-based detection due to error.")
                    self.use_classifier = False
        
        # Only initialize vectors if we're not using classifier or if explicitly requested
        if not self.use_classifier or auto_load_vectors:
            self.vectors = ContrastiveVectors(save_dir=self.save_dir)
            
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
        """Set up target tokens for multiple-choice format (kept for backward compatibility)."""
        # With "last" token strategy, there's no need to set up target tokens
        self.logger.debug("Target tokens setup not needed with 'last' token strategy")
    
    def _initialize_monitor_and_inference(self):
        """Initialize the monitor and inference module."""
        self.logger.info("Setting up activation monitor")
        
        # Create the vectors instance if none exists
        vectors_instance = None
        if self.vectors is not None:
            vectors_instance = self.vectors
        
        # Create monitor with activation hooks
        self.monitor = ActivationMonitor(
            model=self.model,
            layers=self.layers,
            vectors=vectors_instance,
            token_strategy="last",  # Always use last token
            similarity_threshold=self.threshold,
            device=self.device,
            log_level=self.logger.level
        )
        
        # Determine the format to use based on model detection
        format_type = self._detect_format()
        
        # Create inference module
        self.inference = SafeInference(
            model=self.model,
            tokenizer=self.tokenizer,
            monitor=self.monitor,
            device=self.device,
            format_type=format_type,  # Pass detected format
            user_token=self.user_token,
            assistant_token=self.assistant_token,
            log_level=self.logger.level
        )
    
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
    
    def train_on_phrase_pairs(self, phrase_pairs: List[Dict[str, str]], category: str = "harmful"):
        """
        Train contrastive vectors using pairs of phrases.
        
        Args:
            phrase_pairs: List of dictionaries with 'harmful' and 'harmless' keys
            category: Category name for the contrastive vectors
            
        Returns:
            Success flag
        """
        self.logger.info(f"Training on {len(phrase_pairs)} phrase pairs for category '{category}'...")
        
        # Make sure the vectors are initialized
        if self.vectors is None:
            self.vectors = ContrastiveVectors(
                model_name=getattr(self.model.config, "_name_or_path", "unknown"),
                token_strategy="last",  # Always use last token
                similarity_threshold=self.threshold,
                save_dir=self.save_dir,
                log_level=self.logger.level
            )
        
        # Initialize monitor and inference if needed
        if self.monitor is None or self.inference is None:
            self._initialize_monitor_and_inference()
        
        # Convert pairs to multiple choice format to ensure consistent activation patterns
        self.logger.info("Converting phrase pairs to multiple-choice format for consistent activation collection...")
        
        harmful_activations_by_layer = {layer: [] for layer in self.layers}
        harmless_activations_by_layer = {layer: [] for layer in self.layers}
        
        # Process each phrase pair
        for i, pair in enumerate(phrase_pairs):
            harmful_text = pair.get('harmful', '')
            harmless_text = pair.get('harmless', '')
            
            if not harmful_text or not harmless_text:
                self.logger.warning(f"Skipping pair {i}: missing harmful or harmless text")
                continue
            
            # Extract the question part (everything up to the first question mark)
            if '?' in harmful_text:
                question_part = harmful_text.split('?')[0] + '?'
                harmful_answer = harmful_text[len(question_part):].strip()
                harmless_answer = harmless_text[len(question_part):].strip()
            else:
                # If no question mark, use the first half as the question
                question_part = "Which statement is better?"
                harmful_answer = harmful_text
                harmless_answer = harmless_text
            
            # Format as multiple choice
            formatted_harmful = self._format_multiple_choice(question_part, harmless_answer, harmful_answer, answer='B')
            formatted_harmless = self._format_multiple_choice(question_part, harmful_answer, harmless_answer, answer='A')
            
            # Display example if first pair
            if i == 0:
                print("\n==== EXAMPLE TRAINING PAIR ====")
                print(f"HARMFUL: {harmful_text[:100]}...")
                print(f"HARMLESS: {harmless_text[:100]}...")
                print(f"\nFORMATTED HARMFUL MC: {formatted_harmful[:300]}")
            
            # Process harmful example
            # Reset the monitor
            self.monitor.reset()
            
            # Generate tokens for harmful and capture activations 
            result = self.inference.generate(prompt=formatted_harmful, max_new_tokens=1)
            
            # Display activation info for first example
            if i == 0:
                print("\n==== ACTIVATION COLLECTION ====")
                # Get info about the input
                if hasattr(self.inference, 'prompt_ids'):
                    print(f"Input shape: {self.inference.prompt_ids.shape}")
                    # Get the last token ID and text
                    if self.inference.prompt_ids.shape[1] > 0:
                        last_token_id = self.inference.prompt_ids[0, -1].item()
                        last_token_text = self.tokenizer.decode([last_token_id])
                        print(f"Last token ID: {last_token_id} ('{last_token_text}')")
            
            # Get activations from monitor
            harmful_activations = self.monitor.get_activations()
            
            # Store activations by layer
            for layer, activation in harmful_activations.items():
                if layer in harmful_activations_by_layer:
                    harmful_activations_by_layer[layer].append(activation)
            
            # Process harmless example
            # Reset the monitor
            self.monitor.reset()
            
            # Generate tokens for harmless and capture activations
            _ = self.inference.generate(prompt=formatted_harmless, max_new_tokens=1)
            
            # Get activations from monitor
            harmless_activations = self.monitor.get_activations()
            
            # Store activations by layer
            for layer, activation in harmless_activations.items():
                if layer in harmless_activations_by_layer:
                    harmless_activations_by_layer[layer].append(activation)
        
        # Compute contrastive vectors for each layer
        self.logger.info("Computing contrastive vectors...")
        for layer in self.layers:
            if layer in harmful_activations_by_layer and layer in harmless_activations_by_layer:
                harmful_tensors = harmful_activations_by_layer[layer]
                harmless_tensors = harmless_activations_by_layer[layer]
                
                if harmful_tensors and harmless_tensors:
                    self.vectors.add_contrastive_vector(
                        harmful_activations=harmful_tensors,
                        harmless_activations=harmless_tensors,
                        category=category,
                        layer=layer
                    )
        
        # Save vectors
        self.logger.info("Saving vectors to disk...")
        if not self.vectors.save_vectors():
            self.logger.warning("Failed to save vectors")
            return False
        
        # Reinitialize monitor with new vectors
        self.logger.info("Reinitializing monitor with new vectors")
        self._initialize_monitor_and_inference()
        
        # Re-assign vectors to monitor
        self.monitor.vectors = self.vectors
        
        self.logger.info(f"Successfully trained on {len(phrase_pairs)} phrase pairs")
        return True
    
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
    
    def is_harmful(self, text: str) -> bool:
        """
        Check if a given text is harmful.
        
        Args:
            text: Text to check
            
        Returns:
            True if harmful, False if safe
        """
        # Initialize monitor and inference if needed
        if self.monitor is None or self.inference is None:
            self._initialize_monitor_and_inference()

        # If using classifier-based approach
        if hasattr(self, 'use_classifier') and self.use_classifier and hasattr(self, 'classifier') and self.classifier is not None:
            try:
                # Run text through the model to get activations
                success = self._prepare_activations(text)
                if not success:
                    self.logger.error(f"Failed to prepare activations for classification")
                    return False
                    
                # Get activations directly from monitor
                activations = self.monitor.get_activations()
                if not activations:
                    self.logger.error("No activations available")
                    return False
                
                # Get the first monitored layer (typically only one layer is monitored)
                monitored_layer = self.layers[0] if self.layers else None
                if monitored_layer is None or monitored_layer not in activations:
                    self.logger.error(f"Required layer {monitored_layer} not in activations")
                    return False
                
                # Extract the activation tensor for this layer
                activation_tensor = activations[monitored_layer]
                
                # Create a properly formatted activation dictionary for the classifier
                activation_data = {
                    'activations': activation_tensor.detach().flatten(),
                    'layer': monitored_layer,
                    'is_harmful': None  # Will be determined by the classifier
                }
                
                # Make prediction using the classifier
                prediction = self.classifier.predict(activation_data, response_text=text)
                
                is_harmful = prediction.get('is_harmful', False)
                self.logger.debug(f"Classifier prediction: {prediction}, is_harmful: {is_harmful}")
                return is_harmful
                
            except Exception as e:
                self.logger.error(f"Classifier-based detection failed: {e}")
                return False
        
        # Vector-based approach (original)
        try:
            # Prepare activations
            success = self._prepare_activations(text)
            if not success:
                self.logger.warning(f"Failed to prepare activations for text: '{text[:50]}...'")
                return False
            
            # Check activations against harmful vectors
            result = self.monitor.check_activations()
            
            # Find most harmful category
            max_similarity = 0.0
            is_harmful = False
            
            for category, category_result in result.items():
                if category_result["max_similarity"] > max_similarity:
                    max_similarity = category_result["max_similarity"]
                    is_harmful = category_result["is_harmful"]
            
            return is_harmful
            
        except Exception as e:
            self.logger.error(f"Error checking for harmful content: {e}")
            return False
    
    def _is_harmful_threshold_based(self, 
                                    text: str, 
                                    categories: Optional[List[str]] = None) -> bool:
        """
        Use threshold-based method to check if text contains harmful content.
        
        Args:
            text: Text to check for harmful content
            categories: List of categories to check (None for all)
            
        Returns:
            Boolean indicating if harmful content was detected
        """
        # If vectors aren't initialized, we can't use this method
        if self.vectors is None:
            self.logger.error("Cannot use threshold-based detection when configured for classifier-only mode")
            return False
        
        # Get the similarity score
        similarity = self.get_similarity(text, categories)
        
        # Compare to threshold
        result = similarity >= self.threshold
        
        log_fn = self.logger.warning if result else self.logger.debug
        log_fn(f"Similarity: {similarity:.4f}, Threshold: {self.threshold}, Harmful: {result}")
        
        return result
    
    def get_similarity(self, text: str, category: Optional[str] = None) -> float:
        """
        Get similarity score for a text against harmful vectors.
        
        Args:
            text: Text to check
            category: Specific category to check (None for all)
            
        Returns:
            Maximum similarity score
        """
        # Check if vectors are available
        if self.vectors is None:
            self.logger.warning("get_similarity called but this guard is configured for classifier-only mode")
            return 0.0
        
        # If we have no vectors loaded, try to load them
        if not self.vectors.has_any_vectors():
            if not self.load_vectors():
                self.logger.warning("No vectors loaded, cannot check similarity")
                return 0.0
            
        # Make sure monitor and inference are initialized
        if self.monitor is None or self.inference is None:
            self._initialize_monitor_and_inference()
        
        # Prepare activations
        success = self._prepare_activations(text)
        if not success:
            self.logger.warning(f"Failed to prepare activations for similarity check: '{text[:50]}...'")
            return 0.0
        
        # Get similarity scores
        results = self.monitor.check_activations()
        
        # Find maximum similarity for the specified category or across all categories
        max_similarity = 0.0
        
        if category is not None:
            # Check specific category
            if category in results:
                max_similarity = results[category]["max_similarity"]
        else:
            # Find maximum across all categories
            for cat_result in results.values():
                if cat_result["max_similarity"] > max_similarity:
                    max_similarity = cat_result["max_similarity"]
        
        return max_similarity
    
    def _prepare_activations(self, text: str) -> bool:
        """
        Generate tokens from text and capture activations.
        
        Args:
            text: Text to generate tokens from
            
        Returns:
            Success flag
        """
        try:
            # Make sure monitor is initialized
            if self.monitor is None:
                self._initialize_monitor_and_inference()
            
            # Format the prompt properly first
            formatted_text = self._format_prompt(text)
            
            # Tokenize input directly
            inputs = self.tokenizer(formatted_text, return_tensors="pt").to(self.device)
            
            # Run through model to capture activations - generate a single token
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, output_hidden_states=True)
            
            # Use our new method to capture activations
            success = self.monitor.capture_activations_from_forward(outputs, inputs.input_ids)
            
            if not success:
                self.logger.warning("Failed to capture activations from model output")
                
            return success
        except Exception as e:
            self.logger.error(f"Error preparing activations: {e}")
            return False
    
    def generate_safe_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        skip_prompt_check: bool = False,
        token_by_token: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response while monitoring for harmful content.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            skip_prompt_check: Whether to skip the initial prompt safety check
            token_by_token: Whether to generate token-by-token (always True now)
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated text and safety information
        """
        # Initialize monitor and inference if needed
        if self.monitor is None or self.inference is None:
            self._initialize_monitor_and_inference()
        
        # Check if we're in classifier-only mode
        is_classifier_mode = hasattr(self, 'use_classifier') and self.use_classifier and hasattr(self, 'classifier') and self.classifier is not None
        
        # Generate response using inference API
        try:
            result = self.inference.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error during token generation: {e}")
            return {
                "response": f"Error: {str(e)}",
                "token_scores": [],
                "blocked": False,
                "reason": None,
                "harmful_token": None
            }
        
        response = result.get("response", "")
        token_scores = result.get("token_scores", [])
        
        # In classifier-only mode, we evaluate the response as a whole
        if is_classifier_mode:
            try:
                # Reset monitor before preparing activations
                self.monitor.reset()
                
                # Prepare activations using the full response
                formatted_text = self._format_prompt(response)
                self.logger.debug(f"Preparing formatted text for classification: {formatted_text[:50]}...")
                
                # Tokenize the input
                inputs = self.tokenizer(formatted_text, return_tensors="pt").to(self.device)
                
                # Run the model to get activations - important to use model directly for this classifier case
                with torch.no_grad():
                    outputs = self.model(input_ids=inputs.input_ids, output_hidden_states=True)
                
                # Capture activations from the model outputs
                success = self.monitor.capture_activations_from_forward(outputs, inputs.input_ids)
                if not success:
                    self.logger.warning("Failed to capture activations for classification")
                    result["blocked"] = False
                    result["reason"] = None
                    return result
                
                # Manually ensure token data has activation values flag set to true
                if hasattr(self.monitor, 'token_data'):
                    self.monitor.token_data['has_activation_values'] = True
                
                # Double check token data has activation values
                token_data = self.monitor.get_token_data()
                if not token_data.get('has_activation_values', False):
                    self.logger.error("Error during classification: Token data doesn't contain activation values")
                    # Try to fix the issue by manually setting the flag
                    self.monitor.token_data = {'has_activation_values': True, 'token_id': inputs.input_ids[0, -1].item(), 'position': inputs.input_ids.shape[1] - 1}
                    self.logger.info("Manually set token data activation values flag to true")
                
                # Get activations across layers
                activations = self.monitor.get_activations()
                if not activations:
                    self.logger.warning("No activations available for classification")
                    result["blocked"] = False
                    result["reason"] = None
                    return result
                
                # Get the first monitored layer (typically only one layer is monitored)
                monitored_layer = self.layers[0] if self.layers else None
                if monitored_layer is None or monitored_layer not in activations:
                    self.logger.warning(f"Required layer {monitored_layer} not in activations")
                    result["blocked"] = False
                    result["reason"] = None
                    return result
                
                # Extract the activation tensor for this layer
                activation_tensor = activations[monitored_layer]
                
                # Create a properly formatted activation dictionary for the classifier
                activation_data = {
                    'activations': activation_tensor.detach().flatten(),
                    'layer': monitored_layer,
                    'is_harmful': None  # Will be determined by the classifier
                }
                
                # Make prediction using the classifier
                prediction = self.classifier.predict(activation_data, response_text=response)
                
                is_harmful = prediction.get('is_harmful', False)
                probability = prediction.get('score', 0.0)
                
                self.logger.debug(f"Classifier prediction: {prediction}, is_harmful: {is_harmful}")
                
                # Update result with classification
                result["blocked"] = is_harmful
                result["reason"] = f"Response classified as harmful with probability {probability:.4f}" if is_harmful else None
                result["classification_score"] = probability
                result["classification_threshold"] = self.classifier.threshold
                
                # Update token scores
                if token_scores and is_harmful:
                    # Mark all tokens as potentially harmful based on the overall classification
                    for token in token_scores:
                        token["similarity"] = probability
                        token["is_harmful"] = is_harmful
                        token["category"] = "hallucination"  # Default category for classifier
                    
                    # Add harmful token info for the first token (if any)
                    if token_scores:
                        result["harmful_token"] = token_scores[0]
                
                # Check if early termination is enabled and response is harmful
                if self.early_termination and is_harmful:
                    self.logger.info(f"Early termination activated. Response classified as harmful with probability {probability:.4f}")
                    result["response"] = self.placeholder_message
                    result["early_terminated"] = True
                
                # Log harmful response if enabled
                if is_harmful and self.enable_logging:
                    self._log_harmful_response(
                        prompt=prompt,
                        response=response,
                        similarity=probability,
                        category="hallucination",  # Default for classifier
                        harmful_token=result.get("harmful_token")
                    )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error during classification: {e}")
                result["blocked"] = False
                result["reason"] = None
                return result
        
        # For vector-based approach, process token scores to determine if any token is harmful
        blocked = False
        block_reason = None
        harmful_token = None
        first_token_harmful = False
        
        # Check if any token exceeds the threshold
        if 'token_scores' in result:
            for i, token in enumerate(result['token_scores']):
                if token.get('similarity', 0.0) >= self.threshold:
                    blocked = True
                    harmful_category = token.get('category', 'unknown')
                    token_text = token.get('token_text', '')
                    token_position = token.get('position', 0)
                    similarity = token.get('similarity', 0.0)
                    
                    harmful_token = token
                    block_reason = f"Token at position {token_position} ('{token_text}') exceeded threshold: {similarity:.4f} for category '{harmful_category}'"
                    
                    # Check if this is the first token
                    if i == 0:
                        first_token_harmful = True
                    
                    break
        
        # Add safety information to the result
        enhanced_result = {
            "response": result["response"],
            "token_scores": result.get("token_scores", []),
            "blocked": blocked,
            "reason": block_reason,
            "harmful_token": harmful_token
        }
        
        # If early termination is enabled and the first token is harmful, replace the response
        if self.early_termination and blocked and first_token_harmful:
            self.logger.info(f"Early termination activated. First token exceeded threshold: {harmful_token.get('similarity', 0.0):.4f}")
            enhanced_result["response"] = self.placeholder_message
            enhanced_result["early_terminated"] = True
        elif self.early_termination and blocked:
            self.logger.debug(f"Early termination not triggered despite harmful content. first_token_harmful: {first_token_harmful}")
            if harmful_token:
                self.logger.debug(f"Harmful token position: {harmful_token.get('position', -1)}, similarity: {harmful_token.get('similarity', 0.0):.4f}")
        
        # Log harmful response if enabled
        if blocked and self.enable_logging and harmful_token:
            self._log_harmful_response(
                prompt=prompt,
                response=result["response"],
                similarity=harmful_token.get("similarity", 0.0),
                category=harmful_token.get("category", "unknown"),
                harmful_token=harmful_token
            )
        
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
        Get a list of all available categories.
        
        Returns:
            List of category names
        """
        if self.vectors is None:
            # In classifier-only mode, use the classifier's class labels
            if hasattr(self, 'use_classifier') and self.use_classifier and hasattr(self, 'classifier') and self.classifier is not None:
                return ["hallucination"]  # Default category for classifier-only mode
            return []
        
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
    
    def _detect_format(self) -> str:
        """
        Detect the appropriate format to use based on the model.
        
        Returns:
            A string indicating the format: "llama31", "mistral", or "legacy"
        """
        model_name = getattr(self.model.config, "_name_or_path", "unknown").lower()
        
        # Check for specific model types
        if "llama-3" in model_name:
            self.logger.info("Detected Llama 3.1 model. Will use Llama 3.1 prompt format.")
            return "llama31"
        elif "mistral" in model_name:
            self.logger.info("Detected Mistral model. Will use Mistral prompt format.")
            return "mistral"
        else:
            self.logger.info("Using legacy prompt format for model.")
            return "legacy" 
    
    def _log_harmful_response(self, prompt: str, response: str, similarity: float, category: str, harmful_token=None) -> bool:
        """
        Log a harmful response to the JSON log file.
        
        Args:
            prompt: The original prompt
            response: The generated response
            similarity: The similarity score that triggered detection
            category: The category of harmful content detected
            harmful_token: Optional token info that triggered detection
            
        Returns:
            Success flag
        """
        if not self.enable_logging:
            return False
            
        try:
            import json
            import datetime
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "similarity": float(similarity),
                "category": category,
                "threshold": float(self.threshold)
            }
            
            # Add token info if available
            if harmful_token:
                log_entry["token"] = {
                    "text": harmful_token.get("token_text", ""),
                    "position": harmful_token.get("position", 0),
                    "similarity": float(harmful_token.get("similarity", 0.0))
                }
            
            # Read existing log entries
            try:
                with open(self.log_file_path, 'r') as f:
                    log_entries = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self.logger.warning(f"Could not read log file {self.log_file_path}. Creating new log.")
                log_entries = []
            
            # Append new entry
            log_entries.append(log_entry)
            
            # Write updated log
            with open(self.log_file_path, 'w') as f:
                json.dump(log_entries, f, indent=2)
                
            self.logger.info(f"Logged harmful response with similarity {similarity:.4f} for category '{category}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging harmful response: {e}")
            return False
    
    def get_logged_responses(self, limit: Optional[int] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve logged harmful responses from the log file.
        
        Args:
            limit: Maximum number of entries to return (None for all)
            category: Filter by specific category (None for all categories)
            
        Returns:
            List of log entries
        """
        if not self.enable_logging:
            self.logger.warning("Response logging is not enabled")
            return []
            
        try:
            import json
            
            # Check if log file exists
            if not os.path.exists(self.log_file_path):
                self.logger.warning(f"Log file not found: {self.log_file_path}")
                return []
            
            # Read log entries
            with open(self.log_file_path, 'r') as f:
                log_entries = json.load(f)
            
            # Filter by category if specified
            if category is not None:
                log_entries = [entry for entry in log_entries if entry.get("category") == category]
            
            # Sort by timestamp (newest first)
            log_entries.sort(key=lambda entry: entry.get("timestamp", ""), reverse=True)
            
            # Apply limit if specified
            if limit is not None and limit > 0:
                log_entries = log_entries[:limit]
            
            return log_entries
            
        except Exception as e:
            self.logger.error(f"Error retrieving logged responses: {e}")
            return []
            
    def clear_logged_responses(self) -> bool:
        """
        Clear all logged harmful responses.
        
        Returns:
            Success flag
        """
        if not self.enable_logging:
            self.logger.warning("Response logging is not enabled")
            return False
            
        try:
            import json
            
            # Reset log file
            with open(self.log_file_path, 'w') as f:
                json.dump([], f)
                
            self.logger.info(f"Cleared all logged responses from {self.log_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing logged responses: {e}")
            return False 