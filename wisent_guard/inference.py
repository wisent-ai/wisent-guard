"""
Token generation and activation monitoring for Wisent-Guard
"""

import os
import torch
import re
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from .utils.logger import get_logger
from tqdm import tqdm

# Default tokens that can be overridden via environment variables
DEFAULT_USER_TOKEN = "<|user|>"
DEFAULT_ASSISTANT_TOKEN = "<|assistant|>"

# Llama 3.1 special tokens
LLAMA_3_1_BEGIN_TEXT = "<|begin_of_text|>"
LLAMA_3_1_START_HEADER = "<|start_header_id|>"
LLAMA_3_1_END_HEADER = "<|end_header_id|>"
LLAMA_3_1_EOT = "<|eot_id|>"
LLAMA_3_1_EOM = "<|eom_id|>"

# Mistral format tokens
MISTRAL_INST_START = "[INST]"
MISTRAL_INST_END = "[/INST]"

# Model format types
FORMAT_LEGACY = "legacy"
FORMAT_LLAMA31 = "llama31"
FORMAT_MISTRAL = "mistral"

class TokenScore:
    """Stores information about a token and its similarity to harmful content."""
    
    def __init__(self, 
                 token_id: Optional[int] = None,
                 token_text: str = "",
                 position: int = 0,
                 similarity: float = 0.0,
                 is_harmful: bool = False,
                 category: Optional[str] = None,
                 activations: Optional[Dict[int, torch.Tensor]] = None):
        """
        Initialize token score information.
        
        Args:
            token_id: ID of the token
            token_text: String representation of the token
            position: Position in the sequence
            similarity: Similarity score to harmful content
            is_harmful: Whether the token is deemed harmful
            category: Category of harmful content (if harmful)
            activations: Dictionary of layer activations for this token
        """
        self.token_id = token_id
        self.token_text = token_text
        self.position = position
        self.similarity = similarity
        self.is_harmful = is_harmful
        self.category = category
        self.activations = activations or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "token_id": self.token_id,
            "token_text": self.token_text,
            "position": self.position,
            "similarity": float(self.similarity),
            "is_harmful": self.is_harmful,
            "category": self.category,
            # Don't include activations in the dict - they're for internal use
        }

class SafeInference:
    """
    Safe inference wrapper for monitoring model outputs for harmful content.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        monitor: "ActivationMonitor",
        device: Optional[torch.device] = None,
        format_type: str = "legacy",
        user_token: str = "User:",
        assistant_token: str = "Assistant:",
        log_level: str = "info"
    ):
        """
        Initialize safe inference wrapper.
        
        Args:
            model: Hugging Face model
            tokenizer: Hugging Face tokenizer
            monitor: Activation monitor instance
            device: Device to use for inference
            format_type: Prompt format type ("legacy", "llama31", "mistral")
            user_token: Token to use for user messages
            assistant_token: Token to use for assistant messages
            log_level: Logging level
        """
        self.model = model
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.device = device or next(model.parameters()).device
        self.format_type = format_type
        self.user_token = user_token
        self.assistant_token = assistant_token
        
        # Set up logger
        self.logger = get_logger("wisent_guard", log_level)
        
        # Store pre-formatted prompt information
        self.original_prompt = None
        self.formatted_prompt = None
        self.prompt_ids = None
        
        # Log configuration
        self.logger.info(f"Initializing SafeInference for token generation and activation analysis")
        
        # Log special format if used
        if format_type == FORMAT_LLAMA31:
            self.logger.info("Using Llama 3.1 prompt format with special tokens:")
            self.logger.info("  <|begin_of_text|><|start_header_id|>user<|end_header_id|>...")
            self.logger.info("  Note: User/assistant token settings are ignored in Llama 3.1 mode")
        elif format_type == FORMAT_MISTRAL:
            self.logger.info("Using Mistral prompt format with [INST] tokens")
        else:
            self.logger.info(f"Using legacy prompt format with tokens: '{user_token}' / '{assistant_token}'")
            
        # Ensure the tokenizer can handle padding
        if tokenizer.pad_token_id is None:
            self.logger.info("Setting padding token to EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        
    def _detect_format(self, force_format: Optional[str] = None) -> str:
        """
        Determine which format to use based on model name or forced setting.
        
        Args:
            force_format: Force specific format, or None for auto-detect
            
        Returns:
            Format type string: "llama31", "mistral", or "legacy"
        """
        # If format is explicitly forced, respect that setting
        if force_format is not None:
            return force_format
            
        # Try to get model name from the model config
        model_name = getattr(self.model.config, "_name_or_path", "").lower()
        if not model_name and hasattr(self.model.config, "name_or_path"):
            model_name = self.model.config.name_or_path.lower()
            
        # Check environment variable for override
        env_format = os.environ.get("WISENT_FORMAT")
        if env_format is not None:
            return env_format.lower()
            
        # Auto-detection based on model name
        if re.search(r"llama-?3\.?1", model_name, re.IGNORECASE):
            self.logger.info(f"Auto-detected Llama 3.1 model from name: {model_name}")
            return FORMAT_LLAMA31
        elif re.search(r"mistral", model_name, re.IGNORECASE):
            self.logger.info(f"Auto-detected Mistral model from name: {model_name}")
            return FORMAT_MISTRAL
            
        # Default to legacy format if no specific format detected
        return FORMAT_LEGACY
        
    def _format_prompt(self, prompt: str) -> str:
        """
        Format a prompt with the appropriate tokens based on the model format.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Formatted prompt
        """
        if self.format_type == FORMAT_LLAMA31:
            # Llama 3.1 format
            BEGIN_TEXT = "<|begin_of_text|>"
            START_HEADER = "<|start_header_id|>"
            END_HEADER = "<|end_header_id|>"
            EOT = "<|eot_id|>"
            
            # Format the prompt
            formatted = f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{prompt}{EOT}{START_HEADER}assistant{END_HEADER}\n"
            self.logger.debug("Using Llama 3.1 format")
            return formatted
        
        elif self.format_type == FORMAT_MISTRAL:
            # Mistral format
            formatted = f"[INST] {prompt} [/INST]"
            self.logger.debug("Using Mistral format")
            return formatted
        
        else:
            # Legacy format with user/assistant tokens
            formatted = f"{self.user_token}\n{prompt}\n{self.assistant_token}\n"
            self.logger.debug("Using legacy format")
            return formatted
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        min_new_tokens: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response while monitoring for harmful content.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            min_new_tokens: Minimum number of tokens to generate
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated text and token scores
        """
        # Format the prompt if needed (handled in _format_prompt)
        self.logger.info(f"Generating response for prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}' (max_tokens: {max_new_tokens})")
        
        # Store original prompt
        self.original_prompt = prompt
        
        # Format the prompt with the appropriate tokens
        self.formatted_prompt = self._format_prompt(prompt)
        
        # Encode the prompt
        self.prompt_ids = self.tokenizer.encode(self.formatted_prompt, return_tensors="pt").to(self.device)
        self.logger.debug(f"Encoded prompt shape: {self.prompt_ids.shape}")
        
        # Set parameters for generation
        params = {"max_new_tokens": max_new_tokens}
        params.update(kwargs)
        
        # Collect token scores
        token_scores = []
        token_objects = []  # Store actual TokenScore objects, not just dicts
        response_tokens = []
        tokens_generated = 0
        do_early_stopping = False
        early_stop_reason = None
            
        self.logger.info("Beginning token-by-token generation and activation analysis")
        
        # Set up progress bar for lengthy generation
        gen_progress = tqdm(total=max_new_tokens, desc="Generating", disable=max_new_tokens < 20)
        
        # Start token-by-token generation
        current_ids = self.prompt_ids.clone()
        
        while tokens_generated < max_new_tokens and not do_early_stopping:
            try:
                # Generate a single token
                with torch.no_grad():
                    # Get model output for next token prediction
                    outputs = self.model(current_ids, output_hidden_states=True)
                    
                    # Get the next token
                    logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(logits, dim=-1).item()
                
                # Increment token counter
                tokens_generated += 1
                gen_progress.update(1)
                
                # Append to current sequence
                current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)
                
                # Make sure activations are captured
                self.monitor.capture_activations_from_forward(outputs, current_ids)
                
                # Process the token (check for harmful content)
                token_score = self._process_token(next_token_id, tokens_generated)
                
                # Store the token data
                token_objects.append(token_score)
                token_scores.append(token_score.to_dict())
                
                # Add to response tokens
                response_token_text = token_score.token_text
                response_tokens.append(response_token_text)
                
                # Early stopping check 1: Harmful content detector
                if token_score.is_harmful and tokens_generated >= min_new_tokens:
                    category = token_score.category
                    similarity = token_score.similarity 
                    self.logger.warning(f"Harmful content detected in token {tokens_generated}: similarity={similarity:.4f}, category='{category}'")
                    do_early_stopping = True
                    early_stop_reason = f"Harmful content detected: similarity={similarity:.4f}, category='{category}'"
                    break
                
                # Early stopping check 2: EOS token
                if next_token_id == self.tokenizer.eos_token_id and tokens_generated >= min_new_tokens:
                    self.logger.debug("EOS token encountered, stopping generation")
                    do_early_stopping = True
                    early_stop_reason = "EOS token detected"
                    break
                    
            except Exception as e:
                self.logger.error(f"Error during token generation: {e}")
                early_stop_reason = f"Error: {str(e)}"
                break
        
        # Close progress bar
        gen_progress.close()
        self.logger.info(f"Generation complete: {tokens_generated} tokens analyzed")
        
        # Get complete decoded response
        response_text = self.tokenizer.decode(current_ids[0][self.prompt_ids.shape[1]:])
        
        # Return results
        result = {
            "response": response_text,
            "token_scores": token_scores,
            "token_objects": token_objects  # Include full TokenScore objects with activations
        }
        
        if early_stop_reason:
            result["early_stop_reason"] = early_stop_reason
            
        return result

    def _process_token(self, token_id: int, token_position: int) -> TokenScore:
        """
        Process a token, check for harmfulness, and return token information.
        
        Args:
            token_id: ID of the token
            token_position: Position in the sequence
            
        Returns:
            TokenScore object with token information
        """
        token_text = self.tokenizer.decode([token_id])
        
        # Check for harmful content - but only if we have vectors
        if self.monitor.vectors is not None:
            # Get all similarity results for this token
            results = self.monitor.check_activations()
            
            # Find the highest similarity
            max_sim = 0.0
            max_category = None
            is_harmful = False
            
            for category, result in results.items():
                if result["max_similarity"] > max_sim:
                    max_sim = result["max_similarity"]
                    max_category = category
                    is_harmful = result["is_harmful"]
        else:
            # In classifier-only mode, we don't check individual tokens
            max_sim = 0.0
            max_category = None
            is_harmful = False
        
        # Get the current activations for this token
        activations = self.monitor.get_activations()
        
        # Create token score
        token_score = TokenScore(
            token_id=token_id,
            token_text=token_text,
            position=token_position,
            similarity=max_sim,
            is_harmful=is_harmful,
            category=max_category,
            activations=activations  # Store activations for classifier use
        )
        
        return token_score 