"""
Token generation and activation monitoring for Wisent-Guard
"""

import os
import torch
import re
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from .utils.logger import get_logger

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

class SafeInference:
    """
    Simplified class for generating tokens and collecting activations.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        monitor: Any,
        log_level: str = "info",
        user_token: Optional[str] = None,
        assistant_token: Optional[str] = None,
        force_format: Optional[str] = None,
        force_llama_format: Optional[bool] = None  # For backward compatibility
    ):
        """
        Initialize the inference module.
        
        Args:
            model: The transformer model to use for generation
            tokenizer: Tokenizer for the model
            monitor: Monitor instance for tracking activations
            log_level: Logging level ('debug', 'info', 'warning', 'error')
            user_token: Custom user token override (default: from WISENT_USER_TOKEN env var or "<|user|>")
            assistant_token: Custom assistant token override (default: from WISENT_ASSISTANT_TOKEN env var or "<|assistant|>")
            force_format: Force specific format: "llama31", "mistral", "legacy", or None for auto-detect
            force_llama_format: (Deprecated) For backward compatibility
        """
        self.logger = get_logger(name="wisent_guard.inference", level=log_level)
        self.logger.info("Initializing SafeInference for token generation and activation analysis")
        
        # Get user/assistant tokens from parameters, env vars, or defaults
        self.user_token = user_token or os.environ.get("WISENT_USER_TOKEN", DEFAULT_USER_TOKEN)
        self.assistant_token = assistant_token or os.environ.get("WISENT_ASSISTANT_TOKEN", DEFAULT_ASSISTANT_TOKEN)
        
        self.model = model
        self.tokenizer = tokenizer
        self.monitor = monitor

        # Handle force_llama_format for backward compatibility
        if force_llama_format is not None:
            if force_llama_format is True:
                force_format = "llama31"
            elif force_llama_format is False:
                force_format = "legacy"
            if force_format is not None:
                self.logger.warning("force_llama_format is deprecated, use force_format instead")

        # Determine which format to use
        self.format_type = self._detect_format(force_format)
        
        # Log the selected format
        if self.format_type == FORMAT_LLAMA31:
            self.logger.info("Using Llama 3.1 prompt format with special tokens:")
            self.logger.info(f"  {LLAMA_3_1_BEGIN_TEXT}{LLAMA_3_1_START_HEADER}user{LLAMA_3_1_END_HEADER}...")
            self.logger.info(f"  Note: User/assistant token settings are ignored in Llama 3.1 mode")
        elif self.format_type == FORMAT_MISTRAL:
            self.logger.info("Using Mistral prompt format with special tokens:")
            self.logger.info(f"  {MISTRAL_INST_START} instruction {MISTRAL_INST_END} response")
            self.logger.info(f"  Note: User/assistant token settings are ignored in Mistral mode")
        else:
            self.logger.info("Using legacy prompt format with tokens:")
            self.logger.info(f"  User token: {self.user_token}")
            self.logger.info(f"  Assistant token: {self.assistant_token}")
        
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
        
    def _format_prompt_legacy(self, prompt: str) -> str:
        """Format prompt with legacy format (pre-Llama 3.1/Mistral)"""
        return f"{self.user_token}\n{prompt}\n{self.assistant_token}"
        
    def _format_prompt_llama_31(self, prompt: str) -> str:
        """
        Format prompt with Llama 3.1 special tokens.
        
        This implements the format:
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        [message]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        If the prompt contains a system instruction (indicated by "system:" prefix),
        it will be formatted as:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        [system_message]<|eot_id|><|start_header_id|>user<|end_header_id|>
        [user_message]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        # Check if prompt contains a system instruction (prefixed with "system:")
        if prompt.lstrip().startswith("system:"):
            parts = prompt.split("\n", 1)
            system_msg = parts[0].replace("system:", "", 1).strip()
            user_msg = parts[1].strip() if len(parts) > 1 else ""
            
            return f"{LLAMA_3_1_BEGIN_TEXT}{LLAMA_3_1_START_HEADER}system{LLAMA_3_1_END_HEADER}\n{system_msg}{LLAMA_3_1_EOT}{LLAMA_3_1_START_HEADER}user{LLAMA_3_1_END_HEADER}\n{user_msg}{LLAMA_3_1_EOT}{LLAMA_3_1_START_HEADER}assistant{LLAMA_3_1_END_HEADER}"
        else:
            # Regular user message
            return f"{LLAMA_3_1_BEGIN_TEXT}{LLAMA_3_1_START_HEADER}user{LLAMA_3_1_END_HEADER}\n{prompt}{LLAMA_3_1_EOT}{LLAMA_3_1_START_HEADER}assistant{LLAMA_3_1_END_HEADER}"
    
    def _format_prompt_mistral(self, prompt: str) -> str:
        """
        Format prompt with Mistral special tokens.
        
        This implements the format:
        [INST] instruction [/INST] response
        
        If the prompt contains a system instruction (indicated by "system:" prefix),
        it will be formatted as part of the instruction:
        [INST] system: [system_message]
        [user_message] [/INST] response
        """
        # Check if prompt contains a system instruction (prefixed with "system:")
        if prompt.lstrip().startswith("system:"):
            # Keep system message as part of the instruction for Mistral
            return f"{MISTRAL_INST_START} {prompt} {MISTRAL_INST_END}"
        else:
            # Regular user message
            return f"{MISTRAL_INST_START} {prompt} {MISTRAL_INST_END}"
        
    def format_prompt(self, prompt: str) -> str:
        """Format prompt using the appropriate format based on model detection"""
        if self.format_type == FORMAT_LLAMA31:
            return self._format_prompt_llama_31(prompt)
        elif self.format_type == FORMAT_MISTRAL:
            return self._format_prompt_mistral(prompt)
        else:
            return self._format_prompt_legacy(prompt)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate tokens and collect activations.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated text and token activations
        """
        self.logger.info(f"Generating response for prompt: '{prompt[:50]}...' (max_tokens: {max_new_tokens})")
        # Always use token-by-token generation
        return self.generate_token_by_token(prompt, max_new_tokens, **kwargs)
    
    def generate_token_by_token(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text token-by-token and collect activations for each token.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated text and token activations
        """
        self.logger.debug(f"Starting token-by-token generation (max_tokens: {max_new_tokens})")
        try:
            # Filter out Wisent-specific parameters
            model_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['token_by_token', 'return_token_scores', 'skip_prompt_check']}
            self.logger.debug(f"Filtered kwargs: {list(model_kwargs.keys())}")
            
            # Format prompt with appropriate format
            formatted_prompt = self.format_prompt(prompt)
            
            # Initialize generation
            device = next(self.model.parameters()).device
            self.logger.debug(f"Using device: {device}")
            
            input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
            self.logger.debug(f"Encoded prompt length: {len(input_ids[0])} tokens")
            
            generated_sequence = input_ids.clone()
            
            # For tracking token-level scores
            token_scores = []
            
            # Generate one token at a time and collect activations
            self.logger.info("Beginning token-by-token generation and activation analysis")
            for i in range(max_new_tokens):
                # Generate the next token
                with torch.no_grad():
                    self.logger.debug(f"Generating token {i+1}/{max_new_tokens}")
                    outputs = self.model.generate(
                        generated_sequence,
                        max_new_tokens=1,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **model_kwargs  # Use filtered kwargs
                    )
                    
                    # Get the new token
                    next_token = outputs.sequences[:, -1].unsqueeze(-1)
                    
                    # Append to the sequence
                    generated_sequence = torch.cat([generated_sequence, next_token], dim=-1)
                    
                    # Decode the token text
                    token_text = self.tokenizer.decode(next_token[0])
                    self.logger.debug(f"Generated token: '{token_text}' (ID: {next_token.item()})")
                    
                    # Reset monitor and collect activations
                    self.monitor.reset()
                    self.logger.debug("Collecting activations for token")
                    _ = self.model(generated_sequence)
                    
                    # Get activation analysis
                    self.logger.debug("Analyzing token activations")
                    results = self.monitor.check_activations(is_response_token=True)
                    
                    # Find max similarity across all categories
                    max_similarity = 0.0
                    max_category = None
                    
                    for category, category_result in results.items():
                        if category_result["max_similarity"] > max_similarity:
                            max_similarity = category_result["max_similarity"]
                            max_category = category
                    
                    self.logger.debug(f"Token similarity: {max_similarity:.6f} for category: {max_category}")
                    
                    # Record token information
                    token_info = {
                        "token_id": next_token.item(),
                        "token_text": token_text,
                        "position": i,
                        "similarity": float(max_similarity),
                        "category": max_category,
                    }
                    token_scores.append(token_info)
                    
                    # Check if we've reached the end of the sequence
                    if next_token[0].item() == self.tokenizer.eos_token_id:
                        self.logger.debug("Reached end of sequence token")
                        break
                        
                    # Check for format-specific end tokens
                    if self.format_type == FORMAT_LLAMA31:
                        token_str = self.tokenizer.decode(next_token[0])
                        if LLAMA_3_1_EOT in token_str or LLAMA_3_1_EOM in token_str:
                            self.logger.debug("Reached Llama 3.1 end token")
                            break
            
            # Decode the full response (minus the prompt)
            prompt_length = len(input_ids[0])
            response_ids = generated_sequence[0, prompt_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            self.logger.info(f"Generation complete: {len(token_scores)} tokens analyzed")
            self.logger.debug(f"Response: '{response[:50]}...'")
            
            # Return results
            return {
                "response": response,
                "token_scores": token_scores
            }
                
        except Exception as e:
            self.logger.error(f"Error during token generation: {str(e)}")
            return {
                "response": f"Error during generation: {str(e)}",
                "token_scores": []
            } 