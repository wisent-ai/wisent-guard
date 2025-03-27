"""
Token generation and activation monitoring for Wisent-Guard
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from .utils.logger import get_logger

class SafeInference:
    """
    Simplified class for generating tokens and collecting activations.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        monitor: Any,
        log_level: str = "info"
    ):
        """
        Initialize the inference module.
        
        Args:
            model: The transformer model to use for generation
            tokenizer: Tokenizer for the model
            monitor: Monitor instance for tracking activations
            log_level: Logging level ('debug', 'info', 'warning', 'error')
        """
        self.logger = get_logger(name="wisent_guard.inference", level=log_level)
        self.logger.info("Initializing SafeInference for token generation and activation analysis")
        
        self.model = model
        self.tokenizer = tokenizer
        self.monitor = monitor
        
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
            
            # Format prompt
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
            
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