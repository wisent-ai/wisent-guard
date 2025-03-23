"""
Safe inference module for generating text while monitoring for harmful content
"""

import torch
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
from .monitor import ActivationMonitor

class SafeInference:
    """
    Class for generating text while monitoring for harmful content.
    
    This module provides methods for generating text from a model while
    monitoring activations in real-time to detect and block harmful content.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        monitor: ActivationMonitor,
        block_on_harmful: bool = True,
        unsafe_message: str = "I cannot generate content that may be harmful.",
    ):
        """
        Initialize the safe inference module.
        
        Args:
            model: The transformer model to use for generation
            tokenizer: Tokenizer for the model
            monitor: ActivationMonitor instance for monitoring activations
            block_on_harmful: Whether to block generation if harmful content is detected
            unsafe_message: Message to return when harmful content is blocked
        """
        self.model = model
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.block_on_harmful = block_on_harmful
        self.unsafe_message = unsafe_message
        self.blocked_reason = None
        
    def _check_prompt_safety(self, input_text: str) -> bool:
        """
        Check if a prompt is safe before generating a response.
        
        Args:
            input_text: Input prompt to check
            
        Returns:
            True if safe, False if harmful
        """
        # Reset monitor state
        self.monitor.reset()
        
        # Tokenize and encode input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        if self.model.device.type == "cuda":
            input_ids = input_ids.to(self.model.device)
        
        # Run a forward pass through the model to get activations
        with torch.no_grad():
            self.model(input_ids)
        
        # Check if activations match harmful patterns
        if self.monitor.is_harmful():
            harmful_category = self.monitor.get_most_harmful_category()
            if harmful_category:
                category, similarity = harmful_category
                self.blocked_reason = f"Prompt contains potentially harmful content related to '{category}' (similarity: {similarity:.2f})"
            else:
                self.blocked_reason = "Prompt contains potentially harmful content"
            return False
        
        return True
    
    def _token_callback(self, input_ids: torch.Tensor) -> bool:
        """
        Callback function for token-by-token generation monitoring.
        
        Args:
            input_ids: Current sequence of input IDs
            
        Returns:
            True to continue generation, False to stop
        """
        # Check if the current activations match harmful patterns
        if self.monitor.is_harmful():
            harmful_category = self.monitor.get_most_harmful_category()
            if harmful_category:
                category, similarity = harmful_category
                self.blocked_reason = f"Generation was heading toward harmful content related to '{category}' (similarity: {similarity:.2f})"
            else:
                self.blocked_reason = "Generation was heading toward potentially harmful content"
            return False
        
        return True
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        skip_prompt_check: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text while monitoring for harmful content.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            skip_prompt_check: Whether to skip the initial prompt safety check
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated text and safety information
        """
        # Reset monitoring state
        self.monitor.reset()
        self.blocked_reason = None
        
        # Check if the prompt itself is safe
        prompt_is_safe = True
        if not skip_prompt_check:
            prompt_is_safe = self._check_prompt_safety(prompt)
        
        # If prompt is not safe and blocking is enabled, return early
        if not prompt_is_safe and self.block_on_harmful:
            return {
                "text": self.unsafe_message,
                "is_safe": False,
                "blocked": True,
                "reason": self.blocked_reason
            }
        
        # Prepare for generation
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.model.device.type == "cuda":
            input_ids = input_ids.to(self.model.device)
        
        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        
        # Set up token stopping based on callback for streaming generation
        if hasattr(self.model, "streamer"):
            warnings.warn("Streaming generation is used, but monitoring might not catch all harmful content in real-time.")
        
        # Generate text
        try:
            with torch.no_grad():
                # We have to handle token-by-token generation to monitor for harmful content
                generated_text = ""
                current_ids = input_ids
                
                for _ in range(max_new_tokens):
                    # Forward pass
                    outputs = self.model(current_ids)
                    
                    # Get next token
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    
                    # Append to current sequence
                    current_ids = torch.cat([current_ids, next_token_id], dim=-1)
                    
                    # Check for harmful content
                    if not self._token_callback(current_ids):
                        if self.block_on_harmful:
                            break
                    
                    # Check for EOS token
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break
                
                # Decode the generated text
                generated_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
                
                # If harmful content was detected during generation and blocking is enabled
                if self.blocked_reason and self.block_on_harmful:
                    # Return only the prompt part
                    prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    generated_text = prompt_text + "\n\n" + self.unsafe_message
        
        except Exception as e:
            return {
                "text": f"Error during generation: {str(e)}",
                "is_safe": False,
                "blocked": True,
                "reason": f"Exception: {str(e)}"
            }
        
        # Return results
        return {
            "text": generated_text,
            "is_safe": self.blocked_reason is None,
            "blocked": self.blocked_reason is not None and self.block_on_harmful,
            "reason": self.blocked_reason
        } 