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
        
        # Get the model's device and move input_ids to it
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Run a forward pass through the model to get activations
        try:
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
        except Exception as e:
            # Log the error but continue (don't block legitimate content due to errors)
            print(f"Warning: Error during prompt safety check: {e}")
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
            try:
                prompt_is_safe = self._check_prompt_safety(prompt)
            except Exception as e:
                print(f"Warning: Error during prompt safety check: {e}")
                # Continue with generation despite the error
                prompt_is_safe = True
        
        # If prompt is not safe and blocking is enabled, return early
        if not prompt_is_safe and self.block_on_harmful:
            return {
                "response": self.unsafe_message,
                "blocked": True,
                "reason": self.blocked_reason
            }
        
        # Format the prompt if a formatter is available, otherwise use simple format
        if hasattr(self, 'format_prompt') and callable(self.format_prompt):
            formatted_prompt = self.format_prompt(prompt)
        else:
            # Simple format without system prompt
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
        
        # Prepare for generation
        try:
            input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Set up generation parameters
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            # Add other generation parameters
            gen_kwargs.update(kwargs)
            
            # Generate text
            with torch.no_grad():
                # Standard generation process
                outputs = self.model.generate(
                    input_ids,
                    **gen_kwargs
                )
                
                try:
                    # Check for harmful content in the full sequence
                    self.monitor.reset()
                    _ = self.model(outputs.to(device))
                    
                    if self.monitor.is_harmful() and self.block_on_harmful:
                        harmful_category = self.monitor.get_most_harmful_category()
                        if harmful_category:
                            category, similarity = harmful_category
                            self.blocked_reason = f"Response contains potentially harmful content related to '{category}' (similarity: {similarity:.2f})"
                        else:
                            self.blocked_reason = "Response contains potentially harmful content"
                        
                        return {
                            "response": self.unsafe_message,
                            "blocked": True,
                            "reason": self.blocked_reason
                        }
                except Exception as e:
                    print(f"Warning: Error during harmful content check: {e}")
                    # Continue with generation despite the error
                
                # Decode the generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the assistant's response
                if "<|assistant|>" in generated_text:
                    response = generated_text.split("<|assistant|>")[-1].strip()
                else:
                    response = generated_text
                    
        except Exception as e:
            return {
                "response": f"Error during generation: {str(e)}",
                "blocked": True,
                "reason": f"Exception: {str(e)}"
            }
        
        # Return results
        return {
            "response": response,
            "blocked": False,
            "reason": None
        } 