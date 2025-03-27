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
    
    def generate_token_by_token(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        skip_prompt_check: bool = False,
        return_token_scores: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text token-by-token while monitoring each token for harmful content.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            skip_prompt_check: Whether to skip the initial prompt safety check
            return_token_scores: Whether to include similarity scores for each token
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated text, safety information, and token scores
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
                "reason": self.blocked_reason,
                "token_scores": []
            }
            
        # Format the prompt if a formatter is available, otherwise use simple format
        if hasattr(self, 'format_prompt') and callable(self.format_prompt):
            formatted_prompt = self.format_prompt(prompt)
        else:
            # Simple format without system prompt
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
            
        # Prepare for token-by-token generation
        try:
            device = next(self.model.parameters()).device
            input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
            
            # Start with just the input prompt
            generated_sequence = input_ids.clone()
            blocked = False
            blocking_token = None
            blocking_reason = None
            
            # For tracking token-level scores
            token_scores = []
            
            # Generate one token at a time and check each one
            for i in range(max_new_tokens):
                # Generate the next token
                with torch.no_grad():
                    outputs = self.model.generate(
                        generated_sequence,
                        max_new_tokens=1,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **kwargs
                    )
                    
                    # Get the new token
                    next_token = outputs.sequences[:, -1].unsqueeze(-1)
                    
                    # Append to the sequence
                    generated_sequence = torch.cat([generated_sequence, next_token], dim=-1)
                    
                    # Decode the token for display
                    token_text = self.tokenizer.decode(next_token[0])
                    
                    # Process for harmfulness
                    self.monitor.reset()
                    
                    # Run the model on the sequence to get activations
                    _ = self.model(generated_sequence)
                    
                    # Check harmfulness with is_response_token=True since this is a generated token
                    results = self.monitor.check_activations(is_response_token=True)
                    
                    # Find max similarity across all categories
                    max_similarity = 0.0
                    max_category = None
                    
                    for category, category_result in results.items():
                        if category_result["max_similarity"] > max_similarity:
                            max_similarity = category_result["max_similarity"]
                            max_category = category
                    
                    # Check if harmful based on threshold
                    is_harmful = max_similarity >= self.monitor.threshold
                    
                    # Record token score information
                    token_info = {
                        "token_id": next_token.item(),
                        "token_text": token_text,
                        "position": i,
                        "similarity": float(max_similarity),
                        "category": max_category,
                        "is_harmful": is_harmful
                    }
                    token_scores.append(token_info)
                    
                    # Block if harmful and blocking is enabled
                    if is_harmful and self.block_on_harmful and not blocked:
                        blocked = True
                        blocking_token = token_text
                        blocking_reason = f"Token {i+1} ('{token_text}') exceeded similarity threshold: {max_similarity:.4f} for category '{max_category}'"
                        break
                    
                    # Check if we've reached the end of the sequence
                    if next_token[0].item() == self.tokenizer.eos_token_id:
                        break
            
            # Decode the full response (minus the prompt)
            prompt_length = len(input_ids[0])
            response_ids = generated_sequence[0, prompt_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Prepare result
            result = {
                "response": response,
                "blocked": blocked,
                "reason": blocking_reason
            }
            
            # Include token scores if requested
            if return_token_scores:
                result["token_scores"] = token_scores
                
            return result
                
        except Exception as e:
            return {
                "response": f"Error during token-by-token generation: {str(e)}",
                "blocked": True,
                "reason": f"Exception: {str(e)}",
                "token_scores": token_scores if 'token_scores' in locals() else []
            }
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        skip_prompt_check: bool = False,
        token_by_token: bool = False,
        return_token_scores: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text while monitoring for harmful content.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            skip_prompt_check: Whether to skip the initial prompt safety check
            token_by_token: Whether to use token-by-token generation and checking
            return_token_scores: Whether to include similarity scores for each token
            **kwargs: Additional keyword arguments for the generation function
            
        Returns:
            Dictionary containing the generated text and safety information
        """
        # Use token-by-token generation if requested
        if token_by_token:
            return self.generate_token_by_token(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                skip_prompt_check=skip_prompt_check,
                return_token_scores=return_token_scores,
                **kwargs
            )
            
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