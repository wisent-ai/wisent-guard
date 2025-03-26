"""
Patch for wisent-guard to fix MPS device allocation issues.

This patch ensures that tensors created during hook execution are properly
allocated on the MPS device when running on Apple Silicon.
"""

import torch
from wisent_guard.utils.activation_hooks import ActivationHooks
from wisent_guard.inference import SafeInference

# Store the original hook method to patch
original_activation_hook = ActivationHooks._activation_hook

def patched_activation_hook(self, layer_idx):
    """
    Patched version of the activation hook that ensures MPS compatibility
    by explicitly keeping tensors on the same device as the model.
    """
    # Get the original hook
    original_hook = original_activation_hook(self, layer_idx)
    
    # Create a new hook function that ensures device compatibility
    def hook(module, input, output):
        # Get the current device from the module
        device = next(module.parameters()).device
        
        # Call the original hook function
        original_hook(module, input, output)
        
        # Ensure the activation tensor is on the same device as the model
        if layer_idx in self.layer_activations and device.type == 'mps':
            self.layer_activations[layer_idx] = self.layer_activations[layer_idx].to(device)
    
    return hook

# Store original check_prompt_safety method
original_check_prompt_safety = SafeInference._check_prompt_safety

def patched_check_prompt_safety(self, input_text):
    """
    Patched version of check_prompt_safety that ensures MPS compatibility
    by properly handling device placement.
    """
    # Reset monitor state
    self.monitor.reset()
    
    # Tokenize and encode input
    input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
    
    # Get the model's device and move input_ids to it
    device = next(self.model.parameters()).device
    input_ids = input_ids.to(device)
    
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

# Store original generate method
original_generate = SafeInference.generate

def patched_generate(self, prompt, max_new_tokens=100, skip_prompt_check=False, **kwargs):
    """
    Patched version of generate that ensures MPS compatibility by properly handling device placement.
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
            "response": self.unsafe_message,
            "blocked": True,
            "reason": self.blocked_reason
        }
    
    # Prepare for generation - ensure device placement
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
    device = next(self.model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Set up generation parameters
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        "eos_token_id": self.tokenizer.eos_token_id,
    }
    gen_kwargs.update(kwargs)
    
    # Generate text
    try:
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                **gen_kwargs
            )
            
            # Check for harmful content in the full sequence
            self.monitor.reset()
            _ = self.model(outputs)
            
            if self.monitor.is_harmful() and self.block_on_harmful:
                harmful_category = self.monitor.get_most_harmful_category()
                if harmful_category:
                    category, similarity = harmful_category
                    self.blocked_reason = f"Response contains potentially harmful content related to '{category}' (similarity: {similarity:.2f})"
                else:
                    self.blocked_reason = "Response contains potentially harmful content"
                
                # Return the prompt with a warning
                prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                return {
                    "response": self.unsafe_message,
                    "blocked": True,
                    "reason": self.blocked_reason
                }
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # If the prompt is included in the output, extract just the response
            if prompt in generated_text:
                response = generated_text[len(prompt):].strip()
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

# Apply the patches
ActivationHooks._activation_hook = patched_activation_hook
SafeInference._check_prompt_safety = patched_check_prompt_safety
SafeInference.generate = patched_generate

print("✅ Applied MPS compatibility patches to wisent-guard") 