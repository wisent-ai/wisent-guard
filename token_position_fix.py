"""
Proof of concept implementation for fixing the token positioning issue in wisent-guard.

This file demonstrates how to update the ActivationHooks class to support configurable
token position selection for more consistent activation monitoring between training
and inference.
"""

import torch
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
from wisent_guard.utils.activation_hooks import ActivationHooks

class ImprovedActivationHooks(ActivationHooks):
    """
    Enhanced version of ActivationHooks with configurable token position selection.
    """
    def __init__(self, model: torch.nn.Module, model_type: str = None, token_strategy: str = "last"):
        """
        Initialize the hooks manager with token position strategy.
        
        Args:
            model: The transformer model to hook into
            model_type: Type of model ('opt', 'llama', 'gpt2', etc.)
            token_strategy: Default strategy for token selection:
                            - "last": Always use last token (original behavior)
                            - "second_last": Use second-to-last token (similar to CAA)
                            - "all": Store all tokens for later selection
        """
        super().__init__(model, model_type)
        self.token_strategy = token_strategy
        self.boundary_positions = {}  # Store detected boundary positions
    
    def _activation_hook(self, layer_idx: int) -> Callable:
        """
        Create a hook function for the specified layer with configurable token selection.
        
        Args:
            layer_idx: Index of the layer to hook
            
        Returns:
            Hook function
        """
        def hook(module, input, output):
            if layer_idx in self.active_layers:
                # Get the output hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                if not isinstance(hidden_states, torch.Tensor):
                    return
                
                # Store activations based on token strategy
                if self.token_strategy == "last":
                    # Original behavior: last token only
                    last_token_idx = hidden_states.shape[1] - 1
                    self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone()
                
                elif self.token_strategy == "second_last" and hidden_states.shape[1] > 1:
                    # Second-to-last token (like CAA approach)
                    second_last_idx = hidden_states.shape[1] - 2
                    self.layer_activations[layer_idx] = hidden_states[:, second_last_idx, :].detach().clone()
                
                elif self.token_strategy == "all":
                    # Store all tokens for later selection
                    self.layer_activations[layer_idx] = hidden_states.detach().clone()
                
                else:
                    # Default to last token if strategy not recognized or not applicable
                    last_token_idx = hidden_states.shape[1] - 1
                    self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone()
        
        return hook
    
    def get_activations(self, layer_idx: Optional[int] = None, token_idx: Optional[int] = None) -> Dict[int, torch.Tensor]:
        """
        Get the stored activations with optional token selection.
        
        Args:
            layer_idx: Specific layer to get activations for. If None, returns all activations.
            token_idx: For "all" strategy, the specific token to extract.
                       Ignored if activations already contain only one token.
            
        Returns:
            Dictionary mapping layer indices to activation tensors
        """
        result = {}
        
        # Get activations (all or specific layer)
        activations = {}
        if layer_idx is not None:
            if layer_idx in self.layer_activations:
                activations = {layer_idx: self.layer_activations[layer_idx]}
            else:
                return {}
        else:
            activations = self.layer_activations
        
        # Process activations based on token_idx
        for layer, activation in activations.items():
            if token_idx is not None and activation.dim() > 1 and activation.shape[1] > 1:
                # We have multiple tokens stored (all strategy), extract the specific one
                idx = token_idx
                if idx < 0:
                    idx = activation.shape[1] + idx  # Handle negative indexing
                if 0 <= idx < activation.shape[1]:
                    result[layer] = activation[:, idx, :].detach().clone()
            else:
                # Either no token_idx specified or already a single token
                result[layer] = activation.detach().clone()
        
        return result
    
    def find_boundary_position(self, input_ids: torch.Tensor, tokenizer, boundary_token: str = "<|assistant|>") -> int:
        """
        Find a boundary position in the input sequence.
        
        Args:
            input_ids: Input token IDs tensor
            tokenizer: Tokenizer for encoding boundary tokens
            boundary_token: Token string to search for (e.g., "<|assistant|>")
            
        Returns:
            Index of token after the boundary, or -1 if not found
        """
        # Encode the boundary token
        boundary_ids = tokenizer.encode(boundary_token, add_special_tokens=False)
        
        # Convert to list for easier searching
        input_ids_list = input_ids[0].tolist()
        
        # Find the position after the boundary token
        for i in range(len(input_ids_list) - len(boundary_ids)):
            if input_ids_list[i:i+len(boundary_ids)] == boundary_ids:
                # Return position after the boundary token
                boundary_pos = i + len(boundary_ids)
                # Store this position for later use
                self.boundary_positions[input_ids.shape[0]] = boundary_pos
                return boundary_pos
        
        # Boundary not found
        return -1

# Example usage:
"""
# Initialize with token strategy
hooks = ImprovedActivationHooks(model, token_strategy="all")

# Register hooks for specified layers
hooks.register_hooks([10, 15, 20])

# During training:
# 1. Find the boundary between prompt and response
boundary_pos = hooks.find_boundary_position(input_ids, tokenizer)

# 2. Run forward pass
model(input_ids)

# 3. Get activations from the first token after <|assistant|>
if boundary_pos >= 0:
    harmful_activations = hooks.get_activations(token_idx=boundary_pos)
else:
    # Fallback to second-to-last token
    harmful_activations = hooks.get_activations(token_idx=-2)

# Same approach during inference to ensure consistency
"""

def apply_token_position_fix():
    """
    Monkey patch the ActivationHooks class with our improved version.
    """
    from wisent_guard.utils.activation_hooks import ActivationHooks as OriginalHooks
    
    # Save original methods
    original_init = OriginalHooks.__init__
    original_activation_hook = OriginalHooks._activation_hook
    original_get_activations = OriginalHooks.get_activations
    
    # Apply our improved methods
    OriginalHooks.__init__ = ImprovedActivationHooks.__init__
    OriginalHooks._activation_hook = ImprovedActivationHooks._activation_hook
    OriginalHooks.get_activations = ImprovedActivationHooks.get_activations
    OriginalHooks.find_boundary_position = ImprovedActivationHooks.find_boundary_position
    
    print("✅ Applied token position fix to ActivationHooks class")
    
    # Return function to restore original functionality if needed
    def restore_original():
        OriginalHooks.__init__ = original_init
        OriginalHooks._activation_hook = original_activation_hook
        OriginalHooks.get_activations = original_get_activations
        if hasattr(OriginalHooks, "find_boundary_position"):
            delattr(OriginalHooks, "find_boundary_position")
        print("✅ Restored original ActivationHooks functionality")
    
    return restore_original

if __name__ == "__main__":
    print("Token position fix module for wisent-guard")
    print("Import and call apply_token_position_fix() to apply the fix") 