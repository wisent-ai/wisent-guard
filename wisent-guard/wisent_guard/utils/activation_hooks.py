"""
Hooks for capturing activations from various transformer models
"""

import torch
from typing import Dict, List, Callable, Any, Set, Optional, Union
from .helpers import get_layer_name

class ActivationHooks:
    """
    Class for managing activation hooks on transformer models.
    """
    def __init__(self, model: torch.nn.Module, model_type: str = None):
        """
        Initialize the hooks manager.
        
        Args:
            model: The transformer model to hook into
            model_type: Type of model ('opt', 'llama', 'gpt2', etc.)
        """
        self.model = model
        self.model_type = self._detect_model_type(model) if model_type is None else model_type
        self.hooks = {}
        self.layer_activations = {}
        self.active_layers = set()
        
    def _detect_model_type(self, model: torch.nn.Module) -> str:
        """
        Auto-detect the model type based on model architecture.
        
        Args:
            model: Transformer model
            
        Returns:
            String identifier for the model type
        """
        model_name = model.__class__.__name__.lower()
        
        if 'opt' in model_name:
            return 'opt'
        elif 'llama' in model_name:
            return 'llama'
        elif 'gpt2' in model_name:
            return 'gpt2'
        elif 'neox' in model_name:
            return 'gpt_neox'
        elif 'gptj' in model_name:
            return 'gptj'
        elif 't5' in model_name:
            return 't5'
        elif 'bart' in model_name:
            return 'bart'
        else:
            # Default to a generic approach
            return 'generic'
    
    def _get_module_by_name(self, name: str) -> torch.nn.Module:
        """
        Retrieve a module from the model by its name.
        
        Args:
            name: Dot-separated path to the module
            
        Returns:
            The requested module
        """
        module = self.model
        for part in name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _activation_hook(self, layer_idx: int) -> Callable:
        """
        Create a hook function for the specified layer.
        
        Args:
            layer_idx: Index of the layer to hook
            
        Returns:
            Hook function
        """
        def hook(module, input, output):
            # For most transformer models, we want the output of the attention layer
            # or the MLP layer as our activation vector
            if layer_idx in self.active_layers:
                # Get the output hidden states - typically the first element of the output
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Store the hidden states (activations)
                # We may need to handle different shapes based on model architecture
                if isinstance(hidden_states, torch.Tensor):
                    # Get last token's activations for autoregressive models
                    # Shape is typically [batch_size, seq_len, hidden_dim]
                    last_token_idx = hidden_states.shape[1] - 1
                    self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone()
        
        return hook
    
    def register_hooks(self, layers: List[int]) -> None:
        """
        Register activation hooks for the specified layers.
        
        Args:
            layers: List of layer indices to hook
        """
        self.active_layers = set(layers)
        
        # Clear existing hooks
        self.remove_hooks()
        
        # Register new hooks
        for layer_idx in layers:
            layer_name = get_layer_name(self.model_type, layer_idx)
            try:
                module = self._get_module_by_name(layer_name)
                hook = module.register_forward_hook(self._activation_hook(layer_idx))
                self.hooks[layer_idx] = hook
            except Exception as e:
                print(f"Failed to register hook for layer {layer_idx} ({layer_name}): {e}")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}
    
    def get_activations(self, layer_idx: Optional[int] = None) -> Dict[int, torch.Tensor]:
        """
        Get the stored activations.
        
        Args:
            layer_idx: Specific layer to get activations for. If None, returns all activations.
            
        Returns:
            Dictionary mapping layer indices to activation tensors
        """
        if layer_idx is not None:
            if layer_idx in self.layer_activations:
                return {layer_idx: self.layer_activations[layer_idx]}
            else:
                return {}
        return self.layer_activations
    
    def clear_activations(self) -> None:
        """Clear stored activations."""
        self.layer_activations = {} 