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
    def __init__(self, model: torch.nn.Module, model_type: str = None, token_strategy: str = "last"):
        """
        Initialize the hooks manager.
        
        Args:
            model: The transformer model to hook into
            model_type: Type of model ('opt', 'llama', 'gpt2', etc.)
            token_strategy: Strategy for token selection:
                            - "last": Last token in sequence (default)
                            - "target_token": Look for specific tokens like "A" or "B"
                            - "all": Store all tokens for later selection
        """
        self.model = model
        self.model_type = self._detect_model_type(model) if model_type is None else model_type
        self.hooks = {}
        self.layer_activations = {}
        self.active_layers = set()
        self.token_strategy = token_strategy
        self.target_tokens = []  # IDs of tokens to look for (e.g., "A", "B")
        self.last_token_id = None  # ID of the last token processed
        self.last_token_position = -1  # Position of the last token processed
        
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
    
    def set_target_tokens(self, tokenizer, token_texts: List[str]):
        """
        Set target tokens to look for (e.g., "A", "B").
        
        Args:
            tokenizer: The tokenizer to use for encoding
            token_texts: List of token texts to look for
        """
        self.target_tokens = []
        for text in token_texts:
            # Try to encode with and without special tokens to find the right token ID
            tokens_with = tokenizer.encode(text, add_special_tokens=True)
            tokens_without = tokenizer.encode(text, add_special_tokens=False)
            
            # If single token, take it directly
            if len(tokens_without) == 1:
                self.target_tokens.append(tokens_without[0])
            else:
                # Try to find the token in the full encoding
                for token_id in tokens_with:
                    if tokenizer.decode([token_id]) == text:
                        self.target_tokens.append(token_id)
                        break
        
        # Log for debugging
        token_mapping = {token_id: tokenizer.decode([token_id]) for token_id in self.target_tokens}
        print(f"Target tokens set: {token_texts} → IDs: {self.target_tokens} → Decoded: {token_mapping}")
    
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
                    if self.token_strategy == "last":
                        # Original behavior: get last token's activations
                        last_token_idx = hidden_states.shape[1] - 1
                        self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone()
                        self.last_token_position = last_token_idx
                        
                        # Try to get the token ID if available
                        if hasattr(module, '_last_input_ids'):
                            input_ids = module._last_input_ids[0]  # Batch size 1
                            if last_token_idx < len(input_ids):
                                self.last_token_id = input_ids[last_token_idx].item()
                    
                    elif self.token_strategy == "target_token" and len(self.target_tokens) > 0:
                        # Try to find target tokens in the input_ids
                        if hasattr(module, '_last_input_ids'):
                            input_ids = module._last_input_ids[0]  # Batch size 1
                            found = False
                            
                            # Look for target tokens from the end (where A/B would be)
                            for i in range(len(input_ids)-1, -1, -1):
                                token_id = input_ids[i].item()
                                if token_id in self.target_tokens:
                                    self.layer_activations[layer_idx] = hidden_states[:, i, :].detach().clone()
                                    self.last_token_id = token_id
                                    self.last_token_position = i
                                    found = True
                                    break
                            
                            # Fallback to last token if target not found
                            if not found:
                                last_token_idx = hidden_states.shape[1] - 1
                                self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone()
                                if last_token_idx < len(input_ids):
                                    self.last_token_id = input_ids[last_token_idx].item()
                                self.last_token_position = last_token_idx
                    
                    elif self.token_strategy == "all":
                        # Store all tokens for later selection
                        self.layer_activations[layer_idx] = hidden_states.detach().clone()
                        
                        # Save the last token ID for reference
                        if hasattr(module, '_last_input_ids'):
                            input_ids = module._last_input_ids[0]  # Batch size 1
                            last_idx = min(hidden_states.shape[1] - 1, len(input_ids) - 1)
                            self.last_token_id = input_ids[last_idx].item()
                            self.last_token_position = last_idx
                    
                    else:
                        # Default to last token
                        last_token_idx = hidden_states.shape[1] - 1
                        self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone()
                        self.last_token_position = last_token_idx
                        
                        # Try to get the token ID if available
                        if hasattr(module, '_last_input_ids'):
                            input_ids = module._last_input_ids[0]  # Batch size 1
                            if last_token_idx < len(input_ids):
                                self.last_token_id = input_ids[last_token_idx].item()
        
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
                
                # Add a pre-hook to capture input_ids for all strategies
                def forward_pre_hook(module, args):
                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                        # Store the input_ids for use in the activation hook
                        module._last_input_ids = args[0]
                    return args
                
                module.register_forward_pre_hook(forward_pre_hook)
                
                # Add the main activation hook
                hook = module.register_forward_hook(self._activation_hook(layer_idx))
                self.hooks[layer_idx] = hook
            except Exception as e:
                print(f"Failed to register hook for layer {layer_idx} ({layer_name}): {e}")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}
    
    def get_activations(self, layer_idx: Optional[int] = None, token_idx: Optional[int] = None) -> Dict[int, torch.Tensor]:
        """
        Get the stored activations.
        
        Args:
            layer_idx: Specific layer to get activations for. If None, returns all activations.
            token_idx: For "all" strategy, the specific token to extract.
                      Ignored if activations already contain only one token.
            
        Returns:
            Dictionary mapping layer indices to activation tensors
        """
        result = {}
        
        # Get requested activations
        if layer_idx is not None:
            if layer_idx in self.layer_activations:
                activations = {layer_idx: self.layer_activations[layer_idx]}
            else:
                return {}
        else:
            activations = self.layer_activations
        
        # Extract specific token if needed (for "all" strategy)
        for layer, activation in activations.items():
            if token_idx is not None and activation.dim() > 1 and activation.shape[1] > 1:
                # We have multiple tokens stored, extract the specific one
                idx = token_idx
                if idx < 0:
                    idx = activation.shape[1] + idx  # Handle negative indexing
                if 0 <= idx < activation.shape[1]:
                    result[layer] = activation[:, idx, :].detach().clone()
            else:
                result[layer] = activation
        
        return result
    
    def get_last_token_info(self):
        """
        Get information about the last token that was processed.
        
        Returns:
            Dictionary with token ID and position
        """
        return {
            "token_id": self.last_token_id,
            "position": self.last_token_position
        }
    
    def clear_activations(self) -> None:
        """Clear stored activations."""
        self.layer_activations = {}
        self.last_token_id = None
        self.last_token_position = -1 