"""
Hooks for capturing activations from various transformer models
"""

import torch
from typing import Dict, List, Callable, Any, Set, Optional, Union
from ..utils.logger import get_logger
from .helpers import get_layer_name

class ActivationHooks:
    """Manages activation hooks for transformer models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        layers: List[int],
        token_strategy: str = "last",
        log_level: Union[str, int] = "info"
    ):
        """
        Initialize activation hooks for a model.
        
        Args:
            model: Transformer model
            layers: List of layers to monitor
            token_strategy: Strategy for token activations (only 'last' is supported)
            log_level: Logging level
        """
        self.model = model
        self.layers = layers
        self.token_strategy = token_strategy
        self.logger = get_logger("wisent_guard", log_level)
        
        # Determine model type
        model_type = self._detect_model_type(model)
        self.model_type = model_type
        self.logger.info(f"Using model type: {model_type}")
        
        # Always use last token strategy
        self.logger.info("Using last token strategy")
        
        # Initialize hook storage
        self.hooks = {}
        self.activations = {}
        
        # Set up monitoring on specified layers
        self.setup_hooks(layers)
    
    def _detect_model_type(self, model) -> str:
        """Detect model type."""
        model_config = getattr(model, "config", None)
        model_name = getattr(model_config, "_name_or_path", "unknown").lower()
        
        if hasattr(model, "get_input_embeddings"):
            if "llama" in model_name:
                return "llama"
            elif "mistral" in model_name:
                return "mistral"
            elif "mpt" in model_name:
                return "mpt"
        
        return "generic"
    
    def _get_module_by_name(self, name: str) -> torch.nn.Module:
        """
        Retrieve a module from the model by its name.
        
        Args:
            name: Dot-separated path to the module
            
        Returns:
            The requested module
        """
        self.logger.debug(f"Looking up module: {name}")
        module = self.model
        for part in name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def set_target_tokens(self, tokenizer, token_texts: List[str]):
        """
        Set target tokens to look for (kept for backward compatibility).
        
        Args:
            tokenizer: The tokenizer to use for encoding
            token_texts: List of token texts to look for (ignored)
        """
        self.logger.warning("set_target_tokens is ignored with simplified 'last' token strategy")
    
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
                self.logger.debug(f"Hook triggered for layer {layer_idx}")
                
                # Get the output hidden states - typically the first element of the output
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    self.logger.debug(f"Output is tuple, taking first element as hidden states")
                else:
                    hidden_states = output
                    self.logger.debug(f"Output is tensor, using directly as hidden states")
                
                # Store the hidden states (activations)
                # We may need to handle different shapes based on model architecture
                if isinstance(hidden_states, torch.Tensor):
                    # Get the current device for consistent tensor allocation
                    device = hidden_states.device
                    self.logger.debug(f"Hidden states shape: {hidden_states.shape}, device: {device}")
                    
                    # Get last token's activations
                    last_token_idx = hidden_states.shape[1] - 1
                    self.logger.debug(f"Using last token strategy, token position: {last_token_idx}")
                    
                    # Ensure tensor is properly allocated on the device (important for MPS)
                    self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone().to(device)
                    self.last_token_position = last_token_idx
                    
                    # Try to get the token ID if available
                    if hasattr(module, '_last_input_ids'):
                        input_ids = module._last_input_ids[0]  # Batch size 1
                        if last_token_idx < len(input_ids):
                            try:
                                self.last_token_id = input_ids[last_token_idx].item()
                                self.logger.debug(f"Last token ID: {self.last_token_id}, position: {last_token_idx}")
                            except (RuntimeError, ValueError) as e:
                                self.logger.debug(f"Error extracting last token ID: {e}")
                                self.last_token_id = None
                else:
                    self.logger.warning(f"Hidden states is not a tensor: {type(hidden_states)}")
        return hook
    
    def register_hooks(self, layers: List[int]) -> None:
        """
        Register activation hooks for the specified layers.
        
        Args:
            layers: List of layer indices to hook
        """
        self.logger.info(f"Registering hooks for layers: {layers}")
        self.active_layers = set(layers)
        
        # Clear existing hooks
        self.remove_hooks()
        
        # Register new hooks
        hooks_registered = 0
        
        for layer_idx in layers:
            layer_name = get_layer_name(self.model_type, layer_idx)
            self.logger.debug(f"Attempting to register hook for layer {layer_idx} ({layer_name})")
            
            try:
                module = self._get_module_by_name(layer_name)
                
                # Add a pre-hook to capture input_ids for all strategies
                def forward_pre_hook(module, args):
                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                        # Store the input_ids for use in the activation hook
                        module._last_input_ids = args[0]
                        self.logger.debug(f"Pre-hook captured input_ids shape: {args[0].shape}")
                    return args
                
                module.register_forward_pre_hook(forward_pre_hook)
                
                # Add the main activation hook
                hook = module.register_forward_hook(self._activation_hook(layer_idx))
                self.hooks[layer_idx] = hook
                hooks_registered += 1
                
                self.logger.debug(f"Successfully registered hook for layer {layer_idx}")
            except Exception as e:
                self.logger.error(f"Failed to register hook for layer {layer_idx} ({layer_name}): {e}")
        
        self.logger.info(f"Registered {hooks_registered} hooks out of {len(layers)} requested layers")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        if self.hooks:
            self.logger.info(f"Removing {len(self.hooks)} hooks")
            for layer_idx, hook in self.hooks.items():
                self.logger.debug(f"Removing hook for layer {layer_idx}")
                hook.remove()
            self.hooks = {}
        else:
            self.logger.debug("No hooks to remove")
    
    def has_activations(self) -> bool:
        """
        Check if we have collected any activations.
        
        Returns:
            True if activations have been collected, False otherwise
        """
        has_act = bool(self.layer_activations)
        self.logger.debug(f"has_activations check: {has_act}")
        return has_act
    
    def get_activations(self) -> Dict[int, torch.Tensor]:
        """
        Get all activations for registered layers.
        
        Returns:
            Dictionary mapping layer indices to activation tensors
        """
        return self.layer_activations
    
    def get_last_token_info(self) -> Dict[str, Any]:
        """
        Get information about the last token processed.
        
        Returns:
            Dictionary with token_id and position
        """
        return {
            "token_id": self.last_token_id,
            "position": self.last_token_position
        }
    
    def reset(self):
        """Reset stored activations."""
        self.activations = {}
    
    def clear_activations(self):
        """Clear all stored activations (alias for reset)."""
        self.reset()

    def setup_hooks(self, layers: List[int]):
        """
        Set up hooks for specified layers.
        
        Args:
            layers: List of layer indices to hook
        """
        self.logger.info(f"Monitoring {len(layers)} specified layers: {layers}")
        
        registered = 0
        for layer_idx in layers:
            success = self.register_hook(layer_idx)
            if success:
                registered += 1
        
        self.logger.info(f"Registered {registered} hooks out of {len(layers)} requested layers")

    def register_hook(self, layer_idx: int) -> bool:
        """
        Register a hook for a specific layer.
        
        Args:
            layer_idx: Layer index to hook
            
        Returns:
            True if hook was registered successfully, False otherwise
        """
        # Get the layer module
        layer_module = self._get_layer_module(layer_idx)
        if layer_module is None:
            self.logger.warning(f"Could not find layer module for layer {layer_idx}")
            return False
        
        # Create the hook
        def hook_fn(module, input, output):
            # Get the activations
            if isinstance(output, tuple):
                # For layers that return multiple tensors
                hidden_states = output[0]
            else:
                # For layers that return a single tensor
                hidden_states = output
            
            # Store the activations (last token)
            self.activations[layer_idx] = hidden_states[:, -1].detach()
        
        # Register the hook
        hook = layer_module.register_forward_hook(hook_fn)
        self.hooks[layer_idx] = hook
        
        self.logger.debug(f"Registered hook for layer {layer_idx}")
        return True

    def _get_layer_module(self, layer_idx: int) -> Optional[torch.nn.Module]:
        """
        Get the module for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Module for the layer
        """
        if self.model_type == "llama":
            try:
                return self.model.model.layers[layer_idx]
            except (AttributeError, IndexError):
                try:
                    return self.model.layers[layer_idx]
                except (AttributeError, IndexError):
                    self.logger.warning(f"Could not find layer {layer_idx} in Llama model")
                    return None
        elif self.model_type == "mistral":
            try:
                return self.model.model.layers[layer_idx]
            except (AttributeError, IndexError):
                try:
                    return self.model.layers[layer_idx]
                except (AttributeError, IndexError):
                    self.logger.warning(f"Could not find layer {layer_idx} in Mistral model")
                    return None
        else:
            # Generic approach
            try:
                # Try common patterns
                if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    # GPT-2 style
                    return self.model.transformer.h[layer_idx]
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    # Common pattern
                    return self.model.model.layers[layer_idx]
                elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                    # BERT style
                    return self.model.encoder.layer[layer_idx]
                else:
                    self.logger.warning(f"Could not find layer {layer_idx} with generic approach")
                    return None
            except (AttributeError, IndexError) as e:
                self.logger.warning(f"Error accessing layer {layer_idx}: {e}")
                return None

    def get_activation(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Get activation for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Activation tensor for the layer
        """
        return self.activations.get(layer_idx) 