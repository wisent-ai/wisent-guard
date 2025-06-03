#Hooks for capturing activations from various transformer models

import torch
from typing import Dict, List, Callable, Any, Set, Optional, Union
from ..utils.logger import get_logger
from .helpers import get_layer_name

class ActivationHooks:    
    def __init__(
        self,
        model: torch.nn.Module,
        layers: List[int],
        token_strategy: str = "last",
        log_level: Union[str, int] = "info"
    ):
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
        
        # Initialize active layers
        self.active_layers = set()
        
        # Set up monitoring on specified layers
        self.setup_hooks(layers)
    
    def _detect_model_type(self, model) -> str:
        model_config = getattr(model, "config", None)
        model_name = getattr(model_config, "_name_or_path", "unknown").lower()
        
        if hasattr(model, "get_input_embeddings"):
            if "llama" in model_name:
                return "llama"
            elif "mistral" in model_name:
                return "mistral"
            elif "opt" in model_name:
                return "opt"
            elif "mpt" in model_name:
                return "mpt"
        
        return "generic"
    
    def _get_module_by_name(self, name: str) -> torch.nn.Module:
      
        self.logger.debug(f"Looking up module: {name}")
        module = self.model
        for part in name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def set_target_tokens(self, tokenizer, token_texts: List[str]):
      
        self.logger.warning("set_target_tokens is ignored with simplified 'last' token strategy")
    
    def _activation_hook(self, layer_idx: int) -> Callable:
        def hook_fn(module, input, output):
            # For LLaMA models, we want to capture the output of the MLP layer
            if layer_idx in self.active_layers:
                self.logger.debug(f"Hook triggered for layer {layer_idx}")
                
                # For LLaMA, we expect the output to be a tuple with hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    self.logger.debug(f"Output is tuple, taking first element as hidden states")
                else:
                    hidden_states = output
                
                # Store the activations (last token)
                if isinstance(hidden_states, torch.Tensor):
                    # Get the last token's hidden state
                    last_hidden_state = hidden_states[:, -1, :]
                    self.activations[layer_idx] = last_hidden_state
                    self.logger.debug(f"Captured activation for layer {layer_idx}")
                else:
                    self.logger.warning(f"Unexpected output type: {type(hidden_states)}")
        
        return hook_fn
    
    def setup_hooks(self, layers: List[int]) -> None:
        self.logger.info(f"Setting up hooks for layers: {layers}")
        self.active_layers = set(layers)
        
        # Clear existing hooks
        self.remove_hooks()
        
        # Register new hooks
        hooks_registered = 0
        
        for layer_idx in layers:
            layer_module = self._get_layer_module(layer_idx)
            if layer_module is not None:
                hook = layer_module.register_forward_hook(self._activation_hook(layer_idx))
                self.hooks[layer_idx] = hook
                hooks_registered += 1
                self.logger.debug(f"Successfully registered hook for layer {layer_idx}")
            else:
                self.logger.error(f"Failed to register hook for layer {layer_idx}")
        
        self.logger.info(f"Registered {hooks_registered} hooks out of {len(layers)} requested layers")
    
    def remove_hooks(self) -> None:
        if self.hooks:
            self.logger.info(f"Removing {len(self.hooks)} hooks")
            for layer_idx, hook in self.hooks.items():
                self.logger.debug(f"Removing hook for layer {layer_idx}")
                hook.remove()
            self.hooks = {}
        else:
            self.logger.debug("No hooks to remove")
    
    def has_activations(self) -> bool:
        has_act = bool(self.activations)
        self.logger.debug(f"has_activations check: {has_act}")
        return has_act
    
    def get_activations(self) -> Dict[int, torch.Tensor]:
        return self.activations
    
    def get_last_token_info(self) -> Dict[str, Any]:
        return {
            "token_id": None,
            "position": None
        }
    
    def reset(self):
        self.activations = {}
    
    def clear_activations(self):
        pass

    def _get_layer_module(self, layer_idx: int) -> Optional[torch.nn.Module]:
        try:
            if self.model_type == "llama":
                return self.model.model.layers[layer_idx]
            elif self.model_type == "mistral":
                return self.model.model.layers[layer_idx]
            elif self.model_type == "opt":
                return self.model.model.decoder.layers[layer_idx]
            else:
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
        return self.activations.get(layer_idx) 