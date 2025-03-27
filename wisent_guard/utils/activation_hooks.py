"""
Hooks for capturing activations from various transformer models
"""

import torch
from typing import Dict, List, Callable, Any, Set, Optional, Union
from .helpers import get_layer_name
from .logger import get_logger

class ActivationHooks:
    """
    Class for managing activation hooks on transformer models.
    """
    def __init__(self, model: torch.nn.Module, model_type: str = None, token_strategy: str = "last", log_level: str = "info"):
        """
        Initialize the hooks manager.
        
        Args:
            model: The transformer model to hook into
            model_type: Type of model ('opt', 'llama', 'gpt2', etc.)
            token_strategy: Strategy for token selection:
                            - "last": Last token in sequence (default)
                            - "target_token": Look for specific tokens like "A" or "B"
                            - "all": Store all tokens for later selection
            log_level: Logging level ('debug', 'info', 'warning', 'error')
        """
        self.logger = get_logger(name="wisent_guard.hooks", level=log_level)
        self.logger.info("Initializing activation hooks")
        
        self.model = model
        self.model_type = self._detect_model_type(model) if model_type is None else model_type
        self.logger.info(f"Using model type: {self.model_type}")
        
        self.hooks = {}
        self.layer_activations = {}
        self.active_layers = set()
        self.token_strategy = token_strategy
        self.logger.info(f"Using token strategy: {self.token_strategy}")
        
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
        self.logger.debug("Attempting to auto-detect model type")
        model_name = model.__class__.__name__.lower()
        self.logger.debug(f"Model class name: {model_name}")
        
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
            self.logger.warning(f"Could not detect specific model type from '{model_name}', using generic")
            return 'generic'
    
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
        Set target tokens to look for (e.g., "A", "B").
        
        Args:
            tokenizer: The tokenizer to use for encoding
            token_texts: List of token texts to look for
        """
        self.logger.info(f"Setting target tokens: {token_texts}")
        self.target_tokens = []
        for text in token_texts:
            # Try to encode with and without special tokens to find the right token ID
            tokens_with = tokenizer.encode(text, add_special_tokens=True)
            tokens_without = tokenizer.encode(text, add_special_tokens=False)
            
            self.logger.debug(f"Token '{text}' - with special tokens: {tokens_with}")
            self.logger.debug(f"Token '{text}' - without special tokens: {tokens_without}")
            
            # If single token, take it directly
            if len(tokens_without) == 1:
                self.target_tokens.append(tokens_without[0])
                self.logger.debug(f"Found single token ID for '{text}': {tokens_without[0]}")
            else:
                # Try to find the token in the full encoding
                found = False
                for token_id in tokens_with:
                    decoded = tokenizer.decode([token_id])
                    self.logger.debug(f"Checking token ID {token_id}, decodes to: '{decoded}'")
                    if decoded == text:
                        self.target_tokens.append(token_id)
                        self.logger.debug(f"Found matching token ID for '{text}': {token_id}")
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"Could not find exact token ID for '{text}'")
        
        # Log for debugging
        token_mapping = {token_id: tokenizer.decode([token_id]) for token_id in self.target_tokens}
        self.logger.info(f"Target tokens set: {token_texts} → IDs: {self.target_tokens} → Decoded: {token_mapping}")
    
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
                    
                    if self.token_strategy == "last":
                        # Original behavior: get last token's activations
                        last_token_idx = hidden_states.shape[1] - 1
                        self.logger.debug(f"Using last token strategy, token position: {last_token_idx}")
                        
                        # Ensure tensor is properly allocated on the device (important for MPS)
                        self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone().to(device)
                        self.last_token_position = last_token_idx
                        
                        # Try to get the token ID if available
                        if hasattr(module, '_last_input_ids'):
                            input_ids = module._last_input_ids[0]  # Batch size 1
                            if last_token_idx < len(input_ids):
                                self.last_token_id = input_ids[last_token_idx].item()
                                self.logger.debug(f"Last token ID: {self.last_token_id}, position: {last_token_idx}")
                    
                    elif self.token_strategy == "target_token" and len(self.target_tokens) > 0:
                        self.logger.debug(f"Using target token strategy, looking for tokens: {self.target_tokens}")
                        
                        # Try to find target tokens in the input_ids
                        if hasattr(module, '_last_input_ids'):
                            input_ids = module._last_input_ids[0]  # Batch size 1
                            self.logger.debug(f"Input IDs: {input_ids[:10]}... (showing first 10)")
                            
                            found = False
                            
                            # Look for target tokens from the end (where A/B would be)
                            for i in range(len(input_ids)-1, -1, -1):
                                token_id = input_ids[i].item()
                                if token_id in self.target_tokens:
                                    self.logger.debug(f"Found target token {token_id} at position {i}")
                                    # Ensure tensor is properly allocated on the device (important for MPS)
                                    self.layer_activations[layer_idx] = hidden_states[:, i, :].detach().clone().to(device)
                                    self.last_token_id = token_id
                                    self.last_token_position = i
                                    found = True
                                    break
                            
                            # Fallback to last token if target not found
                            if not found:
                                last_token_idx = hidden_states.shape[1] - 1
                                self.logger.debug(f"Target token not found, using last token at position {last_token_idx}")
                                # Ensure tensor is properly allocated on the device (important for MPS)
                                self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone().to(device)
                                if last_token_idx < len(input_ids):
                                    self.last_token_id = input_ids[last_token_idx].item()
                                self.last_token_position = last_token_idx
                    
                    elif self.token_strategy == "all":
                        # Store all tokens for later selection
                        self.logger.debug(f"Using 'all' strategy, storing the entire sequence of length {hidden_states.shape[1]}")
                        # Ensure tensor is properly allocated on the device (important for MPS)
                        self.layer_activations[layer_idx] = hidden_states.detach().clone().to(device)
                        
                        # Save the last token ID for reference
                        if hasattr(module, '_last_input_ids'):
                            input_ids = module._last_input_ids[0]  # Batch size 1
                            last_idx = min(hidden_states.shape[1] - 1, len(input_ids) - 1)
                            self.last_token_id = input_ids[last_idx].item()
                            self.last_token_position = last_idx
                            self.logger.debug(f"Saved last token ID: {self.last_token_id}, position: {last_idx}")
                    
                    else:
                        # Default to last token
                        last_token_idx = hidden_states.shape[1] - 1
                        self.logger.debug(f"Using default strategy (last token), position: {last_token_idx}")
                        # Ensure tensor is properly allocated on the device (important for MPS)
                        self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone().to(device)
                        self.last_token_position = last_token_idx
                        
                        # Try to get the token ID if available
                        if hasattr(module, '_last_input_ids'):
                            input_ids = module._last_input_ids[0]  # Batch size 1
                            if last_token_idx < len(input_ids):
                                self.last_token_id = input_ids[last_token_idx].item()
                                self.logger.debug(f"Last token ID: {self.last_token_id}")
        
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
        if not self.has_activations():
            self.logger.warning("No activations to retrieve")
            return {}
        
        self.logger.debug(f"Getting activations: layer_idx={layer_idx}, token_idx={token_idx}")
            
        result = {}
        
        # Get requested activations
        if layer_idx is not None:
            if layer_idx in self.layer_activations:
                self.logger.debug(f"Retrieving activations for specific layer {layer_idx}")
                activation = self.layer_activations[layer_idx]
                
                # For "all" strategy and if token_idx is specified, get only that token
                if token_idx is not None and self.token_strategy == "all" and len(activation.shape) > 1 and activation.shape[1] > 1:
                    self.logger.debug(f"Getting specific token {token_idx} from 'all' strategy activations")
                    if token_idx < activation.shape[1]:
                        result[layer_idx] = activation[:, token_idx, :]
                        self.logger.debug(f"Retrieved activation for token {token_idx}")
                    else:
                        # If token_idx is out of range, fall back to last token
                        self.logger.warning(f"Token index {token_idx} out of range, using last token instead")
                        result[layer_idx] = activation[:, -1, :]
                else:
                    result[layer_idx] = activation
                    self.logger.debug(f"Retrieved activation with shape: {activation.shape}")
            else:
                self.logger.warning(f"No activations found for layer {layer_idx}")
        else:
            # Return all activations
            self.logger.debug(f"Retrieving activations for all layers: {list(self.layer_activations.keys())}")
            for layer, activation in self.layer_activations.items():
                # For "all" strategy and if token_idx is specified, get only that token
                if token_idx is not None and self.token_strategy == "all" and len(activation.shape) > 1 and activation.shape[1] > 1:
                    self.logger.debug(f"Getting specific token {token_idx} from 'all' strategy activations for layer {layer}")
                    if token_idx < activation.shape[1]:
                        result[layer] = activation[:, token_idx, :]
                    else:
                        # If token_idx is out of range, fall back to last token
                        self.logger.warning(f"Token index {token_idx} out of range for layer {layer}, using last token")
                        result[layer] = activation[:, -1, :]
                else:
                    result[layer] = activation
                    self.logger.debug(f"Retrieved activation for layer {layer} with shape: {activation.shape}")
        
        self.logger.debug(f"Returning activations for {len(result)} layers")
        return result
    
    def get_last_token_info(self):
        """
        Get information about the last token that was processed.
        
        Returns:
            Dictionary with token ID and position
        """
        info = {
            "token_id": self.last_token_id,
            "position": self.last_token_position
        }
        self.logger.debug(f"Last token info: {info}")
        return info
    
    def clear_activations(self) -> None:
        """Clear stored activations."""
        self.logger.debug("Clearing activations")
        self.layer_activations = {}
        self.last_token_id = None
        self.last_token_position = -1 