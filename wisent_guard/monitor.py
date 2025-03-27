"""
Activation monitoring module for tracking activations in real-time
"""

import logging
import torch
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from .utils.activation_hooks import ActivationHooks
from .utils.helpers import cosine_sim, get_layer_count
from .vectors import ContrastiveVectors
from .utils.logger import get_logger
from transformers import PreTrainedModel

class ActivationMonitor:
    """
    Monitors and captures activations from model layers during inference.
    
    This class creates hooks into transformer layers to extract activation patterns
    during text generation, which can be used for hallucination detection.
    """
    
    def __init__(
        self,
        model: "PreTrainedModel",
        layers: List[int],
        vectors: Optional["ContrastiveVectors"] = None,
        token_strategy: str = "last",  # Only 'last' is supported
        similarity_threshold: float = 0.5,
        device: Optional[torch.device] = None,
        log_level: Union[str, int] = "info",
        guard: Optional["ActivationGuard"] = None  # Reference to parent guard
    ):
        """
        Initialize activation monitor.
        
        Args:
            model: Hugging Face model
            layers: Layers to monitor
            vectors: Contrastive vectors for comparisons 
            token_strategy: Strategy for aggregating token activations (only 'last' supported)
            similarity_threshold: Similarity threshold for harmful content detection
            device: Device to use
            log_level: Logging level
            guard: Reference to parent guard
        """
        self.logger = get_logger("wisent_guard", log_level)
        self.logger.info(f"Creating activation monitor for {getattr(model.config, '_name_or_path', 'unknown')}")
        
        self.model = model
        self.layers = layers
        self.vectors = vectors
        self.guard = guard  # Store reference to parent guard for classifier access
        self.similarity_threshold = similarity_threshold
        self.device = device or next(model.parameters()).device
        
        # Set up activation hooks
        self.logger.info("Initializing activation hooks")
        self.hooks = ActivationHooks(
            model=model,
            layers=layers,
            token_strategy="last",  # Always use last token strategy
            log_level=log_level
        )
        
        # Initialize storage for activations
        self.reset()
        
        # Flag to track if activations have been captured
        self.has_activations = False
        self.activations_by_layer = {}
        
        # Token data storage
        self.token_data = {}
        
        # If using vectors, initialize tracking variables
        if self.vectors is not None:
            self.threshold = self.similarity_threshold
            self.max_similarities = {}
            self.triggered_layers = {}
            self.token_strategy = "last"  # Always using "last" strategy
            self.triggering_token_id = None
            self.is_triggered = False
        else:
            self.logger.debug("Operating in classifier-only mode (no vectors)")
        
    def setup_monitoring(self, layers: List[int] = None):
        """
        Set up monitoring on specific layers or all layers.
        
        Args:
            layers: List of layer indices to monitor, or None for all layers
        """
        target_layers = []
        
        # If no layers specified, set up for all layers
        if layers is None:
            # Get layer count based on the detected model type
            model_type = self.hooks.model_type
            layer_count = get_layer_count(self.model, model_type=model_type)
            
            if layer_count > 0:
                target_layers = list(range(layer_count))
                self.logger.info(f"Monitoring all {layer_count} layers")
            else:
                self.logger.warning(f"Could not determine layer count for {model_type}, monitoring first 24 layers")
                target_layers = list(range(24))  # Default to 24 layers
        else:
            target_layers = layers
            self.logger.info(f"Monitoring {len(target_layers)} specified layers: {target_layers}")
        
        # Set up hooks for the layers
        self.hooks.register_hooks(target_layers)
        
    def get_activations(self, layer_idx: int = None) -> Dict[int, torch.Tensor]:
        """
        Get activations after generation.
        
        Args:
            layer_idx: Specific layer to get activations for, or None for all
            
        Returns:
            Dictionary mapping layer indices to activation tensors
        """
        if not self.has_activations:
            self.logger.warning("No activations captured yet")
        
        if layer_idx is not None:
            if layer_idx in self.activations_by_layer:
                return {layer_idx: self.activations_by_layer[layer_idx]}
            else:
                self.logger.warning(f"No activations for layer {layer_idx}")
                return {}
        
        return self.activations_by_layer
    
    def get_activation_for_layer(self, layer: int) -> Optional[torch.Tensor]:
        """
        Get the activation tensor for a specific layer.
        
        Args:
            layer: Layer index to get activation for
            
        Returns:
            Activation tensor for the specified layer
        """
        if not self.has_activations:
            self.logger.warning("No activations captured yet")
            return None
            
        if layer in self.activations_by_layer:
            return self.activations_by_layer[layer]
        else:
            self.logger.warning(f"No activations for layer {layer}")
            return None
    
    def capture_activations(self, input_ids=None, tokenizer=None, target_tokens=None):
        """
        Process the captured activations after generation.
        
        Args:
            input_ids: Optional input_ids to record for reference
            tokenizer: Optional tokenizer for decoding tokens
            target_tokens: Optional target tokens to search for
        """
        # Always processing with last token strategy
        layer_activations = self.hooks.get_layer_activations()
        
        if not layer_activations:
            self.logger.warning("No activations captured by hooks")
            self.has_activations = False
            return
        
        # Store all layer activations
        self.activations_by_layer = layer_activations
        self.has_activations = True
        
        # Debug logging
        if self.logger.isEnabledFor(logging.DEBUG):
            for layer, tensor in self.activations_by_layer.items():
                self.logger.debug(f"Layer {layer} activation tensor shape: {tensor.shape}")
                
            if tokenizer and input_ids is not None:
                decoded = tokenizer.decode(input_ids[0].tolist())
                self.logger.debug(f"Input sequence: {decoded}")
                
                # Log the token position that was captured
                token_pos = self.hooks.last_token_position
                token_id = self.hooks.last_token_id
                
                if token_id is not None and token_pos >= 0:
                    if token_pos < len(input_ids[0]):
                        token = tokenizer.decode([token_id])
                        self.logger.debug(f"Captured token at position {token_pos}: '{token}' (ID: {token_id})")
    
    def reset(self):
        """
        Reset the monitor, clearing all captured activations.
        """
        self.activations_by_layer = {}
        self.has_activations = False
        self.num_tokens = 0
        self.current_token_ids = None
        self.token_data = {'has_activation_values': False}
        self.hooks.reset()
        self.logger.debug("Monitor reset, all activations cleared")

    def check_activations(self) -> Dict[str, Dict[str, Any]]:
        """
        Check current activations against stored vectors.
        
        Returns:
            Dictionary with results per category
        """
        if not self.has_activations:
            self.logger.warning("No activations to check")
            return {}
        
        if self.vectors is None:
            self.logger.warning("No vectors available for comparison")
            return {}
        
        # Get results for each category
        results = {}
        categories = self.vectors.get_available_categories()
        
        for category in categories:
            category_result = {
                "max_similarity": 0.0,
                "is_harmful": False,
                "layer": None
            }
            
            # Check each layer
            for layer in self.layers:
                # Skip if we don't have activations for this layer
                if layer not in self.activations_by_layer:
                    continue
                
                # Get activation
                activation = self.activations_by_layer[layer]
                
                # Get contrastive vector
                contrastive_vector = self.vectors.get_contrastive_vector(category, layer)
                if contrastive_vector is None:
                    continue
                
                # Calculate similarity
                similarity = torch.nn.functional.cosine_similarity(
                    activation.to(self.device).view(1, -1),  # Reshape to [1, hidden_size]
                    contrastive_vector.to(self.device).view(1, -1),  # Reshape to [1, hidden_size]
                    dim=1  # Calculate along the feature dimension
                ).item()  # Get the scalar
                
                # Check if this is the highest similarity
                if similarity > category_result["max_similarity"]:
                    category_result["max_similarity"] = similarity
                    category_result["layer"] = layer
                    category_result["is_harmful"] = similarity >= self.similarity_threshold
            
            # Add category result
            results[category] = category_result
        
        return results
    
    def get_most_harmful_category(self) -> Optional[Tuple[str, float]]:
        """
        Get the category with the highest similarity to harmful patterns.
        
        Returns:
            Tuple of (category_name, similarity) or None if no categories are triggered
        """
        self.logger.debug("Finding most harmful category")
        max_similarity = -1.0
        max_category = None
        
        for category, layer_similarities in self.max_similarities.items():
            if not layer_similarities:
                continue
                
            category_max = max(layer_similarities.values())
            self.logger.debug(f"Category '{category}' max similarity: {category_max:.4f}")
            
            if category_max > max_similarity:
                max_similarity = category_max
                max_category = category
                self.logger.debug(f"New most harmful category: '{category}' ({category_max:.4f})")
        
        if max_category is not None:
            self.logger.info(f"Most harmful category: '{max_category}' with similarity {max_similarity:.4f}")
            return (max_category, max_similarity)
        
        self.logger.debug("No harmful categories found")
        return None
    
    def is_harmful(self, categories: Optional[List[str]] = None, is_response_token: bool = False) -> bool:
        """
        Check if current activations indicate harmful content.
        
        Args:
            categories: Categories to check. If None, all available categories are checked.
            is_response_token: Whether we're checking a token from the model's response.
            
        Returns:
            True if harmful content is detected, False otherwise
        """
        # Check if we're in classifier-only mode (no vectors)
        if not hasattr(self, 'vectors') or self.vectors is None:
            self.logger.debug("Operating in classifier-only mode, returning False for individual token check")
            # In classifier-only mode, individual token checks always return False
            # since we only classify the full response in generate_safe_response
            return False
        
        self.logger.debug("Checking if content is harmful")
        results = self.check_activations()
        
        # Check if any category is flagged as harmful
        for category, category_result in results.items():
            if category_result["is_harmful"]:
                if is_response_token:
                    self.logger.info(f"Response content flagged as harmful in category '{category}'")
                else:
                    self.logger.info(f"Content flagged as harmful in category '{category}'")
                return True
        
        self.logger.debug("Content not flagged as harmful")
        return False
    
    def get_token_classification(self) -> Optional[str]:
        """
        For multiple-choice format, get the classification of the current token.
        
        Returns:
            "truthful" if the token is A, "hallucinatory" if the token is B, or None if not detected
        """
        if self.token_strategy != "target_token" or self.triggering_token_id is None:
            self.logger.debug("Cannot classify token: not using target_token strategy or no triggering token")
            return None
        
        # Get the token text
        token_text = None
        try:
            token_text = self.model.config.tokenizer.decode([self.triggering_token_id])
            self.logger.debug(f"Decoded token text: '{token_text}'")
        except:
            try:
                # Try a different way to access the tokenizer
                if hasattr(self.model, "tokenizer"):
                    token_text = self.model.tokenizer.decode([self.triggering_token_id])
                    self.logger.debug(f"Decoded token text (alternative method): '{token_text}'")
            except:
                self.logger.warning("Failed to decode token text")
                pass
        
        if token_text == "A":
            self.logger.info("Token classified as 'truthful' (A)")
            return "truthful"
        elif token_text == "B":
            self.logger.info("Token classified as 'hallucinatory' (B)")
            return "hallucinatory"
        
        self.logger.debug(f"Token '{token_text}' not classified as either A or B")
        return None

    def capture_activations_from_forward(self, model_output, input_ids=None):
        """
        Capture activations from model output hidden states.
        
        Args:
            model_output: Output from model forward pass
            input_ids: Input token IDs
        
        Returns:
            True if activations were captured, False otherwise
        """
        # Reset previous activations
        self.reset()
        
        # Extract hidden states from output
        if hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:
            hidden_states = model_output.hidden_states
            
            # For each monitored layer, extract and store activations
            for i, layer_idx in enumerate(self.layers):
                # Check if layer index is valid
                if 0 <= layer_idx < len(hidden_states):
                    # Get layer hidden states
                    layer_hidden_states = hidden_states[layer_idx]
                    
                    # Store the current token's hidden state for this layer
                    # Always get the last position since that's the current token being generated
                    self.activations_by_layer[layer_idx] = layer_hidden_states[:, -1].detach()
                
            # Set flag based on whether we captured any activations
            self.has_activations = len(self.activations_by_layer) > 0
            
            # Update token info if input_ids are provided
            if input_ids is not None:
                self.current_token_ids = input_ids
                self.num_tokens = input_ids.shape[1]
                
                # Store token data with activation values
                last_token_id = input_ids[0, -1].item() if input_ids.shape[1] > 0 else None
                self.token_data = {
                    'token_id': last_token_id,
                    'position': input_ids.shape[1] - 1 if input_ids.shape[1] > 0 else 0,
                    'has_activation_values': self.has_activations
                }
            
            return self.has_activations
        else:
            self.logger.warning("No hidden states found in model output")
            return False

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """
        Get activations for all monitored layers.
        
        Returns:
            Dictionary mapping layer indices to activation tensors
        """
        if not self.has_activations:
            self.logger.warning("No activations captured yet")
            return {}
        
        # Mark that token data has activation values
        if hasattr(self, 'token_data'):
            self.token_data['has_activation_values'] = True
        
        return self.activations_by_layer

    def get_token_data(self) -> Dict[str, Any]:
        """
        Get token data associated with current activations.
        
        Returns:
            Dictionary containing token data
        """
        if not hasattr(self, 'token_data') or not self.token_data:
            return {'has_activation_values': False}
        
        # Ensure token data has the activation values flag set correctly
        self.token_data['has_activation_values'] = self.has_activations
        
        return self.token_data """
Activation monitoring module for tracking activations in real-time
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from .utils.activation_hooks import ActivationHooks
from .utils.helpers import cosine_sim
from .vectors import ContrastiveVectors
from .utils.logger import get_logger

class ActivationMonitor:
    """
    Class for monitoring model activations and detecting harmful patterns.
    
    This module tracks model activations during inference and compares them
    to known harmful patterns using contrastive activation vectors.
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        vectors: ContrastiveVectors,
        layers: Optional[List[int]] = None,
        model_type: Optional[str] = None,
        threshold: float = 0.7,
        token_strategy: str = "last",
        log_level: str = "info",
    ):
        """
        Initialize the activation monitor.
        
        Args:
            model: The transformer model to monitor
            vectors: ContrastiveVectors object containing the contrastive vectors
            layers: List of layers to monitor. If None, will use all available layers
                   from the vectors object.
            model_type: Type of the model for correct activation extraction
            threshold: Similarity threshold for detecting harmful content
            token_strategy: Strategy for token selection:
                           - "last": Last token in sequence (default)
                           - "target_token": Look for specific tokens like "A" or "B"
                           - "all": Store all tokens for later selection
            log_level: Logging level ('debug', 'info', 'warning', 'error')
        """
        self.logger = get_logger(name="wisent_guard.monitor", level=log_level)
        self.logger.info("Initializing ActivationMonitor")
        
        self.model = model
        self.vectors = vectors
        self.threshold = threshold
        self.token_strategy = token_strategy
        
        self.logger.debug(f"Using similarity threshold: {self.threshold}")
        self.logger.debug(f"Using token strategy: {self.token_strategy}")
        
        # Determine which layers to monitor
        if layers is None:
            self.layers = self.vectors.get_available_layers()
            self.logger.info(f"Using all available layers from vectors: {self.layers}")
        else:
            self.layers = layers
            self.logger.info(f"Using specified layers: {self.layers}")
        
        # Set up activation hooks
        self.logger.debug("Setting up activation hooks")
        self.hooks = ActivationHooks(model, model_type, token_strategy=token_strategy)
        self.hooks.register_hooks(self.layers)
        
        # Initialize similarity tracking
        self.max_similarities: Dict[str, Dict[int, float]] = {}
        self.triggered_layers: Dict[str, Set[int]] = {}
        self.is_triggered = False
        
        # Track the most recent token that triggered a pattern
        self.triggering_token_id = None
        
        self.logger.info("ActivationMonitor initialized successfully")
        
    def reset(self) -> None:
        """Reset the monitor state."""
        self.logger.debug("Resetting monitor state")
        self.max_similarities = {}
        self.triggered_layers = {}
        self.is_triggered = False
        self.triggering_token_id = None
        self.hooks.clear_activations()
        self.logger.debug("Monitor state reset complete")
    
    def check_activations(self, categories: Optional[List[str]] = None, is_response_token: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Check if current activations are similar to harmful patterns.
        
        Args:
            categories: Categories to check. If None, all available categories are checked.
            is_response_token: Whether we're checking a token from the model's response.
                              When True, the system will focus on the last token position,
                              which is likely the first generated token of the response.
            
        Returns:
            Dictionary with results for each category.
        """
        # Get all available categories if not specified
        if categories is None:
            categories = self.vectors.get_available_categories()
            self.logger.debug(f"Checking all available categories: {categories}")
        else:
            self.logger.debug(f"Checking specific categories: {categories}")
        
        # Get current activations
        activations = self.hooks.get_activations()
        
        # Log token information if available
        token_info = self.hooks.get_last_token_info()
        if token_info["token_id"] is not None:
            self.logger.debug(f"Token ID: {token_info['token_id']}, Position: {token_info['position']}")
            try:
                # Try to decode the token for better logging
                token_text = self.model.config.tokenizer.decode([token_info["token_id"]])
                if is_response_token:
                    self.logger.debug(f"Checking response token: '{token_text}'")
                else:
                    self.logger.debug(f"Token text: '{token_text}'")
            except:
                pass
        
        # If no activations, return empty results
        if not activations:
            self.logger.warning("No activations collected, returning empty results")
            return {cat: {"is_harmful": False, "max_similarity": 0.0, "triggered_layers": [], "layer_similarities": {}} 
                   for cat in categories}
        
        self.logger.debug(f"Collected activations from {len(activations)} layers")
        
        # Initialize result dictionary
        results = {}
        self.is_triggered = False
        
        # Check each category
        for category in categories:
            self.logger.debug(f"Checking category: {category}")
            
            # Initialize category results
            if category not in self.max_similarities:
                self.max_similarities[category] = {}
            
            if category not in self.triggered_layers:
                self.triggered_layers[category] = set()
            
            category_results = {
                "is_harmful": False,
                "max_similarity": 0.0,
                "triggered_layers": [],
                "layer_similarities": {}
            }
            
            # Check each layer
            for layer in self.layers:
                if layer not in activations:
                    self.logger.debug(f"No activations for layer {layer}, skipping")
                    continue
                
                # Get the activation and contrastive vector
                activation = activations[layer]
                contrastive_vector = self.vectors.get_contrastive_vector(category, layer)
                
                if contrastive_vector is None:
                    self.logger.debug(f"No contrastive vector for category '{category}', layer {layer}")
                    continue
                
                self.logger.debug(f"Comparing activation and contrastive vector for layer {layer}")
                self.logger.debug(f"Activation shape: {activation.shape}, Contrastive vector shape: {contrastive_vector.shape}")
                
                # Handle potential dimension mismatches (esp. important for MPS)
                if activation.shape != contrastive_vector.shape:
                    self.logger.debug("Detected shape mismatch, adjusting dimensions")
                    
                    # For MPS, move to CPU for safer dimension handling
                    if activation.device.type == "mps" or contrastive_vector.device.type == "mps":
                        self.logger.debug("Moving tensors to CPU for safer dimension handling")
                        activation = activation.cpu()
                        contrastive_vector = contrastive_vector.cpu()
                        
                    # Flatten tensors if they have different shapes
                    if len(activation.shape) > 1:
                        self.logger.debug(f"Flattening activation from shape {activation.shape}")
                        activation = activation.flatten()
                    if len(contrastive_vector.shape) > 1:
                        self.logger.debug(f"Flattening contrastive vector from shape {contrastive_vector.shape}")
                        contrastive_vector = contrastive_vector.flatten()
                        
                    # If still different shapes, truncate to match
                    if activation.shape != contrastive_vector.shape:
                        min_dim = min(activation.shape[0], contrastive_vector.shape[0])
                        self.logger.debug(f"Truncating both tensors to dimension {min_dim}")
                        activation = activation[:min_dim]
                        contrastive_vector = contrastive_vector[:min_dim]
                
                # Calculate similarity
                similarity = cosine_sim(activation, contrastive_vector)
                self.logger.debug(f"Similarity for layer {layer}: {similarity:.4f}")
                
                # Update max similarity for this layer
                previous_max = self.max_similarities[category].get(layer, -1.0)
                self.max_similarities[category][layer] = max(similarity, previous_max)
                
                if previous_max < similarity:
                    self.logger.debug(f"New max similarity for layer {layer}: {similarity:.4f}")
                
                # Check if threshold is exceeded
                if similarity >= self.threshold:
                    if is_response_token:
                        self.logger.info(f"Harmful response pattern detected in layer {layer} for category '{category}' (similarity: {similarity:.4f})")
                    else:
                        self.logger.info(f"Harmful pattern detected in layer {layer} for category '{category}' (similarity: {similarity:.4f})")
                    
                    self.triggered_layers[category].add(layer)
                    category_results["is_harmful"] = True
                    self.is_triggered = True
                    
                    # If using target_token strategy, try to identify the triggering token
                    if self.token_strategy == "target_token" and hasattr(self.hooks, "last_token_id"):
                        self.triggering_token_id = self.hooks.last_token_id
                        self.logger.debug(f"Triggering token ID: {self.triggering_token_id}")
                        try:
                            # Try to decode the token for better logging
                            token_text = self.model.config.tokenizer.decode([self.triggering_token_id])
                            self.logger.debug(f"Triggering token text: '{token_text}'")
                        except:
                            pass
                
                # Store layer similarity in results
                category_results["layer_similarities"][str(layer)] = float(similarity)
            
            # Update category results
            if self.max_similarities[category]:
                category_max = max(self.max_similarities[category].values())
                category_results["max_similarity"] = category_max
                self.logger.debug(f"Max similarity for category '{category}': {category_max:.4f}")
            
            if self.triggered_layers[category]:
                triggered = list(self.triggered_layers[category])
                category_results["triggered_layers"] = triggered
                self.logger.debug(f"Triggered layers for category '{category}': {triggered}")
            
            results[category] = category_results
            
            # Log the overall result for this category
            if category_results["is_harmful"]:
                self.logger.info(f"Category '{category}' flagged as harmful (max similarity: {category_results['max_similarity']:.4f})")
            else:
                self.logger.debug(f"Category '{category}' not flagged as harmful (max similarity: {category_results['max_similarity']:.4f})")
        
        return results
    
    def get_most_harmful_category(self) -> Optional[Tuple[str, float]]:
        """
        Get the category with the highest similarity to harmful patterns.
        
        Returns:
            Tuple of (category_name, similarity) or None if no categories are triggered
        """
        self.logger.debug("Finding most harmful category")
        max_similarity = -1.0
        max_category = None
        
        for category, layer_similarities in self.max_similarities.items():
            if not layer_similarities:
                continue
                
            category_max = max(layer_similarities.values())
            self.logger.debug(f"Category '{category}' max similarity: {category_max:.4f}")
            
            if category_max > max_similarity:
                max_similarity = category_max
                max_category = category
                self.logger.debug(f"New most harmful category: '{category}' ({category_max:.4f})")
        
        if max_category is not None:
            self.logger.info(f"Most harmful category: '{max_category}' with similarity {max_similarity:.4f}")
            return (max_category, max_similarity)
        
        self.logger.debug("No harmful categories found")
        return None
    
    def is_harmful(self, categories: Optional[List[str]] = None, is_response_token: bool = False) -> bool:
        """
        Check if current activations indicate harmful content.
        
        Args:
            categories: Categories to check. If None, all available categories are checked.
            is_response_token: Whether we're checking a token from the model's response.
            
        Returns:
            True if harmful content is detected, False otherwise
        """
        self.logger.debug("Checking if content is harmful")
        results = self.check_activations(categories, is_response_token=is_response_token)
        
        # Check if any category is flagged as harmful
        for category, category_result in results.items():
            if category_result["is_harmful"]:
                if is_response_token:
                    self.logger.info(f"Response content flagged as harmful in category '{category}'")
                else:
                    self.logger.info(f"Content flagged as harmful in category '{category}'")
                return True
        
        self.logger.debug("Content not flagged as harmful")
        return False
    
    def get_token_classification(self) -> Optional[str]:
        """
        For multiple-choice format, get the classification of the current token.
        
        Returns:
            "truthful" if the token is A, "hallucinatory" if the token is B, or None if not detected
        """
        if self.token_strategy != "target_token" or self.triggering_token_id is None:
            self.logger.debug("Cannot classify token: not using target_token strategy or no triggering token")
            return None
        
        # Get the token text
        token_text = None
        try:
            token_text = self.model.config.tokenizer.decode([self.triggering_token_id])
            self.logger.debug(f"Decoded token text: '{token_text}'")
        except:
            try:
                # Try a different way to access the tokenizer
                if hasattr(self.model, "tokenizer"):
                    token_text = self.model.tokenizer.decode([self.triggering_token_id])
                    self.logger.debug(f"Decoded token text (alternative method): '{token_text}'")
            except:
                self.logger.warning("Failed to decode token text")
                pass
        
        if token_text == "A":
            self.logger.info("Token classified as 'truthful' (A)")
            return "truthful"
        elif token_text == "B":
            self.logger.info("Token classified as 'hallucinatory' (B)")
            return "hallucinatory"
        
        self.logger.debug(f"Token '{token_text}' not classified as either A or B")
        return None 