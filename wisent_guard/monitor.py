"""
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
    
    def check_activations(self, categories: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Check if current activations are similar to harmful patterns.
        
        Args:
            categories: Categories to check. If None, all available categories are checked.
            
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
    
    def is_harmful(self, categories: Optional[List[str]] = None) -> bool:
        """
        Check if current activations indicate harmful content.
        
        Args:
            categories: Categories to check. If None, all available categories are checked.
            
        Returns:
            True if harmful content is detected, False otherwise
        """
        self.logger.debug("Checking if content is harmful")
        results = self.check_activations(categories)
        
        # Check if any category is flagged as harmful
        for category, category_result in results.items():
            if category_result["is_harmful"]:
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