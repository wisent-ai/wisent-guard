"""
Activation monitoring module for tracking activations in real-time
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from .utils.activation_hooks import ActivationHooks
from .utils.helpers import cosine_sim
from .vectors import ContrastiveVectors

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
        """
        self.model = model
        self.vectors = vectors
        self.threshold = threshold
        self.token_strategy = token_strategy
        
        # Determine which layers to monitor
        if layers is None:
            self.layers = self.vectors.get_available_layers()
        else:
            self.layers = layers
        
        # Set up activation hooks
        self.hooks = ActivationHooks(model, model_type, token_strategy=token_strategy)
        self.hooks.register_hooks(self.layers)
        
        # Initialize similarity tracking
        self.max_similarities: Dict[str, Dict[int, float]] = {}
        self.triggered_layers: Dict[str, Set[int]] = {}
        self.is_triggered = False
        
        # Track the most recent token that triggered a pattern
        self.triggering_token_id = None
        
    def reset(self) -> None:
        """Reset the monitor state."""
        self.max_similarities = {}
        self.triggered_layers = {}
        self.is_triggered = False
        self.triggering_token_id = None
        self.hooks.clear_activations()
    
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
        
        # Get current activations
        activations = self.hooks.get_activations()
        
        # Initialize result dictionary
        results = {}
        self.is_triggered = False
        
        # Check each category
        for category in categories:
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
                    continue
                
                # Get the activation and contrastive vector
                activation = activations[layer]
                contrastive_vector = self.vectors.get_contrastive_vector(category, layer)
                
                if contrastive_vector is None:
                    continue
                
                # Calculate similarity
                similarity = cosine_sim(activation, contrastive_vector)
                
                # Update max similarity for this layer
                self.max_similarities[category][layer] = max(
                    similarity,
                    self.max_similarities[category].get(layer, -1.0)
                )
                
                # Check if threshold is exceeded
                if similarity >= self.threshold:
                    self.triggered_layers[category].add(layer)
                    category_results["is_harmful"] = True
                    self.is_triggered = True
                    
                    # If using target_token strategy, try to identify the triggering token
                    if self.token_strategy == "target_token" and hasattr(self.hooks, "last_token_id"):
                        self.triggering_token_id = self.hooks.last_token_id
                
                # Store layer similarity in results
                category_results["layer_similarities"][str(layer)] = float(similarity)
            
            # Update category results
            if self.max_similarities[category]:
                category_results["max_similarity"] = max(self.max_similarities[category].values())
            
            if self.triggered_layers[category]:
                category_results["triggered_layers"] = list(self.triggered_layers[category])
            
            results[category] = category_results
        
        return results
    
    def get_most_harmful_category(self) -> Optional[Tuple[str, float]]:
        """
        Get the category with the highest similarity to harmful patterns.
        
        Returns:
            Tuple of (category_name, similarity) or None if no categories are triggered
        """
        max_similarity = -1.0
        max_category = None
        
        for category, layer_similarities in self.max_similarities.items():
            if not layer_similarities:
                continue
                
            category_max = max(layer_similarities.values())
            if category_max > max_similarity:
                max_similarity = category_max
                max_category = category
        
        if max_category is not None:
            return (max_category, max_similarity)
        
        return None
    
    def is_harmful(self, categories: Optional[List[str]] = None) -> bool:
        """
        Check if current activations indicate harmful content.
        
        Args:
            categories: Categories to check. If None, all available categories are checked.
            
        Returns:
            True if harmful content is detected, False otherwise
        """
        results = self.check_activations(categories)
        
        # Check if any category is flagged as harmful
        for category_result in results.values():
            if category_result["is_harmful"]:
                return True
        
        return False
    
    def get_token_classification(self) -> Optional[str]:
        """
        For multiple-choice format, get the classification of the current token.
        
        Returns:
            "truthful" if the token is A, "hallucinatory" if the token is B, or None if not detected
        """
        if self.token_strategy != "target_token" or self.triggering_token_id is None:
            return None
        
        # Get the token text
        token_text = None
        try:
            token_text = self.model.config.tokenizer.decode([self.triggering_token_id])
        except:
            try:
                # Try a different way to access the tokenizer
                if hasattr(self.model, "tokenizer"):
                    token_text = self.model.tokenizer.decode([self.triggering_token_id])
            except:
                pass
        
        if token_text == "A":
            return "truthful"
        elif token_text == "B":
            return "hallucinatory"
        
        return None 