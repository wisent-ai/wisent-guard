from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from .contrastive_pair_set import ContrastivePairSet
from .aggregation import (
    create_control_vector_from_contrastive_pairs, 
    create_control_vector_from_representations,
    save_control_vector, 
    load_control_vector,
    ControlVectorAggregationMethod
)

class SteeringMethod(ABC):
    """Base class for steering methods that apply directional vectors to model activations."""
    
    def __init__(self, name: str, device: Optional[str] = None):
        self.name = name
        self.device = device
        self.is_trained = False
        
    @abstractmethod
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """Train the steering method on contrastive pairs."""
        pass
    
    @abstractmethod
    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """Apply steering vector to activations during inference."""
        pass
    
    @abstractmethod
    def get_steering_vector(self) -> Optional[torch.Tensor]:
        """Get the computed steering vector."""
        pass


class CAA(SteeringMethod):
    """
    Contrastive Activation Addition (CAA) steering method.
    
    Computes steering vector from positive and negative activations using
    various aggregation methods, then adds this vector to activations
    during inference to steer the model in the desired direction.
    """
    
    def __init__(self, device: Optional[str] = None, aggregation_method: ControlVectorAggregationMethod = ControlVectorAggregationMethod.CAA):
        super().__init__("CAA", device)
        self.steering_vector = None
        self.layer_index = None
        self.aggregation_method = aggregation_method
        self.training_stats = {}
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train CAA by computing control vector from contrastive pairs using the specified aggregation method.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied
            
        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index
        
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
        
        # Use the specified aggregation method to create the control vector
        self.steering_vector, training_stats = create_control_vector_from_contrastive_pairs(
            pos_activations, neg_activations, self.aggregation_method, self.device
        )
        
        # Add layer information to the training stats
        self.training_stats = training_stats.copy()
        self.training_stats['layer_index'] = layer_index
        
        self.is_trained = True
        
        return self.training_stats
    
    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Apply CAA steering vector to activations.
        
        Args:
            activations: Model activations tensor
            strength: Steering strength multiplier (default: 1.0)
            
        Returns:
            Steered activations tensor
        """
        if not self.is_trained or self.steering_vector is None:
            raise ValueError("CAA steering method not trained. Call train() first.")
        
        # Ensure activations are on the same device as steering vector
        if activations.device != self.steering_vector.device:
            steering_vector = self.steering_vector.to(activations.device)
        else:
            steering_vector = self.steering_vector
        
        # Apply steering with strength multiplier
        steered_activations = activations + (strength * steering_vector)
        
        return steered_activations
    
    def get_steering_vector(self) -> Optional[torch.Tensor]:
        """Get the computed steering vector."""
        return self.steering_vector
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()
    
    def save_steering_vector(self, path: str) -> bool:
        """Save steering vector to disk."""
        if not self.is_trained or self.steering_vector is None:
            return False
        
        metadata = {
            'method_name': self.name,
            'aggregation_method': self.aggregation_method.value
        }
        
        return save_control_vector(
            self.steering_vector, 
            self.training_stats, 
            path, 
            self.layer_index, 
            metadata
        )
    
    def load_steering_vector(self, path: str) -> bool:
        """Load steering vector from disk."""
        control_vector, metadata = load_control_vector(path, self.device)
        
        if control_vector is None:
            return False
        
        self.steering_vector = control_vector
        self.training_stats = metadata.get('training_stats', {})
        self.layer_index = metadata.get('layer_index')
        
        # Update aggregation method if available in metadata
        saved_method = metadata.get('metadata', {}).get('aggregation_method')
        if saved_method:
            try:
                self.aggregation_method = ControlVectorAggregationMethod(saved_method)
            except ValueError:
                pass  # Keep current method if saved method is unknown
        
        self.is_trained = True
        return True
