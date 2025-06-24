"""
Contrastive Activation Addition (CAA) steering method.
"""

from typing import Dict, Any, Optional
import torch

from .base import SteeringMethod
from ..contrastive_pairs import ContrastivePairSet
from ..aggregation import ControlVectorAggregationMethod, create_control_vector_from_contrastive_pairs
from ..normalization import VectorNormalizer


class CAA(SteeringMethod):
    """
    Contrastive Activation Addition (CAA) steering method.
    
    Uses simple activation differences between positive and negative examples
    to create steering vectors. Supports various normalization strategies.
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        aggregation_method: ControlVectorAggregationMethod = ControlVectorAggregationMethod.CAA,
        normalization_method: str = "none",
        target_norm: Optional[float] = None
    ):
        super().__init__("CAA", device)
        self.steering_vector = None
        self.layer_index = None
        self.aggregation_method = aggregation_method
        self.normalization_method = normalization_method
        self.target_norm = target_norm
        self.training_stats = {}
        self.normalizer = VectorNormalizer() if normalization_method != "none" else None
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train CAA by computing activation differences.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied
            
        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index
        
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
        
        # Create control vector using aggregation method
        self.steering_vector, training_stats = create_control_vector_from_contrastive_pairs(
            pos_activations, 
            neg_activations, 
            self.aggregation_method, 
            self.device
        )
        
        # Apply normalization if specified
        if self.normalizer and self.normalization_method != "none":
            if self.normalization_method == "cross_behavior":
                # For CAA, we treat each vector as a separate behavior
                vectors = [self.steering_vector]
                normalized_vectors = self.normalizer.normalize_cross_behavior(
                    vectors, 
                    target_norm=self.target_norm
                )
                self.steering_vector = normalized_vectors[0]
            elif self.normalization_method == "l2_unit":
                self.steering_vector = self.normalizer.normalize_l2_unit(self.steering_vector)
            elif self.normalization_method == "layer_wise_mean":
                # For single layer, this is equivalent to l2_unit
                vectors = {layer_index: [self.steering_vector]}
                normalized_vectors = self.normalizer.normalize_layer_wise_mean(vectors)
                self.steering_vector = normalized_vectors[layer_index][0]
        
        # Update training statistics
        self.training_stats = training_stats.copy()
        self.training_stats.update({
            'method': 'CAA',
            'aggregation_method': self.aggregation_method.value,
            'normalization_method': self.normalization_method,
            'target_norm': self.target_norm,
            'layer_index': layer_index,
            'final_vector_norm': torch.norm(self.steering_vector).item(),
        })
        
        self.is_trained = True
        return self.training_stats
    
    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Apply CAA steering using additive intervention.
        
        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier
            
        Returns:
            Steered activations
        """
        if not self.is_trained:
            raise ValueError("CAA method must be trained before applying steering")
        
        # Apply additive steering
        steering_vector = self.steering_vector.to(activations.device)
        
        # Handle different activation shapes
        if len(activations.shape) == 3:  # [batch, seq, hidden]
            # Apply to last token position
            steered = activations.clone()
            steered[:, -1:, :] = steered[:, -1:, :] + strength * steering_vector.unsqueeze(0).unsqueeze(0)
        elif len(activations.shape) == 2:  # [batch, hidden]
            steered = activations + strength * steering_vector.unsqueeze(0)
        else:
            steered = activations + strength * steering_vector
        
        return steered
    
    def get_steering_vector(self) -> torch.Tensor:
        """Return the CAA steering vector."""
        if not self.is_trained:
            raise ValueError("CAA method must be trained before getting steering vector")
        return self.steering_vector
    
    def save_steering_vector(self, path: str) -> bool:
        """Save CAA steering data."""
        if not self.is_trained:
            return False
        try:
            torch.save({
                'steering_vector': self.steering_vector,
                'aggregation_method': self.aggregation_method.value,
                'normalization_method': self.normalization_method,
                'target_norm': self.target_norm,
                'layer_index': self.layer_index,
                'training_stats': self.training_stats,
                'method': 'CAA'
            }, path)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """Load CAA steering data."""
        try:
            data = torch.load(path, map_location=self.device)
            if data.get('method') != 'CAA':
                return False
            self.steering_vector = data['steering_vector']
            self.aggregation_method = ControlVectorAggregationMethod(data.get('aggregation_method', 'CAA'))
            self.normalization_method = data.get('normalization_method', 'none')
            self.target_norm = data.get('target_norm')
            self.layer_index = data.get('layer_index')
            self.training_stats = data.get('training_stats', {})
            self.is_trained = True
            return True
        except Exception:
            return False 