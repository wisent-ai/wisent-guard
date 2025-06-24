"""
Dynamic Activation Composition (DAC) steering method.
"""

from typing import Dict, Any, Optional
import torch

from .base import SteeringMethod
from ..contrastive_pairs import ContrastivePairSet
from ..aggregation import ControlVectorAggregationMethod, create_control_vector_from_contrastive_pairs


class DAC(SteeringMethod):
    """
    Dynamic Activation Composition (DAC) steering method.
    
    Uses information-theoretic principles to dynamically modulate steering
    intensity throughout generation for multi-property control.
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        dynamic_control: bool = True,
        entropy_threshold: float = 1.0,
        aggregation_method: ControlVectorAggregationMethod = ControlVectorAggregationMethod.CAA
    ):
        super().__init__("DAC", device)
        self.steering_vector = None
        self.layer_index = None
        self.dynamic_control = dynamic_control
        self.entropy_threshold = entropy_threshold
        self.aggregation_method = aggregation_method
        self.training_stats = {}
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train DAC by computing activation differences and preparing for dynamic control.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied
            
        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index
        
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
        
        # Use aggregation method to create control vector
        self.steering_vector, training_stats = create_control_vector_from_contrastive_pairs(
            pos_activations, 
            neg_activations, 
            self.aggregation_method, 
            self.device
        )
        
        # Add DAC-specific training stats
        self.training_stats = training_stats.copy()
        self.training_stats.update({
            'method': 'DAC',
            'dynamic_control': self.dynamic_control,
            'entropy_threshold': self.entropy_threshold,
            'layer_index': layer_index
        })
        
        self.is_trained = True
        return self.training_stats
    
    def apply_steering(
        self, 
        activations: torch.Tensor, 
        strength: float = 1.0,
        token_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply DAC steering with optional dynamic intensity control.
        
        Args:
            activations: Input activations to steer
            strength: Base steering strength
            token_probs: Token probabilities for dynamic control (optional)
            
        Returns:
            Steered activations
        """
        if not self.is_trained:
            raise ValueError("DAC method must be trained before applying steering")
        
        # Calculate dynamic strength if enabled and token probabilities provided
        if self.dynamic_control and token_probs is not None:
            # Compute entropy from token probabilities
            entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10), dim=-1)
            
            # Modulate strength based on entropy
            # High entropy (uncertainty) -> reduce steering
            # Low entropy (confidence) -> maintain/increase steering
            entropy_factor = torch.clamp(self.entropy_threshold / (entropy + 1e-10), 0.1, 2.0)
            dynamic_strength = strength * entropy_factor.mean().item()
        else:
            dynamic_strength = strength
        
        # Apply additive steering
        steering_vector = self.steering_vector.to(activations.device)
        
        # Handle different activation shapes
        if len(activations.shape) == 3:  # [batch, seq, hidden]
            # Apply to last token position
            steered = activations.clone()
            steered[:, -1:, :] = steered[:, -1:, :] + dynamic_strength * steering_vector.unsqueeze(0).unsqueeze(0)
        elif len(activations.shape) == 2:  # [batch, hidden]
            steered = activations + dynamic_strength * steering_vector.unsqueeze(0)
        else:
            steered = activations + dynamic_strength * steering_vector
        
        return steered
    
    def get_steering_vector(self) -> torch.Tensor:
        """Return the steering vector."""
        if not self.is_trained:
            raise ValueError("DAC method must be trained before getting steering vector")
        return self.steering_vector
    
    def save_steering_vector(self, path: str) -> bool:
        """Save DAC steering data."""
        if not self.is_trained:
            return False
        try:
            torch.save({
                'steering_vector': self.steering_vector,
                'dynamic_control': self.dynamic_control,
                'entropy_threshold': self.entropy_threshold,
                'aggregation_method': self.aggregation_method.value,
                'layer_index': self.layer_index,
                'training_stats': self.training_stats,
                'method': 'DAC'
            }, path)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """Load DAC steering data."""
        try:
            data = torch.load(path, map_location=self.device)
            if data.get('method') != 'DAC':
                return False
            self.steering_vector = data['steering_vector']
            self.dynamic_control = data.get('dynamic_control', True)
            self.entropy_threshold = data.get('entropy_threshold', 1.0)
            self.layer_index = data.get('layer_index')
            self.training_stats = data.get('training_stats', {})
            self.is_trained = True
            return True
        except Exception:
            return False 