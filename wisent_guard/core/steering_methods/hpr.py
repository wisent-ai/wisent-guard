"""
Householder Pseudo-Rotation (HPR) steering method.
"""

from typing import Dict, Any, Optional
import torch

from .base import SteeringMethod
from ..contrastive_pairs import ContrastivePairSet
from ..aggregation import ControlVectorAggregationMethod, create_control_vector_from_contrastive_pairs


class HPR(SteeringMethod):
    """
    Householder Pseudo-Rotation (HPR) steering method.
    
    Uses Householder transformations to create norm-preserving rotations
    in activation space for more controlled steering interventions.
    """
    
    def __init__(self, device: Optional[str] = None, beta: float = 1.0):
        super().__init__("HPR", device)
        self.steering_vector = None
        self.householder_matrix = None
        self.layer_index = None
        self.beta = beta
        self.training_stats = {}
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train HPR by computing Householder transformation matrix.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied
            
        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index
        
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
        
        # Create initial control vector using CAA
        self.steering_vector, training_stats = create_control_vector_from_contrastive_pairs(
            pos_activations, 
            neg_activations, 
            ControlVectorAggregationMethod.CAA, 
            self.device
        )
        
        # Compute Householder transformation matrix
        # H = I - 2 * (v @ v^T) / (v^T @ v)
        # where v is the normalized steering vector
        vector_norm = torch.norm(self.steering_vector)
        if vector_norm > 1e-8:
            v_normalized = self.steering_vector / vector_norm
            # Create Householder matrix: H = I - 2 * v * v^T
            identity = torch.eye(len(v_normalized), device=self.device, dtype=v_normalized.dtype)
            outer_product = torch.outer(v_normalized, v_normalized)
            self.householder_matrix = identity - 2.0 * outer_product
        else:
            # If steering vector is near zero, use identity matrix
            self.householder_matrix = torch.eye(len(self.steering_vector), device=self.device, dtype=self.steering_vector.dtype)
        
        # Training statistics
        self.training_stats = training_stats.copy()
        self.training_stats.update({
            'method': 'HPR',
            'beta': self.beta,
            'layer_index': layer_index,
            'vector_norm': vector_norm.item(),
            'householder_matrix_norm': torch.norm(self.householder_matrix).item(),
            'householder_matrix_shape': list(self.householder_matrix.shape)
        })
        
        self.is_trained = True
        return self.training_stats
    
    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Apply HPR steering using Householder transformation.
        
        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier
            
        Returns:
            Steered activations with preserved norms
        """
        if not self.is_trained:
            raise ValueError("HPR method must be trained before applying steering")
        
        original_shape = activations.shape
        
        # Flatten to 2D for matrix operations
        if len(original_shape) > 2:
            batch_size = original_shape[0]
            seq_len = original_shape[1] if len(original_shape) > 2 else 1
            hidden_dim = original_shape[-1]
            activations_flat = activations.view(-1, hidden_dim)
        else:
            activations_flat = activations
            batch_size, hidden_dim = activations_flat.shape
        
        # Apply Householder transformation: H @ x
        # Scale by beta and strength
        if strength != 0:
            # Interpolate between identity and Householder matrix
            alpha = self.beta * strength
            identity = torch.eye(hidden_dim, device=activations.device, dtype=activations.dtype)
            effective_matrix = (1 - alpha) * identity + alpha * self.householder_matrix.to(activations.device)
            steered_flat = torch.matmul(activations_flat, effective_matrix.T)
        else:
            steered_flat = activations_flat
        
        # Reshape back to original shape
        steered = steered_flat.view(original_shape)
        
        return steered
    
    def get_steering_vector(self) -> torch.Tensor:
        """Return the steering vector (for compatibility)."""
        if not self.is_trained:
            raise ValueError("HPR method must be trained before getting steering vector")
        return self.steering_vector
    
    def save_steering_vector(self, path: str) -> bool:
        """Save HPR steering data including Householder matrix."""
        if not self.is_trained:
            return False
        try:
            torch.save({
                'steering_vector': self.steering_vector,
                'householder_matrix': self.householder_matrix,
                'beta': self.beta,
                'layer_index': self.layer_index,
                'training_stats': self.training_stats,
                'method': 'HPR'
            }, path)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """Load HPR steering data."""
        try:
            data = torch.load(path, map_location=self.device)
            if data.get('method') != 'HPR':
                return False
            self.steering_vector = data['steering_vector']
            self.householder_matrix = data['householder_matrix']
            self.beta = data.get('beta', 1.0)
            self.layer_index = data.get('layer_index')
            self.training_stats = data.get('training_stats', {})
            self.is_trained = True
            return True
        except Exception:
            return False 