from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from .contrastive_pair_set import ContrastivePairSet

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
    
    Computes steering vector as the average difference between positive and negative
    activations across all contrastive pairs, then adds this vector to activations
    during inference to steer the model in the desired direction.
    """
    
    def __init__(self, device: Optional[str] = None):
        super().__init__("CAA", device)
        self.steering_vector = None
        self.layer_index = None
        self.training_stats = {}
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train CAA by computing average activation difference across contrastive pairs.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied
            
        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index
        
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
        
        if len(pos_activations) != len(neg_activations):
            raise ValueError(f"Mismatch in activation pairs: {len(pos_activations)} vs {len(neg_activations)}")
        
        if len(pos_activations) == 0:
            raise ValueError("No activation pairs found in contrastive pair set")
        
        # Compute differences for each pair
        differences = []
        for pos_act, neg_act in zip(pos_activations, neg_activations):
            # Convert to tensors if needed
            if hasattr(pos_act, 'tensor'):
                pos_tensor = pos_act.tensor
            else:
                pos_tensor = pos_act
                
            if hasattr(neg_act, 'tensor'):
                neg_tensor = neg_act.tensor
            else:
                neg_tensor = neg_act
            
            # Ensure same shape
            if pos_tensor.shape != neg_tensor.shape:
                raise ValueError(f"Shape mismatch in pair: {pos_tensor.shape} vs {neg_tensor.shape}")
            
            # Compute difference: positive - negative (steering toward positive)
            diff = pos_tensor - neg_tensor
            differences.append(diff)
        
        # Average across all pairs to get steering vector
        self.steering_vector = torch.stack(differences).mean(dim=0)
        
        # Move to specified device
        if self.device:
            self.steering_vector = self.steering_vector.to(self.device)
        
        # Compute training statistics
        self.training_stats = {
            'num_pairs': len(differences),
            'vector_norm': torch.norm(self.steering_vector).item(),
            'vector_mean': self.steering_vector.mean().item(),
            'vector_std': self.steering_vector.std().item(),
            'vector_shape': list(self.steering_vector.shape),
            'layer_index': layer_index
        }
        
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
        
        try:
            torch.save({
                'steering_vector': self.steering_vector,
                'training_stats': self.training_stats,
                'layer_index': self.layer_index,
                'method_name': self.name
            }, path)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """Load steering vector from disk."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.steering_vector = checkpoint['steering_vector']
            self.training_stats = checkpoint.get('training_stats', {})
            self.layer_index = checkpoint.get('layer_index')
            
            if self.device:
                self.steering_vector = self.steering_vector.to(self.device)
            
            self.is_trained = True
            return True
        except Exception:
            return False
