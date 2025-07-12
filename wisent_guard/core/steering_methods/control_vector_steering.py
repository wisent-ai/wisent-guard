"""
Control Vector Steering Method - Simple difference-based steering using pre-computed control vectors.
"""

import torch
from typing import Dict, Any, Optional
from .base import SteeringMethod
from ..contrastive_pairs import ContrastivePairSet


class ControlVectorSteering(SteeringMethod):
    """
    Simple control vector steering that applies a pre-computed steering vector.
    
    This method uses a difference vector (harmful - harmless) to steer activations
    away from harmful outputs.
    """
    
    def __init__(self, control_vector: Optional[torch.Tensor] = None, layer: int = 0, device: Optional[str] = None):
        """
        Initialize control vector steering.
        
        Args:
            control_vector: Pre-computed control vector (can be loaded later)
            layer: Layer index where this vector was computed
            device: Device to use
        """
        super().__init__("ControlVectorSteering", device)
        self.control_vector = control_vector.to(self.device) if control_vector is not None else None
        self.layer = layer
        self.is_trained = control_vector is not None
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train by computing the control vector from contrastive pairs.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs
            layer_index: Layer to extract activations from
            
        Returns:
            Training statistics
        """
        from ..layer import Layer
        
        # Create layer object
        layer_obj = Layer(index=layer_index)
        
        # Compute control vector
        self.control_vector = contrastive_pair_set.compute_contrastive_vector(layer_obj)
        
        if self.control_vector is None:
            return {
                "success": False,
                "error": "Failed to compute control vector"
            }
        
        # Move to device
        self.control_vector = self.control_vector.to(self.device)
        self.layer = layer_index
        self.is_trained = True
        
        # Compute some statistics
        vector_norm = torch.norm(self.control_vector).item()
        
        return {
            "success": True,
            "vector_norm": vector_norm,
            "vector_shape": list(self.control_vector.shape),
            "layer": layer_index
        }
    
    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Apply control vector steering by subtracting the scaled vector.
        
        Args:
            activations: Input activations [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
            strength: Scaling factor for the control vector
            
        Returns:
            Steered activations
        """
        if not self.is_trained or self.control_vector is None:
            return activations
        
        # Ensure control vector is on same device
        if self.control_vector.device != activations.device:
            self.control_vector = self.control_vector.to(activations.device)
        
        # Handle different input shapes
        if activations.dim() == 2:
            # [seq_len, hidden_dim] - apply to all positions
            steered = activations - strength * self.control_vector.unsqueeze(0)
        elif activations.dim() == 3:
            # [batch, seq_len, hidden_dim] - apply to all positions in all batches
            steered = activations - strength * self.control_vector.unsqueeze(0).unsqueeze(0)
        else:
            # Fallback - just return original
            return activations
        
        return steered
    
    def get_steering_vector(self) -> torch.Tensor:
        """Get the control vector."""
        return self.control_vector
    
    def save_steering_vector(self, path: str) -> bool:
        """
        Save control vector to file.
        
        Args:
            path: Path to save the vector
            
        Returns:
            Success status
        """
        if self.control_vector is None:
            return False
        
        try:
            torch.save({
                'vector': self.control_vector.cpu(),
                'layer': self.layer,
                'method': 'ControlVectorSteering'
            }, path)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """
        Load control vector from file.
        
        Args:
            path: Path to load from
            
        Returns:
            Success status
        """
        try:
            data = torch.load(path, map_location=self.device)
            self.control_vector = data['vector'].to(self.device)
            self.layer = data.get('layer', 0)
            self.is_trained = True
            return True
        except Exception:
            return False
    
    def set_control_vector(self, vector: torch.Tensor, layer: int):
        """
        Set the control vector directly.
        
        Args:
            vector: Control vector to use
            layer: Layer index where this vector applies
        """
        self.control_vector = vector.to(self.device)
        self.layer = layer
        self.is_trained = True