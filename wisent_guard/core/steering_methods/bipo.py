"""
Bi-directional Preference Optimization (BiPO) steering method.
"""

from typing import Dict, Any, Optional
import torch

from .base import SteeringMethod
from ..contrastive_pairs import ContrastivePairSet


class BiPO(SteeringMethod):
    """
    Bi-directional Preference Optimization (BiPO) steering method.
    
    Uses preference optimization to learn steering vectors as trainable parameters
    rather than computing them from activation differences.
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        beta: float = 0.1,
        learning_rate: float = 5e-4,
        num_epochs: int = 100
    ):
        super().__init__("BiPO", device)
        self.steering_vector = None
        self.layer_index = None
        self.beta = beta
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.training_stats = {}
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train BiPO by optimizing steering vector as learnable parameter.
        
        Note: This is a simplified implementation. Full BiPO requires DPO training
        with preference data and specialized loss functions.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied
            
        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index
        
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
        
        # Initialize steering vector as learnable parameter (zero initialization like BiPO)
        if len(pos_activations) > 0:
            vector_dim = pos_activations[0].shape[-1]
        else:
            raise ValueError("No positive activations provided for training")
        
        # Initialize with zeros (BiPO approach)
        self.steering_vector = torch.zeros(vector_dim, device=self.device, requires_grad=True)
        
        # Simple optimization loop (simplified version of BiPO training)
        optimizer = torch.optim.AdamW([self.steering_vector], lr=self.learning_rate)
        
        losses = []
        for epoch in range(min(10, self.num_epochs)):  # Simplified training
            optimizer.zero_grad()
            
            # Compute preference-based loss (simplified)
            pos_scores = []
            neg_scores = []
            
            for pos_act, neg_act in zip(pos_activations[:10], neg_activations[:10]):  # Limit for efficiency
                pos_act = pos_act.to(self.device)
                neg_act = neg_act.to(self.device)
                
                # Compute alignment scores
                pos_score = torch.dot(pos_act.flatten(), self.steering_vector.flatten())
                neg_score = torch.dot(neg_act.flatten(), self.steering_vector.flatten())
                
                pos_scores.append(pos_score)
                neg_scores.append(neg_score)
            
            if pos_scores and neg_scores:
                pos_scores = torch.stack(pos_scores)
                neg_scores = torch.stack(neg_scores)
                
                # DPO-style loss (simplified)
                loss = -torch.mean(torch.log(torch.sigmoid(self.beta * (pos_scores - neg_scores))))
                
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        
        # Detach from computation graph
        self.steering_vector = self.steering_vector.detach()
        
        # Training statistics
        self.training_stats = {
            'num_pairs': len(pos_activations),
            'vector_norm': torch.norm(self.steering_vector).item(),
            'vector_mean': self.steering_vector.mean().item(),
            'vector_std': self.steering_vector.std().item(),
            'vector_shape': list(self.steering_vector.shape),
            'method': 'BiPO',
            'beta': self.beta,
            'learning_rate': self.learning_rate,
            'num_epochs_trained': len(losses),
            'final_loss': losses[-1] if losses else 0.0,
            'layer_index': layer_index
        }
        
        self.is_trained = True
        return self.training_stats
    
    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Apply BiPO steering using learned steering vector.
        
        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier
            
        Returns:
            Steered activations
        """
        if not self.is_trained:
            raise ValueError("BiPO method must be trained before applying steering")
        
        # Apply additive steering with learned vector
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
        """Return the learned steering vector."""
        if not self.is_trained:
            raise ValueError("BiPO method must be trained before getting steering vector")
        return self.steering_vector
    
    def save_steering_vector(self, path: str) -> bool:
        """Save BiPO steering data."""
        if not self.is_trained:
            return False
        try:
            torch.save({
                'steering_vector': self.steering_vector,
                'beta': self.beta,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'layer_index': self.layer_index,
                'training_stats': self.training_stats,
                'method': 'BiPO'
            }, path)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """Load BiPO steering data."""
        try:
            data = torch.load(path, map_location=self.device)
            if data.get('method') != 'BiPO':
                return False
            self.steering_vector = data['steering_vector']
            self.beta = data.get('beta', 0.1)
            self.learning_rate = data.get('learning_rate', 5e-4)
            self.num_epochs = data.get('num_epochs', 100)
            self.layer_index = data.get('layer_index')
            self.training_stats = data.get('training_stats', {})
            self.is_trained = True
            return True
        except Exception:
            return False 