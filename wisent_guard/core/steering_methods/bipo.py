"""
Bi-directional Preference Optimization (BiPO) steering method.

This implementation follows the reference BiPO approach which trains a single
steering vector that can be applied bidirectionally (with positive or negative
multipliers) to steer model behavior in opposite directions.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from .base import SteeringMethod
from ..contrastive_pairs import ContrastivePairSet


class BiPO(SteeringMethod):
    """
    Bi-directional Preference Optimization (BiPO) steering method.
    
    Trains a single steering vector using DPO loss with random sign flipping.
    The vector can be applied with positive or negative multipliers to achieve
    bidirectional control over model behavior.
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        beta: float = 0.1,
        learning_rate: float = 5e-4,
        num_epochs: int = 100,
        batch_size: int = 16,
        reference_free: bool = True
    ):
        super().__init__("BiPO", device)
        self.steering_vector = None
        self.layer_index = None
        self.beta = beta
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.reference_free = reference_free  # Whether to use reference-free DPO
        self.training_stats = {}
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train BiPO by optimizing a bidirectional steering vector.
        
        The key innovation is that during training, the vector is applied with
        randomly alternating signs, teaching it to work in both directions.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied
            
        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index
        
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
        
        if len(pos_activations) == 0:
            raise ValueError("No positive activations provided for training")
        
        # Initialize steering vector as learnable parameter (zero initialization)
        vector_dim = pos_activations[0].shape[-1]
        self.steering_vector = torch.zeros(vector_dim, device=self.device, requires_grad=True)
        
        # Convert activations to tensors
        pos_acts = torch.stack([act.to(self.device) for act in pos_activations])
        neg_acts = torch.stack([act.to(self.device) for act in neg_activations])
        
        # Flatten if needed
        if len(pos_acts.shape) > 2:
            pos_acts = pos_acts.view(pos_acts.shape[0], -1)
            neg_acts = neg_acts.view(neg_acts.shape[0], -1)
        
        # Optimizer
        optimizer = torch.optim.AdamW([self.steering_vector], lr=self.learning_rate)
        
        # Training loop with bidirectional application
        losses = []
        dataset_size = len(pos_acts)
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle data
            perm = torch.randperm(dataset_size)
            pos_acts = pos_acts[perm]
            neg_acts = neg_acts[perm]
            
            for i in range(0, dataset_size, self.batch_size):
                batch_pos = pos_acts[i:i + self.batch_size]
                batch_neg = neg_acts[i:i + self.batch_size]
                batch_size_actual = batch_pos.shape[0]
                
                optimizer.zero_grad()
                
                # Key BiPO mechanism: randomly choose multiplier for this batch
                # This teaches the vector to work bidirectionally
                multiplier = np.random.choice([-1.0, 1.0])
                
                # Apply steering vector with the chosen multiplier
                # In the reference, this would be done by the model wrapper
                # Here we simulate it by computing the scores with the multiplier
                steered_pos = batch_pos + multiplier * self.steering_vector.unsqueeze(0)
                steered_neg = batch_neg + multiplier * self.steering_vector.unsqueeze(0)
                
                # Compute DPO loss
                # The loss should account for the direction of steering
                if self.reference_free:
                    # Reference-free DPO (simplified for steering vectors)
                    # We use the dot product as a proxy for log probability change
                    pos_logprobs = torch.sum(steered_pos * batch_pos, dim=-1) / vector_dim
                    neg_logprobs = torch.sum(steered_neg * batch_neg, dim=-1) / vector_dim
                else:
                    # Standard DPO would require model outputs, so we approximate
                    # by using the alignment between steered and original activations
                    pos_logprobs = F.cosine_similarity(steered_pos, batch_pos, dim=-1)
                    neg_logprobs = F.cosine_similarity(steered_neg, batch_neg, dim=-1)
                
                # DPO loss: -log(sigmoid(beta * (log p(y_w) - log p(y_l))))
                # With bidirectional training, the multiplier affects which direction is "preferred"
                if multiplier > 0:
                    # Positive multiplier: prefer positive over negative
                    loss = -torch.mean(torch.log(torch.sigmoid(self.beta * (pos_logprobs - neg_logprobs))))
                else:
                    # Negative multiplier: prefer negative over positive (flipped)
                    loss = -torch.mean(torch.log(torch.sigmoid(self.beta * (neg_logprobs - pos_logprobs))))
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            losses.append(avg_loss)
        
        # Detach from computation graph
        self.steering_vector = self.steering_vector.detach()
        
        # Compute final statistics
        with torch.no_grad():
            # Test both directions
            pos_scores = torch.matmul(pos_acts, self.steering_vector)
            neg_scores = torch.matmul(neg_acts, self.steering_vector)
            
            # Positive direction performance
            pos_dir_correct = (pos_scores > neg_scores).float().mean().item()
            
            # Negative direction performance (flipped)
            neg_dir_correct = (neg_scores > pos_scores).float().mean().item()
        
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
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'reference_free': self.reference_free,
            'final_loss': losses[-1] if losses else 0.0,
            'pos_direction_accuracy': pos_dir_correct,
            'neg_direction_accuracy': neg_dir_correct,
            'bidirectional': True,
            'layer_index': layer_index
        }
        
        self.is_trained = True
        return self.training_stats
    
    def apply_steering(
        self, 
        activations: torch.Tensor, 
        strength: float = 1.0,
        direction: Optional[str] = "positive"
    ) -> torch.Tensor:
        """
        Apply BiPO steering using learned steering vector.
        
        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier (magnitude)
            direction: Steering direction - "positive", "negative", or "bidirectional"
                      - "positive": apply with +strength
                      - "negative": apply with -strength  
                      - "bidirectional": for compatibility, same as "positive"
            
        Returns:
            Steered activations
        """
        if not self.is_trained:
            raise ValueError("BiPO method must be trained before applying steering")
        
        # Determine multiplier based on direction
        if direction == "negative":
            multiplier = -strength
        else:  # "positive" or "bidirectional" 
            multiplier = strength
        
        # Apply additive steering with learned vector
        steering_vector = self.steering_vector.to(activations.device)
        
        # Handle different activation shapes  
        if len(activations.shape) == 3:  # [batch, seq, hidden]
            # Apply to second-to-last token position (reference behavior)
            steered = activations.clone()
            if activations.shape[1] > 1:
                # Use second-to-last token if sequence has more than 1 token
                steered[:, -2:-1, :] = steered[:, -2:-1, :] + multiplier * steering_vector.unsqueeze(0).unsqueeze(0)
            else:
                # Fallback to last token for single-token sequences
                steered[:, -1:, :] = steered[:, -1:, :] + multiplier * steering_vector.unsqueeze(0).unsqueeze(0)
        elif len(activations.shape) == 2:  # [batch, hidden]
            steered = activations + multiplier * steering_vector.unsqueeze(0)
        else:
            steered = activations + multiplier * steering_vector
        
        return steered
    
    def get_steering_vector(self) -> torch.Tensor:
        """Return the learned steering vector."""
        if not self.is_trained:
            raise ValueError("BiPO method must be trained before getting steering vector")
        return self.steering_vector
    
    def get_bidirectional_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return both positive and negative direction vectors.
        
        Returns:
            Tuple of (positive_vector, negative_vector)
        """
        if not self.is_trained:
            raise ValueError("BiPO method must be trained before getting steering vectors")
        return self.steering_vector, -self.steering_vector
    
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
                'batch_size': self.batch_size,
                'reference_free': self.reference_free,
                'layer_index': self.layer_index,
                'training_stats': self.training_stats,
                'method': 'BiPO',
                'bidirectional': True
            }, path)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """Load BiPO steering data."""
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
            if data.get('method') != 'BiPO':
                return False
            self.steering_vector = data['steering_vector']
            self.beta = data.get('beta', 0.1)
            self.learning_rate = data.get('learning_rate', 5e-4)
            self.num_epochs = data.get('num_epochs', 100)
            self.batch_size = data.get('batch_size', 16)
            self.reference_free = data.get('reference_free', True)
            self.layer_index = data.get('layer_index')
            self.training_stats = data.get('training_stats', {})
            self.is_trained = True
            return True
        except Exception:
            return False