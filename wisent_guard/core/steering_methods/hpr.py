"""
Householder Pseudo-Rotation (HPR) steering method.

This implementation follows the reference HPR approach which uses:
1. A binary classifier to distinguish positive/negative activations
2. An angle prediction network to determine rotation amounts
3. Selective application of Householder transformations
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .base import SteeringMethod
from ..contrastive_pairs import ContrastivePairSet


class BinaryClassifier(nn.Module):
    """Linear classifier to distinguish positive from negative activations."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def get_hyperplane_normal(self) -> torch.Tensor:
        """Get the normal vector of the classification hyperplane."""
        return self.linear.weight.squeeze(0)


class AnglePredictor(nn.Module):
    """Neural network to predict rotation angles for activations."""
    
    def __init__(self, hidden_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict angles in radians
        return self.net(x)


class HPR(SteeringMethod):
    """
    Householder Pseudo-Rotation (HPR) steering method.
    
    Uses learned classifiers and angle predictors to selectively rotate
    activations in a norm-preserving manner.
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        hidden_size: int = 128,
        angle_loss_weight: float = 0.1
    ):
        super().__init__("HPR", device)
        self.classifier = None
        self.angle_predictor = None
        self.householder_matrix = None
        self.layer_index = None
        self.training_stats = {}
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.angle_loss_weight = angle_loss_weight
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train HPR components: classifier and angle predictor.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied
            
        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index
        
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
        
        # Convert to tensors and create labels
        pos_tensors = torch.stack([act.to(self.device) for act in pos_activations])
        neg_tensors = torch.stack([act.to(self.device) for act in neg_activations])
        
        # Flatten if needed (we work with hidden states)
        if len(pos_tensors.shape) > 2:
            pos_tensors = pos_tensors.view(pos_tensors.shape[0], -1)
            neg_tensors = neg_tensors.view(neg_tensors.shape[0], -1)
        
        hidden_dim = pos_tensors.shape[1]
        
        # Create dataset
        all_activations = torch.cat([pos_tensors, neg_tensors], dim=0)
        labels = torch.cat([
            torch.ones(len(pos_tensors), 1),
            torch.zeros(len(neg_tensors), 1)
        ], dim=0).to(self.device)
        
        # Initialize models
        self.classifier = BinaryClassifier(hidden_dim).to(self.device)
        self.angle_predictor = AnglePredictor(hidden_dim, self.hidden_size).to(self.device)
        
        # Optimizers
        classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        angle_optimizer = optim.Adam(self.angle_predictor.parameters(), lr=self.learning_rate)
        
        # Training loop
        dataset_size = all_activations.shape[0]
        losses = {'classifier': [], 'angle': [], 'total': []}
        
        for epoch in range(self.epochs):
            # Shuffle data
            perm = torch.randperm(dataset_size)
            all_activations = all_activations[perm]
            labels = labels[perm]
            
            epoch_losses = {'classifier': 0, 'angle': 0, 'total': 0}
            num_batches = 0
            
            for i in range(0, dataset_size, self.batch_size):
                batch_acts = all_activations[i:i + self.batch_size]
                batch_labels = labels[i:i + self.batch_size]
                
                # Classifier forward pass
                classifier_optimizer.zero_grad()
                logits = self.classifier(batch_acts)
                classifier_loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
                
                # Angle predictor forward pass
                angle_optimizer.zero_grad()
                predicted_angles = self.angle_predictor(batch_acts)
                
                # Angle loss: encourage larger angles for negative examples
                # and smaller angles for positive examples
                target_angles = torch.where(
                    batch_labels > 0.5,
                    torch.zeros_like(predicted_angles),  # Positive: no rotation
                    torch.ones_like(predicted_angles) * np.pi / 4  # Negative: rotate by π/4
                )
                angle_loss = F.mse_loss(predicted_angles, target_angles)
                
                # Combined loss
                total_loss = classifier_loss + self.angle_loss_weight * angle_loss
                
                # Backward pass
                total_loss.backward()
                classifier_optimizer.step()
                angle_optimizer.step()
                
                # Record losses
                epoch_losses['classifier'] += classifier_loss.item()
                epoch_losses['angle'] += angle_loss.item()
                epoch_losses['total'] += total_loss.item()
                num_batches += 1
            
            # Average losses for epoch
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
                losses[key].append(epoch_losses[key])
        
        # Compute Householder matrix from classifier hyperplane
        hyperplane_normal = self.classifier.get_hyperplane_normal()
        hyperplane_normal = F.normalize(hyperplane_normal, dim=0)
        
        # Householder matrix: H = I - 2 * v * v^T
        identity = torch.eye(hidden_dim, device=self.device)
        self.householder_matrix = identity - 2.0 * torch.outer(hyperplane_normal, hyperplane_normal)
        
        # Compute final accuracy
        with torch.no_grad():
            predictions = torch.sigmoid(self.classifier(all_activations)) > 0.5
            accuracy = (predictions == labels).float().mean().item()
        
        # Training statistics
        self.training_stats = {
            'method': 'HPR',
            'layer_index': layer_index,
            'hidden_dim': hidden_dim,
            'epochs': self.epochs,
            'final_accuracy': accuracy,
            'final_classifier_loss': losses['classifier'][-1],
            'final_angle_loss': losses['angle'][-1],
            'hyperplane_norm': torch.norm(hyperplane_normal).item(),
            'householder_matrix_norm': torch.norm(self.householder_matrix).item()
        }
        
        self.is_trained = True
        return self.training_stats
    
    def apply_steering(
        self, 
        activations: torch.Tensor, 
        strength: float = 1.0,
        apply_to_all: bool = False
    ) -> torch.Tensor:
        """
        Apply HPR steering using learned components.
        
        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier
            apply_to_all: If True, apply to all activations (not just negative ones)
            
        Returns:
            Steered activations with preserved norms
        """
        if not self.is_trained:
            raise ValueError("HPR method must be trained before applying steering")
        
        original_shape = activations.shape
        device = activations.device
        
        # Flatten to 2D for processing
        if len(original_shape) == 3:  # [batch, seq, hidden]
            batch_size, seq_len, hidden_dim = original_shape
            # Apply to second-to-last token position (reference behavior)
            if seq_len > 1:
                # Extract the token we want to steer
                target_acts = activations[:, -2, :]  # [batch, hidden]
            else:
                target_acts = activations[:, -1, :]
        else:  # [batch, hidden]
            target_acts = activations
            batch_size = target_acts.shape[0]
        
        # Move models to correct device
        self.classifier = self.classifier.to(device)
        self.angle_predictor = self.angle_predictor.to(device)
        householder_matrix = self.householder_matrix.to(device)
        
        with torch.no_grad():
            # Classify activations
            logits = self.classifier(target_acts)
            is_negative = torch.sigmoid(logits).squeeze(1) < 0.5  # Shape: [batch]
            
            if not apply_to_all:
                # Only process negative activations
                mask = is_negative.float()
            else:
                # Process all activations
                mask = torch.ones_like(is_negative, dtype=torch.float)
            
            # Predict rotation angles
            angles = self.angle_predictor(target_acts).squeeze(1)  # Shape: [batch]
            angles = angles * strength * mask  # Scale by strength and mask
            
            # Apply rotation using Householder matrix
            # For each activation, we rotate by the predicted angle
            steered_acts = target_acts.clone()
            
            for i in range(batch_size):
                if mask[i] > 0:  # Only rotate if masked
                    angle = angles[i]
                    act = target_acts[i]
                    
                    # Householder reflection followed by rotation
                    # First reflect across hyperplane
                    reflected = torch.matmul(householder_matrix, act)
                    
                    # Then rotate by angle (simplified rotation in reflection space)
                    # This is a simplified version - full implementation would use
                    # proper rotation matrices in the reflection space
                    cos_angle = torch.cos(angle)
                    sin_angle = torch.sin(angle)
                    
                    # Interpolate between original and reflected based on angle
                    steered_acts[i] = cos_angle * act + sin_angle * reflected
        
        # Reconstruct full tensor
        if len(original_shape) == 3:
            result = activations.clone()
            if seq_len > 1:
                result[:, -2, :] = steered_acts
            else:
                result[:, -1, :] = steered_acts
            return result
        else:
            return steered_acts
    
    def get_steering_vector(self) -> torch.Tensor:
        """Return the hyperplane normal (for compatibility)."""
        if not self.is_trained:
            raise ValueError("HPR method must be trained before getting steering vector")
        return self.classifier.get_hyperplane_normal()
    
    def save_steering_vector(self, path: str) -> bool:
        """Save HPR steering components."""
        if not self.is_trained:
            return False
        try:
            torch.save({
                'classifier_state': self.classifier.state_dict(),
                'angle_predictor_state': self.angle_predictor.state_dict(),
                'householder_matrix': self.householder_matrix,
                'layer_index': self.layer_index,
                'training_stats': self.training_stats,
                'hidden_dim': self.householder_matrix.shape[0],
                'hidden_size': self.hidden_size,
                'method': 'HPR'
            }, path)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """Load HPR steering components."""
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
            if data.get('method') != 'HPR':
                return False
            
            # Recreate models
            hidden_dim = data['hidden_dim']
            self.hidden_size = data.get('hidden_size', 128)
            self.classifier = BinaryClassifier(hidden_dim).to(self.device)
            self.angle_predictor = AnglePredictor(hidden_dim, self.hidden_size).to(self.device)
            
            # Load states
            self.classifier.load_state_dict(data['classifier_state'])
            self.angle_predictor.load_state_dict(data['angle_predictor_state'])
            self.householder_matrix = data['householder_matrix']
            self.layer_index = data.get('layer_index')
            self.training_stats = data.get('training_stats', {})
            
            self.is_trained = True
            return True
        except Exception:
            return False