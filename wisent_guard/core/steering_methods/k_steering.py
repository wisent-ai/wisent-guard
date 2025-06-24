"""
K-Steering method for steering language models in multiple directions simultaneously.

Based on the paper "Steering Language Models in Multiple Directions Simultaneously"
by lukemarks, Narmeen, and Amirali Abdullah.
"""

from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SteeringMethod
from ..contrastive_pair_set import ContrastivePairSet


class KSteeringClassifier(nn.Module):
    """
    Multi-label classifier for K-steering.
    
    Architecture: f(x) = W^(3) * σ(W^(2) * σ(W^(1) * x + b^(1)) + b^(2)) + b^(3)
    """
    
    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for each label."""
        return self.layers(x)


class KSteering(SteeringMethod):
    """
    K-Steering method for multi-directional steering.
    
    Uses a multilabel classifier to steer activations in multiple directions
    simultaneously by optimizing a steering loss through gradient descent.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        num_labels: int = 6,
        hidden_dim: int = 512,
        learning_rate: float = 1e-3,
        classifier_epochs: int = 100,
        target_labels: Optional[List[int]] = None,
        avoid_labels: Optional[List[int]] = None,
        alpha: float = 50.0
    ):
        super().__init__("K-Steering", device)
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.classifier_epochs = classifier_epochs
        self.target_labels = target_labels or []
        self.avoid_labels = avoid_labels or []
        self.alpha = alpha
        
        # Components
        self.classifier = None
        self.layer_index = None
        self.training_stats = {}
        
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train K-steering by first training a multilabel classifier on activations.
        
        Args:
            contrastive_pair_set: Set of contrastive pairs with activations and labels
            layer_index: Layer index where steering will be applied
            
        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index
        
        # Get activations and labels - handle different data structures
        try:
            pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
            all_activations = pos_activations + neg_activations
        except AttributeError:
            # Fallback: extract activations directly from pairs
            all_activations = []
            for pair in contrastive_pair_set.pairs:
                if hasattr(pair, 'positive_activations') and pair.positive_activations:
                    if layer_index in pair.positive_activations:
                        all_activations.append(pair.positive_activations[layer_index])
                if hasattr(pair, 'negative_activations') and pair.negative_activations:
                    if layer_index in pair.negative_activations:
                        all_activations.append(pair.negative_activations[layer_index])
        
        if not all_activations:
            raise ValueError("No activations provided for training")
        
        # Get activation dimension and device
        activation_dim = all_activations[0].shape[-1]
        activation_device = all_activations[0].device
        
        # Update device to match activations if not explicitly set
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = activation_device
        elif self.device == "cpu" and activation_device != torch.device("cpu"):
            self.device = activation_device
        
        # Initialize classifier
        self.classifier = KSteeringClassifier(
            input_dim=activation_dim,
            num_labels=self.num_labels,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Create synthetic labels for training (simplified approach)
        # In practice, you'd have real multilabel data from different tones/behaviors
        labels = []
        for i, activation in enumerate(all_activations):
            # Create synthetic multilabel targets
            # Positive activations get target labels, negative get avoid labels
            label = torch.zeros(self.num_labels)
            if i < len(pos_activations):
                # Positive examples - assign target labels
                if self.target_labels:
                    for target_idx in self.target_labels:
                        if target_idx < self.num_labels:
                            label[target_idx] = 1.0
                else:
                    # Default: assign first label
                    label[0] = 1.0
            else:
                # Negative examples - assign avoid labels or different labels
                if self.avoid_labels:
                    for avoid_idx in self.avoid_labels:
                        if avoid_idx < self.num_labels:
                            label[avoid_idx] = 1.0
                else:
                    # Default: assign last label
                    label[-1] = 1.0
            labels.append(label)
        
        # Convert to tensors and ensure they're on the correct device
        activation_tensor = torch.stack([act.to(self.device) for act in all_activations])
        label_tensor = torch.stack(labels).to(self.device)
        
        # Move classifier to the same device as the data
        self.classifier = self.classifier.to(self.device)
        
        # Train classifier
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        classifier_losses = []
        self.classifier.train()
        
        for epoch in range(self.classifier_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.classifier(activation_tensor)
            loss = criterion(logits, label_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            classifier_losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Classifier training epoch {epoch}, loss: {loss.item():.4f}")
        
        # Set to eval mode
        self.classifier.eval()
        
        # Mark as trained first
        self.is_trained = True
        
        # Get a representative vector for compatibility
        representative_vector = self.get_steering_vector()
        
        # Training statistics
        self.training_stats = {
            'method': 'K-Steering',
            'num_labels': self.num_labels,
            'target_labels': self.target_labels,
            'avoid_labels': self.avoid_labels,
            'alpha': self.alpha,
            'layer_index': layer_index,
            'classifier_epochs': self.classifier_epochs,
            'final_classifier_loss': classifier_losses[-1] if classifier_losses else 0.0,
            'num_training_samples': len(all_activations),
            'activation_dim': activation_dim,
            'vector_norm': float(torch.norm(representative_vector).item()),
            'vector_shape': tuple(representative_vector.shape),
            'num_pairs': len(all_activations) // 2  # Approximate since we don't have true pairs
        }
        return self.training_stats
    
    def compute_steering_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute the K-steering loss: L(x) = -1/|T| * Σ(f_k(x)) + 1/|A| * Σ(f_k(x))
        
        Args:
            activations: Input activations
            
        Returns:
            Steering loss
        """
        if not self.is_trained:
            raise ValueError("K-steering must be trained before computing steering loss")
        
        # CRITICAL FIX: Enable gradients explicitly since they're globally disabled
        with torch.enable_grad():
            print(f"      🎯 compute_steering_loss DEBUG:")
            print(f"         📊 Input activations shape: {activations.shape}")
            print(f"         📊 Input activations requires_grad: {activations.requires_grad}")
            print(f"         📊 Target labels: {self.target_labels}")
            print(f"         📊 Avoid labels: {self.avoid_labels}")
            print(f"         ⚡ torch.is_grad_enabled(): {torch.is_grad_enabled()}")
            print(f"         📊 Classifier in training mode: {self.classifier.training}")
            
            # Now that activations preserve gradients, we should have proper computational graph
            if not activations.requires_grad:
                print(f"         🔧 Input activations don't have gradients, enabling them...")
                activations = activations.requires_grad_(True)
                print(f"         ✅ Enabled gradients on input activations")
            
            print(f"         📊 Final input activations requires_grad: {activations.requires_grad}")
            print(f"         📊 Final input activations grad_fn: {activations.grad_fn}")
            
            # CRITICAL FIX: Ensure ALL classifier parameters have requires_grad=True
            print(f"         🔧 Ensuring classifier parameters have gradients enabled...")
            for name, param in self.classifier.named_parameters():
                if not param.requires_grad:
                    print(f"            ⚠️  Parameter {name} had requires_grad=False, fixing...")
                    param.requires_grad = True
            
            print(f"         📊 Classifier training mode: {self.classifier.training}")
            
            # 🧠 Running forward pass through classifier...
            print(f"         🧠 Running forward pass through classifier...")
            
            # Reshape activations for classifier input
            activations_2d = activations.view(activations.shape[0], -1)  # [batch_size, hidden_dim]
            
            # Ensure activations_2d preserves gradients
            if not activations_2d.requires_grad:
                print(f"         🔧 View operation lost gradients, enabling them...")
                activations_2d = activations_2d.requires_grad_(True)
                print(f"         ✅ Enabled gradients on activations_2d")
            
            print(f"         📊 Activations_2d requires_grad: {activations_2d.requires_grad}")
            print(f"         📊 Activations_2d grad_fn: {activations_2d.grad_fn}")
            
            # Forward pass through classifier - this should preserve gradients
            logits = self.classifier(activations_2d)
            print(f"         📊 Raw logits requires_grad: {logits.requires_grad}")
            print(f"         📊 Raw logits grad_fn: {logits.grad_fn}")
            
            # Reshape logits back if needed
            if len(activations.shape) == 3:
                # Original shape was [batch_size, seq_len, hidden_dim]
                # Logits are [batch_size, num_labels]  
                # Reshape to [batch_size, 1, num_labels] to match original sequence structure
                logits = logits.view(activations.shape[0], 1, self.num_labels)
                print(f"         📊 Reshaped logits back to {logits.shape}")
                print(f"         📊 Final logits requires_grad: {logits.requires_grad}")
                print(f"         📊 Final logits grad_fn: {logits.grad_fn}")
                    
            print(f"         📊 Logits shape: {logits.shape}")
            print(f"         📊 Logits requires_grad: {logits.requires_grad}")
            print(f"         📊 Logits grad_fn: {logits.grad_fn}")
            print(f"         📊 Logits mean: {logits.mean().item()}")
            
            # Build loss components without initializing with requires_grad tensor
            loss_components = []
            
            # Target labels - we want to maximize these (negative in loss)
            if self.target_labels:
                target_logits = logits[:, :, self.target_labels]
                target_mean = target_logits.mean()
                print(f"         📈 Target logits shape: {target_logits.shape}")
                print(f"         📈 Target logits mean: {target_mean.item()}")
                print(f"         📈 Target logits requires_grad: {target_mean.requires_grad}")
                print(f"         📈 Target logits grad_fn: {target_mean.grad_fn}")
                loss_components.append(-target_mean)
            
            # Avoid labels - we want to minimize these (positive in loss)  
            if self.avoid_labels:
                avoid_logits = logits[:, :, self.avoid_labels]
                avoid_mean = avoid_logits.mean()
                print(f"         📉 Avoid logits shape: {avoid_logits.shape}")
                print(f"         📉 Avoid logits mean: {avoid_mean.item()}")
                print(f"         📉 Avoid logits requires_grad: {avoid_mean.requires_grad}")
                print(f"         📉 Avoid logits grad_fn: {avoid_mean.grad_fn}")
                loss_components.append(avoid_mean)
            
            # Combine loss components properly
            if loss_components:
                loss = sum(loss_components)
            else:
                # Fallback: use mean of all logits if no specific labels
                loss = logits.mean()
            
            print(f"         ✅ Final loss: {loss.item()}")
            print(f"         ✅ Final loss requires_grad: {loss.requires_grad}")
            print(f"         ✅ Final loss grad_fn: {loss.grad_fn}")
            
            return loss
    
    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Apply K-steering by taking gradient steps on activations.
        
        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier (scales alpha)
            
        Returns:
            Steered activations: x - α * ∇_x L(x)
        """
        if not self.is_trained:
            raise ValueError("K-steering method must be trained before applying steering")
        
        # CRITICAL FIX: Enable gradients explicitly since they're globally disabled
        with torch.enable_grad():
            print(f"\n🔍 K-STEERING DEBUG - apply_steering:")
            print(f"   📊 Input activations shape: {activations.shape}")
            print(f"   📊 Input activations device: {activations.device}")
            print(f"   📊 Input activations dtype: {activations.dtype}")
            print(f"   📊 Input activations requires_grad: {activations.requires_grad}")
            print(f"   📊 Input activations grad_fn: {activations.grad_fn}")
            print(f"   📊 Steering strength: {strength}")
            print(f"   📊 Alpha parameter: {self.alpha}")
            print(f"   ⚡ torch.is_grad_enabled(): {torch.is_grad_enabled()}")
            
            # Ensure the input activations require gradients for steering
            if not activations.requires_grad:
                print(f"   🔧 Input activations don't have gradients, enabling them...")
                steered_activations = activations.requires_grad_(True)
                print(f"   ✅ Enabled gradients on input activations")
            else:
                steered_activations = activations
            
            print(f"   ✅ Prepared steered_activations:")
            print(f"      📊 Shape: {steered_activations.shape}")
            print(f"      📊 Device: {steered_activations.device}")
            print(f"      📊 Requires_grad: {steered_activations.requires_grad}")
            print(f"      📊 Grad_fn: {steered_activations.grad_fn}")
            
            # Ensure classifier parameters have gradients enabled
            print(f"   🧠 Classifier state:")
            print(f"      📊 Training mode: {self.classifier.training}")
            classifier_params_with_grad = sum(1 for p in self.classifier.parameters() if p.requires_grad)
            total_classifier_params = sum(1 for p in self.classifier.parameters())
            print(f"      📊 Parameters with gradients: {classifier_params_with_grad}/{total_classifier_params}")
            
            # Force classifier into training mode to enable gradient computation
            original_classifier_training = self.classifier.training
            self.classifier.train()
            print(f"   🔧 Set classifier to training mode")
            
            try:
                # Clear any existing gradients
                if steered_activations.grad is not None:
                    steered_activations.grad.zero_()
                
                # Compute steering loss
                print(f"   🎯 Computing steering loss...")
                loss = self.compute_steering_loss(steered_activations)
                print(f"      📊 Loss value: {loss.item()}")
                print(f"      📊 Loss requires_grad: {loss.requires_grad}")
                print(f"      📊 Loss grad_fn: {loss.grad_fn}")
                
                # Compute gradients with respect to activations
                print(f"   📈 Computing gradients...")
                try:
                    # Use autograd.grad for more explicit control
                    import torch.autograd as autograd
                    gradients = autograd.grad(
                        outputs=loss,
                        inputs=steered_activations,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True
                    )[0]
                    
                    print(f"      ✅ Gradient computation successful")
                    print(f"      📊 Gradient shape: {gradients.shape}")
                    print(f"      📊 Gradient norm: {torch.norm(gradients).item()}")
                    print(f"      📊 Gradient mean: {gradients.mean().item()}")
                    print(f"      📊 Gradient requires_grad: {gradients.requires_grad}")
                    
                    # Apply gradient step: x' = x - α * ∇_x L(x)
                    print(f"   🎯 Applying gradient step...")
                    with torch.no_grad():
                        effective_alpha = self.alpha * strength
                        print(f"      📊 Effective alpha: {effective_alpha}")
                        
                        grad_step = effective_alpha * gradients
                        print(f"      📊 Gradient step norm: {torch.norm(grad_step).item()}")
                        result = steered_activations - grad_step
                        print(f"      ✅ Applied gradient step")
                        
                        # Preserve original requires_grad setting
                        result = result.requires_grad_(activations.requires_grad)
                        
                except RuntimeError as e:
                    print(f"      ❌ Gradient computation failed: {e}")
                    print(f"      🔄 Returning original activations")
                    result = activations.clone()
                
                print(f"   ✅ Final steered activations:")
                print(f"      📊 Shape: {result.shape}")
                print(f"      📊 Requires_grad: {result.requires_grad}")
                
                return result
                
            finally:
                # Restore original classifier training mode
                self.classifier.train(original_classifier_training)
                print(f"   🔄 Restored classifier training mode to: {original_classifier_training}")
    
    def get_steering_vector(self) -> torch.Tensor:
        """
        Return a representative steering vector (for compatibility).
        Note: K-steering doesn't use a fixed vector but computes gradients dynamically.
        """
        if not self.is_trained:
            raise ValueError("K-steering method must be trained before getting steering vector")
        
        # Return the first layer weights as a representative vector
        return self.classifier.layers[0].weight.mean(dim=0).detach()
    
    def set_targets(self, target_labels: List[int], avoid_labels: Optional[List[int]] = None):
        """
        Set target and avoid labels for steering.
        
        Args:
            target_labels: List of label indices to steer toward
            avoid_labels: List of label indices to steer away from (optional)
        """
        self.target_labels = target_labels
        self.avoid_labels = avoid_labels or []
    
    def get_classifier_probabilities(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Get classifier probabilities for the given activations.
        
        Args:
            activations: Input activations
            
        Returns:
            Probabilities for each label
        """
        if not self.is_trained:
            raise ValueError("K-steering method must be trained before getting probabilities")
        
        with torch.no_grad():
            logits = self.classifier(activations)
            probabilities = torch.sigmoid(logits)
        
        return probabilities
    
    def save_steering_vector(self, path: str) -> bool:
        """Save K-steering data including classifier."""
        if not self.is_trained:
            return False
        try:
            torch.save({
                'classifier_state_dict': self.classifier.state_dict(),
                'num_labels': self.num_labels,
                'hidden_dim': self.hidden_dim,
                'target_labels': self.target_labels,
                'avoid_labels': self.avoid_labels,
                'alpha': self.alpha,
                'layer_index': self.layer_index,
                'training_stats': self.training_stats,
                'method': 'K-Steering'
            }, path)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """Load K-steering data."""
        try:
            data = torch.load(path, map_location=self.device)
            if data.get('method') != 'K-Steering':
                return False
            
            # Reconstruct classifier
            self.num_labels = data['num_labels']
            self.hidden_dim = data['hidden_dim']
            
            # We need to know the input dimension to reconstruct the classifier
            # For now, we'll store it in training_stats
            input_dim = data['training_stats'].get('activation_dim', 4096)
            
            self.classifier = KSteeringClassifier(
                input_dim=input_dim,
                num_labels=self.num_labels,
                hidden_dim=self.hidden_dim
            ).to(self.device)
            
            self.classifier.load_state_dict(data['classifier_state_dict'])
            self.target_labels = data['target_labels']
            self.avoid_labels = data['avoid_labels']
            self.alpha = data['alpha']
            self.layer_index = data['layer_index']
            self.training_stats = data['training_stats']
            self.is_trained = True
            return True
        except Exception:
            return False 