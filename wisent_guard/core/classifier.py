import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import List, Dict, Any, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)

class LogisticModel(nn.Module):
    """Simple logistic regression model for activation classification."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure output has proper dimensions by keeping the batch dimension
        logits = self.linear(x)
        # Keep output shape consistent regardless of batch size
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(1)
        return self.sigmoid(logits)

class MLPModel(nn.Module):
    """Multi-layer perceptron model for activation classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        # Keep output shape consistent regardless of batch size
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(1)
        return logits

def calculate_roc_auc(y_true: List[float], y_scores: List[float]) -> float:
    """
    Calculate the ROC AUC score without using scikit-learn.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted scores
        
    Returns:
        ROC AUC score
    """
    if len(y_true) != len(y_scores):
        raise ValueError("Length of y_true and y_scores must match")
    
    if len(set(y_true)) != 2:
        # Not a binary classification problem or only one class in the data
        return 0.5
    
    # Pair the scores with their true labels and sort by score in descending order
    pair_list = sorted(zip(y_scores, y_true), reverse=True)
    
    # Count the number of positive and negative samples
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        # Only one class present, ROC AUC is not defined
        return 0.5
    
    # Count the number of correctly ranked pairs
    auc = 0.0
    pos_so_far = 0
    
    # Iterate through pairs
    for i, (_, label) in enumerate(pair_list):
        if label == 1:
            # This is a positive example
            pos_so_far += 1
        else:
            # This is a negative example, add the number of positive examples seen so far
            auc += pos_so_far
    
    # Normalize by the number of positive-negative pairs
    auc /= (n_pos * n_neg)
    
    return auc

class Classifier:
    def __init__(self, model_type='logistic', device=None, threshold=0.5, model_path=None):
        self.model_type = model_type
        self.threshold = threshold
        self.model_path = model_path
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.model = None
        
        # Load model if path provided
        if model_path is not None:
            self.load_model(model_path)

    def _extract_features(self, activation_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features from activation tensor."""
        # Convert to PyTorch tensor if not already
        if not isinstance(activation_tensor, torch.Tensor):
            activation_tensor = torch.tensor(activation_tensor, dtype=torch.float32, device=self.device)
        else:
            # If it's already a tensor, ensure it's the right type for MPS compatibility
            if self.device == 'mps' and activation_tensor.dtype != torch.float32:
                activation_tensor = activation_tensor.to(dtype=torch.float32)
            
            # Move to the correct device if needed
            activation_tensor = activation_tensor.to(device=self.device, dtype=torch.float32)
        
        # Flatten if needed
        if len(activation_tensor.shape) > 1:
            activation_tensor = activation_tensor.flatten()
        
        return activation_tensor

    def fit(self, X, y, test_size=0.2, num_epochs=50, batch_size=32, learning_rate=0.001, 
            early_stopping_patience=10, random_state=42, **model_params):
        """
        Train the classifier on activation data.
        
        Args:
            X: Input activations (tensor or list of tensors)
            y: Labels (tensor or list)
            test_size: Proportion of data for testing
            num_epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Epochs to wait for improvement
            random_state: Random seed
            **model_params: Additional model parameters
        """
        print(f"Preparing to train {self.model_type} classifier with {len(X)} samples")
        
        # Set random seed for reproducibility
        torch.manual_seed(random_state)
        
        # Process inputs
        if isinstance(X, list):
            X = torch.stack([self._extract_features(x) for x in X])
        else:
            X = self._extract_features(X)
            
        if isinstance(y, list):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(device=self.device, dtype=torch.float32)
        
        # Get input dimension
        input_dim = X.shape[1] if len(X.shape) > 1 else X.shape[0]
        
        # Create TensorDataset
        dataset = TensorDataset(X, y)
        
        # Split into train and test
        test_count = int(test_size * len(dataset))
        train_count = len(dataset) - test_count
        train_dataset, test_dataset = random_split(dataset, [train_count, test_count])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        if self.model_type.lower() == "mlp":
            hidden_dim = model_params.get('hidden_dim', 128)
            self.model = MLPModel(input_dim, hidden_dim=hidden_dim).to(self.device)
        else:  # default to logistic
            self.model = LogisticModel(input_dim).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_accuracy = 0.0
        best_model_state = None
        early_stopping_counter = 0
        
        print(f"Starting training for up to {num_epochs} epochs...")
        
        metrics = {
            'train_loss': [],
            'test_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.view(-1)  # Flatten to match labels
                labels = labels.view(-1)    # Ensure labels are flattened too
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            metrics['train_loss'].append(train_loss)
            
            # Evaluation phase
            self.model.eval()
            test_loss = 0.0
            all_preds = []
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = self.model(inputs)
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    
                    probs = outputs.cpu()
                    preds = (probs >= self.threshold).float()
                    
                    all_probs.extend(probs.tolist())
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.cpu().tolist())
            
            test_loss /= len(test_loader)
            metrics['test_loss'].append(test_loss)
            
            # Calculate metrics
            test_accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_preds)
            metrics['accuracy'].append(test_accuracy)
            
            # Calculate precision, recall, F1
            true_positives = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
            false_positives = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
            false_negatives = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            
            # Calculate AUC
            try:
                auc = calculate_roc_auc(all_labels, all_probs)
                metrics['auc'].append(auc)
            except Exception as e:
                metrics['auc'].append(0.0)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                      f"Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}")
            
            # Check for improvement
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_state = self.model.state_dict().copy()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in accuracy")
                break
        
        # Load the best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        self.model.eval()
        y_pred = []
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                outputs = outputs.view(-1)
                labels = labels.view(-1)
                probs = outputs.cpu()
                preds = (probs >= self.threshold).float()
                
                y_prob.extend(probs.tolist())
                y_pred.extend(preds.tolist())
                y_true.extend(labels.cpu().tolist())
        
        # Final metrics
        test_accuracy = sum(1 for p, l in zip(y_pred, y_true) if p == l) / len(y_pred)
        
        true_positives = sum(1 for p, l in zip(y_pred, y_true) if p == 1 and l == 1)
        false_positives = sum(1 for p, l in zip(y_pred, y_true) if p == 1 and l == 0)
        false_negatives = sum(1 for p, l in zip(y_pred, y_true) if p == 0 and l == 1)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUC
        try:
            auc = calculate_roc_auc(y_true, y_prob)
        except Exception:
            auc = 0.0
        
        # Final results
        training_results = {
            'accuracy': float(test_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'epochs': epoch + 1,
            'input_dim': input_dim,
            'model_type': self.model_type,
            'metrics_history': {k: [float(val) for val in v] for k, v in metrics.items()},
            'best_epoch': metrics['accuracy'].index(max(metrics['accuracy'])) + 1
        }
        
        print("\nTraining complete!")
        print(f"Final metrics - Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return training_results

    def predict(self, X):
        """
        Predict using the trained model.
        
        Args:
            X: Input data (tensor or list of tensors)
            
        Returns:
            Predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded. Call fit() or load_model() first.")
        
        self.model.eval()
        
        # Process input
        if isinstance(X, list):
            X = torch.stack([self._extract_features(x) for x in X])
        else:
            X = self._extract_features(X)
        
        # Reshape for model if needed
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = outputs.view(-1)
            predictions = (probabilities >= self.threshold).int()
        
        return predictions.cpu().numpy() if len(predictions) > 1 else predictions.item()

    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Input data (tensor or list of tensors)
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded. Call fit() or load_model() first.")
        
        self.model.eval()
        
        # Process input
        if isinstance(X, list):
            X = torch.stack([self._extract_features(x) for x in X])
        else:
            X = self._extract_features(X)
        
        # Reshape for model if needed
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = outputs.view(-1)
        
        return probabilities.cpu().numpy() if len(probabilities) > 1 else probabilities.item()

    def save_model(self, save_path: str) -> None:
        """
        Save the trained classifier model to disk.
        
        Args:
            save_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Extract model metadata
        model_type = 'mlp' if isinstance(self.model, MLPModel) else 'logistic'
        input_dim = next(self.model.parameters()).shape[1]
        
        # For MLP, extract hidden_dim
        hidden_dim = None
        if model_type == 'mlp':
            # Extract hidden dim from first layer
            hidden_dim = self.model.network[0].out_features
        
        # Create save dictionary with metadata
        save_dict = {
            'model': self.model.state_dict(),
            'model_type': model_type,
            'input_dim': input_dim,
            'threshold': self.threshold
        }
        
        if hidden_dim is not None:
            save_dict['hidden_dim'] = hidden_dim
        
        # Save the model
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, model_path: str) -> None:
        """
        Load a trained classifier model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is not a valid classifier
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load the PyTorch model with weights_only=False for backward compatibility
            loaded_data = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract the model and metadata
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                # Model was saved with metadata
                model_state = loaded_data['model']
                self.model_type = loaded_data.get('model_type', 'logistic')
                input_dim = loaded_data.get('input_dim', None)
                
                # Create model instance based on type
                if self.model_type == 'mlp':
                    hidden_dim = loaded_data.get('hidden_dim', 128)
                    self.model = MLPModel(input_dim, hidden_dim=hidden_dim)
                else:  # default to logistic
                    self.model = LogisticModel(input_dim)
                
                # Load state dictionary
                self.model.load_state_dict(model_state)
            else:
                # Model was saved directly
                self.model = loaded_data
                
            self.model = self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            
            print(f"Loaded {self.model_type} model from {model_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")

    def set_threshold(self, threshold: float) -> None:
        """
        Update the classification threshold.
        
        Args:
            threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test input data
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded.")
        
        # Get predictions
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        # Convert to lists if needed
        if not isinstance(y_pred, list):
            y_pred = [y_pred] if not hasattr(y_pred, '__iter__') else list(y_pred)
        if not isinstance(y_prob, list):
            y_prob = [y_prob] if not hasattr(y_prob, '__iter__') else list(y_prob)
        if not isinstance(y, list):
            y = [y] if not hasattr(y, '__iter__') else list(y)
        
        # Calculate metrics
        accuracy = sum(1 for p, l in zip(y_pred, y) if p == l) / len(y)
        
        true_positives = sum(1 for p, l in zip(y_pred, y) if p == 1 and l == 1)
        false_positives = sum(1 for p, l in zip(y_pred, y) if p == 1 and l == 0)
        false_negatives = sum(1 for p, l in zip(y_pred, y) if p == 0 and l == 1)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUC
        try:
            auc = calculate_roc_auc(y, y_prob)
        except Exception:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

class ActivationClassifier:
    """
    Classifier for detecting harmful content from activations.
    Integrated into the classifier primitive.
    """
    
    def __init__(
        self,
        model_type: str = "logistic",
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """Initialize activation classifier."""
        self.classifier = Classifier(
            model_type=model_type,
            threshold=threshold,
            device=device
        )
        self.is_trained = False
    
    def train_on_activations(
        self,
        harmful_activations: List['Activations'],
        harmless_activations: List['Activations']
    ) -> Dict[str, Any]:
        """Train classifier on activation data."""
        from .activations import Activations
        
        # Prepare training data
        X = []
        y = []
        
        # Add harmful examples (label = 1)
        for activation in harmful_activations:
            features = activation.extract_features_for_classifier()
            X.append(features)
            y.append(1)
        
        # Add harmless examples (label = 0)
        for activation in harmless_activations:
            features = activation.extract_features_for_classifier()
            X.append(features)
            y.append(0)
        
        # Train classifier
        results = self.classifier.fit(X, y)
        self.is_trained = True
        return results
    
    def predict_from_activations(self, activations: 'Activations') -> Dict[str, Any]:
        """Predict if activations represent harmful content."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        
        features = activations.extract_features_for_classifier()
        prediction = self.classifier.predict([features])[0]
        probability = self.classifier.predict_proba([features])[0]
        
        return {
            "is_harmful": bool(prediction),
            "probability": float(probability),
            "confidence": abs(probability - 0.5) * 2
        }
    
    def evaluate_on_activations(
        self,
        test_harmful: List['Activations'],
        test_harmless: List['Activations']
    ) -> Dict[str, float]:
        """Evaluate classifier performance on activation data."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before evaluation")
        
        # Prepare test data
        X_test = []
        y_test = []
        
        for activation in test_harmful:
            X_test.append(activation.extract_features_for_classifier())
            y_test.append(1)
        
        for activation in test_harmless:
            X_test.append(activation.extract_features_for_classifier())
            y_test.append(0)
        
        return self.classifier.evaluate(X_test, y_test)
    
    def save(self, path: str) -> None:
        """Save trained classifier."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained classifier")
        self.classifier.save_model(path)
    
    def load(self, path: str) -> None:
        """Load trained classifier."""
        self.classifier.load_model(path)
        self.is_trained = True
    
    def set_threshold(self, threshold: float) -> None:
        """Set classification threshold."""
        self.classifier.set_threshold(threshold) 