"""
Classifier module for analyzing activation patterns with PyTorch-based machine learning models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, Any, Tuple, List, Optional, Union
import math

class LogisticModel(nn.Module):
    """Simple PyTorch logistic regression model"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure output has proper dimensions by keeping the batch dimension
        logits = self.linear(x)
        # Keep output shape consistent regardless of batch size
        # This ensures output is [batch_size, 1] even for single samples
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(1)
        return self.sigmoid(logits)

class MLPModel(nn.Module):
    """PyTorch Multi-Layer Perceptron model for binary classification"""
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

class ActivationClassifier:
    """
    PyTorch-based classifier for activation pattern analysis.
    
    This classifier analyzes activation patterns from language models to detect various types
    of content or behaviors. It can be used for hallucination detection, toxicity detection,
    topic classification, or any other classification task based on activation patterns.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        positive_class_label: str = "positive",
        model: Optional[nn.Module] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the activation classifier.
        
        Args:
            model_path: Path to the trained classifier model (PyTorch format)
            threshold: Classification threshold (default: 0.5)
            positive_class_label: Label for the positive class (default: "positive")
            model: Pre-trained PyTorch model (optional)
            device: Device to run the model on (e.g., 'cuda', 'cpu')
        """
        self.model_path = model_path
        self.threshold = threshold
        self.positive_class_label = positive_class_label
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        
        if model is not None:
            # Use provided model directly
            self.model = model.to(self.device)
        elif model_path is not None:
            # Load model from path
            self.model = self._load_model(model_path)
        else:
            # No model provided or path given - will need to be trained
            self.model = None
        
    def _load_model(self, model_path: str) -> nn.Module:
        """
        Load a trained classifier model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded PyTorch classifier model
        
        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is not a valid classifier
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load the PyTorch model
            loaded_data = torch.load(model_path, map_location=self.device)
            
            # Extract the model and metadata
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                # Model was saved with metadata
                model_state = loaded_data['model']
                model_type = loaded_data.get('model_type', 'logistic')
                input_dim = loaded_data.get('input_dim', None)
                
                # Create model instance based on type
                if model_type == 'mlp':
                    hidden_dim = loaded_data.get('hidden_dim', 128)
                    model = MLPModel(input_dim, hidden_dim=hidden_dim)
                else:  # default to logistic
                    model = LogisticModel(input_dim)
                
                # Load state dictionary
                model.load_state_dict(model_state)
            else:
                # Model was saved directly
                model = loaded_data
                
            model = model.to(self.device)
            model.eval()  # Set model to evaluation mode
            
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def _extract_features(self, token_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract features from token activation data.
        
        Args:
            token_data: Dictionary containing token activation data
            
        Returns:
            PyTorch tensor of features for classification
        """
        # Get the token's activation values
        activations = token_data.get('activations', None)
        
        if activations is None:
            raise ValueError("Token data doesn't contain activation values")
        
        # Convert to PyTorch tensor if not already
        if not isinstance(activations, torch.Tensor):
            # Check if it's a numpy array and convert accordingly
            if hasattr(activations, 'dtype'):  # likely a numpy array
                activations = torch.tensor(activations, dtype=torch.float32, device=self.device)
            else:
                activations = torch.tensor(activations, dtype=torch.float32, device=self.device)
        
        # Move to the correct device if needed
        activations = activations.to(self.device)
        
        # Flatten if needed
        if len(activations.shape) > 1:
            activations = activations.flatten()
        
        return activations
    
    def extract_features(self, token_data: Dict[str, Any]) -> torch.Tensor:
        """
        Public method to extract features from token activation data.
        
        Args:
            token_data: Dictionary containing token activation data
            
        Returns:
            PyTorch tensor of features for classification
        """
        return self._extract_features(token_data)
    
    def predict(self, token_data: Dict[str, Any], response_text: str = None) -> Dict[str, Any]:
        """
        Predict whether token activations match the target class.
        
        Args:
            token_data: Dictionary containing token activation data
            response_text: Optional response text for logging
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("No model has been loaded or trained. Call train() or load a model first.")
            
        # Extract features
        features = self._extract_features(token_data)
        
        # Reshape for model (expects 2D tensor)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get prediction probability
        with torch.no_grad():
            outputs = self.model(features)
            probability = outputs.view(-1).item()  # Ensure we get a scalar value
        
        # Create result dictionary
        result = {
            'score': float(probability),
            'is_harmful': bool(probability >= self.threshold),
            'threshold': float(self.threshold)
        }
        
        return result
    
    def batch_predict(self, tokens_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict on a batch of token activations.
        
        Args:
            tokens_data: List of token activation dictionaries
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for token_data in tokens_data:
            results.append(self.predict(token_data))
        return results
    
    def set_threshold(self, threshold: float) -> None:
        """
        Update the classification threshold.
        
        Args:
            threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
    
    def train(self, 
              harmful_activations: List[Dict[str, Any]], 
              harmless_activations: List[Dict[str, Any]], 
              model_type: str = "logistic", 
              test_size: float = 0.2, 
              random_state: int = 42,
              batch_size: int = 64,
              num_epochs: int = 100,
              learning_rate: float = 0.001,
              early_stopping_patience: int = 10,
              **model_params) -> Dict[str, Any]:
        """
        Train the classifier on harmful and harmless activation patterns.
        
        Args:
            harmful_activations: List of activation dictionaries labeled as harmful (class 1)
            harmless_activations: List of activation dictionaries labeled as harmless (class 0)
            model_type: Type of model to train: "logistic" or "mlp" (default: "logistic")
            test_size: Proportion of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            batch_size: Size of batches for training (default: 64)
            num_epochs: Maximum number of training epochs (default: 100)
            learning_rate: Learning rate for optimizer (default: 0.001)
            early_stopping_patience: Number of epochs to wait for improvement (default: 10)
            **model_params: Additional parameters to pass to the model constructor
                           
        Returns:
            Dictionary containing training metrics and results
        """
        print(f"Preparing to train {model_type} classifier with {len(harmful_activations)} harmful "
              f"and {len(harmless_activations)} harmless samples")
        
        # Set random seed for reproducibility
        torch.manual_seed(random_state)
        
        # Extract features from activations and convert to tensors
        X_harmful = torch.stack([self._extract_features(act) for act in harmful_activations])
        X_harmless = torch.stack([self._extract_features(act) for act in harmless_activations])
        
        # Create labels
        y_harmful = torch.ones(len(harmful_activations), dtype=torch.float32, device=self.device)
        y_harmless = torch.zeros(len(harmless_activations), dtype=torch.float32, device=self.device)
        
        # Combine data
        X = torch.cat([X_harmful, X_harmless], dim=0)
        y = torch.cat([y_harmful, y_harmless], dim=0)
        
        # Get input dimension
        input_dim = X.shape[1]
        
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
        if model_type.lower() == "mlp":
            hidden_dim = model_params.get('hidden_dim', 128)
            self.model = MLPModel(input_dim, hidden_dim=hidden_dim).to(self.device)
        else:  # default to logistic
            self.model = LogisticModel(input_dim).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_test_loss = float('inf')
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
                # Ensure outputs and labels have matching dimensions
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
                    # Ensure dimensions match for evaluation too
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
                print(f"Error calculating AUC: {e}")
                metrics['auc'].append(0.0)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                      f"Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}")
            
            # Check for improvement
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_test_loss = test_loss
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
                # Ensure dimensions match for evaluation too
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
        except Exception as e:
            print(f"Error calculating final AUC: {e}")
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
            'model_type': model_type,
            'metrics_history': {k: [float(val) for val in v] for k, v in metrics.items()},
            'best_epoch': metrics['accuracy'].index(max(metrics['accuracy'])) + 1
        }
        
        print("\nTraining complete!")
        print(f"Final metrics - Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return training_results
    
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
            'threshold': self.threshold,
            'positive_class_label': self.positive_class_label
        }
        
        if hidden_dim is not None:
            save_dict['hidden_dim'] = hidden_dim
        
        # Save the model
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def create_from_activations(cls, 
                               harmful_activations: List[Dict[str, Any]], 
                               harmless_activations: List[Dict[str, Any]], 
                               model_type: str = "logistic",
                               save_path: Optional[str] = None,
                               threshold: float = 0.5,
                               positive_class_label: str = "harmful",
                               device: Optional[str] = None,
                               **model_params) -> 'ActivationClassifier':
        """
        Create and train a classifier from activation data.
        
        Args:
            harmful_activations: List of activation dictionaries labeled as harmful
            harmless_activations: List of activation dictionaries labeled as harmless
            model_type: Type of model to train: "logistic" or "mlp" (default: "logistic")
            save_path: Path to save the trained model (optional)
            threshold: Classification threshold (default: 0.5)
            positive_class_label: Label for the positive class (default: "harmful")
            device: Device to run the model on (e.g., 'cuda', 'cpu')
            **model_params: Additional parameters for the training process
            
        Returns:
            Trained ActivationClassifier instance
        """
        # Create classifier instance
        classifier = cls(threshold=threshold, 
                         positive_class_label=positive_class_label,
                         device=device)
        
        # Train the classifier
        results = classifier.train(
            harmful_activations=harmful_activations,
            harmless_activations=harmless_activations,
            model_type=model_type,
            **model_params
        )
        
        # Save model if path provided
        if save_path is not None:
            classifier.save_model(save_path)
        
        return classifier

# For backward compatibility
HallucinationClassifier = ActivationClassifier 