"""
Classifier module for analyzing activation patterns with machine learning models.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

from sklearn.base import BaseEstimator
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class ActivationClassifier:
    """
    Machine learning classifier for activation pattern analysis.
    
    This classifier analyzes activation patterns from language models to detect various types
    of content or behaviors. It can be used for hallucination detection, toxicity detection,
    topic classification, or any other classification task based on activation patterns.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        positive_class_label: str = "positive",
        model: Optional[BaseEstimator] = None
    ):
        """
        Initialize the activation classifier.
        
        Args:
            model_path: Path to the trained classifier model (joblib or pickle format)
            threshold: Classification threshold (default: 0.5)
            positive_class_label: Label for the positive class (default: "positive")
            model: Pre-trained scikit-learn compatible model (optional)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.positive_class_label = positive_class_label
        
        if model is not None:
            # Use provided model directly
            self.model = model
        elif model_path is not None:
            # Load model from path
            self.model = self._load_model(model_path)
        else:
            # No model provided or path given - will need to be trained
            self.model = None
        
    def _load_model(self, model_path: str) -> BaseEstimator:
        """
        Load a trained classifier model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded scikit-learn compatible classifier model
        
        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is not a valid classifier
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Try joblib first (preferred for scikit-learn models)
            model = joblib.load(model_path)
        except Exception:
            try:
                # Fall back to pickle if joblib fails
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load model from {model_path}: {e}")
        
        # Verify the model has the necessary methods
        if not (hasattr(model, 'predict') or hasattr(model, 'predict_proba')):
            raise ValueError(f"Loaded model doesn't have required prediction methods")
        
        return model
    
    def _extract_features(self, token_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from token activation data.
        
        Args:
            token_data: Dictionary containing token activation data
            
        Returns:
            Numpy array of features for classification
        """
        # Get the token's activation values
        activations = token_data.get('activations', None)
        
        if activations is None:
            raise ValueError("Token data doesn't contain activation values")
        
        # Convert to numpy array if not already
        if not isinstance(activations, np.ndarray):
            activations = np.array(activations)
        
        # Reshape if needed
        if len(activations.shape) > 1:
            activations = activations.flatten()
        
        return activations
    
    def extract_features(self, token_data: Dict[str, Any]) -> np.ndarray:
        """
        Public method to extract features from token activation data.
        
        Args:
            token_data: Dictionary containing token activation data
            
        Returns:
            Numpy array of features for classification
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
        
        # Reshape for sklearn (expects 2D array)
        features = features.reshape(1, -1)
        
        # Get prediction probability
        probability = 0.0
        
        if hasattr(self.model, 'predict_proba'):
            try:
                # Get probability of positive class
                proba = self.model.predict_proba(features)
                
                # Find the index of the positive class
                if hasattr(self.model, 'classes_'):
                    pos_idx = np.where(self.model.classes_ == 1)[0]
                    if len(pos_idx) > 0:
                        probability = proba[0, pos_idx[0]]
                    else:
                        # If no class is labeled as 1, use the second column 
                        # (typically class 1 in binary classification)
                        probability = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
                else:
                    # Default to the second column if classes are not defined
                    probability = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
            except Exception:
                # Fall back to decision function if predict_proba fails
                if hasattr(self.model, 'decision_function'):
                    decision = self.model.decision_function(features)
                    # Convert decision function to probability-like score (0-1 range)
                    probability = 1 / (1 + np.exp(-decision[0]))
                else:
                    # If all else fails, use binary prediction
                    probability = float(self.model.predict(features)[0])
        else:
            # For models without predict_proba, use decision_function if available
            if hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(features)
                # Convert decision function to probability-like score (0-1 range)
                probability = 1 / (1 + np.exp(-decision[0]))
            else:
                # If all else fails, use binary prediction
                probability = float(self.model.predict(features)[0])
        
        # Create result dictionary
        result = {
            'score': float(probability),
            'is_harmful': bool(probability >= self.threshold),
            'threshold': float(self.threshold)
        }
        
        return result
    
    def batch_predict(self, tokens_data: List[Dict[str, Any]]) -> List[Tuple[float, bool]]:
        """
        Predict on a batch of token activations.
        
        Args:
            tokens_data: List of token activation dictionaries
            
        Returns:
            List of tuples (probability, classification_result)
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
              **model_params) -> Dict[str, Any]:
        """
        Train the classifier on harmful and harmless activation patterns.
        
        Args:
            harmful_activations: List of activation dictionaries labeled as harmful (class 1)
            harmless_activations: List of activation dictionaries labeled as harmless (class 0)
            model_type: Type of model to train: "logistic" or "mlp" (default: "logistic")
            test_size: Proportion of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            **model_params: Additional parameters to pass to the model constructor
                           
        Returns:
            Dictionary containing training metrics and results
        """
        print(f"Preparing to train {model_type} classifier with {len(harmful_activations)} harmful "
              f"and {len(harmless_activations)} harmless samples")
        
        # Extract features from activations
        X_harmful = np.vstack([self._extract_features(act).reshape(1, -1) for act in harmful_activations])
        X_harmless = np.vstack([self._extract_features(act).reshape(1, -1) for act in harmless_activations])
        
        # Create labels
        y_harmful = np.ones(len(harmful_activations))
        y_harmless = np.zeros(len(harmless_activations))
        
        # Combine data
        X = np.vstack([X_harmful, X_harmless])
        y = np.concatenate([y_harmful, y_harmless])
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
        
        # Create the model based on model_type
        if model_type.lower() == "logistic":
            # Default parameters for LogisticRegression if not provided
            default_params = {
                'C': 1.0,
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': random_state
            }
            # Override defaults with any provided parameters
            model_params = {**default_params, **model_params}
            model = LogisticRegression(**model_params)
            print(f"Created LogisticRegression model with parameters: {model_params}")
            
        elif model_type.lower() == "mlp":
            # Default parameters for MLPClassifier if not provided
            default_params = {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'max_iter': 1000,
                'random_state': random_state
            }
            # Override defaults with any provided parameters
            model_params = {**default_params, **model_params}
            model = MLPClassifier(**model_params)
            print(f"Created MLPClassifier model with parameters: {model_params}")
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'logistic' or 'mlp'.")
        
        # Train the model
        print("Training the model...")
        model.fit(X_train, y_train)
        
        # Evaluate the model
        print("Evaluating the model...")
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            # If model doesn't have predict_proba, use decision function if available
            if hasattr(model, 'decision_function'):
                decisions = model.decision_function(X_test)
                auc = roc_auc_score(y_test, decisions)
            else:
                auc = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        # Store the trained model
        self.model = model
        
        # Create results dictionary
        results = {
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'test_size': test_size,
            'harmful_samples': len(harmful_activations),
            'harmless_samples': len(harmless_activations),
            'feature_dim': X.shape[1],
            'model_params': model_params
        }
        
        # Format AUC string properly for display
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        
        print(f"Training complete with accuracy: {accuracy:.4f}, precision: {precision:.4f}, "
              f"recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_str}")
        
        return results
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            save_path: Path to save the model file
        
        Raises:
            ValueError: If no model has been trained or loaded
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded. Cannot save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Save using joblib (preferred for scikit-learn models)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")
        
        # Update the model path
        self.model_path = save_path
    
    @classmethod
    def create_from_activations(cls, 
                               harmful_activations: List[Dict[str, Any]], 
                               harmless_activations: List[Dict[str, Any]], 
                               model_type: str = "logistic",
                               save_path: Optional[str] = None,
                               threshold: float = 0.5,
                               positive_class_label: str = "harmful",
                               **model_params) -> 'ActivationClassifier':
        """
        Create and train a new classifier from activation data.
        
        Args:
            harmful_activations: List of activation dictionaries labeled as harmful (class 1)
            harmless_activations: List of activation dictionaries labeled as harmless (class 0)
            model_type: Type of model to train: "logistic" or "mlp" (default: "logistic")
            save_path: Path to save the trained model (optional)
            threshold: Classification threshold (default: 0.5)
            positive_class_label: Label for the positive class (default: "harmful")
            **model_params: Additional parameters to pass to the model constructor
            
        Returns:
            Trained ActivationClassifier instance
        """
        # Create a new classifier instance
        classifier = cls(threshold=threshold, positive_class_label=positive_class_label)
        
        # Train the model
        classifier.train(
            harmful_activations=harmful_activations,
            harmless_activations=harmless_activations,
            model_type=model_type,
            **model_params
        )
        
        # Save the model if a path is provided
        if save_path:
            classifier.save_model(save_path)
        
        return classifier

# For backward compatibility
HallucinationClassifier = ActivationClassifier 