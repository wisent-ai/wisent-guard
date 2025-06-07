"""
Guard training pipeline using wisent-guard's classifier system.
"""

import logging
import os
import torch
from typing import List, Dict, Any, Tuple, Optional
from ..classifier import ActivationClassifier
from .generate import load_model_and_tokenizer, extract_hidden_states
from .data import prepare_prompts_from_docs

logger = logging.getLogger(__name__)

class GuardPipeline:
    """
    Pipeline for training and using activation-based guards.
    """
    
    def __init__(
        self, 
        model_name: str,
        layer: int = 15,
        device: Optional[str] = None,
        classifier_type: str = "logistic",
        save_dir: str = "./wg_harness_models"
    ):
        """
        Initialize the guard pipeline.
        
        Args:
            model_name: Language model name
            layer: Layer to extract activations from
            device: Device to run on
            classifier_type: Type of classifier ("logistic" or "mlp")
            save_dir: Directory to save trained models
        """
        self.model_name = model_name
        self.layer = layer
        self.classifier_type = classifier_type
        self.save_dir = save_dir
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model = None
        self.tokenizer = None
        self.classifier = None
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Initialized GuardPipeline for {model_name} layer {layer} on {device}")
    
    def _load_model_if_needed(self):
        """Load model and tokenizer if not already loaded."""
        if self.model is None or self.tokenizer is None:
            logger.info("Loading language model...")
            self.model, self.tokenizer = load_model_and_tokenizer(self.model_name, self.device)
    
    def _extract_activations_from_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract activations from a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of activation dictionaries
        """
        self._load_model_if_needed()
        
        activations = []
        
        for text in texts:
            try:
                # Use the model to get hidden states for the text
                # We'll simulate a prompt-response scenario where the text is the response
                prompt = f"Question: Please respond.\nAnswer: {text}"
                
                _, hidden_state = extract_hidden_states(
                    self.model, self.tokenizer, prompt, self.layer, max_new_tokens=1
                )
                
                # Format as activation data expected by classifier
                # The classifier expects an 'activations' key with the tensor data
                activation_data = {
                    'activations': hidden_state,  # Direct tensor, classifier will handle device/shape
                    'layer': self.layer,
                    'token_strategy': 'last'
                }
                
                activations.append(activation_data)
                
            except Exception as e:
                logger.warning(f"Failed to extract activation for text: {e}")
                # Create zero activation as fallback
                zero_activation = {
                    'activations': torch.zeros(self.model.config.hidden_size),
                    'layer': self.layer,
                    'token_strategy': 'last'
                }
                activations.append(zero_activation)
        
        return activations
    
    def fit(self, train_triples: List[Tuple[str, str, str]]) -> None:
        """
        Train the guard classifier on labeled data.
        
        Args:
            train_triples: List of (prompt, good_response, bad_response) tuples
        """
        logger.info(f"Training guard classifier on {len(train_triples)} examples")
        
        # Extract good and bad responses
        good_responses = [triple[1] for triple in train_triples]
        bad_responses = [triple[2] for triple in train_triples]
        
        logger.info("Extracting activations for good responses...")
        good_activations = self._extract_activations_from_texts(good_responses)
        
        logger.info("Extracting activations for bad responses...")
        bad_activations = self._extract_activations_from_texts(bad_responses)
        
        # Train classifier
        logger.info(f"Training {self.classifier_type} classifier...")
        
        self.classifier = ActivationClassifier.create_from_activations(
            harmful_activations=bad_activations,  # Bad responses are "harmful"
            harmless_activations=good_activations,  # Good responses are "harmless"
            model_type=self.classifier_type,
            threshold=0.5,
            positive_class_label="bad_response",
            device=self.device,
            test_size=0.2,
            num_epochs=50,
            batch_size=32,
            learning_rate=0.001
        )
        
        logger.info("Guard classifier training completed")
    
    def predict(self, hidden_state: torch.Tensor) -> int:
        """
        Predict if a hidden state represents bad behavior.
        
        Args:
            hidden_state: Hidden state tensor
            
        Returns:
            0 for good, 1 for bad
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        
        # Format hidden state as activation data
        activation_data = {
            'activations': hidden_state,
            'layer': self.layer,
            'token_strategy': 'last'
        }
        
        result = self.classifier.predict(activation_data)
        
        # Return binary prediction - use 'is_harmful' key from classifier result
        return 1 if result['is_harmful'] else 0
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Predict if a text represents bad behavior.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction and confidence
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        
        # Extract activation
        activations = self._extract_activations_from_texts([text])
        activation_data = activations[0]
        
        # Get prediction
        result = self.classifier.predict(activation_data, text)
        
        return {
            'prediction': 1 if result['is_harmful'] else 0,
            'confidence': result['score'],
            'text': text
        }
    
    def batch_predict(self, hidden_states: List[torch.Tensor]) -> List[int]:
        """
        Predict for multiple hidden states.
        
        Args:
            hidden_states: List of hidden state tensors
            
        Returns:
            List of predictions (0 for good, 1 for bad)
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        
        predictions = []
        
        for hidden_state in hidden_states:
            pred = self.predict(hidden_state)
            predictions.append(pred)
        
        return predictions
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained classifier.
        
        Args:
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved model
        """
        if self.classifier is None:
            raise ValueError("No classifier to save. Train first with fit().")
        
        if filename is None:
            # Generate filename based on model and layer
            model_short = self.model_name.split('/')[-1]  # Get last part of model name
            filename = f"guard_{model_short}_layer{self.layer}_{self.classifier_type}.pkl"
        
        save_path = os.path.join(self.save_dir, filename)
        self.classifier.save_model(save_path)
        
        logger.info(f"Saved guard classifier to {save_path}")
        return save_path
    
    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained classifier.
        
        Args:
            model_path: Path to the saved model
        """
        logger.info(f"Loading guard classifier from {model_path}")
        
        self.classifier = ActivationClassifier(
            model_path=model_path,
            threshold=0.5,
            positive_class_label="bad_response",
            device=self.device
        )
        
        logger.info("Guard classifier loaded successfully")
    
    def evaluate_on_test_set(
        self, 
        test_triples: List[Tuple[str, str, str]]
    ) -> Dict[str, float]:
        """
        Evaluate the classifier on a test set.
        
        Args:
            test_triples: List of (prompt, good_response, bad_response) tuples
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        
        logger.info(f"Evaluating classifier on {len(test_triples)} test examples")
        
        # Prepare test data
        all_texts = []
        true_labels = []
        
        for prompt, good, bad in test_triples:
            all_texts.extend([good, bad])
            true_labels.extend([0, 1])  # 0 = good, 1 = bad
        
        # Get predictions
        predictions = []
        confidences = []
        
        for text in all_texts:
            result = self.predict_text(text)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
        
        # Calculate metrics
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
        except ImportError:
            # Fallback calculation without sklearn
            tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
            tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
            fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
            
            accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Count true/false positives/negatives
        tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total_samples': len(true_labels),
            'bad_samples_detected': tp,
            'bad_samples_total': sum(true_labels),
            'detection_rate': tp / sum(true_labels) if sum(true_labels) > 0 else 0.0
        }
        
        logger.info(f"Evaluation complete - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        logger.info(f"Detection rate: {metrics['detection_rate']:.2%} ({tp}/{sum(true_labels)} bad samples)")
        
        return metrics 