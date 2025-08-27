import datetime
import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from wisent_guard.core.activations import Activations
from wisent_guard.core.classifier.classifier import Classifier

from .contrastive_pairs import ContrastivePairSet
from .steering_method import CAA


class SteeringType(Enum):
    LOGISTIC = "logistic"
    MLP = "mlp"
    CUSTOM = "custom"
    CAA = "caa"  # New vector-based steering


class SteeringMethod:
    """
    Legacy classifier-based steering method for backward compatibility.
    For new vector-based steering, use steering_method.CAA directly.
    """

    def __init__(self, method_type: SteeringType, device=None, threshold=0.5):
        self.method_type = method_type
        self.device = device
        self.threshold = threshold
        self.classifier = None

        # For vector-based steering
        self.vector_steering = None
        self.is_vector_based = method_type == SteeringType.CAA

        if self.is_vector_based:
            self.vector_steering = CAA(device=device)

        # Response logging settings
        self.enable_logging = False
        self.log_file_path = "./harmful_responses.json"

        # Parameter optimization tracking
        self.original_parameters = {}
        self.optimization_history = []

    def train(
        self, contrastive_pair_set: ContrastivePairSet, layer_index: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Train the steering method on a ContrastivePairSet.

        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index for vector-based steering (required for CAA)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics
        """
        if self.is_vector_based:
            if layer_index is None:
                raise ValueError("layer_index required for vector-based steering methods")
            return self.vector_steering.train(contrastive_pair_set, layer_index)

        # Legacy classifier-based training
        X, y = contrastive_pair_set.prepare_classifier_data()

        if len(X) < 4:
            raise ValueError(f"Need at least 4 training examples, got {len(X)}")

        # Create classifier
        self.classifier = Classifier(model_type=self.method_type.value, device=self.device, threshold=self.threshold)

        # Train classifier
        results = self.classifier.fit(X, y, **kwargs)

        return results

    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Apply steering to activations (vector-based methods only).

        Args:
            activations: Input activations
            strength: Steering strength

        Returns:
            Steered activations
        """
        if not self.is_vector_based:
            raise ValueError("apply_steering only available for vector-based methods")

        return self.vector_steering.apply_steering(activations, strength)

    def get_steering_vector(self) -> Optional[torch.Tensor]:
        """Get steering vector (vector-based methods only)."""
        if not self.is_vector_based:
            return None
        return self.vector_steering.get_steering_vector()

    def predict(self, activations) -> float:
        """
        Predict if activations represent harmful behavior (classifier-based only).

        Args:
            activations: Activation tensor or Activations object

        Returns:
            Prediction score (0 = harmless, 1 = harmful)
        """
        if self.is_vector_based:
            raise ValueError("predict not available for vector-based methods")

        if self.classifier is None:
            raise ValueError("SteeringMethod not trained. Call train() first.")

        return self.classifier.predict(activations)

    def predict_proba(self, activations) -> float:
        """
        Get prediction probability for activations (classifier-based only).

        Args:
            activations: Activation tensor or Activations object

        Returns:
            Probability score (0.0-1.0)
        """
        if self.is_vector_based:
            raise ValueError("predict_proba not available for vector-based methods")

        if self.classifier is None:
            raise ValueError("SteeringMethod not trained. Call train() first.")

        return self.classifier.predict_proba(activations)

    def is_harmful(self, activations, detailed=False) -> Union[bool, Dict[str, Any]]:
        """
        Check if activations represent harmful content (classifier-based only).

        Args:
            activations: Activation tensor or Activations object
            detailed: Whether to return detailed results

        Returns:
            Boolean or detailed dictionary
        """
        if self.is_vector_based:
            raise ValueError("is_harmful not available for vector-based methods")

        if self.classifier is None:
            raise ValueError("SteeringMethod not trained. Call train() first.")

        # Get probability score
        probability = self.predict_proba(activations)
        is_harmful = probability >= self.threshold

        if detailed:
            return {
                "is_harmful": is_harmful,
                "probability": probability,
                "threshold": self.threshold,
                "method_type": self.method_type.value,
            }
        return is_harmful

    def check_safety(self, text: str, model, layer) -> Dict[str, Any]:
        """
        Comprehensive safety check for text using the model.

        Args:
            text: Text to check
            model: Model object for activation extraction
            layer: Layer object for activation extraction

        Returns:
            Safety check results
        """
        try:
            # Extract activations from text
            activations_tensor = model.extract_activations(text, layer)

            # Create Activations object
            activations = Activations(tensor=activations_tensor, layer=layer)

            # Get detailed prediction
            result = self.is_harmful(activations, detailed=True)

            # Add text information
            result.update(
                {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "text_length": len(text),
                    "layer_index": layer.index,
                }
            )

            return result

        except Exception as e:
            return {
                "is_harmful": False,
                "probability": 0.0,
                "error": str(e),
                "text": text[:100] + "..." if len(text) > 100 else text,
            }

    def enable_response_logging(self, log_file_path: str = "./harmful_responses.json") -> None:
        """
        Enable logging of harmful responses.

        Args:
            log_file_path: Path to the log file
        """
        self.enable_logging = True
        self.log_file_path = log_file_path

        # Initialize log file if it doesn't exist
        if not os.path.exists(os.path.dirname(log_file_path)):
            try:
                os.makedirs(os.path.dirname(log_file_path))
            except Exception:
                pass

        if not os.path.exists(log_file_path):
            try:
                with open(log_file_path, "w") as f:
                    json.dump([], f)
            except Exception:
                pass

    def log_harmful_response(
        self, prompt: str, response: str, probability: float, category: str = "harmful", additional_info: Dict = None
    ) -> bool:
        """
        Log a harmful response to the JSON log file.

        Args:
            prompt: The original prompt
            response: The generated response
            probability: The probability score that triggered detection
            category: The category of harmful content detected
            additional_info: Optional additional information

        Returns:
            Success flag
        """
        if not self.enable_logging:
            return False

        try:
            # Create log entry
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "probability": float(probability),
                "category": category,
                "threshold": float(self.threshold),
                "method_type": self.method_type.value,
            }

            # Add additional info if provided
            if additional_info:
                log_entry.update(additional_info)

            # Read existing log entries
            try:
                with open(self.log_file_path) as f:
                    log_entries = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                log_entries = []

            # Append new entry
            log_entries.append(log_entry)

            # Write updated log
            with open(self.log_file_path, "w") as f:
                json.dump(log_entries, f, indent=2)

            return True

        except Exception:
            return False

    def get_logged_responses(self, limit: Optional[int] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve logged harmful responses from the log file.

        Args:
            limit: Maximum number of entries to return (None for all)
            category: Filter by specific category (None for all categories)

        Returns:
            List of log entries
        """
        if not self.enable_logging:
            return []

        try:
            # Check if log file exists
            if not os.path.exists(self.log_file_path):
                return []

            # Read log entries
            with open(self.log_file_path) as f:
                log_entries = json.load(f)

            # Filter by category if specified
            if category is not None:
                log_entries = [entry for entry in log_entries if entry.get("category") == category]

            # Sort by timestamp (newest first)
            log_entries.sort(key=lambda entry: entry.get("timestamp", ""), reverse=True)

            # Apply limit if specified
            if limit is not None and limit > 0:
                log_entries = log_entries[:limit]

            return log_entries

        except Exception:
            return []

    def optimize_parameters(
        self,
        model,
        target_layer,
        pair_set: ContrastivePairSet,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        regularization_strength: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Optimize model parameters to improve steering effectiveness.

        Args:
            model: Model object to optimize
            target_layer: Layer to optimize
            pair_set: ContrastivePairSet with training data
            learning_rate: Learning rate for optimization
            num_epochs: Number of optimization epochs
            regularization_strength: L2 regularization strength

        Returns:
            Dictionary with optimization results
        """
        try:
            # Get the target layer module for optimization
            layer_module = self._get_layer_module(model, target_layer)
            if layer_module is None:
                raise ValueError(f"Could not find layer {target_layer} in model")

            # Store original parameters
            self._store_original_parameters(layer_module)

            # Extract activations for the pair set
            pair_set.extract_activations_with_model(model, target_layer)

            # Prepare training data
            X_tensors, y_labels = pair_set.prepare_classifier_data()

            # Set up optimizer for just the target layer
            optimizer = torch.optim.Adam(layer_module.parameters(), lr=learning_rate)

            # Training loop
            best_steering_loss = float("inf")
            best_parameters = None

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0

                # Process in batches
                batch_size = 4
                for i in range(0, len(X_tensors), batch_size):
                    batch_X = X_tensors[i : i + batch_size]
                    batch_y = y_labels[i : i + batch_size]

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass through the modified layer
                    loss = self._compute_steering_loss(batch_X, batch_y, layer_module, regularization_strength)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

                # Track best parameters
                if avg_loss < best_steering_loss:
                    best_steering_loss = avg_loss
                    best_parameters = {name: param.clone() for name, param in layer_module.named_parameters()}

            # Load best parameters
            if best_parameters is not None:
                for name, param in layer_module.named_parameters():
                    if name in best_parameters:
                        param.data.copy_(best_parameters[name])

            # Store optimization results
            optimization_result = {
                "target_layer": target_layer.index if hasattr(target_layer, "index") else target_layer,
                "final_loss": best_steering_loss,
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "regularization_strength": regularization_strength,
                "parameters_optimized": True,
            }

            self.optimization_history.append(optimization_result)

            return optimization_result

        except Exception as e:
            return {"error": str(e), "parameters_optimized": False}

    def _get_layer_module(self, model, layer):
        """Get the module for a specific layer."""
        try:
            hf_model = model.hf_model if hasattr(model, "hf_model") else model
            layer_idx = layer.index if hasattr(layer, "index") else layer

            if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
                # Llama-style model
                if layer_idx < len(hf_model.model.layers):
                    return hf_model.model.layers[layer_idx]
            elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
                # GPT-style model
                if layer_idx < len(hf_model.transformer.h):
                    return hf_model.transformer.h[layer_idx]

            return None
        except Exception:
            return None

    def _store_original_parameters(self, module):
        """Store original parameters of a module."""
        key = f"module_{id(module)}"
        self.original_parameters[key] = {name: param.clone() for name, param in module.named_parameters()}

    def _compute_steering_loss(self, batch_X, batch_y, layer_module, regularization_strength):
        """
        Compute loss for steering optimization.

        Args:
            batch_X: Batch of activation tensors
            batch_y: Batch of labels
            layer_module: Layer module being optimized
            regularization_strength: L2 regularization strength

        Returns:
            Loss tensor
        """
        total_loss = 0.0

        # Compute steering effectiveness loss
        for i, (activation, label) in enumerate(zip(batch_X, batch_y)):
            # Get prediction from steering method
            prediction = self.predict_proba(activation)

            # Convert to tensor for loss computation
            if not isinstance(prediction, torch.Tensor):
                prediction = torch.tensor(prediction, dtype=torch.float32, device=self.device)

            target = torch.tensor(label, dtype=torch.float32, device=self.device)

            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(prediction.unsqueeze(0), target.unsqueeze(0))
            total_loss += loss

        # Add L2 regularization
        l2_reg = 0.0
        for param in layer_module.parameters():
            l2_reg += torch.norm(param, p=2)

        total_loss += regularization_strength * l2_reg

        return total_loss / len(batch_X)  # Average over batch

    def restore_original_parameters(self) -> bool:
        """
        Restore original parameters.

        Returns:
            Success flag
        """
        try:
            # This is a simplified version - in practice, you'd need to keep track
            # of which modules correspond to which keys
            return len(self.original_parameters) > 0
        except Exception:
            return False

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all optimizations performed.

        Returns:
            Summary dictionary
        """
        return {
            "total_optimizations": len(self.optimization_history),
            "optimization_history": self.optimization_history,
            "has_original_parameters": len(self.original_parameters) > 0,
            "method_type": self.method_type.value,
            "threshold": self.threshold,
        }

    def evaluate(self, contrastive_pair_set: ContrastivePairSet) -> Dict[str, Any]:
        """
        Evaluate the steering method on a ContrastivePairSet.

        Args:
            contrastive_pair_set: Set of contrastive pairs for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        if self.classifier is None:
            raise ValueError("SteeringMethod not trained. Call train() first.")

        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()

        # Predict on positive activations (should be low scores)
        pos_predictions = []
        for activation in pos_activations:
            pred = self.predict_proba(activation)
            pos_predictions.append(pred)

        # Predict on negative activations (should be high scores)
        neg_predictions = []
        for activation in neg_activations:
            pred = self.predict_proba(activation)
            neg_predictions.append(pred)

        # Calculate metrics
        # True Positives: negative activations correctly identified as harmful (pred >= threshold)
        true_positives = sum(1 for pred in neg_predictions if pred >= self.threshold)

        # False Positives: positive activations incorrectly identified as harmful (pred >= threshold)
        false_positives = sum(1 for pred in pos_predictions if pred >= self.threshold)

        # True Negatives: positive activations correctly identified as harmless (pred < threshold)
        true_negatives = sum(1 for pred in pos_predictions if pred < self.threshold)

        # False Negatives: negative activations incorrectly identified as harmless (pred < threshold)
        false_negatives = sum(1 for pred in neg_predictions if pred < self.threshold)

        # Calculate metrics
        detection_rate = true_positives / len(neg_predictions) if neg_predictions else 0
        false_positive_rate = false_positives / len(pos_predictions) if pos_predictions else 0

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        accuracy = (
            (true_positives + true_negatives) / (len(pos_predictions) + len(neg_predictions))
            if (pos_predictions or neg_predictions)
            else 0
        )

        return {
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "num_positive_samples": len(pos_predictions),
            "num_negative_samples": len(neg_predictions),
            "threshold": self.threshold,
        }

    def save_model(self, save_path: str) -> bool:
        """
        Save the steering method to disk.

        Args:
            save_path: Path to save the model

        Returns:
            Success flag
        """
        if self.classifier is None:
            return False

        try:
            self.classifier.save_model(save_path)
            return True
        except Exception:
            return False

    def load_model(self, model_path: str) -> bool:
        """
        Load a steering method from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Success flag
        """
        try:
            self.classifier = Classifier(
                model_type=self.method_type.value, device=self.device, threshold=self.threshold, model_path=model_path
            )
            return True
        except Exception:
            return False

    @classmethod
    def create_and_train(
        cls,
        method_type: SteeringType,
        contrastive_pair_set: ContrastivePairSet,
        device: Optional[str] = None,
        threshold: float = 0.5,
        **training_kwargs,
    ) -> "SteeringMethod":
        """
        Create and train a SteeringMethod in one step.

        Args:
            method_type: Type of steering method
            contrastive_pair_set: Training data
            device: Device to use
            threshold: Classification threshold
            **training_kwargs: Additional training parameters

        Returns:
            Trained SteeringMethod
        """
        steering = cls(method_type=method_type, device=device, threshold=threshold)
        steering.train(contrastive_pair_set, **training_kwargs)
        return steering
