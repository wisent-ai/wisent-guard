"""
Probe optimization module for detecting correctness from model activations.

This module handles training and optimizing probes that can detect when the model
gives incorrect answers by analyzing activation patterns.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from . import data_utils

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Configuration for probe training and optimization."""

    layers: List[int]
    c_values: List[float] = None  # Regularization strength values to search
    # TODO: Add L1 regularization support for feature selection
    # l1_ratios: List[float] = None  # For elastic net regularization
    # penalty: str = 'l2'  # 'l1', 'l2', 'elasticnet'
    max_iter: int = 1000
    random_state: int = 42

    def __post_init__(self):
        if self.c_values is None:
            self.c_values = [0.1, 1.0, 10.0]


@dataclass
class ProbeTrainingResult:
    """Results from probe training on a specific layer and configuration."""

    layer: int
    c_value: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    probe: LogisticRegression
    training_samples: int


class ProbeOptimizer:
    """
    Optimizes probes for detecting correctness from model activations.

    The probe optimization process:
    1. Train probes on different layers using training data
    2. Evaluate probe performance on validation data
    3. Select best layer and hyperparameters based on validation performance
    4. Test final probe on test data
    """

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def train_probes(
        self,
        model,
        tokenizer,
        train_samples: List[Dict],
        device: str,
        batch_size: int = 32,
        max_length: int = 512,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Dict[str, Dict[str, ProbeTrainingResult]]:
        """
        Train probes on all specified layers and configurations.

        Args:
            model: Language model for activation extraction
            tokenizer: Model tokenizer
            train_samples: Training samples for probe training
            device: Device to run on
            batch_size: Batch size for processing
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping layer -> C value -> ProbeTrainingResult
        """
        self.logger.info(f"Training probes on {len(self.config.layers)} layers with {len(train_samples)} samples")

        results = {}

        for layer in tqdm(self.config.layers, desc="Training probes on layers"):
            self.logger.info(f"Training probes for layer {layer}")
            layer_key = f"layer_{layer}"
            results[layer_key] = {}

            # Extract training data for this layer
            X_train, y_train = data_utils.create_probe_training_data(
                model, tokenizer, train_samples, layer, batch_size, max_length, device, task_name, max_new_tokens
            )

            if len(X_train) < 4:
                self.logger.warning(f"Insufficient training data for layer {layer}: {len(X_train)} samples")
                continue

            # Train probes with different C values
            for c_value in tqdm(self.config.c_values, desc=f"Layer {layer} C values", leave=False):
                self.logger.debug(f"Training probe for layer {layer}, C={c_value}")

                probe = LogisticRegression(
                    C=c_value, max_iter=self.config.max_iter, random_state=self.config.random_state
                )

                try:
                    probe.fit(X_train, y_train)

                    # Evaluate on training data (for monitoring)
                    y_pred = probe.predict(X_train)
                    y_pred_proba = probe.predict_proba(X_train)[:, 1]

                    result = ProbeTrainingResult(
                        layer=layer,
                        c_value=c_value,
                        accuracy=accuracy_score(y_train, y_pred),
                        precision=precision_score(y_train, y_pred, zero_division=0),
                        recall=recall_score(y_train, y_pred, zero_division=0),
                        f1=f1_score(y_train, y_pred, zero_division=0),
                        auc=roc_auc_score(y_train, y_pred_proba) if len(np.unique(y_train)) > 1 else 0.5,
                        probe=probe,
                        training_samples=len(X_train),
                    )

                    results[layer_key][f"C_{c_value}"] = result
                    self.logger.debug(f"Layer {layer}, C={c_value}: Acc={result.accuracy:.3f}, AUC={result.auc:.3f}")

                except Exception as e:
                    self.logger.error(f"Failed to train probe for layer {layer}, C={c_value}: {e}")
                    continue

        self.logger.info(f"Completed probe training. Trained {self._count_successful_probes(results)} probes")
        return results

    def evaluate_probes(
        self,
        probe_results: Dict[str, Dict[str, ProbeTrainingResult]],
        model,
        tokenizer,
        validation_samples: List[Dict],
        device: str,
        batch_size: int = 32,
        max_length: int = 512,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
        """
        Evaluate trained probes on validation data and select best configuration.

        Args:
            probe_results: Results from probe training
            model: Language model
            tokenizer: Model tokenizer
            validation_samples: Validation samples
            device: Device to run on
            batch_size: Batch size
            max_length: Maximum sequence length

        Returns:
            Tuple of (best_probe_config, all_validation_metrics)
        """
        self.logger.info(f"Evaluating probes on {len(validation_samples)} validation samples")

        best_config = None
        best_score = -1
        all_metrics = {}

        for layer_key, layer_probes in tqdm(probe_results.items(), desc="Evaluating probes"):
            layer = int(layer_key.split("_")[1])
            all_metrics[layer_key] = {}

            # Extract validation data for this layer
            X_val, y_val = data_utils.create_probe_training_data(
                model, tokenizer, validation_samples, layer, batch_size, max_length, device, task_name, max_new_tokens
            )

            if len(X_val) < 2:
                self.logger.warning(f"Insufficient validation data for layer {layer}: {len(X_val)} samples")
                continue

            for c_key, probe_result in tqdm(layer_probes.items(), desc=f"Layer {layer} validation", leave=False):
                probe = probe_result.probe

                try:
                    y_pred = probe.predict(X_val)
                    y_pred_proba = probe.predict_proba(X_val)[:, 1]

                    metrics = {
                        "accuracy": accuracy_score(y_val, y_pred),
                        "precision": precision_score(y_val, y_pred, zero_division=0),
                        "recall": recall_score(y_val, y_pred, zero_division=0),
                        "f1": f1_score(y_val, y_pred, zero_division=0),
                        "auc": roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5,
                    }

                    all_metrics[layer_key][c_key] = metrics

                    # Use AUC as primary metric for probe selection
                    score = metrics["auc"]
                    if score > best_score:
                        best_score = score
                        best_config = {"layer": layer, "C": probe_result.c_value, "probe": probe, "metrics": metrics}

                    self.logger.debug(
                        f"Validation - Layer {layer}, C={probe_result.c_value}: "
                        f"Acc={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to evaluate probe for layer {layer}, C={probe_result.c_value}: {e}")
                    continue

        if best_config is None:
            self.logger.warning("No valid probe configuration found")
            # Return a default configuration
            best_config = {
                "layer": self.config.layers[0] if self.config.layers else 0,
                "C": self.config.c_values[0],
                "probe": None,
                "metrics": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.5},
            }
        else:
            self.logger.info(
                f"Best probe configuration: Layer {best_config['layer']}, C={best_config['C']}, "
                f"AUC={best_config['metrics']['auc']:.3f}"
            )

        return best_config, all_metrics

    def _count_successful_probes(self, results: Dict[str, Dict[str, ProbeTrainingResult]]) -> int:
        """Count the number of successfully trained probes."""
        count = 0
        for layer_results in results.values():
            count += len(layer_results)
        return count
