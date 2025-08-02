"""
Steering vector trainer that orchestrates the training process.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..contrastive_pairs import ContrastivePairSet
from ..data_loaders.steering_data_extractor import ContrastivePair
from ..steering_methods.caa import CAA
from .activation_collector import ActivationCollector, ActivationData

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for steering vector training."""

    model_name: str = "distilgpt2"
    target_layers: List[int] = None
    steering_method: str = "CAA"
    batch_size: int = 8
    device: str = "auto"
    max_length: int = 512

    # Steering method specific parameters
    aggregation_method: str = "CAA"
    normalization_method: str = "none"
    target_norm: Optional[float] = None

    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = [5]  # Default to layer 5 for distilgpt2 (6 layers: 0-5)


@dataclass
class TrainingResults:
    """Results from steering vector training."""

    steering_vectors: Dict[int, torch.Tensor]
    training_stats: Dict[str, Any]
    config: TrainingConfig
    model_info: Dict[str, Any]
    training_time: float
    timestamp: str


class SteeringVectorTrainer:
    """Orchestrates steering vector training from contrastive pairs."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.activation_collector = None
        self.steering_method = None

        # Available steering methods
        self.steering_methods = {
            "CAA": CAA,
            # Add other methods as needed
        }

        logger.info(f"Initialized SteeringVectorTrainer with {config.steering_method} method")

    def train(self, contrastive_pairs: List[ContrastivePair]) -> TrainingResults:
        """
        Train steering vectors from contrastive pairs.

        Args:
            contrastive_pairs: List of contrastive pairs for training

        Returns:
            TrainingResults object with trained vectors and metadata
        """
        start_time = datetime.now()
        logger.info(f"Starting steering vector training with {len(contrastive_pairs)} pairs")

        try:
            # Initialize components
            self._initialize_components()

            # Collect activations
            activation_data = self._collect_activations(contrastive_pairs)

            # Train steering vectors
            steering_vectors, training_stats = self._train_steering_vectors(activation_data)

            # Create results
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            results = TrainingResults(
                steering_vectors=steering_vectors,
                training_stats=training_stats,
                config=self.config,
                model_info=self.activation_collector.get_model_info(),
                training_time=training_time,
                timestamp=start_time.isoformat(),
            )

            logger.info(f"Training completed in {training_time:.2f} seconds")
            return results

        finally:
            # Clean up resources
            self._cleanup()

    def _initialize_components(self):
        """Initialize activation collector and steering method."""

        # Initialize activation collector
        self.activation_collector = ActivationCollector(
            model_name=self.config.model_name, device=self.config.device, max_length=self.config.max_length
        )

        # Initialize steering method
        if self.config.steering_method not in self.steering_methods:
            raise ValueError(f"Unknown steering method: {self.config.steering_method}")

        steering_method_class = self.steering_methods[self.config.steering_method]
        self.steering_method = steering_method_class(
            device=self.config.device,
            normalization_method=self.config.normalization_method,
            target_norm=self.config.target_norm,
        )

        logger.info(f"Initialized components: {self.config.model_name} + {self.config.steering_method}")

    def _collect_activations(self, contrastive_pairs: List[ContrastivePair]) -> List[ActivationData]:
        """Collect activations from contrastive pairs."""
        logger.info(f"Collecting activations for layers {self.config.target_layers}")

        activation_data = self.activation_collector.collect_contrastive_activations(
            contrastive_pairs=contrastive_pairs,
            target_layers=self.config.target_layers,
            batch_size=self.config.batch_size,
        )

        logger.info(f"Collected {len(activation_data)} activation datasets")
        return activation_data

    def _train_steering_vectors(
        self, activation_data: List[ActivationData]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
        """Train steering vectors from activation data."""

        steering_vectors = {}
        all_training_stats = {}

        # Group activation data by layer
        layer_data = {}
        for data in activation_data:
            layer_idx = data.layer_idx
            if layer_idx not in layer_data:
                layer_data[layer_idx] = []
            layer_data[layer_idx].append(data)

        # Train steering vector for each layer
        for layer_idx, layer_activations in layer_data.items():
            logger.info(f"Training steering vector for layer {layer_idx}")

            # Convert to ContrastivePairSet format
            contrastive_set = self._create_contrastive_set(layer_activations)

            # Train steering vector
            training_stats = self.steering_method.train(contrastive_set, layer_idx)

            # Get the trained vector
            steering_vector = self.steering_method.steering_vector

            if steering_vector is not None:
                steering_vectors[layer_idx] = steering_vector.detach().cpu()
                all_training_stats[f"layer_{layer_idx}"] = training_stats

                logger.info(
                    f"Trained steering vector for layer {layer_idx}: "
                    f"shape={steering_vector.shape}, "
                    f"norm={torch.norm(steering_vector).item():.4f}"
                )
            else:
                logger.warning(f"No steering vector produced for layer {layer_idx}")

        return steering_vectors, all_training_stats

    def _create_contrastive_set(self, layer_activations: List[ActivationData]) -> ContrastivePairSet:
        """Create ContrastivePairSet from activation data."""

        positive_activations = []
        negative_activations = []

        for data in layer_activations:
            positive_activations.append(data.positive_activations)
            negative_activations.append(data.negative_activations)

        # Stack activations
        positive_stack = torch.stack(positive_activations)  # (num_pairs, hidden_size)
        negative_stack = torch.stack(negative_activations)  # (num_pairs, hidden_size)

        # Create ContrastivePairSet
        # Note: We're using the existing ContrastivePairSet interface
        # The actual implementation might need to be adapted
        contrastive_set = ContrastivePairSet(name="steering_training_set")

        # Add activations to the set
        # This is a simplified interface - the actual implementation
        # would need to handle the activation tensors properly
        contrastive_set.positive_activations = positive_stack
        contrastive_set.negative_activations = negative_stack

        return contrastive_set

    def _cleanup(self):
        """Clean up resources."""
        if self.activation_collector:
            self.activation_collector.cleanup()
            self.activation_collector = None

        self.steering_method = None

        logger.info("SteeringVectorTrainer cleaned up")

    def save_results(self, results: TrainingResults, save_directory: str):
        """
        Save training results to directory.

        Args:
            results: TrainingResults object
            save_directory: Directory to save results
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save steering vectors
        for layer_idx, steering_vector in results.steering_vectors.items():
            # Create filename with model, layer, and date
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"steering_vector_{results.config.model_name}_{layer_idx}_{date_str}.pt"
            filepath = os.path.join(save_directory, filename)

            torch.save(steering_vector, filepath)
            logger.info(f"Saved steering vector for layer {layer_idx} to {filepath}")

            # Save metadata
            metadata = {
                "layer_idx": layer_idx,
                "model_name": results.config.model_name,
                "steering_method": results.config.steering_method,
                "training_time": results.training_time,
                "timestamp": results.timestamp,
                "vector_shape": list(steering_vector.shape),
                "vector_norm": float(torch.norm(steering_vector).item()),
                "training_stats": results.training_stats.get(f"layer_{layer_idx}", {}),
                "model_info": results.model_info,
                "config": {
                    "batch_size": results.config.batch_size,
                    "max_length": results.config.max_length,
                    "aggregation_method": results.config.aggregation_method,
                    "normalization_method": results.config.normalization_method,
                    "target_norm": results.config.target_norm,
                },
            }

            metadata_filename = f"steering_vector_{results.config.model_name}_{layer_idx}_{date_str}.json"
            metadata_filepath = os.path.join(save_directory, metadata_filename)

            import json

            with open(metadata_filepath, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved metadata for layer {layer_idx} to {metadata_filepath}")

        # Save overall training summary
        summary = {
            "training_summary": {
                "total_layers": len(results.steering_vectors),
                "layers_trained": list(results.steering_vectors.keys()),
                "training_time": results.training_time,
                "timestamp": results.timestamp,
                "model_info": results.model_info,
                "config": results.config.__dict__,
            }
        }

        summary_filepath = os.path.join(save_directory, "training_summary.json")
        with open(summary_filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved training summary to {summary_filepath}")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> "SteeringVectorTrainer":
        """Create trainer from configuration dictionary."""
        config = TrainingConfig(**config_dict)
        return cls(config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
