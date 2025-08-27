"""
Activation pre-generation module for efficient Optuna-based classifier optimization.

This module generates activations once and stores them for reuse across all Optuna trials,
significantly improving optimization performance by avoiding redundant activation extraction.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

import torch
import numpy as np

from wisent_guard.core.activations.activation_collection_method import ActivationCollectionLogic

logger = logging.getLogger(__name__)


@dataclass
class ActivationData:
    """Container for pre-generated activation data."""

    activations: torch.Tensor
    labels: torch.Tensor
    layer: int
    aggregation: str
    metadata: dict[str, Any]

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays for sklearn compatibility."""
        X = self.activations.detach().cpu().numpy()
        y = self.labels.detach().cpu().numpy()
        return X, y

    def to_tensors(self, device: str = None, dtype: torch.dtype = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tensors directly for PyTorch classifiers."""
        # Use specified dtype, or preserve original dtype if not specified
        target_dtype = dtype if dtype is not None else self.activations.dtype

        if device:
            X = self.activations.to(device=device, dtype=target_dtype)
            y = self.labels.to(device=device, dtype=target_dtype)
        else:
            X = self.activations.to(dtype=target_dtype)
            y = self.labels.to(dtype=target_dtype)
        return X, y


@dataclass
class GenerationConfig:
    """Configuration for activation generation."""

    layer_search_range: tuple[int, int]
    aggregation_methods: list[str]
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None  # Auto-detect if None
    batch_size: int = 32

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = "./activation_cache"
        if not self.aggregation_methods:
            self.aggregation_methods = ["average", "final", "first", "max", "min"]


class ActivationGenerator:
    """
    Generates and caches activations for efficient classifier optimization.

    Key features:
    - Pre-generates activations once for all layers and aggregation methods
    - Caches results to disk for reuse across optimization runs
    - Memory-efficient batch processing
    - Supports both contrastive pairs and labeled datasets
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_from_contrastive_pairs(
        self, model, contrastive_pairs: list, task_name: str, model_name: str, limit: int
    ) -> dict[str, ActivationData]:
        """
        Generate activations from contrastive pairs.

        Args:
            model: Language model
            contrastive_pairs: List of contrastive pairs
            task_name: Name of the task
            model_name: Name of the model
            limit: Data limit used

        Returns:
            Dict mapping (layer, aggregation) keys to ActivationData
        """
        # Create cache key
        cache_key = self._create_cache_key(model_name, task_name, limit, "contrastive")

        # Try to load from cache
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            self.logger.info(f"Loaded pre-generated activations from cache: {cache_key}")
            return cached_data

        self.logger.info(f"Generating activations for {len(contrastive_pairs)} contrastive pairs")

        # Initialize activation collector
        collector = ActivationCollectionLogic(model=model)
        activation_data = {}

        for layer in range(self.config.layer_search_range[0], self.config.layer_search_range[1] + 1):
            self.logger.info(f"Processing layer {layer}")

            try:
                # Extract activations for this layer
                processed_pairs = collector.collect_activations_batch(
                    pairs=contrastive_pairs, layer_index=layer, device=self.config.device
                )

                # Convert to tensor format
                positive_activations = []
                negative_activations = []

                for pair in processed_pairs:
                    if hasattr(pair, "positive_activations") and pair.positive_activations is not None:
                        positive_activations.append(pair.positive_activations.detach().cpu())
                    if hasattr(pair, "negative_activations") and pair.negative_activations is not None:
                        negative_activations.append(pair.negative_activations.detach().cpu())

                if not positive_activations or not negative_activations:
                    self.logger.warning(f"Insufficient activations for layer {layer}")
                    continue

                # Stack activations
                pos_stack = torch.stack(positive_activations)  # [n_samples, hidden_dim]
                neg_stack = torch.stack(negative_activations)  # [n_samples, hidden_dim]

                # Apply aggregation methods
                for aggregation in self.config.aggregation_methods:
                    try:
                        # Apply aggregation to activations
                        pos_aggregated = self._apply_aggregation(pos_stack, aggregation)
                        neg_aggregated = self._apply_aggregation(neg_stack, aggregation)

                        # Combine positive (label=0) and negative (label=1)
                        X = torch.cat([pos_aggregated, neg_aggregated], dim=0)
                        y = torch.cat([torch.zeros(len(pos_aggregated)), torch.ones(len(neg_aggregated))], dim=0)

                        # Create activation data
                        key = f"layer_{layer}_agg_{aggregation}"
                        activation_data[key] = ActivationData(
                            activations=X,
                            labels=y,
                            layer=layer,
                            aggregation=aggregation,
                            metadata={
                                "task_name": task_name,
                                "model_name": model_name,
                                "n_positive": len(pos_aggregated),
                                "n_negative": len(neg_aggregated),
                                "feature_dim": X.shape[1] if len(X.shape) > 1 else X.shape[0],
                            },
                        )

                        self.logger.debug(f"Layer {layer}, aggregation {aggregation}: {X.shape[0]} samples")

                    except Exception as e:
                        self.logger.warning(f"Failed to apply aggregation {aggregation} for layer {layer}: {e}")
                        continue

            except Exception as e:
                self.logger.warning(f"Failed to process layer {layer}: {e}")
                continue

        # Cache the results
        self._save_to_cache(cache_key, activation_data)

        self.logger.info(f"Generated activations for {len(activation_data)} layer-aggregation combinations")
        return activation_data

    def _apply_aggregation(self, activations: torch.Tensor, method: str) -> torch.Tensor:
        """
        Apply aggregation method to activations.

        Args:
            activations: Tensor of shape [n_samples, ...] or [n_samples, n_tokens, hidden_dim]
            method: Aggregation method

        Returns:
            Aggregated activations of shape [n_samples, hidden_dim]
        """
        if len(activations.shape) == 2:
            # Already aggregated, just flatten if needed
            return activations
        elif len(activations.shape) == 3:
            # [n_samples, n_tokens, hidden_dim] -> [n_samples, hidden_dim]
            if method == "average":
                return torch.mean(activations, dim=1)
            elif method == "final":
                return activations[:, -1, :]
            elif method == "first":
                return activations[:, 0, :]
            elif method == "max":
                return torch.max(activations, dim=1)[0]
            elif method == "min":
                return torch.min(activations, dim=1)[0]
            else:
                # Default to average
                return torch.mean(activations, dim=1)
        else:
            # Flatten to [n_samples, -1]
            return activations.view(activations.shape[0], -1)

    def _create_cache_key(self, model_name: str, task_name: str, limit: int, data_type: str) -> str:
        """Create a unique cache key for the given parameters."""
        key_components = [
            model_name.replace("/", "_"),
            task_name,
            str(limit),
            data_type,
            f"{self.config.layer_search_range[0]}-{self.config.layer_search_range[1]}",
            str(sorted(self.config.aggregation_methods)),
        ]
        key_string = "_".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[dict[str, ActivationData]]:
        """Load activation data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            self.logger.debug(f"Loaded {len(data)} activation datasets from cache")
            return data

        except Exception as e:
            self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: dict[str, ActivationData]) -> None:
        """Save activation data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)

            self.logger.info(f"Saved {len(data)} activation datasets to cache: {cache_file}")

        except Exception as e:
            self.logger.error(f"Failed to save cache file {cache_file}: {e}")

    def clear_cache(self) -> None:
        """Clear all cached activation data."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                self.logger.info(f"Removed cache file: {cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        self.logger.info(f"Cleared {len(cache_files)} cache files")

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about cached data."""
        cache_files = list(self.cache_dir.glob("*.pkl"))

        info = {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            "files": [],
        }

        for cache_file in cache_files:
            try:
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                info["files"].append(
                    {"name": cache_file.name, "size_mb": size_mb, "modified": cache_file.stat().st_mtime}
                )
            except Exception as e:
                self.logger.warning(f"Failed to get info for {cache_file}: {e}")

        return info
