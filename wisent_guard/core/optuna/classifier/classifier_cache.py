"""
Classifier model caching system for efficient Optuna optimization.

This module provides intelligent caching of trained classifier models to avoid
retraining identical configurations across optimization runs and trials.
"""

import hashlib
import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from wisent_guard.core.classifier.classifier import Classifier

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadata for cached classifier models."""

    cache_key: str
    model_name: str
    task_name: str
    model_type: str
    layer: int
    aggregation: str
    threshold: float
    hyperparameters: dict[str, Any]
    performance_metrics: dict[str, float]
    training_samples: int
    data_hash: str
    timestamp: float
    file_size_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CacheConfig:
    """Configuration for classifier cache."""

    cache_dir: str = "./classifier_cache"
    max_cache_size_gb: float = 5.0
    max_age_days: float = 30.0
    memory_cache_size: int = 10  # Number of models to keep in memory

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


class ClassifierCache:
    """
    Intelligent caching system for trained classifier models.

    Features:
    - Hash-based cache keys for deterministic caching
    - Persistent disk storage with metadata
    - In-memory hot cache for frequently used models
    - Automatic cleanup based on size and age limits
    - Performance metrics tracking
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.memory_cache: dict[str, Classifier] = {}
        self.access_times: dict[str, float] = {}

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Load existing metadata
        self.metadata = self._load_metadata()

        # Cleanup old/large cache if needed
        self._cleanup_cache()

    def get_cache_key(
        self,
        model_name: str,
        task_name: str,
        model_type: str,
        layer: int,
        aggregation: str,
        threshold: float,
        hyperparameters: dict[str, Any],
        data_hash: str,
    ) -> str:
        """
        Generate deterministic cache key for classifier configuration.

        Args:
            model_name: Name of the base model
            task_name: Task being optimized
            model_type: Type of classifier ("logistic", "mlp")
            layer: Layer index used
            aggregation: Token aggregation method
            threshold: Classification threshold
            hyperparameters: Model-specific hyperparameters
            data_hash: Hash of the training data

        Returns:
            Unique cache key string
        """
        # Normalize model name
        clean_model_name = model_name.replace("/", "_").replace(":", "_")

        # Sort hyperparameters for consistent hashing
        sorted_hyperparams = json.dumps(hyperparameters, sort_keys=True)

        # Create cache key components
        key_components = [
            clean_model_name,
            task_name,
            model_type,
            str(layer),
            aggregation,
            f"{threshold:.3f}",
            sorted_hyperparams,
            data_hash,
        ]

        # Generate hash
        key_string = "_".join(key_components)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:16]  # First 16 chars

        return cache_key

    def has_cached_model(self, cache_key: str) -> bool:
        """Check if a model with the given cache key exists."""
        return cache_key in self.metadata or cache_key in self.memory_cache

    def save_classifier(
        self,
        cache_key: str,
        classifier: Classifier,
        model_name: str,
        task_name: str,
        layer: int,
        aggregation: str,
        threshold: float,
        hyperparameters: dict[str, Any],
        performance_metrics: dict[str, float],
        training_samples: int,
        data_hash: str,
    ) -> None:
        """
        Save a trained classifier to cache.

        Args:
            cache_key: Unique cache key
            classifier: Trained classifier model
            model_name: Name of base model
            task_name: Task name
            layer: Layer index
            aggregation: Aggregation method
            threshold: Classification threshold
            hyperparameters: Model hyperparameters
            performance_metrics: Training/validation metrics
            training_samples: Number of training samples
            data_hash: Hash of training data
        """
        try:
            # Save model to disk
            model_file = self.cache_dir / f"{cache_key}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(classifier, f)

            # Calculate file size
            file_size_mb = model_file.stat().st_size / (1024 * 1024)

            # Create metadata
            metadata = CacheMetadata(
                cache_key=cache_key,
                model_name=model_name,
                task_name=task_name,
                model_type=classifier.model_type,
                layer=layer,
                aggregation=aggregation,
                threshold=threshold,
                hyperparameters=hyperparameters,
                performance_metrics=performance_metrics,
                training_samples=training_samples,
                data_hash=data_hash,
                timestamp=time.time(),
                file_size_mb=file_size_mb,
            )

            # Update metadata
            self.metadata[cache_key] = metadata
            self._save_metadata()

            # Add to memory cache if space available
            if len(self.memory_cache) < self.config.memory_cache_size:
                self.memory_cache[cache_key] = classifier
                self.access_times[cache_key] = time.time()

            self.logger.info(
                f"Cached classifier {cache_key}: {model_name}/{task_name} "
                f"layer_{layer} {classifier.model_type} ({file_size_mb:.2f}MB)"
            )

        except Exception as e:
            self.logger.error(f"Failed to save classifier {cache_key}: {e}")
            raise

    def load_classifier(self, cache_key: str) -> Optional[Classifier]:
        """
        Load a cached classifier model.

        Args:
            cache_key: Cache key to load

        Returns:
            Loaded classifier or None if not found
        """
        # Try memory cache first
        if cache_key in self.memory_cache:
            self.access_times[cache_key] = time.time()
            self.logger.debug(f"Loaded classifier {cache_key} from memory cache")
            return self.memory_cache[cache_key]

        # Try disk cache
        if cache_key not in self.metadata:
            return None

        model_file = self.cache_dir / f"{cache_key}.pkl"
        if not model_file.exists():
            self.logger.warning(f"Cache file missing for {cache_key}")
            # Remove from metadata
            del self.metadata[cache_key]
            self._save_metadata()
            return None

        try:
            with open(model_file, "rb") as f:
                classifier = pickle.load(f)

            # Add to memory cache (evict oldest if needed)
            if len(self.memory_cache) >= self.config.memory_cache_size:
                # Evict oldest accessed model
                oldest_key = min(self.access_times.keys(), key=self.access_times.get)
                del self.memory_cache[oldest_key]
                del self.access_times[oldest_key]

            self.memory_cache[cache_key] = classifier
            self.access_times[cache_key] = time.time()

            self.logger.debug(f"Loaded classifier {cache_key} from disk cache")
            return classifier

        except Exception as e:
            self.logger.error(f"Failed to load classifier {cache_key}: {e}")
            return None

    def get_cache_info(self) -> dict[str, Any]:
        """Get comprehensive cache information."""
        total_size_mb = sum(metadata.file_size_mb for metadata in self.metadata.values())

        # Group by task and model type
        task_counts = {}
        model_type_counts = {}

        for metadata in self.metadata.values():
            task_counts[metadata.task_name] = task_counts.get(metadata.task_name, 0) + 1
            model_type_counts[metadata.model_type] = model_type_counts.get(metadata.model_type, 0) + 1

        return {
            "total_models": len(self.metadata),
            "total_size_mb": total_size_mb,
            "memory_cache_size": len(self.memory_cache),
            "cache_dir": str(self.cache_dir),
            "task_distribution": task_counts,
            "model_type_distribution": model_type_counts,
            "oldest_cache_age_hours": (
                time.time() - min((m.timestamp for m in self.metadata.values()), default=time.time())
            )
            / 3600,
            "config": asdict(self.config),
        }

    def find_similar_models(
        self,
        model_name: str,
        task_name: str,
        model_type: Optional[str] = None,
        layer: Optional[int] = None,
        top_k: int = 5,
    ) -> list[tuple[str, CacheMetadata, float]]:
        """
        Find similar cached models based on configuration.

        Args:
            model_name: Base model name
            task_name: Task name
            model_type: Optional model type filter
            layer: Optional layer filter
            top_k: Maximum number of results

        Returns:
            List of (cache_key, metadata, similarity_score) tuples
        """
        candidates = []

        for cache_key, metadata in self.metadata.items():
            # Calculate similarity score
            score = 0.0

            # Model name match (highest weight)
            if metadata.model_name == model_name:
                score += 0.4

            # Task name match
            if metadata.task_name == task_name:
                score += 0.3

            # Model type match
            if model_type and metadata.model_type == model_type:
                score += 0.2

            # Layer proximity
            if layer is not None:
                layer_diff = abs(metadata.layer - layer)
                layer_score = max(0, 1.0 - layer_diff / 10.0)  # Decay with distance
                score += 0.1 * layer_score

            # Only include models with some similarity
            if score > 0.1:
                candidates.append((cache_key, metadata, score))

        # Sort by similarity score and return top_k
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

    def clear_cache(self, keep_recent_hours: float = 0) -> int:
        """
        Clear cached models.

        Args:
            keep_recent_hours: Keep models newer than this many hours

        Returns:
            Number of models removed
        """
        cutoff_time = time.time() - (keep_recent_hours * 3600)
        removed_count = 0

        keys_to_remove = []
        for cache_key, metadata in self.metadata.items():
            if metadata.timestamp < cutoff_time:
                keys_to_remove.append(cache_key)

        for cache_key in keys_to_remove:
            try:
                # Remove from disk
                model_file = self.cache_dir / f"{cache_key}.pkl"
                if model_file.exists():
                    model_file.unlink()

                # Remove from memory cache
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]

                # Remove from metadata
                del self.metadata[cache_key]
                removed_count += 1

            except Exception as e:
                self.logger.warning(f"Failed to remove cached model {cache_key}: {e}")

        self._save_metadata()
        self.logger.info(f"Cleared {removed_count} cached models")
        return removed_count

    def _load_metadata(self) -> dict[str, CacheMetadata]:
        """Load cache metadata from disk."""
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file) as f:
                data = json.load(f)

            metadata = {}
            for cache_key, metadata_dict in data.items():
                metadata[cache_key] = CacheMetadata.from_dict(metadata_dict)

            self.logger.debug(f"Loaded metadata for {len(metadata)} cached models")
            return metadata

        except Exception as e:
            self.logger.warning(f"Failed to load cache metadata: {e}")
            return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            data = {}
            for cache_key, metadata in self.metadata.items():
                data[cache_key] = metadata.to_dict()

            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")

    def _cleanup_cache(self) -> None:
        """Clean up cache based on size and age limits."""
        current_time = time.time()
        total_size_mb = sum(metadata.file_size_mb for metadata in self.metadata.values())

        # Remove old models
        old_threshold = current_time - (self.config.max_age_days * 24 * 3600)
        old_models = [cache_key for cache_key, metadata in self.metadata.items() if metadata.timestamp < old_threshold]

        if old_models:
            for cache_key in old_models:
                try:
                    model_file = self.cache_dir / f"{cache_key}.pkl"
                    if model_file.exists():
                        model_file.unlink()
                    del self.metadata[cache_key]
                except Exception as e:
                    self.logger.warning(f"Failed to remove old model {cache_key}: {e}")

            self.logger.info(f"Removed {len(old_models)} old cached models")
            total_size_mb = sum(metadata.file_size_mb for metadata in self.metadata.values())

        # Remove largest models if over size limit
        if total_size_mb > self.config.max_cache_size_gb * 1024:
            # Sort by size (largest first)
            models_by_size = sorted(self.metadata.items(), key=lambda x: x[1].file_size_mb, reverse=True)

            removed_count = 0
            for cache_key, metadata in models_by_size:
                if total_size_mb <= self.config.max_cache_size_gb * 1024:
                    break

                try:
                    model_file = self.cache_dir / f"{cache_key}.pkl"
                    if model_file.exists():
                        model_file.unlink()

                    total_size_mb -= metadata.file_size_mb
                    del self.metadata[cache_key]
                    removed_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to remove large model {cache_key}: {e}")

            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} large cached models to free space")

        # Save updated metadata
        self._save_metadata()

    def compute_data_hash(self, X: torch.Tensor, y: torch.Tensor) -> str:
        """
        Compute hash of training data for cache key generation.

        Args:
            X: Training features (torch tensor)
            y: Training labels (torch tensor)

        Returns:
            Hash string representing the data
        """
        # Work directly with tensors - no numpy conversion needed
        # Use shape and sample of data for hashing (efficient for large datasets)
        x_hash = hashlib.md5(str(tuple(X.shape)).encode()).hexdigest()[:8]
        y_hash = hashlib.md5(str(tuple(y.shape)).encode()).hexdigest()[:8]

        # Sample some data points for more unique hash (tensor operations)
        if X.size(0) > 10:
            # Use tensor indexing instead of numpy.linspace
            sample_indices = torch.linspace(0, X.size(0) - 1, 10, dtype=torch.long)
            x_sample = X[sample_indices].flatten()[:100]  # First 100 values
            y_sample = y[sample_indices]
        else:
            x_sample = X.flatten()[:100]
            y_sample = y

        # Convert tensor data to bytes for hashing
        x_sample_bytes = x_sample.detach().cpu().numpy().tobytes()
        y_sample_bytes = y_sample.detach().cpu().numpy().tobytes()

        x_sample_hash = hashlib.md5(x_sample_bytes).hexdigest()[:8]
        y_sample_hash = hashlib.md5(y_sample_bytes).hexdigest()[:8]

        return f"{x_hash}_{y_hash}_{x_sample_hash}_{y_sample_hash}"
