"""
Vector Normalization Module for Steering Methods

This module provides normalization utilities for steering vectors, including:
- Cross-behavior normalization (like reference CAA)
- Individual vector normalization
- Batch normalization operations
- Integration with existing aggregation methods

The cross-behavior normalization is crucial for CAA consistency, ensuring all
behaviors have the same vector magnitude per layer for fair comparison and
consistent steering strength across different behaviors.
"""

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class VectorNormalizationMethod(Enum):
    """Different normalization strategies for steering vectors."""

    NONE = "none"  # No normalization
    L2_UNIT = "l2_unit"  # Normalize to unit length (L2 norm = 1) TODO: Warning - layers do not have same magnitude, therefore steering strength is not consistent across layers!!! Consider remove this normalization in future.
    CROSS_BEHAVIOR = "cross_behavior"  # Normalize across behaviors to same magnitude (CAA style)
    LAYER_WISE_MEAN = "layer_wise_mean"  # Normalize to layer-wise mean magnitude
    MIN_MAX = "min_max"  # Min-max normalization to [0, 1] range
    Z_SCORE = "z_score"  # Z-score normalization (mean=0, std=1)


class VectorNormalizer:
    """
    Main class for vector normalization operations.

    Supports both individual vector normalization and cross-behavior normalization
    following the reference CAA implementation approach.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.normalization_stats = {}

    def normalize_vector(
        self,
        vector: torch.Tensor,
        method: VectorNormalizationMethod = VectorNormalizationMethod.L2_UNIT,
        target_norm: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Normalize a single vector using the specified method.

        Args:
            vector: Input vector to normalize
            method: Normalization method to use
            target_norm: Target norm for certain methods (optional)

        Returns:
            Tuple of (normalized_vector, normalization_stats)
        """
        if method == VectorNormalizationMethod.NONE:
            return vector, {"method": "none", "original_norm": torch.norm(vector).item()}

        original_norm = torch.norm(vector, p=2).item()
        original_device = vector.device

        # Handle edge cases
        if original_norm < 1e-10:
            return vector, {
                "method": method.value,
                "original_norm": original_norm,
                "warning": "Vector norm too small for normalization",
            }

        try:
            if method == VectorNormalizationMethod.L2_UNIT:
                normalized = vector / torch.norm(vector, p=2)
                target_norm_value = 1.0

            elif method == VectorNormalizationMethod.LAYER_WISE_MEAN:
                if target_norm is None:
                    target_norm = 1.0
                normalized = vector * (target_norm / original_norm)
                target_norm_value = target_norm

            elif method == VectorNormalizationMethod.MIN_MAX:
                vector_min = vector.min()
                vector_max = vector.max()
                if vector_max > vector_min:
                    normalized = (vector - vector_min) / (vector_max - vector_min)
                else:
                    normalized = vector
                target_norm_value = torch.norm(normalized, p=2).item()

            elif method == VectorNormalizationMethod.Z_SCORE:
                vector_mean = vector.mean()
                vector_std = vector.std()
                if vector_std > 1e-10:
                    normalized = (vector - vector_mean) / vector_std
                else:
                    normalized = vector - vector_mean
                target_norm_value = torch.norm(normalized, p=2).item()

            else:
                raise ValueError(f"Unsupported normalization method: {method}")

            # Ensure result is on original device
            normalized = normalized.to(original_device)
            final_norm = torch.norm(normalized, p=2).item()

            stats = {
                "method": method.value,
                "original_norm": original_norm,
                "target_norm": target_norm_value,
                "final_norm": final_norm,
                "scaling_factor": final_norm / original_norm if original_norm > 0 else 1.0,
            }

            return normalized, stats

        except Exception as e:
            print(f"Error normalizing vector with method {method}: {e}")
            return vector, {"method": method.value, "original_norm": original_norm, "error": str(e)}

    def normalize_cross_behavior(
        self,
        behavior_vectors: Dict[str, torch.Tensor],
        method: VectorNormalizationMethod = VectorNormalizationMethod.CROSS_BEHAVIOR,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Normalize vectors across behaviors to have consistent magnitudes.

        This implements the reference CAA normalization approach where all behaviors
        for a given layer are normalized to have the same magnitude.

        Args:
            behavior_vectors: Dictionary mapping behavior names to vectors
            method: Cross-behavior normalization method

        Returns:
            Tuple of (normalized_vectors_dict, normalization_stats)
        """
        if not behavior_vectors:
            return {}, {"error": "No vectors provided"}

        # Calculate norms for all behaviors
        norms = {}
        for behavior, vector in behavior_vectors.items():
            norms[behavior] = torch.norm(vector, p=2).item()

        if method == VectorNormalizationMethod.CROSS_BEHAVIOR:
            # Reference CAA approach: normalize to mean norm
            mean_norm = np.mean(list(norms.values()))
            target_norm = mean_norm
        elif method == VectorNormalizationMethod.L2_UNIT:
            # Normalize all to unit length
            target_norm = 1.0
        else:
            raise ValueError(f"Method {method} not supported for cross-behavior normalization")

        # Normalize each vector to target norm
        normalized_vectors = {}
        scaling_factors = {}

        for behavior, vector in behavior_vectors.items():
            original_norm = norms[behavior]
            if original_norm > 1e-10:
                scaling_factor = target_norm / original_norm
                normalized_vectors[behavior] = vector * scaling_factor
                scaling_factors[behavior] = scaling_factor
            else:
                normalized_vectors[behavior] = vector
                scaling_factors[behavior] = 1.0

        stats = {
            "method": method.value,
            "num_behaviors": len(behavior_vectors),
            "original_norms": norms,
            "target_norm": target_norm,
            "mean_original_norm": np.mean(list(norms.values())),
            "std_original_norm": np.std(list(norms.values())),
            "scaling_factors": scaling_factors,
            "final_norms": {
                behavior: torch.norm(vector, p=2).item() for behavior, vector in normalized_vectors.items()
            },
        }

        return normalized_vectors, stats

    def normalize_layer_wise(
        self,
        layer_behavior_vectors: Dict[int, Dict[str, torch.Tensor]],
        method: VectorNormalizationMethod = VectorNormalizationMethod.CROSS_BEHAVIOR,
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, Any]]]:
        """
        Normalize vectors across behaviors for each layer separately.

        This is the full CAA normalization approach where each layer's vectors
        across all behaviors are normalized independently.

        Args:
            layer_behavior_vectors: Nested dict {layer_idx: {behavior: vector}}
            method: Normalization method to apply per layer

        Returns:
            Tuple of (normalized_layer_vectors, layer_normalization_stats)
        """
        normalized_layers = {}
        layer_stats = {}

        for layer_idx, behavior_vectors in layer_behavior_vectors.items():
            if behavior_vectors:
                normalized_vectors, stats = self.normalize_cross_behavior(behavior_vectors, method)
                normalized_layers[layer_idx] = normalized_vectors
                layer_stats[layer_idx] = stats
            else:
                normalized_layers[layer_idx] = {}
                layer_stats[layer_idx] = {"warning": "No vectors for this layer"}

        return normalized_layers, layer_stats

    def save_normalized_vectors(
        self,
        layer_behavior_vectors: Dict[int, Dict[str, torch.Tensor]],
        base_save_dir: str,
        method: VectorNormalizationMethod = VectorNormalizationMethod.CROSS_BEHAVIOR,
        save_stats: bool = True,
    ) -> bool:
        """
        Normalize and save vectors to disk following CAA directory structure.

        Args:
            layer_behavior_vectors: Nested dict {layer_idx: {behavior: vector}}
            base_save_dir: Base directory for saving normalized vectors
            method: Normalization method to use
            save_stats: Whether to save normalization statistics

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create normalized vectors directory
            normalized_dir = os.path.join(base_save_dir, "normalized_vectors")
            os.makedirs(normalized_dir, exist_ok=True)

            # Normalize vectors
            normalized_layers, layer_stats = self.normalize_layer_wise(layer_behavior_vectors, method)

            # Save normalized vectors
            for layer_idx, behavior_vectors in normalized_layers.items():
                for behavior, vector in behavior_vectors.items():
                    # Create behavior directory
                    behavior_dir = os.path.join(normalized_dir, behavior)
                    os.makedirs(behavior_dir, exist_ok=True)

                    # Save vector
                    vector_path = os.path.join(behavior_dir, f"vec_layer_{layer_idx}.pt")
                    torch.save(vector, vector_path)

            # Save normalization statistics if requested
            if save_stats:
                stats_path = os.path.join(normalized_dir, "normalization_stats.pt")
                torch.save(
                    {
                        "method": method.value,
                        "layer_stats": layer_stats,
                        "total_layers": len(layer_stats),
                        "total_behaviors": len(next(iter(normalized_layers.values()), {})),
                    },
                    stats_path,
                )

            return True

        except Exception as e:
            print(f"Error saving normalized vectors: {e}")
            return False

    def load_normalized_vectors(
        self, base_load_dir: str, layers: Optional[List[int]] = None, behaviors: Optional[List[str]] = None
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Optional[Dict[str, Any]]]:
        """
        Load normalized vectors from disk.

        Args:
            base_load_dir: Base directory containing normalized vectors
            layers: Specific layers to load (None for all)
            behaviors: Specific behaviors to load (None for all)

        Returns:
            Tuple of (layer_behavior_vectors, normalization_stats)
        """
        try:
            normalized_dir = os.path.join(base_load_dir, "normalized_vectors")

            if not os.path.exists(normalized_dir):
                return {}, None

            # Load normalization stats if available
            stats_path = os.path.join(normalized_dir, "normalization_stats.pt")
            stats = None
            if os.path.exists(stats_path):
                stats = torch.load(stats_path, map_location=self.device)

            # Discover available behaviors and layers
            available_behaviors = []
            if behaviors is None:
                for item in os.listdir(normalized_dir):
                    behavior_path = os.path.join(normalized_dir, item)
                    if os.path.isdir(behavior_path):
                        available_behaviors.append(item)
            else:
                available_behaviors = behaviors

            # Load vectors
            layer_behavior_vectors = {}

            for behavior in available_behaviors:
                behavior_dir = os.path.join(normalized_dir, behavior)
                if not os.path.exists(behavior_dir):
                    continue

                # Find vector files
                for file in os.listdir(behavior_dir):
                    if file.startswith("vec_layer_") and file.endswith(".pt"):
                        # Extract layer index
                        layer_str = file.replace("vec_layer_", "").replace(".pt", "")
                        try:
                            layer_idx = int(layer_str)
                            if layers is None or layer_idx in layers:
                                # Load vector
                                vector_path = os.path.join(behavior_dir, file)
                                vector = torch.load(vector_path, map_location=self.device)

                                # Store in nested dict
                                if layer_idx not in layer_behavior_vectors:
                                    layer_behavior_vectors[layer_idx] = {}
                                layer_behavior_vectors[layer_idx][behavior] = vector
                        except ValueError:
                            continue

            return layer_behavior_vectors, stats

        except Exception as e:
            print(f"Error loading normalized vectors: {e}")
            return {}, None


def normalize_control_vector(
    control_vector: torch.Tensor,
    method: VectorNormalizationMethod = VectorNormalizationMethod.L2_UNIT,
    target_norm: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Convenience function to normalize a single control vector.

    Args:
        control_vector: Vector to normalize
        method: Normalization method
        target_norm: Target norm for applicable methods

    Returns:
        Tuple of (normalized_vector, stats)
    """
    normalizer = VectorNormalizer()
    return normalizer.normalize_vector(control_vector, method, target_norm)


def create_caa_normalized_vectors(
    behavior_vectors: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Convenience function to apply CAA-style cross-behavior normalization.

    Args:
        behavior_vectors: Dictionary mapping behavior names to vectors

    Returns:
        Tuple of (normalized_vectors, normalization_stats)
    """
    normalizer = VectorNormalizer()
    return normalizer.normalize_cross_behavior(behavior_vectors, VectorNormalizationMethod.CROSS_BEHAVIOR)


def integrate_normalization_with_aggregation(
    control_vector: torch.Tensor,
    training_stats: Dict[str, Any],
    normalization_method: VectorNormalizationMethod = VectorNormalizationMethod.NONE,
    target_norm: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Integrate normalization with existing aggregation pipeline.

    This function can be called after control vector creation to apply
    normalization and update training statistics.

    Args:
        control_vector: Raw control vector from aggregation
        training_stats: Original training statistics
        normalization_method: Method to apply
        target_norm: Target norm for applicable methods

    Returns:
        Tuple of (normalized_vector, updated_stats)
    """
    if normalization_method == VectorNormalizationMethod.NONE:
        return control_vector, training_stats

    # Apply normalization
    normalized_vector, norm_stats = normalize_control_vector(control_vector, normalization_method, target_norm)

    # Update training statistics
    updated_stats = training_stats.copy()
    updated_stats["normalization"] = norm_stats
    updated_stats["normalized"] = True
    updated_stats["final_vector_norm"] = norm_stats.get("final_norm", torch.norm(normalized_vector).item())

    return normalized_vector, updated_stats


# Utility functions for backward compatibility and convenience


def l2_normalize(vector: torch.Tensor) -> torch.Tensor:
    """Simple L2 normalization to unit length."""
    norm = torch.norm(vector, p=2)
    if norm > 1e-10:
        return vector / norm
    return vector


def get_vector_statistics(vectors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Get comprehensive statistics about a set of vectors."""
    if not vectors:
        return {}

    norms = [torch.norm(v, p=2).item() for v in vectors.values()]

    return {
        "num_vectors": len(vectors),
        "vector_names": list(vectors.keys()),
        "norms": dict(zip(vectors.keys(), norms)),
        "mean_norm": np.mean(norms),
        "std_norm": np.std(norms),
        "min_norm": np.min(norms),
        "max_norm": np.max(norms),
        "norm_ratio": np.max(norms) / np.min(norms) if np.min(norms) > 0 else float("inf"),
    }
