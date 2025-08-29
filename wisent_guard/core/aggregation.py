import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from .normalization import VectorNormalizationMethod, integrate_normalization_with_aggregation


class ControlVectorAggregationMethod(Enum):
    CAA = "caa"  # Contrastive Activation Addition (mean of differences)
    # Additional methods can be added here in the future


def create_control_vector_from_contrastive_pairs(
    pos_activations: List,
    neg_activations: List,
    method: ControlVectorAggregationMethod = ControlVectorAggregationMethod.CAA,
    device: str = None,
    normalization_method: VectorNormalizationMethod = VectorNormalizationMethod.NONE,
    target_norm: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Create a control vector from contrastive activation pairs using various aggregation methods.

    Args:
        pos_activations: List of positive (desirable) activations
        neg_activations: List of negative (undesirable) activations
        method: Aggregation method to use for creating the control vector
        device: Target device for the control vector
        normalization_method: Normalization method to apply after aggregation
        target_norm: Target norm for certain normalization methods

    Returns:
        Tuple of (control_vector, training_stats)
    """
    if len(pos_activations) != len(neg_activations):
        raise ValueError(f"Mismatch in activation pairs: {len(pos_activations)} vs {len(neg_activations)}")

    if len(pos_activations) == 0:
        raise ValueError("No activation pairs found in contrastive pair set")

    # Compute differences for each pair
    differences = []
    for pos_act, neg_act in zip(pos_activations, neg_activations):
        # Convert to tensors if needed
        if hasattr(pos_act, "tensor"):
            pos_tensor = pos_act.tensor
        else:
            pos_tensor = pos_act

        if hasattr(neg_act, "tensor"):
            neg_tensor = neg_act.tensor
        else:
            neg_tensor = neg_act

        # Ensure same shape
        if pos_tensor.shape != neg_tensor.shape:
            raise ValueError(f"Shape mismatch in pair: {pos_tensor.shape} vs {neg_tensor.shape}")

        # Compute difference: positive - negative (steering toward positive)
        diff = pos_tensor - neg_tensor
        differences.append(diff)

    # Apply the specified aggregation method
    if method == ControlVectorAggregationMethod.CAA:
        # CAA: average of differences
        control_vector = torch.stack(differences).mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    # Move to specified device
    if device:
        control_vector = control_vector.to(device)

    # Compute training statistics
    training_stats = {
        "num_pairs": len(differences),
        "vector_norm": torch.norm(control_vector).item(),
        "vector_mean": control_vector.mean().item(),
        "vector_std": control_vector.std().item(),
        "vector_shape": list(control_vector.shape),
        "aggregation_method": method.value,
    }

    # Apply normalization if requested
    if normalization_method != VectorNormalizationMethod.NONE:
        control_vector, training_stats = integrate_normalization_with_aggregation(
            control_vector, training_stats, normalization_method, target_norm
        )

    return control_vector, training_stats


def create_control_vector_from_representations(
    positive_representation: torch.Tensor,
    negative_representation: torch.Tensor,
    method: ControlVectorAggregationMethod = ControlVectorAggregationMethod.CAA,
    device: str = None,
    normalization_method: VectorNormalizationMethod = VectorNormalizationMethod.NONE,
    target_norm: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Create a control vector from pre-aggregated positive and negative representations.

    Args:
        positive_representation: Pre-aggregated positive activation representation
        negative_representation: Pre-aggregated negative activation representation
        method: Aggregation method (currently only CAA supported for pre-aggregated)
        device: Target device for the control vector
        normalization_method: Normalization method to apply after aggregation
        target_norm: Target norm for certain normalization methods

    Returns:
        Tuple of (control_vector, training_stats)
    """
    if positive_representation.shape != negative_representation.shape:
        raise ValueError(f"Shape mismatch: {positive_representation.shape} vs {negative_representation.shape}")

    # For pre-aggregated representations, we just compute the difference
    if method == ControlVectorAggregationMethod.CAA:
        control_vector = positive_representation - negative_representation
    else:
        raise ValueError(f"Method {method} not supported for pre-aggregated representations")

    # Move to specified device
    if device:
        control_vector = control_vector.to(device)

    # Compute training statistics
    training_stats = {
        "num_pairs": 1,  # Single pair of pre-aggregated representations
        "vector_norm": torch.norm(control_vector).item(),
        "vector_mean": control_vector.mean().item(),
        "vector_std": control_vector.std().item(),
        "vector_shape": list(control_vector.shape),
        "aggregation_method": method.value,
        "source": "pre_aggregated_representations",
    }

    # Apply normalization if requested
    if normalization_method != VectorNormalizationMethod.NONE:
        control_vector, training_stats = integrate_normalization_with_aggregation(
            control_vector, training_stats, normalization_method, target_norm
        )

    return control_vector, training_stats


def save_control_vector(
    control_vector: torch.Tensor,
    training_stats: Dict[str, Any],
    save_path: str,
    layer_index: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Save a control vector to disk with metadata.

    Args:
        control_vector: The control vector tensor to save
        training_stats: Training statistics from vector creation
        save_path: Path to save the vector
        layer_index: Optional layer index where vector will be applied
        metadata: Optional additional metadata to save

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        save_data = {
            "control_vector": control_vector,
            "training_stats": training_stats,
            "layer_index": layer_index,
            "metadata": metadata or {},
        }

        torch.save(save_data, save_path)
        return True

    except Exception as e:
        print(f"Error saving control vector: {e}")
        return False


def load_control_vector(
    load_path: str, device: Optional[str] = None
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """
    Load a control vector from disk.

    Args:
        load_path: Path to load the vector from
        device: Target device for the loaded vector

    Returns:
        Tuple of (control_vector, metadata) or (None, None) if failed
    """
    try:
        checkpoint = torch.load(load_path, map_location=device)

        control_vector = checkpoint["control_vector"]

        # Move to specified device if provided
        if device:
            control_vector = control_vector.to(device)

        # Combine all metadata
        metadata = {
            "training_stats": checkpoint.get("training_stats", {}),
            "layer_index": checkpoint.get("layer_index"),
            "metadata": checkpoint.get("metadata", {}),
        }

        return control_vector, metadata

    except Exception as e:
        print(f"Error loading control vector: {e}")
        return None, None


def list_saved_control_vectors(directory: str) -> List[str]:
    """
    List all saved control vector files in a directory.

    Args:
        directory: Directory to search for control vector files

    Returns:
        List of file paths
    """
    if not os.path.exists(directory):
        return []

    vector_files = []
    for file in os.listdir(directory):
        if file.endswith(".pt") or file.endswith(".pth"):
            file_path = os.path.join(directory, file)
            # Try to load and check if it's a valid control vector
            try:
                checkpoint = torch.load(file_path, map_location="cpu")
                if "control_vector" in checkpoint:
                    vector_files.append(file_path)
            except:
                continue

    return vector_files
