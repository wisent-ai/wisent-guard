"""
Model utilities for DAC steering validation tests.

This module provides utilities for working with DAC steering vectors
using our wisent-guard tensor-based implementation and comparing with reference vectors.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

from const import (
    COSINE_SIMILARITY_THRESHOLD,
    MAX_EXAMPLES,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    REFERENCE_DATA_PATH,
    TORCH_DTYPE,
)

from wisent_guard.core.contrastive_pairs import ContrastivePair, ContrastivePairSet
from wisent_guard.core.response import NegativeResponse, PositiveResponse
from wisent_guard.core.steering_methods_tensor.dac_attention import DAC


def load_dac_datasets_for_testing() -> Tuple[List[Dict], List[Dict]]:
    """Load ITA and ENG datasets for DAC testing."""
    # Load ITA dataset
    ita_path = REFERENCE_DATA_PATH / "ita_train.json"
    with open(ita_path, encoding="utf-8") as f:
        ita_data = json.load(f)

    # Load ENG dataset
    eng_path = REFERENCE_DATA_PATH / "eng_train.json"
    with open(eng_path, encoding="utf-8") as f:
        eng_data = json.load(f)

    return ita_data, eng_data


def create_contrastive_pairs_for_dac(ita_data: List[Dict], eng_data: List[Dict]) -> ContrastivePairSet:
    """
    Create ContrastivePairSet for DAC testing.

    Args:
        ita_data: Italian dataset
        eng_data: English dataset

    Returns:
        ContrastivePairSet with language pairs
    """
    pairs = []
    max_pairs = min(len(ita_data), len(eng_data), MAX_EXAMPLES)

    for i in range(max_pairs):
        ita_item = ita_data[i]
        eng_item = eng_data[i]

        # Ensure both have the same input
        if ita_item["input"] != eng_item["input"]:
            continue

        prompt = ita_item["input"]
        ita_response = ita_item["output"]
        eng_response = eng_item["output"]

        # Create responses (ITA as positive, ENG as negative)
        pos_response = PositiveResponse(text=ita_response)
        neg_response = NegativeResponse(text=eng_response)

        # Create contrastive pair
        pair = ContrastivePair(
            prompt=prompt, positive_response=pos_response, negative_response=neg_response, label="language_steering"
        )

        pairs.append(pair)

    # Create ContrastivePairSet
    pair_set = ContrastivePairSet(name="ita_eng_language_steering", pairs=pairs, task_type="language_steering")

    return pair_set


def generate_dac_tensor(
    model_name: str = MODEL_NAME,
    max_examples: int = MAX_EXAMPLES,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "auto",
) -> Tuple[DAC, Dict[str, Any]]:
    """
    Generate DAC tensor using the tensor-based implementation.

    Args:
        model_name: Model name to use
        max_examples: Maximum number of examples
        max_new_tokens: Maximum new tokens
        device: Device to use

    Returns:
        Tuple of (DAC instance, training statistics)
    """
    # Load datasets
    ita_data, eng_data = load_dac_datasets_for_testing()

    # Create contrastive pairs
    contrastive_pairs = create_contrastive_pairs_for_dac(ita_data, eng_data)

    # Initialize DAC
    dac_method = DAC(
        device=device,
        model_name=model_name,
        max_examples=max_examples,
        max_new_tokens=max_new_tokens,
        torch_dtype=TORCH_DTYPE,
    )

    # Train the DAC method
    training_stats = dac_method.train_property("language_steering", contrastive_pairs)

    return dac_method, training_stats


def validate_dac_tensor(dac_tensor: torch.Tensor, reference_path: Path) -> Dict[str, Any]:
    """
    Validate DAC tensor against reference.

    Args:
        dac_tensor: Our generated DAC tensor
        reference_path: Path to reference tensor

    Returns:
        Dictionary with validation metrics
    """
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference tensor not found: {reference_path}")

    # Load reference tensor
    ref_tensor = torch.load(reference_path, map_location="cpu")

    # Move tensors to CPU for comparison
    our_tensor_cpu = dac_tensor.cpu()
    ref_tensor_cpu = ref_tensor.cpu()

    # Calculate metrics
    cosine_sim = F.cosine_similarity(our_tensor_cpu.flatten(), ref_tensor_cpu.flatten(), dim=0).item()
    mse = F.mse_loss(our_tensor_cpu, ref_tensor_cpu).item()

    # Shape and norm comparisons
    shapes_match = our_tensor_cpu.shape == ref_tensor_cpu.shape
    our_norm = torch.norm(our_tensor_cpu).item()
    ref_norm = torch.norm(ref_tensor_cpu).item()
    norm_ratio = our_norm / ref_norm if ref_norm > 0 else 0

    validation_results = {
        "cosine_similarity": cosine_sim,
        "mse": mse,
        "our_shape": list(our_tensor_cpu.shape),
        "ref_shape": list(ref_tensor_cpu.shape),
        "shapes_match": shapes_match,
        "our_norm": our_norm,
        "ref_norm": ref_norm,
        "norm_ratio": norm_ratio,
        "passes_threshold": cosine_sim > COSINE_SIMILARITY_THRESHOLD,
        "high_quality": cosine_sim > 0.99 and mse < 1e-6 and shapes_match,
    }

    return validation_results


def compute_dac_tensor_similarity(our_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Compute similarity metrics between two DAC tensors.

    Args:
        our_tensor: Our generated tensor
        ref_tensor: Reference tensor

    Returns:
        Dictionary with similarity metrics
    """
    # Move to CPU for comparison
    our_cpu = our_tensor.cpu()
    ref_cpu = ref_tensor.cpu()

    # Overall similarity
    cosine_sim = F.cosine_similarity(our_cpu.flatten(), ref_cpu.flatten(), dim=0).item()
    mse = F.mse_loss(our_cpu, ref_cpu).item()

    results = {"overall": {"cosine_similarity": cosine_sim, "mse": mse, "shapes_match": our_cpu.shape == ref_cpu.shape}}

    # Per-layer analysis if shapes match
    if our_cpu.shape == ref_cpu.shape and len(our_cpu.shape) == 4:  # [steps, layers, heads, d_head]
        layer_similarities = []
        for layer_idx in range(our_cpu.shape[1]):
            layer_our = our_cpu[:, layer_idx, :, :].flatten()
            layer_ref = ref_cpu[:, layer_idx, :, :].flatten()

            layer_cosine = F.cosine_similarity(layer_our, layer_ref, dim=0).item()
            layer_mse = F.mse_loss(layer_our, layer_ref).item()

            layer_similarities.append({"cosine_similarity": layer_cosine, "mse": layer_mse})

            results[f"layer_{layer_idx}"] = {
                "cosine_similarity": layer_cosine,
                "mse": layer_mse,
                "norm_ratio": torch.norm(layer_our).item() / torch.norm(layer_ref).item(),
            }

        # Average layer metrics
        avg_cosine = sum(s["cosine_similarity"] for s in layer_similarities) / len(layer_similarities)
        results["overall"]["average_layer_cosine"] = avg_cosine
        results["overall"]["layers_analyzed"] = len(layer_similarities)

    return results
