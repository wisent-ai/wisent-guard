#!/usr/bin/env python3
"""
Test DAC vector generation quality by comparing with reference tensors.

This test validates that our DAC implementation generates steering vectors
with sufficient quality (cosine similarity > 0.9) compared to the original
DAC implementation reference tensors.

It needs Dynamic-Activation-Composition repository cloned to the parent directory.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
import pytest

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

# Import wisent-guard DAC implementation
from wisent_guard.core.steering_methods_tensor.dac_attention import DAC
from wisent_guard.core.contrastive_pairs import ContrastivePairSet, ContrastivePair
from wisent_guard.core.response import PositiveResponse, NegativeResponse

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from const import (
    MODEL_NAME,
    DATASET_A_NAME,
    DATASET_B_NAME,
    REFERENCE_DATA_PATH,
    TORCH_DTYPE,
    ICL_EXAMPLES,
    MAX_NEW_TOKENS,
)

# Import utils
utils_path = Path(__file__).parent.parent / "utils.py"
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import aggressive_memory_cleanup

# Store original directory for path management
ORIGINAL_CWD = os.getcwd()

# Test configuration
MIN_COSINE_SIMILARITY = 0.9
MAX_EXAMPLES = 20


def _load_datasets() -> Tuple[list, list]:
    """Load ITA and ENG datasets."""
    ita_path = REFERENCE_DATA_PATH / "ita_train.json"
    eng_path = REFERENCE_DATA_PATH / "eng_train.json"

    if not ita_path.exists() or not eng_path.exists():
        pytest.skip(
            f"Required datasets not found. Please run generate_data_with_original_dac_implementation.py first. "
            f"Missing: {ita_path if not ita_path.exists() else eng_path}"
        )

    with open(ita_path, "r", encoding="utf-8") as f:
        ita_data = json.load(f)
    with open(eng_path, "r", encoding="utf-8") as f:
        eng_data = json.load(f)

    return ita_data, eng_data


def _create_contrastive_pairs(ita_data: list, eng_data: list, max_examples: int = MAX_EXAMPLES) -> ContrastivePairSet:
    """Create contrastive pairs for DAC training."""
    pairs = []
    max_pairs = min(len(ita_data), len(eng_data), max_examples)

    for i in range(max_pairs):
        ita_item = ita_data[i]
        eng_item = eng_data[i]

        # Ensure inputs match
        assert ita_item["input"] == eng_item["input"], f"Mismatched inputs at index {i}"

        prompt = ita_item["input"]
        ita_response = ita_item["output"]
        eng_response = eng_item["output"]

        pos_response = PositiveResponse(text=ita_response)
        neg_response = NegativeResponse(text=eng_response)

        pair = ContrastivePair(
            prompt=prompt, positive_response=pos_response, negative_response=neg_response, label="language_steering"
        )
        pairs.append(pair)

    return ContrastivePairSet(name="ita_eng_language_steering", pairs=pairs, task_type="language_steering")


@pytest.mark.slow
@pytest.mark.heavy
class TestDACVectorGeneration:
    """Test DAC vector generation quality against reference tensors."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_vector_similarity_vs_reference(self):
        """Test that our DAC implementation generates vectors with >0.9 cosine similarity to reference."""
        config_name = f"icl{ICL_EXAMPLES}_tok{MAX_NEW_TOKENS}"
        ref_tensor_path = REFERENCE_DATA_PATH / f"diff_activations_{config_name}.pt"

        # Check if reference tensor exists
        if not ref_tensor_path.exists():
            pytest.skip(
                f"Reference tensor not found: {ref_tensor_path}\n"
                f"Please run generate_data_with_original_dac_implementation.py first to generate reference data."
            )

        try:
            # Load reference tensor
            ref_tensor = torch.load(ref_tensor_path, map_location="cpu")
            print(f"Loaded reference tensor from: {ref_tensor_path}")
            print(f"Reference tensor shape: {ref_tensor.shape}")
            print(f"Reference tensor norm: {torch.norm(ref_tensor).item():.4f}")

            # Load datasets and create contrastive pairs
            ita_data, eng_data = _load_datasets()
            contrastive_pairs = _create_contrastive_pairs(ita_data, eng_data, max_examples=MAX_EXAMPLES)
            print(f"Created {len(contrastive_pairs.pairs)} contrastive pairs")

            # Initialize DAC with test configuration (matching debug script exactly)
            print(f"Initializing DAC with config: {config_name}")
            dac = DAC(
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                model_name=MODEL_NAME,
                max_examples=MAX_EXAMPLES,
                max_new_tokens=MAX_NEW_TOKENS,
                torch_dtype=TORCH_DTYPE,
                icl_examples=ICL_EXAMPLES,
                original_dac_format=True,  # Use original DAC format for consistency
            )

            # Train DAC
            print(f"Training DAC with {config_name} configuration...")
            start_time = time.time()
            training_stats = dac.train_property("language_steering", contrastive_pairs)
            training_time = time.time() - start_time

            assert training_stats["success"], f"DAC training failed: {training_stats}"
            print(f"DAC training completed in {training_time:.1f} seconds")

            # Get the generated tensor
            our_tensor = dac.get_steering_tensor()
            print(f"Generated tensor shape: {our_tensor.shape}")
            print(f"Generated tensor norm: {torch.norm(our_tensor).item():.4f}")

            # Clean up DAC to free memory
            del dac
            aggressive_memory_cleanup()

            # Compare tensors (matching debug script logic exactly)
            comparison_results = self._compare_tensors(our_tensor, ref_tensor, config_name)

            # Log results
            print(f"\n{'=' * 50}")
            print(f"TENSOR COMPARISON RESULTS: {config_name}")
            print(f"{'=' * 50}")
            print(f"Shapes match: {comparison_results['shapes_match']}")
            print(f"Cosine similarity: {comparison_results['cosine_similarity']:.6f}")
            print(f"MSE: {comparison_results['mse']:.8f}")
            print(f"Norm ratio: {comparison_results['norm_ratio']:.4f}")
            print(f"Our tensor norm: {comparison_results['our_norm']:.4f}")
            print(f"Reference norm: {comparison_results['ref_norm']:.4f}")
            print(f"Tensors match (strict): {comparison_results['tensors_match']}")

            # Assert quality requirements
            assert comparison_results["cosine_similarity"] > MIN_COSINE_SIMILARITY, (
                f"Cosine similarity {comparison_results['cosine_similarity']:.6f} should be > {MIN_COSINE_SIMILARITY}"
            )

        finally:
            aggressive_memory_cleanup()

    def _compare_tensors(self, our_tensor: torch.Tensor, ref_tensor: torch.Tensor, config_name: str) -> Dict[str, Any]:
        """
        Compare our generated tensor with reference tensor.
        Matches debug_dac_tensor_comparison.py logic exactly.
        """
        # Ensure tensors are on CPU for comparison
        our_tensor_cpu = our_tensor.cpu()
        ref_tensor_cpu = ref_tensor.cpu()

        # Basic shape and norm comparison
        shapes_match = our_tensor.shape == ref_tensor.shape
        our_norm = torch.norm(our_tensor_cpu).item()
        ref_norm = torch.norm(ref_tensor_cpu).item()

        if shapes_match:
            # Compute cosine similarity and MSE
            cosine_sim = F.cosine_similarity(our_tensor_cpu.flatten(), ref_tensor_cpu.flatten(), dim=0).item()
            mse = F.mse_loss(our_tensor_cpu, ref_tensor_cpu).item()

            # Compute norm ratio
            norm_ratio = our_norm / ref_norm if ref_norm != 0 else float("inf")

            # Determine if tensors are close
            tensors_match = cosine_sim > 0.99 and mse < 1e-6 and 0.95 < norm_ratio < 1.05

            return {
                "config_name": config_name,
                "shapes_match": shapes_match,
                "cosine_similarity": cosine_sim,
                "mse": mse,
                "our_norm": our_norm,
                "ref_norm": ref_norm,
                "norm_ratio": norm_ratio,
                "tensors_match": tensors_match,
                "our_shape": list(our_tensor.shape),
                "ref_shape": list(ref_tensor.shape),
            }
        else:
            return {
                "config_name": config_name,
                "shapes_match": shapes_match,
                "cosine_similarity": None,
                "mse": None,
                "our_norm": our_norm,
                "ref_norm": ref_norm,
                "norm_ratio": None,
                "tensors_match": False,
                "our_shape": list(our_tensor.shape),
                "ref_shape": list(ref_tensor.shape),
            }
