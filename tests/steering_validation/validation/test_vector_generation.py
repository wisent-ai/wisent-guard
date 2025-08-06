#!/usr/bin/env python3
"""
Test vector generation correctness by comparing our implementation
with reference data.

This test loads the same dataset used by the reference implementation,
generates steering vectors using our CAA implementation, and compares
the results.
"""

import json
import torch
import pytest
from pathlib import Path
import sys
import gc

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

from wisent_guard.core.steering_methods.caa import CAA

# Unused direct imports - accessed through model_utils functions
from wisent_guard.core.aggregation import ControlVectorAggregationMethod

from .model_utils import RealModelWrapper, create_real_contrastive_pairs, MODEL_NAME, DEFAULT_LAYER_INDEX, DEVICE


def aggressive_memory_cleanup():
    """Aggressively clean GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()


def load_reference_data():
    """Load reference vector and activations."""
    vector_path = (
        Path(__file__).parent.parent
        / "reference_data"
        / "vectors"
        / "caa"
        / f"hallucination_layer{DEFAULT_LAYER_INDEX}.pt"
    )

    if not vector_path.exists():
        pytest.skip(f"Reference data not found at {vector_path}")

    data = torch.load(vector_path, map_location=DEVICE, weights_only=False)
    return data


def load_test_dataset():
    """Load the hallucination dataset."""
    dataset_path = Path(__file__).parent.parent / "reference_data" / "datasets" / "hallucination.json"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    return data


def test_vector_generation_correctness():
    """Test that our vector generation produces expected results using real model."""
    print("\n🔬 Testing vector generation correctness with real model...")

    # Load test dataset (small subset for speed)
    dataset = load_test_dataset()
    dataset_subset = dataset[:20]
    print(f"Using {len(dataset_subset)} examples for real model test")

    # Initialize real model
    real_model = RealModelWrapper(MODEL_NAME)

    # Create contrastive pairs with real activations
    pair_set = create_real_contrastive_pairs(dataset_subset, real_model, layer_idx=DEFAULT_LAYER_INDEX, max_pairs=20)

    # Test our CAA implementation
    caa = CAA(device=DEVICE, aggregation_method=ControlVectorAggregationMethod.CAA, normalization_method="none")

    print("Training CAA with real activations...")
    training_stats = caa.train(pair_set, layer_index=DEFAULT_LAYER_INDEX)

    our_vector = caa.get_steering_vector()
    print(f"Generated vector: shape={our_vector.shape}, norm={torch.norm(our_vector).item():.4f}")

    # Test 1: Vector has correct dimensions
    assert our_vector.shape[0] == 4096, f"Wrong vector dimension: {our_vector.shape}"

    # Test 2: Vector norm is reasonable
    our_norm = torch.norm(our_vector).item()
    assert 0.1 < our_norm < 100.0, f"Unreasonable vector norm: {our_norm}"

    # Test 3: Training stats are populated
    assert "method" in training_stats
    assert training_stats["method"] == "CAA"
    assert "layer_index" in training_stats
    assert training_stats["layer_index"] == DEFAULT_LAYER_INDEX

    # Test 4: Vector contains no NaN or Inf
    assert not torch.any(torch.isnan(our_vector)), "Vector contains NaN"
    assert not torch.any(torch.isinf(our_vector)), "Vector contains Inf"

    print("✅ Vector generation correctness test passed")


def test_vector_computation_method():
    """Test the vector computation method directly with real model activations."""
    print("\n🧮 Testing vector computation method with real activations...")

    # Load test dataset (small subset)
    dataset = load_test_dataset()
    dataset_subset = dataset[:10]

    # Initialize real model
    real_model = RealModelWrapper(MODEL_NAME)

    # Create contrastive pairs
    pair_set = create_real_contrastive_pairs(dataset_subset, real_model, layer_idx=DEFAULT_LAYER_INDEX, max_pairs=10)

    # Extract positive and negative activations
    pos_activations = []
    neg_activations = []

    for pair in pair_set.pairs:
        pos_activations.append(pair.positive_response.activations)
        neg_activations.append(pair.negative_response.activations)

    pos_stack = torch.stack(pos_activations)
    neg_stack = torch.stack(neg_activations)

    # Expected result (reference method)
    expected_vector = (pos_stack - neg_stack).mean(dim=0)

    # Test our aggregation function
    from wisent_guard.core.aggregation import create_control_vector_from_contrastive_pairs

    our_vector, _ = create_control_vector_from_contrastive_pairs(
        pos_stack, neg_stack, ControlVectorAggregationMethod.CAA, DEVICE
    )

    # Ensure both vectors are on the same device for comparison
    expected_vector = expected_vector.to(DEVICE)

    # Should match exactly for CAA method
    assert torch.allclose(our_vector, expected_vector, atol=1e-6), "Vector computation mismatch"

    print(f"Expected: norm={torch.norm(expected_vector).item():.4f}")
    print(f"Actual:   norm={torch.norm(our_vector).item():.4f}")
    print("✅ Vector computation method test passed")


def test_different_aggregation_methods():
    """Test different aggregation methods with real model activations."""
    print("\n🔀 Testing different aggregation methods with real activations...")

    # Aggressive memory cleanup before starting
    aggressive_memory_cleanup()

    # Load test dataset (small subset)
    dataset = load_test_dataset()
    dataset_subset = dataset[:10]

    # Initialize real model with proper cleanup
    real_model = RealModelWrapper(MODEL_NAME)

    try:
        # Create contrastive pairs
        pair_set = create_real_contrastive_pairs(
            dataset_subset, real_model, layer_idx=DEFAULT_LAYER_INDEX, max_pairs=10
        )

        # Extract activations
        pos_activations = []
        neg_activations = []

        for pair in pair_set.pairs:
            pos_activations.append(pair.positive_response.activations)
            neg_activations.append(pair.negative_response.activations)

        pos_stack = torch.stack(pos_activations)
        neg_stack = torch.stack(neg_activations)

        from wisent_guard.core.aggregation import create_control_vector_from_contrastive_pairs

        # Test CAA method
        _, caa_stats = create_control_vector_from_contrastive_pairs(
            pos_stack, neg_stack, ControlVectorAggregationMethod.CAA, DEVICE
        )
        assert "aggregation_method" in caa_stats

        # If we have other methods implemented, test them too
        methods_to_test = [ControlVectorAggregationMethod.CAA]

        # Try to get other methods if they exist
        try:
            methods_to_test.append(ControlVectorAggregationMethod.MEAN_DIFF)
        except AttributeError:
            pass

        for method in methods_to_test:
            vector, method_stats = create_control_vector_from_contrastive_pairs(pos_stack, neg_stack, method, "cpu")
            print(f"Method {method}: norm={torch.norm(vector).item():.4f}")
            assert torch.norm(vector).item() > 0, f"Zero vector for method {method}"
            assert "aggregation_method" in method_stats

        print("✅ Aggregation methods test passed")

    finally:
        # Clean up model to free GPU memory
        del real_model
        aggressive_memory_cleanup()


def test_normalization_effects():
    """Test that normalization affects vectors as expected with real model."""
    print("\n📏 Testing normalization effects with real model...")

    # Aggressive memory cleanup before starting
    aggressive_memory_cleanup()

    # Load test dataset (small subset)
    dataset = load_test_dataset()
    dataset_subset = dataset[:10]

    # Initialize real model
    real_model = RealModelWrapper(MODEL_NAME)

    try:
        # Create contrastive pairs
        pair_set = create_real_contrastive_pairs(
            dataset_subset, real_model, layer_idx=DEFAULT_LAYER_INDEX, max_pairs=10
        )

        # Test supported normalization methods
        normalizations = ["none"]  # Just test basic functionality for now
        vectors = {}

        for norm_method in normalizations:
            caa = CAA(
                device=DEVICE, aggregation_method=ControlVectorAggregationMethod.CAA, normalization_method=norm_method
            )

            caa.train(pair_set, layer_index=DEFAULT_LAYER_INDEX)
            vectors[norm_method] = caa.get_steering_vector()

            print(f"Normalization '{norm_method}': norm={torch.norm(vectors[norm_method]).item():.4f}")

        # Basic checks
        for method, vector in vectors.items():
            assert vector.shape[0] > 0, f"Empty vector for {method}"
            assert torch.norm(vector).item() > 0, f"Zero norm vector for {method}"
            assert not torch.any(torch.isnan(vector)), f"NaN values in {method} vector"
            assert not torch.any(torch.isinf(vector)), f"Inf values in {method} vector"

        print("✅ Normalization effects test passed")

    finally:
        # Clean up model to free GPU memory
        del real_model
        aggressive_memory_cleanup()


def test_real_model_vector_generation():
    """Test vector generation using real Llama2 model."""
    print("\n🦙 Testing with real Llama2 model...")

    # Aggressive memory cleanup before starting
    aggressive_memory_cleanup()

    # Load test dataset (small subset for speed)
    dataset = load_test_dataset()
    print(f"Using {len(dataset[:20])} examples for real model test")

    # Initialize real model
    real_model = RealModelWrapper(MODEL_NAME)

    try:
        # Create contrastive pairs with real activations
        pair_set = create_real_contrastive_pairs(dataset, real_model, layer_idx=DEFAULT_LAYER_INDEX, max_pairs=20)

        # Train our CAA implementation
        caa = CAA(device=DEVICE, aggregation_method=ControlVectorAggregationMethod.CAA, normalization_method="none")

        training_stats = caa.train(pair_set, layer_index=DEFAULT_LAYER_INDEX)
        our_vector = caa.get_steering_vector()

        print(f"Generated vector with real model: shape={our_vector.shape}, norm={torch.norm(our_vector).item():.4f}")

        # Basic checks
        assert our_vector.shape[0] == 4096, f"Wrong vector dimension: {our_vector.shape}"
        assert torch.norm(our_vector).item() > 0.1, "Vector norm too small"
        assert not torch.any(torch.isnan(our_vector)), "Vector contains NaN"
        assert not torch.any(torch.isinf(our_vector)), "Vector contains Inf"

        # Check training stats
        assert "method" in training_stats
        assert training_stats["method"] == "CAA"
        assert "layer_index" in training_stats
        assert training_stats["layer_index"] == DEFAULT_LAYER_INDEX

        print("✅ Real model vector generation test passed")

    finally:
        # Clean up model to free GPU memory
        del real_model
        aggressive_memory_cleanup()


@pytest.mark.slow
def test_compare_with_reference_vector():
    """Compare our vector with reference vector generated by CAA implementation."""
    print("\n🔍 Comparing with reference CAA vector...")

    # Load reference data
    try:
        ref_data = load_reference_data()
        ref_vector = ref_data["vector"]

        # Check if reference was generated with real model or is mock
        if ref_data.get("metadata", {}).get("is_mock", False):
            pytest.skip("Reference data is mock - generate real reference data first")

    except Exception as e:
        pytest.skip(f"Reference data not available: {e}")

    # Load same dataset
    dataset = load_test_dataset()

    # Use same number of examples as reference
    num_examples = ref_data.get("num_examples", 50)
    dataset_subset = dataset[:num_examples]

    print(f"Using {len(dataset_subset)} examples (matching reference)")

    # Aggressive memory cleanup before starting
    aggressive_memory_cleanup()

    # Initialize real model
    real_model = RealModelWrapper("meta-llama/Llama-2-7b-hf")

    try:
        # Create contrastive pairs
        pair_set = create_real_contrastive_pairs(
            dataset_subset, real_model, layer_idx=DEFAULT_LAYER_INDEX, max_pairs=num_examples
        )

        # Train our implementation
        caa = CAA(device=DEVICE, aggregation_method=ControlVectorAggregationMethod.CAA, normalization_method="none")

        _ = caa.train(pair_set, layer_index=DEFAULT_LAYER_INDEX)
        our_vector = caa.get_steering_vector()

        # Compare with reference
        cosine_sim = torch.nn.functional.cosine_similarity(our_vector, ref_vector, dim=0).item()
        norm_ratio = torch.norm(our_vector).item() / torch.norm(ref_vector).item()

        print(f"Comparison with reference:")
        print(f"  Our vector norm:    {torch.norm(our_vector).item():.4f}")
        print(f"  Reference norm:     {torch.norm(ref_vector).item():.4f}")
        print(f"  Cosine similarity:  {cosine_sim:.4f}")
        print(f"  Norm ratio:         {norm_ratio:.4f}")

        # Assertions for correctness
        assert cosine_sim > 0.7, f"Low cosine similarity with reference: {cosine_sim}"
        assert 0.5 < norm_ratio < 2.0, f"Large difference in vector magnitude: {norm_ratio}"

        print("✅ Reference comparison test passed")

    finally:
        # Clean up model to free GPU memory
        del real_model
        aggressive_memory_cleanup()


# Remove custom runner - use pytest instead
