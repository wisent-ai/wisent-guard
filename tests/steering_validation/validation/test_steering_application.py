#!/usr/bin/env python3
"""
Test steering application correctness by comparing how our implementation
applies steering vectors to activations.

This test verifies that our steering application logic works as expected
and produces reasonable changes to activations.
"""

import torch
import pytest
from pathlib import Path
import sys

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

from .caa_copy import CAA

from .model_utils import RealModelWrapper, create_real_contrastive_pairs, MODEL_NAME, DEFAULT_LAYER_INDEX, DEVICE


def load_reference_vector():
    """Load reference steering vector."""
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
    return data["vector"]


def create_test_activations(batch_size=2, seq_len=10, hidden_dim=4096):
    """Create realistic test activations."""
    torch.manual_seed(42)

    # Create activations that look like real transformer outputs
    activations = torch.randn(batch_size, seq_len, hidden_dim) * 0.5

    # Add some structure (like residual connections)
    for i in range(seq_len):
        activations[:, i, :] += torch.randn(hidden_dim) * 0.1

    # Normalize to reasonable ranges
    activations = torch.clamp(activations, -10, 10)

    return activations


def test_steering_application_shapes():
    """Test that steering application preserves activation shapes."""
    print("\n📐 Testing steering application shapes...")

    # Load reference vector
    steering_vector = load_reference_vector()
    print(f"Steering vector shape: {steering_vector.shape}")

    # Create test activations
    test_cases = [
        (1, 5, 4096),  # Single batch, short sequence
        (2, 10, 4096),  # Multiple batches, medium sequence
        (1, 50, 4096),  # Single batch, long sequence
    ]

    caa = CAA(device=DEVICE)
    caa.steering_vector = steering_vector
    caa.is_trained = True

    for batch_size, seq_len, hidden_dim in test_cases:
        activations = create_test_activations(batch_size, seq_len, hidden_dim)
        original_shape = activations.shape

        # Apply steering
        steered = caa.apply_steering(activations, strength=1.0)

        # Check shape preservation
        assert steered.shape == original_shape, f"Shape changed: {original_shape} -> {steered.shape}"

        print(f"✓ Shape preserved for {original_shape}")

    print("✅ Steering application shapes test passed")


def test_steering_application_position():
    """Test that steering is applied to the correct position."""
    print("\n🎯 Testing steering application position...")

    # Load reference vector
    steering_vector = load_reference_vector()

    # Create test activations
    batch_size, seq_len, hidden_dim = 1, 10, 4096
    activations = create_test_activations(batch_size, seq_len, hidden_dim)
    original_activations = activations.clone()

    caa = CAA(device=DEVICE)
    caa.steering_vector = steering_vector
    caa.is_trained = True

    # Apply steering with strength 1.0
    steered = caa.apply_steering(activations, strength=1.0)

    # Check that ALL positions were modified (CAA reference behavior)
    # Our implementation now matches CAA's fallback behavior when instruction detection fails
    # (from_pos=-1 steers ALL positions)

    # ALL positions should be different
    for pos in range(seq_len):
        pos_changed = not torch.allclose(steered[0, pos, :], original_activations[0, pos, :], atol=1e-6)
        assert pos_changed, f"Position {pos} was not modified (expected ALL positions to be modified)"

    print(f"✓ ALL {seq_len} positions were modified (CAA reference behavior with from_pos=-1)")

    print("✅ Steering application position test passed")


def test_steering_strength_scaling():
    """Test that steering strength scales the effect proportionally."""
    print("\n⚖️ Testing steering strength scaling...")

    # Load reference vector
    steering_vector = load_reference_vector()

    # Create test activations
    activations = create_test_activations(1, 5, 4096)
    original_activations = activations.clone()

    caa = CAA(device=DEVICE)
    caa.steering_vector = steering_vector
    caa.is_trained = True

    # Test different strengths
    strengths = [0.0, 0.5, 1.0, 2.0]
    effects = {}

    for strength in strengths:
        steered = caa.apply_steering(activations.clone(), strength=strength)

        # Calculate effect magnitude at position -2
        effect = steered[0, -2, :] - original_activations[0, -2, :]
        effect_norm = torch.norm(effect).item()
        effects[strength] = effect_norm

        print(f"Strength {strength}: effect norm = {effect_norm:.4f}")

    # Check proportional scaling
    assert effects[0.0] < 1e-6, "Zero strength should produce no effect"

    # Check that higher strengths produce proportionally larger effects
    for i in range(len(strengths) - 1):
        s1, s2 = strengths[i], strengths[i + 1]
        if s1 > 0 and s2 > 0:  # Skip zero strength
            ratio_expected = s2 / s1
            ratio_actual = effects[s2] / effects[s1] if effects[s1] > 0 else float("inf")

            # Allow some tolerance due to numerical precision
            assert abs(ratio_actual - ratio_expected) < 0.1, (
                f"Scaling not proportional: {ratio_actual} vs {ratio_expected}"
            )

    print("✅ Steering strength scaling test passed")


def test_steering_direction_consistency():
    """Test that steering consistently moves activations in the same direction."""
    print("\n🧭 Testing steering direction consistency...")

    # Load reference vector
    steering_vector = load_reference_vector()

    # Create multiple test cases
    num_tests = 5
    directions = []

    caa = CAA(device=DEVICE)
    caa.steering_vector = steering_vector
    caa.is_trained = True

    for i in range(num_tests):
        # Create different activations each time
        torch.manual_seed(100 + i)
        activations = create_test_activations(1, 5, 4096)
        original_activations = activations.clone()

        # Apply steering
        steered = caa.apply_steering(activations, strength=1.0)

        # Calculate direction of change at position -2
        change = steered[0, -2, :] - original_activations[0, -2, :]
        change_normalized = change / torch.norm(change)
        directions.append(change_normalized)

        print(f"Test {i + 1}: change norm = {torch.norm(change).item():.4f}")

    # Check that directions are consistent (high cosine similarity)
    for i in range(len(directions) - 1):
        cosine_sim = torch.dot(directions[i], directions[i + 1]).item()
        print(f"Cosine similarity {i} vs {i + 1}: {cosine_sim:.4f}")

        # Direction should be consistent (cosine similarity > 0.8)
        assert cosine_sim > 0.8, f"Inconsistent steering direction: {cosine_sim}"

    print("✅ Steering direction consistency test passed")


def test_activation_range_preservation():
    """Test that steering doesn't create unrealistic activation values."""
    print("\n📊 Testing activation range preservation...")

    # Load reference vector
    steering_vector = load_reference_vector()

    # Create test activations in realistic range
    activations = create_test_activations(2, 8, 4096)
    original_max = torch.abs(activations).max().item()
    original_mean = torch.abs(activations).mean().item()

    caa = CAA(device=DEVICE)
    caa.steering_vector = steering_vector
    caa.is_trained = True

    # Test with different strengths
    for strength in [0.5, 1.0, 2.0]:
        steered = caa.apply_steering(activations.clone(), strength=strength)

        steered_max = torch.abs(steered).max().item()
        steered_mean = torch.abs(steered).mean().item()

        print(f"Strength {strength}: max {steered_max:.2f}, mean {steered_mean:.4f}")

        # Check that values don't become unrealistically large
        assert steered_max < original_max * 10, f"Activations too large: {steered_max}"

        # Check for NaN or inf values
        assert not torch.any(torch.isnan(steered)), "NaN values in steered activations"
        assert not torch.any(torch.isinf(steered)), "Inf values in steered activations"

    print("✅ Activation range preservation test passed")


def test_batch_consistency():
    """Test that steering is applied consistently across batches."""
    print("\n📦 Testing batch consistency...")

    # Load reference vector
    steering_vector = load_reference_vector()

    # Create batch of activations (same sequence repeated)
    single_activations = create_test_activations(1, 5, 4096)
    batch_size = 3
    batch_activations = single_activations.repeat(batch_size, 1, 1)

    caa = CAA(device=DEVICE)
    caa.steering_vector = steering_vector
    caa.is_trained = True

    # Apply steering to batch
    steered_batch = caa.apply_steering(batch_activations, strength=1.0)

    # Apply steering to single example
    steered_single = caa.apply_steering(single_activations, strength=1.0)

    # Check that each batch element matches the single result
    for b in range(batch_size):
        batch_element = steered_batch[b : b + 1]  # Keep batch dimension

        # Should be identical
        assert torch.allclose(batch_element, steered_single, atol=1e-6), f"Batch element {b} differs from single result"

    print("✅ Batch consistency test passed")


def test_real_model_steering_application():
    """Test steering application using real Llama2 model activations."""
    print("\n🦙 Testing steering application with real model...")

    # Load test dataset
    import json

    dataset_path = Path(__file__).parent.parent / "reference_data" / "datasets" / "hallucination.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Use small subset for speed
    dataset_subset = dataset[:5]
    print(f"Using {len(dataset_subset)} examples for real model steering test")

    # Initialize real model
    real_model = RealModelWrapper(MODEL_NAME)

    # Create contrastive pairs
    pair_set = create_real_contrastive_pairs(dataset_subset, real_model, layer_idx=DEFAULT_LAYER_INDEX, max_pairs=5)

    # Train CAA with real activations
    caa = CAA(device=DEVICE)
    training_stats = caa.train(pair_set, layer_index=DEFAULT_LAYER_INDEX)

    print(f"Trained CAA with real activations: vector norm = {torch.norm(caa.get_steering_vector()).item():.4f}")

    # Test steering application on real activations
    # Get a fresh activation from the model
    test_text = dataset_subset[0]["question"] + "\n" + dataset_subset[0]["answer_matching_behavior"]
    real_activations = real_model.get_activations([test_text], layer_idx=DEFAULT_LAYER_INDEX, position=-2)

    # Apply steering
    original_activation = real_activations[0].unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
    steered_activation = caa.apply_steering(original_activation, strength=1.0)

    # Check that steering was applied
    activation_change = steered_activation - original_activation
    change_norm = torch.norm(activation_change).item()

    print(f"Activation change norm: {change_norm:.4f}")

    # Basic checks
    assert steered_activation.shape == original_activation.shape, "Shape changed during steering"
    assert change_norm > 0.1, "Steering effect too small"
    assert change_norm < 50.0, "Steering effect too large"
    assert not torch.any(torch.isnan(steered_activation)), "NaN in steered activation"
    assert not torch.any(torch.isinf(steered_activation)), "Inf in steered activation"

    print("✅ Real model steering application test passed")


# Remove custom runner - use pytest instead
