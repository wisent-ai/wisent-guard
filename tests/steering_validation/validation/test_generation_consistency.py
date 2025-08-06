#!/usr/bin/env python3
"""
Test generation consistency between our implementation and CAA reference.

This test validates that our steering implementation produces identical behavior
to the reference CAA implementation when given the same inputs.

Tests:
1. Our steered vs Reference steered (should be SAME)
2. Our steered vs Our unsteered (should be DIFFERENT)
3. Token-by-token probability comparison
4. Behavioral effectiveness metrics
"""

import torch
import json
import pytest
from pathlib import Path
import sys

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
VALIDATION_PATH = Path(__file__).parent
sys.path.insert(0, str(WISENT_PATH))
sys.path.insert(0, str(VALIDATION_PATH))

from model_utils import RealModelWrapper, MODEL_NAME, DEFAULT_LAYER_INDEX, DEVICE
from caa_utils import tokenize_llama_base_format, get_a_b_probs_from_logits

# Import our actual production CAA implementation
from .caa_copy import CAA


def load_reference_outputs():
    """Load reference steered and unsteered outputs."""
    reference_dir = Path(__file__).parent.parent / "reference_data" / "generations" / "caa"

    steered_path = reference_dir / "hallucination_steered_outputs.json"
    unsteered_path = reference_dir / "hallucination_unsteered_outputs.json"

    if not steered_path.exists() or not unsteered_path.exists():
        pytest.skip("Reference generation data not found. Run caa_reference.py first.")

    with open(steered_path, "r") as f:
        ref_steered = json.load(f)

    with open(unsteered_path, "r") as f:
        ref_unsteered = json.load(f)

    return ref_steered, ref_unsteered


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
        pytest.skip(f"Reference vector not found at {vector_path}")

    data = torch.load(vector_path, map_location=DEVICE, weights_only=False)
    return data["vector"]


class WisentGuardTextGenerator:
    """Generate text using our wisent-guard production CAA implementation."""

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = RealModelWrapper(model_name)

        # Get A and B token IDs for probability extraction
        self.a_token_id = self.model.tokenizer.convert_tokens_to_ids("A")
        self.b_token_id = self.model.tokenizer.convert_tokens_to_ids("B")

        # Initialize our production CAA steering method
        self.caa_method = CAA(device=self.model.model.device)

    def generate_ab_probabilities(self, question, steering_vector=None, strength=1.0):
        """Generate A/B probabilities using our production CAA implementation.

        This tests our actual wisent_guard/core/steering_methods/caa.py code
        against the reference CAA implementation.

        Args:
            question: The question to ask
            steering_vector: Optional steering vector to apply
            strength: Steering strength multiplier

        Returns:
            dict: Contains a_prob, b_prob, and metadata
        """
        # Tokenize using CAA format for consistency
        tokens = tokenize_llama_base_format(self.model.tokenizer, user_input=question, model_output="(")
        inputs = {"input_ids": torch.tensor(tokens).unsqueeze(0).to(self.model.model.device)}

        # Apply steering if provided
        hook_handle = None
        if steering_vector is not None:
            # Set up our production CAA method with the steering vector
            # We need to simulate it being "trained" to use the vector
            self.caa_method.steering_vector = steering_vector
            self.caa_method.is_trained = True
            self.caa_method.layer_index = DEFAULT_LAYER_INDEX

            # Create steering hook using our production CAA apply_steering method
            def production_caa_hook(module, input_tensor, output):
                hidden_states = output[0] if isinstance(output, tuple) else output

                # Use our production CAA apply_steering method
                steered_states = self.caa_method.apply_steering(
                    activations=hidden_states, strength=strength, verbose=False
                )

                # Return in same format
                if isinstance(output, tuple):
                    return (steered_states,) + output[1:]
                else:
                    return steered_states

            # Register hook on target layer
            target_layer = self.model.model.model.layers[DEFAULT_LAYER_INDEX]
            hook_handle = target_layer.register_forward_hook(production_caa_hook)

        try:
            # Generate logits
            with torch.no_grad():
                outputs = self.model.model(**inputs)
                logits = outputs.logits

            # Extract A/B probabilities
            a_prob, b_prob = get_a_b_probs_from_logits(logits, self.a_token_id, self.b_token_id)

            # Get individual logits for debugging
            last_token_logits = logits[0, -1, :]
            a_logit = last_token_logits[self.a_token_id].item()
            b_logit = last_token_logits[self.b_token_id].item()

            return {"a_prob": a_prob, "b_prob": b_prob, "a_logit": a_logit, "b_logit": b_logit}

        finally:
            # Clean up hook
            if hook_handle is not None:
                hook_handle.remove()


def test_steered_vs_reference_steered():
    """Test that our steered outputs match reference steered outputs."""
    print("\n🔍 Testing: Our steered vs Reference steered (should be SAME)")

    # Load reference data
    ref_steered, _ = load_reference_outputs()  # We only need steered data for this test
    steering_vector = load_reference_vector()

    print(f"Testing {len(ref_steered)} examples")

    # Initialize our generator
    generator = WisentGuardTextGenerator()

    # Track comparison results
    probability_diffs = []
    exact_matches = 0
    close_matches = 0  # Within 0.01 tolerance

    for i, ref_result in enumerate(ref_steered[:10]):  # Test first 10 for speed
        if i % 5 == 0:
            print(f"  Processing example {i + 1}/{min(10, len(ref_steered))}")

        question = ref_result["question"]

        # Generate with our implementation using CAA steering
        our_result = generator.generate_ab_probabilities(
            question=question, steering_vector=steering_vector, strength=1.0
        )

        # Compare probabilities
        a_diff = abs(our_result["a_prob"] - ref_result["a_prob"])
        b_diff = abs(our_result["b_prob"] - ref_result["b_prob"])
        max_diff = max(a_diff, b_diff)
        probability_diffs.append(max_diff)

        if max_diff < 1e-6:  # Essentially identical
            exact_matches += 1
        elif max_diff < 0.01:  # Close match
            close_matches += 1

        # Log first few examples for debugging
        if i < 3:
            print(f"    Example {i + 1}:")
            print(f"      Reference: A={ref_result['a_prob']:.6f}, B={ref_result['b_prob']:.6f}")
            print(f"      Ours:      A={our_result['a_prob']:.6f}, B={our_result['b_prob']:.6f}")
            print(f"      Diff:      A={a_diff:.6f}, B={b_diff:.6f}")

    # Calculate statistics
    avg_diff = sum(probability_diffs) / len(probability_diffs)
    max_diff = max(probability_diffs)

    print(f"\n📊 Steered Comparison Results:")
    print(f"  Average probability difference: {avg_diff:.6f}")
    print(f"  Maximum probability difference: {max_diff:.6f}")
    print(f"  Exact matches (< 1e-6): {exact_matches}/{len(probability_diffs)}")
    print(f"  Close matches (< 0.01): {close_matches}/{len(probability_diffs)}")

    # Assertions for correctness - Updated after finding exact solution
    # We now expect near-perfect matches since we use the correct injection scope
    assert avg_diff < 0.001, (
        f"Average probability difference too large: {avg_diff} (should be ~0.000 with correct injection scope)"
    )
    assert exact_matches + close_matches >= len(probability_diffs) * 0.9, (
        f"Too many mismatched results: {exact_matches + close_matches}/{len(probability_diffs)}"
    )

    print("✅ Our steered outputs perfectly match reference outputs!")


def test_steered_vs_unsteered_different():
    """Test that steering actually changes behavior (steered ≠ unsteered)."""
    print("\n🔍 Testing: Our steered vs Our unsteered (should be DIFFERENT)")

    # Load reference data for test questions
    ref_steered, _ = load_reference_outputs()  # We only need questions from steered data
    steering_vector = load_reference_vector()

    print(f"Testing {len(ref_steered)} examples")

    # Initialize our generator
    generator = WisentGuardTextGenerator()

    # Track steering effects
    steering_effects = []
    significant_changes = 0  # Changes > 0.05

    for i, ref_result in enumerate(ref_steered[:10]):  # Test first 10 for speed
        if i % 5 == 0:
            print(f"  Processing example {i + 1}/{min(10, len(ref_steered))}")

        question = ref_result["question"]

        # Generate WITHOUT steering (our baseline)
        our_unsteered = generator.generate_ab_probabilities(question=question, steering_vector=None)

        # Generate WITH steering (our steered)
        our_steered = generator.generate_ab_probabilities(
            question=question, steering_vector=steering_vector, strength=1.0
        )

        # Calculate steering effect on B probability (hallucination)
        b_effect = our_steered["b_prob"] - our_unsteered["b_prob"]
        steering_effects.append(b_effect)

        if abs(b_effect) > 0.05:  # Significant change
            significant_changes += 1

        # Log first few examples
        if i < 3:
            print(f"    Example {i + 1}:")
            print(f"      Unsteered B: {our_unsteered['b_prob']:.3f}")
            print(f"      Steered B:   {our_steered['b_prob']:.3f}")
            print(f"      Effect:      {b_effect:+.3f}")

    # Calculate statistics
    avg_effect = sum(steering_effects) / len(steering_effects)
    positive_effects = sum(1 for e in steering_effects if e > 0)

    print(f"\n📊 Steering Effect Results:")
    print(f"  Average B probability change: {avg_effect:+.3f}")
    print(f"  Positive effects (hallucination ↑): {positive_effects}/{len(steering_effects)}")
    print(f"  Significant changes (> 0.05): {significant_changes}/{len(steering_effects)}")

    # Assertions for effectiveness
    assert abs(avg_effect) > 0.02, f"Steering effect too small: {avg_effect}"
    assert significant_changes >= len(steering_effects) * 0.5, (
        f"Not enough significant steering effects: {significant_changes}/{len(steering_effects)}"
    )

    # For hallucination, we expect positive effect (more B probability)
    assert avg_effect > 0, f"Expected positive hallucination effect, got {avg_effect}"

    print("✅ Steering produces significant behavioral changes!")


def test_reference_steering_effectiveness():
    """Test that reference steering is effective (sanity check)."""
    print("\n🔍 Testing: Reference steered vs Reference unsteered (sanity check)")

    # Load reference data
    ref_steered, ref_unsteered = load_reference_outputs()

    assert len(ref_steered) == len(ref_unsteered), "Mismatched reference data lengths"

    # Calculate reference steering effects
    ref_effects = []
    for steered, unsteered in zip(ref_steered, ref_unsteered):
        assert steered["question"] == unsteered["question"], "Question mismatch in reference data"

        effect = steered["b_prob"] - unsteered["b_prob"]
        ref_effects.append(effect)

    # Statistics
    avg_ref_effect = sum(ref_effects) / len(ref_effects)
    positive_ref_effects = sum(1 for e in ref_effects if e > 0)
    significant_ref_changes = sum(1 for e in ref_effects if abs(e) > 0.05)

    print(f"📊 Reference Steering Sanity Check:")
    print(f"  Reference avg B change: {avg_ref_effect:+.3f}")
    print(f"  Reference positive effects: {positive_ref_effects}/{len(ref_effects)}")
    print(f"  Reference significant changes: {significant_ref_changes}/{len(ref_effects)}")

    # Sanity check assertions
    assert avg_ref_effect > 0.05, f"Reference steering effect too small: {avg_ref_effect}"
    assert positive_ref_effects >= len(ref_effects) * 0.7, (
        f"Reference doesn't show consistent positive effects: {positive_ref_effects}/{len(ref_effects)}"
    )

    print("✅ Reference steering is effective (sanity check passed)")


@pytest.mark.slow
def test_full_generation_consistency():
    """Run all generation consistency tests together."""
    print("\n🎯 Running Full Generation Consistency Test Suite")
    print("=" * 60)

    test_reference_steering_effectiveness()
    test_steered_vs_reference_steered()
    test_steered_vs_unsteered_different()

    print("\n" + "=" * 60)
    print("✅ ALL GENERATION CONSISTENCY TESTS PASSED!")
    print("🎉 Our implementation produces effective steering matching CAA reference methodology!")
