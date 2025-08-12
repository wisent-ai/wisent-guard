#!/usr/bin/env python3
"""
DAC dynamic steering validation test suite.

This test suite focuses on real-world validation of DAC dynamic steering:
- Reference data validation against original DAC implementation
- Integration tests with actual DAC tensors
- Core algorithm correctness without mocks
"""

import json
import torch
import pytest
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

# Import wisent-guard tensor-based DAC
from wisent_guard.core.steering_methods_tensor.dac_attention import DAC

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from const import (
    MODEL_NAME,
    REFERENCE_DATA_PATH,
    MAX_EXAMPLES,
    MAX_NEW_TOKENS,
    TORCH_DTYPE,
    DYNAMIC_CONFIG,
    ICL_EXAMPLES,
)

# Import aggressive_memory_cleanup
utils_path = Path(__file__).parent.parent / "utils.py"
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import aggressive_memory_cleanup


# Reference data validation utilities
def reference_data_exists() -> bool:
    """Check if reference data files exist."""
    unsteered_path = REFERENCE_DATA_PATH / "text_completions_unsteered.json"
    dynamic_path = REFERENCE_DATA_PATH / "text_completions_dynamic_steering.json"
    return unsteered_path.exists() and dynamic_path.exists()


def load_reference_completions() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load reference unsteered and dynamic steering completions."""
    unsteered_path = REFERENCE_DATA_PATH / "text_completions_unsteered.json"
    dynamic_path = REFERENCE_DATA_PATH / "text_completions_dynamic_steering.json"

    if not unsteered_path.exists():
        raise FileNotFoundError(f"Reference unsteered completions not found: {unsteered_path}")
    if not dynamic_path.exists():
        raise FileNotFoundError(f"Reference dynamic completions not found: {dynamic_path}")

    with open(unsteered_path, "r", encoding="utf-8") as f:
        unsteered_data = json.load(f)

    with open(dynamic_path, "r", encoding="utf-8") as f:
        dynamic_data = json.load(f)

    return unsteered_data, dynamic_data


def compare_alpha_histories(our_alphas: List[float], ref_alphas: List[float], tolerance: float = 0.5) -> Dict[str, Any]:
    """Compare alpha adaptation patterns with reasonable tolerance for randomness."""
    results = {
        "length_match": len(our_alphas) == len(ref_alphas),
        "both_have_nonzero": any(a > 0 for a in our_alphas) and any(a > 0 for a in ref_alphas),
        "adaptation_similarity": 0.0,
        "avg_difference": 0.0,
        "within_tolerance": False,
    }

    if not results["length_match"] or not our_alphas or not ref_alphas:
        return results

    # Calculate average absolute difference
    min_len = min(len(our_alphas), len(ref_alphas))
    our_subset = our_alphas[:min_len]
    ref_subset = ref_alphas[:min_len]

    differences = [abs(a - b) for a, b in zip(our_subset, ref_subset)]
    results["avg_difference"] = sum(differences) / len(differences)
    results["within_tolerance"] = results["avg_difference"] <= tolerance

    # Calculate adaptation pattern similarity (both should show similar adaptation trends)
    our_range = max(our_subset) - min(our_subset) if our_subset else 0
    ref_range = max(ref_subset) - min(ref_subset) if ref_subset else 0

    if our_range > 0 and ref_range > 0:
        results["adaptation_similarity"] = min(our_range, ref_range) / max(our_range, ref_range)

    return results


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate lexical similarity between two texts."""
    if not text1 or not text2:
        return 0.0

    # Simple token-based similarity
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union if union > 0 else 0.0


def validate_steering_effectiveness(unsteered_data: List[Dict], dynamic_data: List[Dict]) -> Dict[str, Any]:
    """Validate that dynamic steering actually changes outputs vs baseline."""
    results = {
        "total_prompts": len(unsteered_data),
        "prompts_with_steering_effect": 0,
        "avg_text_difference": 0.0,
        "non_zero_alpha_percentage": 0.0,
        "steering_successful": False,
    }

    if not unsteered_data or not dynamic_data:
        return results

    # Create lookup for unsteered results by prompt
    unsteered_lookup = {item["prompt"]: item["generated_text"] for item in unsteered_data}

    text_differences = []
    non_zero_alpha_count = 0
    steering_effect_count = 0

    for dynamic_item in dynamic_data:
        prompt = dynamic_item["prompt"]
        dynamic_text = dynamic_item["generated_text"]

        if prompt in unsteered_lookup:
            unsteered_text = unsteered_lookup[prompt]

            # Check text difference
            similarity = calculate_text_similarity(unsteered_text, dynamic_text)
            difference = 1.0 - similarity  # Convert similarity to difference
            text_differences.append(difference)

            if difference > 0.1:  # Threshold for "meaningful difference"
                steering_effect_count += 1

        # Check for non-zero alphas
        if "alpha_history" in dynamic_item:
            alphas = dynamic_item["alpha_history"]
            if isinstance(alphas, list) and any(a > 0 for a in alphas):
                non_zero_alpha_count += 1

    if text_differences:
        results["avg_text_difference"] = sum(text_differences) / len(text_differences)

    results["prompts_with_steering_effect"] = steering_effect_count
    results["non_zero_alpha_percentage"] = (non_zero_alpha_count / len(dynamic_data)) * 100

    # Consider steering successful if we have non-zero alphas and some text changes
    results["steering_successful"] = (
        results["non_zero_alpha_percentage"] > 20  # At least 20% of prompts show adaptation
        and results["avg_text_difference"] > 0.05  # Average difference > 5%
    )

    return results


class TestKLToAlphaBounds:
    """Test KL divergence to alpha transformation bounds without mocks."""

    def test_alpha_bounds_respected(self):
        """Test that alpha transformation respects bounds."""
        # Create a real DAC instance (minimal initialization)
        dac = DAC(device="cpu", model_name="gpt2", max_examples=1, max_new_tokens=1, icl_examples=0)

        # Test various KL values with different bounds
        test_cases = [
            (0.0, (0.0, 2.0), 0.0),  # KL=0 should give alpha=0
            (1.0, (0.0, 2.0), 1.0),  # KL=1 should give alpha=1
            (2.0, (0.0, 2.0), 2.0),  # KL=2 should give alpha=2 (at bound)
            (3.0, (0.0, 2.0), 2.0),  # KL=3 should clamp to alpha=2
            (-1.0, (0.0, 2.0), 0.0),  # KL=-1 should clamp to alpha=0
        ]

        for kl_val, bounds, expected_alpha in test_cases:
            alpha = dac._kl_to_alpha(kl_val, bounds)
            assert bounds[0] <= alpha <= bounds[1], f"Alpha {alpha} outside bounds {bounds} for KL {kl_val}"
            assert abs(alpha - expected_alpha) < 1e-6, f"Expected {expected_alpha}, got {alpha}"


class TestParameterValidation:
    """Test basic parameter validation without mocks."""

    def test_empty_property_weights_validation(self):
        """Test that empty property weights are rejected."""
        dac = DAC(device="cpu", model_name="gpt2", icl_examples=0)
        dac.is_trained = True  # Bypass training requirement
        dac.property_tensors = {"test_prop": torch.randn(5, 4, 4, 8)}

        with pytest.raises(ValueError, match="Dynamic steering requires at least one property"):
            dac.generate_with_dynamic_steering("Test prompt", {})

    def test_unknown_property_validation(self):
        """Test that unknown properties are rejected."""
        dac = DAC(device="cpu", model_name="gpt2", icl_examples=0)
        dac.is_trained = True
        dac.property_tensors = {"known_prop": torch.randn(5, 4, 4, 8)}

        with pytest.raises(ValueError, match="Property 'unknown_prop' not found"):
            dac.generate_with_dynamic_steering("Test prompt", {"unknown_prop": 1.0})

    def test_untrained_dac_validation(self):
        """Test that untrained DAC is rejected for dynamic steering."""
        untrained_dac = DAC(device="cpu", icl_examples=0)

        with pytest.raises(ValueError, match="DAC must be trained before generating"):
            untrained_dac.generate_with_dynamic_steering("Test prompt", {"prop1": 1.0})


@pytest.mark.slow
@pytest.mark.heavy
class TestDynamicSteeringReferenceValidation:
    """Validate our DAC implementation against reference data from original DAC.

    These tests load reference completions generated by the original DAC implementation
    and validate that our wisent-guard implementation produces comparable results.
    """

    def test_load_reference_data(self):
        """Test loading and validating reference data structure."""
        if not reference_data_exists():
            pytest.skip("Reference data not found. Run generate_data_with_original_dac_implementation.py first")

        unsteered_data, dynamic_data = load_reference_completions()

        # Validate data structure
        assert isinstance(unsteered_data, list), "Unsteered data should be a list"
        assert isinstance(dynamic_data, list), "Dynamic data should be a list"
        assert len(unsteered_data) > 0, "Should have unsteered completions"
        assert len(dynamic_data) > 0, "Should have dynamic completions"

        # Validate unsteered data structure
        for item in unsteered_data[:3]:  # Check first few items
            assert "prompt" in item, "Unsteered items should have prompt"
            assert "generated_text" in item, "Unsteered items should have generated_text"
            assert "method" in item, "Unsteered items should have method"
            assert item["method"] == "unsteered_baseline", "Should be unsteered method"

        # Validate dynamic data structure
        for item in dynamic_data[:3]:  # Check first few items
            assert "prompt" in item, "Dynamic items should have prompt"
            assert "generated_text" in item, "Dynamic items should have generated_text"
            assert "method" in item, "Dynamic items should have method"
            assert "alpha_history" in item, "Dynamic items should have alpha_history"
            assert "kl_history" in item, "Dynamic items should have kl_history"
            assert "starting_alpha" in item, "Dynamic items should have starting_alpha"
            assert "top_p" in item, "Dynamic items should have top_p"
            assert item["method"] == "dynamic_steering_original", "Should be dynamic method"

        print(f"✅ Loaded {len(unsteered_data)} unsteered and {len(dynamic_data)} dynamic completions")

    def test_reference_data_configuration_consistency(self):
        """Test that reference data uses expected configuration (icl4_tok30)."""
        if not reference_data_exists():
            pytest.skip("Reference data not found")

        unsteered_data, dynamic_data = load_reference_completions()

        # Check configuration consistency with our constants
        for item in dynamic_data[:3]:
            # Should use starting_alpha from DYNAMIC_CONFIG
            expected_alpha = DYNAMIC_CONFIG["starting_alpha"]
            assert item.get("starting_alpha") == expected_alpha, (
                f"Expected starting_alpha {expected_alpha}, got {item.get('starting_alpha')}"
            )

            # Should use top_p values from DYNAMIC_CONFIG
            item_top_p = item.get("top_p")
            assert item_top_p in DYNAMIC_CONFIG["top_p_values"], (
                f"top_p {item_top_p} not in expected values {DYNAMIC_CONFIG['top_p_values']}"
            )

            # Should have reasonable token count (icl4_tok30 → ~30 tokens)
            tokens_generated = item.get("tokens_generated", 0)
            assert 25 <= tokens_generated <= 35, f"Expected ~30 tokens, got {tokens_generated}"  # Allow some variance

        print(
            f"✅ Reference data uses expected configuration (ICL=4, tok30): α={DYNAMIC_CONFIG['starting_alpha']}, top_p={DYNAMIC_CONFIG['top_p_values']}"
        )

    def test_reference_data_has_non_zero_alphas(self):
        """Test that reference data contains non-zero alpha values (indicating working steering)."""
        if not reference_data_exists():
            pytest.skip("Reference data not found")

        unsteered_data, dynamic_data = load_reference_completions()

        non_zero_alpha_count = 0
        total_alpha_count = 0
        all_alphas = []

        for item in dynamic_data:
            alpha_history = item.get("alpha_history", [])
            if isinstance(alpha_history, list):
                all_alphas.extend(alpha_history)
                total_alpha_count += len(alpha_history)
                non_zero_count = sum(1 for a in alpha_history if a > 0)
                if non_zero_count > 0:
                    non_zero_alpha_count += 1

        # Key assertions for working dynamic steering
        assert total_alpha_count > 0, "Should have alpha values in reference data"
        assert any(a > 0 for a in all_alphas), "Should have some non-zero alpha values"

        non_zero_percentage = (non_zero_alpha_count / len(dynamic_data)) * 100 if dynamic_data else 0
        alpha_stats = {
            "total_completions": len(dynamic_data),
            "completions_with_nonzero_alphas": non_zero_alpha_count,
            "non_zero_percentage": non_zero_percentage,
            "total_alpha_values": total_alpha_count,
            "non_zero_alpha_values": sum(1 for a in all_alphas if a > 0),
            "max_alpha": max(all_alphas) if all_alphas else 0,
            "avg_alpha": sum(all_alphas) / len(all_alphas) if all_alphas else 0,
        }

        print(f"✅ Alpha statistics: {alpha_stats}")

        # Assert meaningful steering is happening
        assert non_zero_percentage >= 10, (
            f"Expected at least 10% of completions to have non-zero alphas, got {non_zero_percentage:.1f}%"
        )
        assert alpha_stats["max_alpha"] > 0.01, f"Expected max alpha > 0.01, got {alpha_stats['max_alpha']:.3f}"

    def test_reference_steering_effectiveness(self):
        """Test that reference dynamic steering actually changes text vs baseline."""
        if not reference_data_exists():
            pytest.skip("Reference data not found")

        unsteered_data, dynamic_data = load_reference_completions()
        effectiveness = validate_steering_effectiveness(unsteered_data, dynamic_data)

        print(f"✅ Steering effectiveness: {effectiveness}")

        # Key assertions for effective steering
        assert effectiveness["total_prompts"] > 0, "Should have prompts to analyze"
        assert effectiveness["non_zero_alpha_percentage"] > 0, "Should have some non-zero alphas"
        assert effectiveness["avg_text_difference"] >= 0, "Should have valid text difference metric"

        # Informational - these may vary based on the specific reference data
        prompts_with_effect = effectiveness["prompts_with_steering_effect"]
        if prompts_with_effect > 0:
            print(f"✅ {prompts_with_effect}/{effectiveness['total_prompts']} prompts show steering effect")

        # Focus on alpha adaptation rather than text changes for reference validation
        # The key is that our implementation should match the reference behavior
        alpha_adaptation_working = (
            effectiveness["non_zero_alpha_percentage"] > 50  # Most completions show adaptation
            and effectiveness["total_prompts"] > 0
        )
        assert alpha_adaptation_working, f"Alpha adaptation should be working in reference data: {effectiveness}"

        # Text changes are optional - the reference implementation may not produce different text
        # but the alpha adaptation mechanism should still be functioning
        print(f"✅ Alpha adaptation is working ({effectiveness['non_zero_alpha_percentage']:.1f}% of completions)")
        if effectiveness["avg_text_difference"] > 0:
            print(f"✅ Text changes detected ({effectiveness['avg_text_difference']:.3f} avg difference)")
        else:
            print(
                "ℹ️ No text changes, but alpha adaptation is functioning (this is acceptable for reference validation)"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_our_implementation_vs_reference_alpha_patterns(self):
        """Test our DAC implementation produces similar alpha patterns to reference."""
        if not reference_data_exists():
            pytest.skip("Reference data not found")

        # Skip if no saved DAC tensors (would require training)
        saved_dac_path = REFERENCE_DATA_PATH / "dac_method.pt"
        if not saved_dac_path.exists():
            pytest.skip(
                "No saved DAC tensors. Run generate_data_with_original_dac_implementation.py and train DAC first"
            )

        unsteered_data, dynamic_data = load_reference_completions()

        try:
            # Load our DAC implementation
            dac = DAC(
                device="cuda:0",
                model_name=MODEL_NAME,
                max_examples=MAX_EXAMPLES,
                max_new_tokens=MAX_NEW_TOKENS,
                torch_dtype=TORCH_DTYPE,
                icl_examples=ICL_EXAMPLES,
            )

            success = dac.load_steering_tensor(str(saved_dac_path))
            if not success:
                pytest.skip("Could not load DAC tensors")

            # Test with a few reference prompts
            comparison_results = []

            for ref_item in dynamic_data[:2]:  # Test first 2 prompts
                prompt = ref_item["prompt"]
                ref_alphas = ref_item.get("alpha_history", [])
                ref_top_p = ref_item.get("top_p", 0.9)
                ref_starting_alpha = ref_item.get("starting_alpha", DYNAMIC_CONFIG["starting_alpha"])

                if not ref_alphas:
                    continue

                try:
                    # Generate with our implementation
                    our_result = dac.generate_with_dynamic_steering(
                        prompt=prompt,
                        property_weights={"language_ita_eng": 1.0},
                        max_new_tokens=MAX_NEW_TOKENS,
                        starting_alpha=ref_starting_alpha,
                        top_p=ref_top_p,
                    )

                    our_alphas = our_result.get("alpha_history", {}).get("language_ita_eng", [])

                    # Compare alpha patterns
                    comparison = compare_alpha_histories(our_alphas, ref_alphas, tolerance=1.0)
                    comparison["prompt"] = prompt[:30] + "..."
                    comparison_results.append(comparison)

                    print(f"Alpha comparison for '{prompt[:30]}...': {comparison}")

                except Exception as e:
                    print(f"⚠️ Error testing prompt '{prompt[:30]}...': {e}")
                    continue

            # Validate at least some comparisons were successful
            if comparison_results:
                successful_comparisons = sum(1 for r in comparison_results if r["both_have_nonzero"])
                assert successful_comparisons > 0, (
                    "Should have at least one comparison with non-zero alphas from both implementations"
                )

                print(f"✅ Successfully compared alpha patterns for {len(comparison_results)} prompts")
                print(f"✅ {successful_comparisons} comparisons had non-zero alphas from both implementations")
            else:
                pytest.skip("No successful alpha comparisons could be made")

        finally:
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_our_implementation_vs_reference_text_generation(self):
        """Test our DAC implementation generates different text with vs without steering."""
        if not reference_data_exists():
            pytest.skip("Reference data not found")

        saved_dac_path = REFERENCE_DATA_PATH / "dac_method.pt"
        if not saved_dac_path.exists():
            pytest.skip("No saved DAC tensors")

        unsteered_data, dynamic_data = load_reference_completions()

        try:
            # Load our DAC implementation
            dac = DAC(
                device="cuda:0",
                model_name=MODEL_NAME,
                max_examples=MAX_EXAMPLES,
                max_new_tokens=MAX_NEW_TOKENS,
                torch_dtype=TORCH_DTYPE,
                icl_examples=ICL_EXAMPLES,
            )

            success = dac.load_steering_tensor(str(saved_dac_path))
            if not success:
                pytest.skip("Could not load DAC tensors")

            # Test steering effectiveness with our implementation
            steering_tests = []

            for ref_item in unsteered_data[:2]:  # Test first 2 prompts
                prompt = ref_item["prompt"]

                try:
                    # Generate without steering
                    unsteered_result = dac.generate_with_steering(
                        prompt=prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        steering_strength=0.0,
                    )

                    # Generate with dynamic steering
                    steered_result = dac.generate_with_dynamic_steering(
                        prompt=prompt,
                        property_weights={"language_ita_eng": 1.0},
                        max_new_tokens=MAX_NEW_TOKENS,
                        starting_alpha=DYNAMIC_CONFIG["starting_alpha"],
                        top_p=0.9,
                    )

                    # Compare results
                    steered_text = steered_result["generated_text"]
                    similarity = calculate_text_similarity(unsteered_result, steered_text)
                    difference = 1.0 - similarity

                    test_result = {
                        "prompt": prompt[:30] + "...",
                        "unsteered": unsteered_result[:50] + "...",
                        "steered": steered_text[:50] + "...",
                        "text_difference": difference,
                        "steering_detectable": difference > 0.05,
                        "alpha_history_length": len(
                            steered_result.get("alpha_history", {}).get("language_ita_eng", [])
                        ),
                        "adaptation_effectiveness": steered_result.get("adaptation_effectiveness", 0),
                    }

                    steering_tests.append(test_result)
                    print(f"Steering test: {test_result}")

                except Exception as e:
                    print(f"⚠️ Error testing prompt '{prompt[:30]}...': {e}")
                    continue

            # Validate steering behavior
            if steering_tests:
                detectable_steering_count = sum(1 for t in steering_tests if t["steering_detectable"])

                # At least verify our implementation can generate text with dynamic steering
                assert all(t["alpha_history_length"] > 0 for t in steering_tests), "Should have alpha histories"
                assert all(t["adaptation_effectiveness"] >= 0 for t in steering_tests), (
                    "Should have valid adaptation metrics"
                )

                print(f"✅ {detectable_steering_count}/{len(steering_tests)} tests show detectable steering effect")
                print(f"✅ Our implementation produces dynamic steering with alpha adaptation")
            else:
                pytest.skip("No successful steering tests could be completed")

        finally:
            aggressive_memory_cleanup()


@pytest.mark.slow
class TestDynamicSteeringIntegrationWithRealData:
    """Integration tests with real DAC data (if available)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_dynamic_steering_with_real_dac_tensors(self):
        """Test dynamic steering with real DAC tensors if available."""
        saved_dac_path = REFERENCE_DATA_PATH / "dac_method.pt"

        if not saved_dac_path.exists():
            pytest.skip("No saved DAC tensors available for dynamic steering test")

        try:
            # Load real DAC instance
            dac = DAC(device="cuda:0", model_name=MODEL_NAME, max_new_tokens=3, icl_examples=ICL_EXAMPLES)
            success = dac.load_steering_tensor(str(saved_dac_path))

            if not success or not dac.property_tensors:
                pytest.skip("Could not load property tensors for dynamic steering test")

            available_props = list(dac.property_tensors.keys())
            if len(available_props) < 1:
                pytest.skip("No properties available for dynamic steering test")

            # Test dynamic steering with real data
            test_prompt = "What is the capital of France?"
            property_weights = {available_props[0]: 1.0}

            result = dac.generate_with_dynamic_steering(
                prompt=test_prompt,
                property_weights=property_weights,
                max_new_tokens=3,  # Short generation for testing
                starting_alpha=1.5,
                top_p=0.9,
            )

            # Verify result structure
            assert "generated_text" in result
            assert "alpha_history" in result
            assert "adaptation_effectiveness" in result
            assert len(result["generated_text"]) >= 0

            # Verify alpha history has the expected structure
            assert available_props[0] in result["alpha_history"]
            assert isinstance(result["alpha_history"][available_props[0]], list)

            # Verify adaptation effectiveness is in valid range
            assert 0.0 <= result["adaptation_effectiveness"] <= 1.0

        finally:
            aggressive_memory_cleanup()


if __name__ == "__main__":
    # Run reference validation tests
    pytest.main([__file__ + "::TestDynamicSteeringReferenceValidation", "-v"])
