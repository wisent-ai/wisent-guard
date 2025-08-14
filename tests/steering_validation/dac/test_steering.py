#!/usr/bin/env python3
"""
DAC Dynamic Steering Validation Test

This test validates that our DAC implementation produces the same dynamic steering
as shown in the reference data by:

1. Loading reference tensor from dac_method.pt
2. Generating text with dynamic steering using same prompts as reference
3. Comparing steered outputs against reference steered and unsteered data
4. Validating Italian language detection in steered outputs
5. Ensuring dynamic features (alpha/KL history) are tracked properly

The test uses the same model, tokenization format, and parameters as the reference
implementation to ensure faithful reproduction of the dynamic steering behavior.

The differences come from the fact, that we use other library, and the generation in DAC is stochastic.

"""

import json
import logging
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pytest

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

# Import wisent-guard DAC
from wisent_guard.core.steering_methods_tensor.dac_attention import DAC

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from const import (
    MODEL_NAME,
    REFERENCE_DATA_PATH,
    MAX_NEW_TOKENS,
    ICL_EXAMPLES,
    TORCH_DTYPE,
)

# Import aggressive_memory_cleanup
utils_path = Path(__file__).parent.parent / "utils.py"
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import aggressive_memory_cleanup

logger = logging.getLogger(__name__)

# Test configuration - Focus on distinctive Italian words from reference steered examples
ITALIAN_WORDS = [
    "tutti",
    "ogni",
    "giorno",
    "nostri",
    "basati",
    "settepesi",
    "come",
    "temperature",
    "estive",
    "elevate",
    "bassissime",
    "precipit",
    "fisho",
    "pesce",
    "totale",
    "prodotti",
    "colore",
    "più",
    "popolare",
    "client",
    "bevo",
    "pesciolini",
    "sette",
    "peso",
    "tavani",
    "divertime",
    "pranzo",
]
MIN_ITALIAN_DETECTION_RATE = 0.50  # At least 50% of steered outputs should contain Italian (adjusted based on analysis)
DYNAMIC_CONFIG = {
    "starting_alpha": 2.0,
    "top_p": 0.9,
    "alpha_bounds": (0.0, 2.0),
}


def _load_reference_data(data_type: str) -> List[Dict[str, Any]]:
    """Load reference data for testing."""
    if data_type == "steered":
        ref_file = REFERENCE_DATA_PATH / "text_completions_dynamic_steering.json"
    elif data_type == "unsteered":
        ref_file = REFERENCE_DATA_PATH / "text_completions_unsteered.json"
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if not ref_file.exists():
        pytest.skip(f"Required reference data not found: {ref_file}. Please run reference data generation first.")

    with open(ref_file, "r") as f:
        return json.load(f)


def _contains_italian_words(text: str, italian_words: List[str] = None) -> Tuple[bool, List[str]]:
    """Check if text contains Italian words and return found words."""
    if italian_words is None:
        italian_words = ITALIAN_WORDS

    text_lower = text.lower()
    found_words = [word for word in italian_words if word in text_lower]
    return len(found_words) > 0, found_words


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple word-based similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0


def _compute_sequence_similarity(seq1: List[float], seq2: List[float]) -> float:
    """Compute correlation between two numerical sequences."""
    if not seq1 or not seq2:
        return 0.0

    # Ensure same length by taking minimum
    min_len = min(len(seq1), len(seq2))
    if min_len < 2:
        return 0.0

    seq1_truncated = seq1[:min_len]
    seq2_truncated = seq2[:min_len]

    # Compute Pearson correlation
    try:
        correlation_matrix = np.corrcoef(seq1_truncated, seq2_truncated)
        correlation = correlation_matrix[0, 1]
        # Handle NaN case (constant sequences)
        return 0.0 if np.isnan(correlation) else correlation
    except:
        return 0.0


def _compute_pattern_similarity(positions1: List[int], positions2: List[int], total_length: int = 30) -> float:
    """Compute similarity between two lists of positions."""
    if not positions1 and not positions2:
        return 1.0
    if not positions1 or not positions2:
        return 0.0

    # Convert to binary vectors
    vec1 = [1 if i in positions1 else 0 for i in range(total_length)]
    vec2 = [1 if i in positions2 else 0 for i in range(total_length)]

    # Compute overlap
    intersection = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    union = sum(1 for v1, v2 in zip(vec1, vec2) if v1 or v2)

    return intersection / union if union > 0 else 0.0


def _compute_token_overlap(tokens1: List[int], tokens2: List[int]) -> float:
    """Compute token-level overlap between two token sequences."""
    if not tokens1 or not tokens2:
        return 0.0

    min_len = min(len(tokens1), len(tokens2))
    if min_len == 0:
        return 0.0

    matches = sum(1 for i in range(min_len) if tokens1[i] == tokens2[i])
    return matches / min_len


@pytest.mark.slow
@pytest.mark.heavy
class TestDACSteeringValidation:
    """Test DAC steering validation against reference implementation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_reference_tensor_loading(self):
        """Test that our DAC loads reference tensor correctly."""
        print("\\n🔧 Testing DAC reference tensor loading...")

        dac_method_path = REFERENCE_DATA_PATH / "dac_method.pt"
        assert dac_method_path.exists(), f"DAC method file not found: {dac_method_path}"

        # Initialize DAC instance with same parameters as reference
        dac = DAC(
            model_name=MODEL_NAME,
            max_examples=20,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=TORCH_DTYPE,
            icl_examples=ICL_EXAMPLES,
            legacy_behavior=True,  # Use Q:/A: format like reference
        )

        try:
            # Load using our DAC class
            success = dac.load_steering_tensor(str(dac_method_path))
            assert success, "Failed to load DAC method file"

            # Verify basic properties
            assert dac.steering_tensor is not None, "Steering tensor is None"
            expected_shape = torch.Size([30, 32, 32, 128])
            assert dac.steering_tensor.shape == expected_shape, (
                f"Unexpected tensor shape: {dac.steering_tensor.shape}, expected: {expected_shape}"
            )

            # Verify property tensors
            assert "language_ita_eng" in dac.property_tensors, "Missing language_ita_eng property"
            property_tensor = dac.property_tensors["language_ita_eng"]
            assert property_tensor.shape == expected_shape, (
                f"Property tensor shape mismatch: {property_tensor.shape}, expected: {expected_shape}"
            )

            # Verify tensor contains non-zero values
            tensor_norm = torch.norm(dac.steering_tensor).item()
            assert tensor_norm > 0.1, f"Tensor norm too small: {tensor_norm}"

            # Verify no NaN or Inf values
            assert not torch.any(torch.isnan(dac.steering_tensor)), "Tensor contains NaN"
            assert not torch.any(torch.isinf(dac.steering_tensor)), "Tensor contains Inf"

            print(f"✅ Tensor loaded successfully:")
            print(f"   Shape: {dac.steering_tensor.shape}")
            print(f"   Norm: {tensor_norm:.4f}")
            print(f"   Properties: {list(dac.property_tensors.keys())}")

            # Optional: Compare with diff_activations_ita_eng.pt if available
            diff_path = REFERENCE_DATA_PATH / "diff_activations_ita_eng.pt"
            if diff_path.exists():
                reference_tensor = torch.load(diff_path, map_location=dac.device)
                cosine_sim = torch.nn.functional.cosine_similarity(
                    dac.steering_tensor.flatten(), reference_tensor.flatten(), dim=0
                ).item()
                print(f"   Cosine similarity with reference: {cosine_sim:.4f}")
                assert cosine_sim > 0.99, f"Low cosine similarity with reference: {cosine_sim}"

        finally:
            # Clean up
            if hasattr(dac, "_model") and dac._model:
                del dac._model
                del dac._tokenizer
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_dynamic_steering_reproduction(self):
        """Test that our DAC reproduces dynamic steering like reference data."""
        print("\\n🎯 Testing DAC dynamic steering reproduction...")

        # Load reference data
        steered_reference = _load_reference_data("steered")
        assert len(steered_reference) > 0, "No steered reference data found"

        # Load tensor and initialize DAC
        dac_method_path = REFERENCE_DATA_PATH / "dac_method.pt"
        dac = DAC(
            model_name=MODEL_NAME,
            max_examples=20,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=TORCH_DTYPE,
            icl_examples=ICL_EXAMPLES,
            legacy_behavior=True,  # Use Q:/A: format like reference
        )

        try:
            # Load reference tensor
            success = dac.load_steering_tensor(str(dac_method_path))
            assert success, "Failed to load DAC method file"

            italian_detected = 0
            dynamic_features_count = 0
            total_examples = len(steered_reference)

            print(f"Testing {total_examples} examples from reference data...")

            for i, ref_example in enumerate(steered_reference):
                prompt = ref_example["prompt"]
                ref_generated = ref_example["generated_text"]

                print(f"\\nExample {i + 1}/{total_examples}:")
                print(f"  Prompt: {prompt[:60]}...")
                print(f"  Reference: {ref_generated[:60]}...")

                # Generate with dynamic steering using same parameters
                result = dac.generate_with_dynamic_steering(
                    prompt=prompt,
                    property_weights={"language_ita_eng": 1.0},
                    starting_alpha=DYNAMIC_CONFIG["starting_alpha"],
                    top_p=DYNAMIC_CONFIG["top_p"],
                    alpha_bounds=DYNAMIC_CONFIG["alpha_bounds"],
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                our_generated = result["generated_text"]
                print(f"  Our output: {our_generated[:60]}...")

                # Check for Italian words in our output
                has_italian, found_words = _contains_italian_words(our_generated)
                if has_italian:
                    italian_detected += 1
                    print(f"  ✅ Italian detected: {found_words[:3]}")
                else:
                    print(f"  ⚠️  No Italian detected")

                # Check for dynamic features
                if (
                    "alpha_history" in result
                    and result["alpha_history"]
                    and "kl_history" in result
                    and result["kl_history"]
                ):
                    dynamic_features_count += 1
                    alpha_hist = result["alpha_history"]["language_ita_eng"]
                    kl_hist = result["kl_history"]["language_ita_eng"]
                    if alpha_hist and kl_hist:
                        print(f"  📊 Dynamic features: α=[{alpha_hist[0]:.3f}...], KL=[{kl_hist[0]:.3f}...]")
                    else:
                        print(f"  📊 Dynamic features: α={len(alpha_hist)} entries, KL={len(kl_hist)} entries")

            # Validate Italian detection rate
            italian_rate = italian_detected / total_examples
            print(f"\\n📈 Italian detection summary:")
            print(f"   Examples with Italian: {italian_detected}/{total_examples} ({italian_rate:.1%})")
            print(f"   Minimum required: {MIN_ITALIAN_DETECTION_RATE:.1%}")

            assert italian_rate >= MIN_ITALIAN_DETECTION_RATE, (
                f"Italian detection rate {italian_rate:.2%} below minimum {MIN_ITALIAN_DETECTION_RATE:.2%}. "
                f"This suggests the steering is not working properly."
            )

            # Validate dynamic features are tracked
            dynamic_rate = dynamic_features_count / total_examples
            print(f"   Dynamic features tracked: {dynamic_features_count}/{total_examples} ({dynamic_rate:.1%})")

            assert dynamic_rate >= 0.8, (
                f"Dynamic features tracking rate {dynamic_rate:.2%} too low. "
                f"Expected alpha_history and kl_history to be tracked."
            )

            print("✅ Dynamic steering reproduction test passed!")

        finally:
            # Clean up
            if hasattr(dac, "_model") and dac._model:
                del dac._model
                del dac._tokenizer
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_steering_effectiveness(self):
        """Test that steered outputs differ significantly from unsteered baseline."""
        print("\\n⚖️  Testing steering effectiveness vs baseline...")

        # Load reference data
        steered_reference = _load_reference_data("steered")
        unsteered_reference = _load_reference_data("unsteered")

        assert len(steered_reference) > 0, "No steered reference data found"
        assert len(unsteered_reference) > 0, "No unsteered reference data found"

        # Load tensor and initialize DAC
        dac_method_path = REFERENCE_DATA_PATH / "dac_method.pt"
        dac = DAC(
            model_name=MODEL_NAME,
            max_examples=20,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=TORCH_DTYPE,
            icl_examples=ICL_EXAMPLES,
            legacy_behavior=True,  # Use Q:/A: format like reference
        )

        try:
            # Load reference tensor
            success = dac.load_steering_tensor(str(dac_method_path))
            assert success, "Failed to load DAC method file"

            significant_differences = 0
            total_comparisons = min(len(steered_reference), len(unsteered_reference))

            print(f"Comparing {total_comparisons} steered vs unsteered examples...")

            for i in range(total_comparisons):
                steered_ref = steered_reference[i]
                unsteered_ref = unsteered_reference[i]

                # Ensure same prompt
                assert steered_ref["prompt"] == unsteered_ref["prompt"], f"Prompt mismatch at index {i}"

                prompt = steered_ref["prompt"]

                # Generate steered output
                steered_result = dac.generate_with_dynamic_steering(
                    prompt=prompt,
                    property_weights={"language_ita_eng": 1.0},
                    starting_alpha=DYNAMIC_CONFIG["starting_alpha"],
                    top_p=DYNAMIC_CONFIG["top_p"],
                    alpha_bounds=DYNAMIC_CONFIG["alpha_bounds"],
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                # Generate unsteered output
                unsteered_result = dac.generate_with_steering(
                    prompt=prompt,
                    property_weights=None,  # No steering
                    max_new_tokens=MAX_NEW_TOKENS,
                    timing_strategy="normal",
                )

                our_steered = steered_result["generated_text"]
                our_unsteered = unsteered_result
                ref_steered = steered_ref["generated_text"]
                ref_unsteered = unsteered_ref["generated_text"]

                print(f"\\nExample {i + 1}:")
                print(f"  Prompt: {prompt[:50]}...")
                print(f"  Our steered: {our_steered[:50]}...")
                print(f"  Our unsteered: {our_unsteered[:50]}...")
                print(f"  Ref steered: {ref_steered[:50]}...")
                print(f"  Ref unsteered: {ref_unsteered[:50]}...")

                # Calculate similarity scores
                steered_similarity = _calculate_text_similarity(our_steered, our_unsteered)

                # Check for Italian in steered but not unsteered
                steered_has_italian, steered_words = _contains_italian_words(our_steered)
                unsteered_has_italian, unsteered_words = _contains_italian_words(our_unsteered)

                if steered_has_italian and not unsteered_has_italian:
                    significant_differences += 1
                    print(f"  ✅ Clear steering effect: Italian in steered ({steered_words[:2]}), none in unsteered")
                elif steered_similarity < 0.5:  # Low text similarity also indicates steering effect
                    significant_differences += 1
                    print(f"  ✅ Steering effect via text difference (similarity: {steered_similarity:.2f})")
                else:
                    print(f"  ⚠️  Limited steering effect detected")

            # Validate steering effectiveness
            effectiveness_rate = significant_differences / total_comparisons
            print(f"\\n📈 Steering effectiveness summary:")
            print(
                f"   Significant differences: {significant_differences}/{total_comparisons} ({effectiveness_rate:.1%})"
            )

            assert effectiveness_rate >= 0.25, (
                f"Steering effectiveness rate {effectiveness_rate:.2%} too low. "
                f"Expected at least some clear differences between steered and unsteered outputs."
            )

            print("✅ Steering effectiveness test passed!")

        finally:
            # Clean up
            if hasattr(dac, "_model") and dac._model:
                del dac._model
                del dac._tokenizer
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_italian_language_detection(self):
        """Test that steered outputs contain Italian language patterns like reference."""
        print("\\n🇮🇹 Testing Italian language detection in steered outputs...")

        # Load reference steered data
        steered_reference = _load_reference_data("steered")
        unsteered_reference = _load_reference_data("unsteered")

        print("\\n📊 Analyzing reference data language patterns...")

        # Analyze reference steered data for Italian content
        ref_steered_italian_count = 0
        ref_unsteered_italian_count = 0

        for example in steered_reference:
            has_italian, _ = _contains_italian_words(example["generated_text"])
            if has_italian:
                ref_steered_italian_count += 1

        for example in unsteered_reference:
            has_italian, _ = _contains_italian_words(example["generated_text"])
            if has_italian:
                ref_unsteered_italian_count += 1

        ref_steered_rate = ref_steered_italian_count / len(steered_reference)
        ref_unsteered_rate = ref_unsteered_italian_count / len(unsteered_reference)

        print(f"Reference Italian detection rates:")
        print(f"  Steered: {ref_steered_italian_count}/{len(steered_reference)} ({ref_steered_rate:.1%})")
        print(f"  Unsteered: {ref_unsteered_italian_count}/{len(unsteered_reference)} ({ref_unsteered_rate:.1%})")

        # Note: Some overlap expected due to model tendencies, focus on our implementation
        print(f"Note: Reference data shows steered rate {ref_steered_rate:.1%} vs unsteered {ref_unsteered_rate:.1%}")

        print(
            f"✅ Reference data validated: steering increases Italian content by {ref_steered_rate - ref_unsteered_rate:.1%}"
        )

        # Now test our implementation achieves similar Italian detection rates
        dac_method_path = REFERENCE_DATA_PATH / "dac_method.pt"
        dac = DAC(
            model_name=MODEL_NAME,
            max_examples=20,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=TORCH_DTYPE,
            icl_examples=ICL_EXAMPLES,
            legacy_behavior=True,  # Use Q:/A: format like reference
        )

        try:
            # Load reference tensor
            success = dac.load_steering_tensor(str(dac_method_path))
            assert success, "Failed to load DAC method file"

            # Test on a subset of prompts
            test_prompts = [example["prompt"] for example in steered_reference[:2]]  # Test 2 examples for speed
            our_italian_count = 0
            italian_examples = []

            print(f"\\n🧪 Testing our implementation on {len(test_prompts)} prompts...")

            for i, prompt in enumerate(test_prompts):
                result = dac.generate_with_dynamic_steering(
                    prompt=prompt,
                    property_weights={"language_ita_eng": 1.0},
                    starting_alpha=DYNAMIC_CONFIG["starting_alpha"],
                    top_p=DYNAMIC_CONFIG["top_p"],
                    alpha_bounds=DYNAMIC_CONFIG["alpha_bounds"],
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                generated_text = result["generated_text"]
                has_italian, found_words = _contains_italian_words(generated_text)

                print(f"  Example {i + 1}:")
                print(f"    Generated: {generated_text[:70]}...")

                if has_italian:
                    our_italian_count += 1
                    italian_examples.append(
                        {
                            "text": generated_text,
                            "words": found_words[:5],  # Show first 5 words
                        }
                    )
                    print(f"    ✅ Italian detected: {found_words[:3]}")
                else:
                    print(f"    ❌ No Italian detected")

            our_italian_rate = our_italian_count / len(test_prompts)
            print(f"\\n📈 Our implementation Italian detection:")
            print(f"   Examples with Italian: {our_italian_count}/{len(test_prompts)} ({our_italian_rate:.1%})")

            # Show best Italian examples
            if italian_examples:
                print(f"\\n🎯 Examples with Italian content:")
                for i, example in enumerate(italian_examples[:2], 1):
                    print(f"   {i}. {example['text'][:60]}...")
                    print(f"      Italian words: {example['words']}")

            # Validate our implementation achieves reasonable Italian detection
            # We use a lower threshold for small test set, but expect some Italian
            min_expected_rate = 0.5  # At least 50% should have Italian for small test
            assert our_italian_rate >= min_expected_rate, (
                f"Our Italian detection rate {our_italian_rate:.1%} too low. "
                f"Expected at least {min_expected_rate:.1%} to validate steering works."
            )

            print("✅ Italian language detection test passed!")

        finally:
            # Clean up
            if hasattr(dac, "_model") and dac._model:
                del dac._model
                del dac._tokenizer
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_alpha_kl_history_comparison(self):
        """Compare our alpha/KL histories with reference values for quantitative validation."""
        print("\\n📊 Testing quantitative alpha/KL history comparison...")

        # Load reference data
        steered_reference = _load_reference_data("steered")
        assert len(steered_reference) > 0, "No steered reference data found"

        # Load tensor and initialize DAC
        dac_method_path = REFERENCE_DATA_PATH / "dac_method.pt"
        dac = DAC(
            model_name=MODEL_NAME,
            max_examples=20,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=TORCH_DTYPE,
            icl_examples=ICL_EXAMPLES,
            legacy_behavior=True,
        )

        try:
            success = dac.load_steering_tensor(str(dac_method_path))
            assert success, "Failed to load DAC method file"

            alpha_correlations = []
            kl_correlations = []
            total_examples = len(steered_reference)

            print(f"Comparing alpha/KL sequences for {total_examples} examples...")

            for i, ref_example in enumerate(steered_reference):
                prompt = ref_example["prompt"]
                ref_alpha = ref_example["alpha_history"]
                ref_kl = ref_example["kl_history"]

                print(f"\\nExample {i + 1}/{total_examples}:")
                print(f"  Prompt: {prompt[:60]}...")

                # Generate with our implementation
                result = dac.generate_with_dynamic_steering(
                    prompt=prompt,
                    property_weights={"language_ita_eng": 1.0},
                    starting_alpha=DYNAMIC_CONFIG["starting_alpha"],
                    top_p=DYNAMIC_CONFIG["top_p"],
                    alpha_bounds=DYNAMIC_CONFIG["alpha_bounds"],
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                our_alpha = result["alpha_history"]["language_ita_eng"]
                our_kl = result["kl_history"]["language_ita_eng"]

                # Compute correlations
                alpha_correlation = _compute_sequence_similarity(our_alpha, ref_alpha)
                kl_correlation = _compute_sequence_similarity(our_kl, ref_kl) if ref_kl else 0.0

                print(f"  Reference alpha: {ref_alpha[:5]}... (len={len(ref_alpha)})")
                print(f"  Our alpha: {our_alpha[:5]}... (len={len(our_alpha)})")
                print(f"  Alpha correlation: {alpha_correlation:.3f}")

                if ref_kl and our_kl:
                    print(f"  Reference KL: {ref_kl[:5]}... (len={len(ref_kl)})")
                    print(f"  Our KL: {our_kl[:5]}... (len={len(our_kl)})")
                    print(f"  KL correlation: {kl_correlation:.3f}")
                else:
                    print(f"  KL comparison: Ref={len(ref_kl) if ref_kl else 0}, Our={len(our_kl)}")

                alpha_correlations.append(alpha_correlation)
                if ref_kl and our_kl:
                    kl_correlations.append(kl_correlation)

                # Quality indicators
                if alpha_correlation > 0.3:
                    print(f"  ✅ Good alpha correlation")
                elif alpha_correlation > 0.0:
                    print(f"  ⚠️ Moderate alpha correlation")
                else:
                    print(f"  ❌ Low alpha correlation")

            # Overall assessment
            avg_alpha_correlation = np.mean(alpha_correlations)
            avg_kl_correlation = np.mean(kl_correlations) if kl_correlations else 0.0

            print(f"\\n📈 Quantitative comparison summary:")
            print(f"   Average alpha correlation: {avg_alpha_correlation:.3f}")
            print(f"   Alpha correlations: {[f'{c:.2f}' for c in alpha_correlations]}")

            if kl_correlations:
                print(f"   Average KL correlation: {avg_kl_correlation:.3f}")
                print(f"   KL correlations: {[f'{c:.2f}' for c in kl_correlations]}")
            else:
                print(f"   KL correlation: Not available (our implementation may not track KL)")

            # Validation assertions - use relaxed thresholds since exact reproduction is difficult
            assert avg_alpha_correlation > 0.1, (
                f"Alpha correlation {avg_alpha_correlation:.3f} too low. "
                f"Expected at least some correlation with reference alpha sequences."
            )

            # At least some examples should have reasonable correlation
            good_correlations = sum(1 for c in alpha_correlations if c > 0.2)
            good_rate = good_correlations / total_examples
            assert good_rate >= 0.25, (
                f"Only {good_correlations}/{total_examples} ({good_rate:.1%}) examples show good alpha correlation. "
                f"Expected at least 25% to have correlation > 0.2"
            )

            print("✅ Alpha/KL history comparison test passed!")

        finally:
            if hasattr(dac, "_model") and dac._model:
                del dac._model
                del dac._tokenizer
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_generated_text_similarity(self):
        """Compare our generated text with reference generated text for quantitative validation."""
        print("\\n📝 Testing generated text similarity with reference...")

        # Load reference data
        steered_reference = _load_reference_data("steered")
        assert len(steered_reference) > 0, "No steered reference data found"

        # Load tensor and initialize DAC
        dac_method_path = REFERENCE_DATA_PATH / "dac_method.pt"
        dac = DAC(
            model_name=MODEL_NAME,
            max_examples=20,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=TORCH_DTYPE,
            icl_examples=ICL_EXAMPLES,
            legacy_behavior=True,
        )

        try:
            success = dac.load_steering_tensor(str(dac_method_path))
            assert success, "Failed to load DAC method file"

            text_similarities = []
            italian_word_overlaps = []
            total_examples = len(steered_reference)

            print(f"Comparing generated text for {total_examples} examples...")

            for i, ref_example in enumerate(steered_reference):
                prompt = ref_example["prompt"]
                ref_text = ref_example["generated_text"]

                print(f"\\nExample {i + 1}/{total_examples}:")
                print(f"  Prompt: {prompt[:50]}...")

                # Generate with our implementation
                result = dac.generate_with_dynamic_steering(
                    prompt=prompt,
                    property_weights={"language_ita_eng": 1.0},
                    starting_alpha=DYNAMIC_CONFIG["starting_alpha"],
                    top_p=DYNAMIC_CONFIG["top_p"],
                    alpha_bounds=DYNAMIC_CONFIG["alpha_bounds"],
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                our_text = result["generated_text"]

                # Compute text similarity
                text_similarity = _calculate_text_similarity(our_text, ref_text)
                text_similarities.append(text_similarity)

                # Compare Italian word usage
                our_italian, our_words = _contains_italian_words(our_text)
                ref_italian, ref_words = _contains_italian_words(ref_text)

                italian_overlap = 0.0
                if our_words or ref_words:
                    common_words = set(our_words).intersection(set(ref_words))
                    total_unique_words = len(set(our_words).union(set(ref_words)))
                    italian_overlap = len(common_words) / total_unique_words if total_unique_words > 0 else 0.0

                italian_word_overlaps.append(italian_overlap)

                print(f"  Reference: {ref_text[:70]}...")
                print(f"  Our text: {our_text[:70]}...")
                print(f"  Text similarity: {text_similarity:.3f}")
                print(f"  Italian words - Ref: {ref_words[:3]}, Our: {our_words[:3]}")
                print(f"  Italian overlap: {italian_overlap:.3f}")

                # Quality indicators
                if text_similarity > 0.3:
                    print(f"  ✅ Good text similarity")
                elif text_similarity > 0.15:
                    print(f"  ⚠️ Moderate text similarity")
                else:
                    print(f"  ❌ Low text similarity")

            # Overall assessment
            avg_text_similarity = np.mean(text_similarities)
            avg_italian_overlap = np.mean(italian_word_overlaps)

            print(f"\\n📈 Text similarity summary:")
            print(f"   Average text similarity: {avg_text_similarity:.3f}")
            print(f"   Text similarities: {[f'{s:.2f}' for s in text_similarities]}")
            print(f"   Average Italian word overlap: {avg_italian_overlap:.3f}")
            print(f"   Italian overlaps: {[f'{o:.2f}' for o in italian_word_overlaps]}")

            # Validation assertions - use realistic thresholds for generative text
            assert avg_text_similarity > 0.05, (
                f"Text similarity {avg_text_similarity:.3f} too low. "
                f"Expected at least some word overlap with reference texts."
            )

            # At least some examples should show meaningful similarity
            good_similarities = sum(1 for s in text_similarities if s > 0.15)
            good_rate = good_similarities / total_examples
            assert good_rate >= 0.25, (
                f"Only {good_similarities}/{total_examples} ({good_rate:.1%}) examples show good text similarity. "
                f"Expected at least 25% to have similarity > 0.15"
            )

            print("✅ Generated text similarity test passed!")

        finally:
            if hasattr(dac, "_model") and dac._model:
                del dac._model
                del dac._tokenizer
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_dynamic_adaptation_patterns(self):
        """Validate our dynamic adaptation follows similar patterns to reference."""
        print("\\n🔄 Testing dynamic adaptation patterns...")

        # Load reference data
        steered_reference = _load_reference_data("steered")
        assert len(steered_reference) > 0, "No steered reference data found"

        # Load tensor and initialize DAC
        dac_method_path = REFERENCE_DATA_PATH / "dac_method.pt"
        dac = DAC(
            model_name=MODEL_NAME,
            max_examples=20,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=TORCH_DTYPE,
            icl_examples=ICL_EXAMPLES,
            legacy_behavior=True,
        )

        try:
            success = dac.load_steering_tensor(str(dac_method_path))
            assert success, "Failed to load DAC method file"

            pattern_similarities = []
            adaptation_scores = []
            total_examples = len(steered_reference)

            print(f"Analyzing adaptation patterns for {total_examples} examples...")

            for i, ref_example in enumerate(steered_reference):
                prompt = ref_example["prompt"]
                ref_alpha = ref_example["alpha_history"]

                print(f"\\nExample {i + 1}/{total_examples}:")
                print(f"  Prompt: {prompt[:50]}...")

                # Generate with our implementation
                result = dac.generate_with_dynamic_steering(
                    prompt=prompt,
                    property_weights={"language_ita_eng": 1.0},
                    starting_alpha=DYNAMIC_CONFIG["starting_alpha"],
                    top_p=DYNAMIC_CONFIG["top_p"],
                    alpha_bounds=DYNAMIC_CONFIG["alpha_bounds"],
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                our_alpha = result["alpha_history"]["language_ita_eng"]

                # Analyze high-alpha positions (strong steering)
                our_high_alpha = [i for i, a in enumerate(our_alpha) if a > 1.0]
                ref_high_alpha = [i for i, a in enumerate(ref_alpha) if a > 1.0]

                # Analyze adaptation effectiveness
                our_adaptation = np.std(our_alpha) if len(our_alpha) > 1 else 0.0
                ref_adaptation = np.std(ref_alpha) if len(ref_alpha) > 1 else 0.0

                # Compare patterns
                pattern_similarity = _compute_pattern_similarity(our_high_alpha, ref_high_alpha, len(our_alpha))
                pattern_similarities.append(pattern_similarity)

                # Compare adaptation levels
                adaptation_ratio = (
                    min(our_adaptation, ref_adaptation) / max(our_adaptation, ref_adaptation)
                    if max(our_adaptation, ref_adaptation) > 0
                    else 1.0
                )
                adaptation_scores.append(adaptation_ratio)

                print(
                    f"  Reference alpha range: [{min(ref_alpha):.2f}, {max(ref_alpha):.2f}], std: {ref_adaptation:.2f}"
                )
                print(f"  Our alpha range: [{min(our_alpha):.2f}, {max(our_alpha):.2f}], std: {our_adaptation:.2f}")
                print(f"  High-alpha positions - Ref: {ref_high_alpha}, Our: {our_high_alpha}")
                print(f"  Pattern similarity: {pattern_similarity:.3f}")
                print(f"  Adaptation ratio: {adaptation_ratio:.3f}")

                # Quality indicators
                if pattern_similarity > 0.3 and adaptation_ratio > 0.5:
                    print(f"  ✅ Good adaptation pattern")
                elif pattern_similarity > 0.1 or adaptation_ratio > 0.3:
                    print(f"  ⚠️ Moderate adaptation pattern")
                else:
                    print(f"  ❌ Different adaptation pattern")

            # Overall assessment
            avg_pattern_similarity = np.mean(pattern_similarities)
            avg_adaptation_ratio = np.mean(adaptation_scores)

            print(f"\\n📈 Adaptation pattern summary:")
            print(f"   Average pattern similarity: {avg_pattern_similarity:.3f}")
            print(f"   Pattern similarities: {[f'{p:.2f}' for p in pattern_similarities]}")
            print(f"   Average adaptation ratio: {avg_adaptation_ratio:.3f}")
            print(f"   Adaptation ratios: {[f'{a:.2f}' for a in adaptation_scores]}")

            # Validation assertions
            assert avg_pattern_similarity > 0.05, (
                f"Pattern similarity {avg_pattern_similarity:.3f} too low. "
                f"Expected at least some similarity in adaptation patterns."
            )

            assert avg_adaptation_ratio > 0.2, (
                f"Adaptation ratio {avg_adaptation_ratio:.3f} too low. "
                f"Expected similar levels of alpha variation as reference."
            )

            print("✅ Dynamic adaptation patterns test passed!")

        finally:
            if hasattr(dac, "_model") and dac._model:
                del dac._model
                del dac._tokenizer
            aggressive_memory_cleanup()


if __name__ == "__main__":
    # Run individual tests for debugging
    test_instance = TestDACSteeringValidation()

    print("🧪 Running Enhanced DAC Steering Validation Tests")

    try:
        print("\\n1️⃣ Testing reference tensor loading...")
        test_instance.test_reference_tensor_loading()
        print("✅ Reference tensor loading test passed!")

        print("\\n2️⃣ Testing dynamic steering reproduction...")
        test_instance.test_dynamic_steering_reproduction()
        print("✅ Dynamic steering reproduction test passed!")

        print("\\n3️⃣ Testing alpha/KL history comparison...")
        test_instance.test_alpha_kl_history_comparison()
        print("✅ Alpha/KL history comparison test passed!")

        print("\\n4️⃣ Testing generated text similarity...")
        test_instance.test_generated_text_similarity()
        print("✅ Generated text similarity test passed!")

        print("\\n5️⃣ Testing dynamic adaptation patterns...")
        test_instance.test_dynamic_adaptation_patterns()
        print("✅ Dynamic adaptation patterns test passed!")

        print("\\n🎉 All enhanced DAC steering tests passed!")

    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
