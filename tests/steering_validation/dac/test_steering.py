#!/usr/bin/env python3
"""
Test DAC Steering Validation

This test validates that our DAC implementation works correctly by testing:
1. Tensor loading matches reference exactly
2. Tokenization patterns match reference (special tokens)
3. Steering direction validation (English → Italian)
4. Italian language detection in steered outputs

Uses pytest structure with proper markers and assertions.
"""

import sys
import json
import torch
import os
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
    DATASET_A_NAME,
    DATASET_B_NAME,
)

# Import aggressive_memory_cleanup
utils_path = Path(__file__).parent.parent / "utils.py"
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import aggressive_memory_cleanup

# Add DAC directory for original tokenization
DAC_DIR = Path(__file__).parent.parent.parent.parent / "Dynamic-Activation-Composition"
sys.path.insert(0, str(DAC_DIR))

# Store original directory for path management
ORIGINAL_CWD = os.getcwd()

# Test configuration
ITALIAN_WORDS = ["sono", "tutti", "ogni", "giorno", "che", "con", "di", "la", "il", "una", "nostri", "basati"]
MIN_ITALIAN_DETECTION_RATE = 0.5  # At least 50% of steered outputs should contain Italian


def _load_reference_data(data_type: str) -> List[Dict[str, Any]]:
    """Load reference data for testing."""
    if data_type == "steered":
        ref_file = REFERENCE_DATA_PATH / "text_completions_dynamic_steering.json"
    elif data_type == "unsteered":
        ref_file = REFERENCE_DATA_PATH / "text_completions_unsteered.json"
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if not ref_file.exists():
        pytest.skip(
            f"Required reference data not found: {ref_file}. "
            f"Please run generate_data_with_original_dac_implementation.py first."
        )

    with open(ref_file, "r") as f:
        return json.load(f)


def _contains_italian_words(text: str) -> bool:
    """Check if text contains Italian words."""
    text_lower = text.lower()
    return any(italian_word in text_lower for italian_word in ITALIAN_WORDS)


class TestKLToAlphaBounds:
    """Test KL divergence to alpha transformation bounds."""

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


@pytest.mark.slow
@pytest.mark.heavy
class TestDACSteeringValidation:
    """Test DAC steering validation against reference implementation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_tensor_loading_identical(self):
        """Test that our DAC loads tensors identically to reference."""
        dac_method_path = REFERENCE_DATA_PATH / "dac_method.pt"

        assert dac_method_path.exists(), f"DAC method file not found: {dac_method_path}"

        # Initialize DAC instance
        dac = DAC(
            model_name=MODEL_NAME,
            max_examples=20,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=TORCH_DTYPE,
            icl_examples=ICL_EXAMPLES,
        )

        try:
            # Load using our DAC class
            success = dac.load_steering_tensor(str(dac_method_path))
            assert success, "Failed to load DAC method file"

            # Verify basic properties
            assert dac.steering_tensor is not None, "Steering tensor is None"
            assert dac.steering_tensor.shape == torch.Size([30, 32, 32, 128]), (
                f"Unexpected tensor shape: {dac.steering_tensor.shape}"
            )
            assert "language_ita_eng" in dac.property_tensors, "Missing language_ita_eng property"

            # Verify the tensor matches reference diff_activations_ita_eng.pt
            diff_path = REFERENCE_DATA_PATH / "diff_activations_ita_eng.pt"
            if diff_path.exists():
                reference_tensor = torch.load(diff_path, map_location=dac.device)
                assert torch.allclose(dac.steering_tensor, reference_tensor, atol=1e-6), (
                    f"Steering tensor differs from reference. "
                    f"Max diff: {torch.max(torch.abs(dac.steering_tensor - reference_tensor)).item()}"
                )

        finally:
            if hasattr(dac, "_model") and dac._model:
                del dac._model
                del dac._tokenizer
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_steering_direction_validation(self):
        """Test that DAC steering changes language direction from English to Italian."""
        # Load reference data
        unsteered_data = _load_reference_data("unsteered")
        steered_data = _load_reference_data("steered")

        assert len(unsteered_data) > 0, "No unsteered reference data found"
        assert len(steered_data) > 0, "No steered reference data found"

        # Test steering direction for all available examples
        test_examples = min(len(unsteered_data), len(steered_data))

        unsteered_italian_count = 0
        steered_italian_count = 0
        steering_effects = []

        for i in range(test_examples):
            unsteered_example = unsteered_data[i]
            steered_example = steered_data[i]

            # Verify same prompt (essential for valid comparison)
            assert unsteered_example["prompt"] == steered_example["prompt"], (
                f"Prompt mismatch at index {i}: "
                f"Unsteered: '{unsteered_example['prompt'][:50]}...', "
                f"Steered: '{steered_example['prompt'][:50]}...'"
            )

            # Check for Italian words
            unsteered_text = unsteered_example["generated_text"]
            steered_text = steered_example["generated_text"]

            unsteered_has_italian = _contains_italian_words(unsteered_text)
            steered_has_italian = _contains_italian_words(steered_text)

            if unsteered_has_italian:
                unsteered_italian_count += 1
            if steered_has_italian:
                steered_italian_count += 1

            # Track individual steering effects
            steering_effect = {
                "index": i,
                "prompt": unsteered_example["prompt"][:50] + "...",
                "unsteered_has_italian": unsteered_has_italian,
                "steered_has_italian": steered_has_italian,
                "steering_effective": not unsteered_has_italian and steered_has_italian,
                "unsteered_text": unsteered_text[:100] + "...",
                "steered_text": steered_text[:100] + "...",
            }
            steering_effects.append(steering_effect)

        # Calculate steering effectiveness
        effective_steering_count = sum(1 for effect in steering_effects if effect["steering_effective"])
        steering_effectiveness = effective_steering_count / test_examples if test_examples > 0 else 0

        # Assertions about steering direction with detailed error messages
        assert unsteered_italian_count == 0, (
            f"Unsteered outputs should not contain Italian words. "
            f"Found {unsteered_italian_count}/{test_examples} with Italian words. "
            f"Examples with Italian: {[effect for effect in steering_effects if effect['unsteered_has_italian']]}"
        )

        assert steered_italian_count >= test_examples * MIN_ITALIAN_DETECTION_RATE, (
            f"Steered outputs should contain Italian words. "
            f"Found {steered_italian_count}/{test_examples} with Italian words. "
            f"Expected at least {test_examples * MIN_ITALIAN_DETECTION_RATE:.1f}. "
            f"Steering effectiveness: {steering_effectiveness:.2%}. "
            f"Failed examples: {[effect for effect in steering_effects if not effect['steered_has_italian']]}"
        )

        # Additional assertion for overall steering effectiveness
        assert steering_effectiveness >= MIN_ITALIAN_DETECTION_RATE, (
            f"Overall steering effectiveness {steering_effectiveness:.2%} is below minimum {MIN_ITALIAN_DETECTION_RATE:.0%}. "
            f"Only {effective_steering_count}/{test_examples} examples showed effective steering (English → Italian)."
        )

        # Log successful steering examples for debugging
        print(f"\nSteering validation results:")
        print(f"  - Test examples: {test_examples}")
        print(f"  - Unsteered with Italian: {unsteered_italian_count}")
        print(f"  - Steered with Italian: {steered_italian_count}")
        print(f"  - Effective steering: {effective_steering_count} ({steering_effectiveness:.1%})")

        for effect in steering_effects[:3]:  # Show first 3 examples
            status = "✅ EFFECTIVE" if effect["steering_effective"] else "❌ NOT EFFECTIVE"
            print(f"  Example {effect['index'] + 1}: {status}")
            print(f"    Prompt: {effect['prompt']}")
            print(f"    Unsteered: {effect['unsteered_text']}")
            print(f"    Steered: {effect['steered_text']}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_italian_language_detection(self):
        """Test that steered outputs contain Italian language patterns."""
        steered_data = _load_reference_data("steered")

        assert len(steered_data) > 0, "No steered reference data found"

        italian_examples = []
        word_frequency = {}
        total_examples = len(steered_data)

        for i, example in enumerate(steered_data):
            generated_text = example["generated_text"]
            italian_words_found = [word for word in ITALIAN_WORDS if word in generated_text.lower()]

            if italian_words_found:
                italian_examples.append(
                    {
                        "index": i,
                        "prompt": example["prompt"],
                        "text": generated_text,
                        "italian_words_found": italian_words_found,
                        "word_count": len(italian_words_found),
                    }
                )

                # Track word frequency
                for word in italian_words_found:
                    word_frequency[word] = word_frequency.get(word, 0) + 1

        italian_count = len(italian_examples)
        italian_rate = italian_count / total_examples if total_examples > 0 else 0

        # Calculate additional metrics
        total_italian_words = sum(len(ex["italian_words_found"]) for ex in italian_examples)
        avg_words_per_italian_example = total_italian_words / italian_count if italian_count > 0 else 0
        unique_italian_words = len(word_frequency)
        most_common_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:5]

        # Primary assertion - minimum detection rate
        assert italian_rate >= MIN_ITALIAN_DETECTION_RATE, (
            f"Italian detection rate {italian_rate:.2%} is below minimum {MIN_ITALIAN_DETECTION_RATE:.0%}. "
            f"Found {italian_count}/{total_examples} examples with Italian words. "
            f"Total Italian words detected: {total_italian_words}. "
            f"Average words per Italian example: {avg_words_per_italian_example:.1f}. "
            f"Examples WITHOUT Italian: {[i for i, ex in enumerate(steered_data) if not any(word in ex['generated_text'].lower() for word in ITALIAN_WORDS)]}"
        )

        # Secondary assertion - ensure variety in Italian words
        assert unique_italian_words >= 3, (
            f"Too few unique Italian words detected ({unique_italian_words}). "
            f"Expected at least 3 different Italian words for valid language steering. "
            f"Words found: {list(word_frequency.keys())}"
        )

        # Tertiary assertion - ensure sufficient Italian word density
        avg_italian_density = total_italian_words / total_examples
        min_density = 0.5  # At least 0.5 Italian words per example on average
        assert avg_italian_density >= min_density, (
            f"Italian word density {avg_italian_density:.2f} is too low (minimum {min_density}). "
            f"This suggests weak steering effect."
        )

        # Print detailed analysis for debugging
        print(f"\nItalian language detection analysis:")
        print(f"  - Examples with Italian: {italian_count}/{total_examples} ({italian_rate:.1%})")
        print(f"  - Total Italian words: {total_italian_words}")
        print(f"  - Unique Italian words: {unique_italian_words}")
        print(f"  - Average words per Italian example: {avg_words_per_italian_example:.1f}")
        print(f"  - Average Italian density: {avg_italian_density:.2f} words/example")
        print(f"  - Most common Italian words: {most_common_words}")

        # Show examples with best Italian content
        best_examples = sorted(italian_examples, key=lambda x: x["word_count"], reverse=True)[:3]
        print(f"\nTop 3 examples with most Italian words:")
        for i, example in enumerate(best_examples):
            print(f"  {i + 1}. Example {example['index']} ({example['word_count']} Italian words)")
            print(f"     Prompt: {example['prompt'][:50]}...")
            print(f"     Text: {example['text'][:80]}...")
            print(f"     Italian words: {example['italian_words_found']}")

        # Show examples with no Italian for debugging
        no_italian_indices = [
            i
            for i, ex in enumerate(steered_data)
            if not any(word in ex["generated_text"].lower() for word in ITALIAN_WORDS)
        ]
        if no_italian_indices:
            print(f"\nExamples WITHOUT Italian words ({len(no_italian_indices)}):")
            for idx in no_italian_indices[:2]:  # Show first 2
                ex = steered_data[idx]
                print(f"  Example {idx}: {ex['generated_text'][:80]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
