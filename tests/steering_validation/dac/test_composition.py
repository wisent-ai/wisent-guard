#!/usr/bin/env python3
"""
Test DAC Multiproperty Steering Composition

This test validates DAC's ability to compose multiple steering properties:
1. Safety steering (ITunsafe vs ITsafe) - ITunsafe as positive direction
2. Language steering (ENG vs ITA) - ITA as positive direction

Key test: Higher English steering should limit Italian words.
Evaluation on ITsafe prompts to test steering away from default safe behavior.

Uses pytest structure with detailed user-facing output.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

# Import wisent-guard DAC
from wisent_guard.core.contrastive_pairs import ContrastivePair, ContrastivePairSet
from wisent_guard.core.response import NegativeResponse, PositiveResponse
from wisent_guard.core.steering_methods_tensor.dac_attention import DAC

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from const import (
    ICL_EXAMPLES,
    MAX_EXAMPLES,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    REFERENCE_DATA_PATH,
    TORCH_DTYPE,
)

# Import aggressive_memory_cleanup
utils_path = Path(__file__).parent.parent / "utils.py"
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import aggressive_memory_cleanup

# Test configuration for composition
ITALIAN_WORDS = [
    "sono",
    "tutti",
    "ogni",
    "giorno",
    "che",
    "con",
    "di",
    "la",
    "il",
    "una",
    "nostri",
    "basati",
    "è",
    "dei",
    "delle",
    "nel",
    "per",
    "sul",
    "alla",
    "dello",
]
TEST_PROMPTS_COUNT = 3  # Use first 3 prompts from ITunsafe for manageable testing


def _load_english_safety_datasets() -> Tuple[List[Dict], List[Dict]]:
    """Load ENsafe and ENunsafe datasets from reference_data."""
    ensafe_path = REFERENCE_DATA_PATH / "ensafe_train.json"
    enunsafe_path = REFERENCE_DATA_PATH / "enunsafe_train.json"

    if not ensafe_path.exists():
        pytest.skip(f"ENsafe dataset not found: {ensafe_path}. Run create_ensafe_dataset.py first.")

    if not enunsafe_path.exists():
        pytest.skip(f"ENunsafe dataset not found: {enunsafe_path}. Run limit_safety_datasets.py first.")

    with open(ensafe_path, encoding="utf-8") as f:
        ensafe_data = json.load(f)

    with open(enunsafe_path, encoding="utf-8") as f:
        enunsafe_data = json.load(f)

    return ensafe_data, enunsafe_data


def _load_language_datasets() -> Tuple[List[Dict], List[Dict]]:
    """Load ENG and ITA datasets from reference_data."""
    eng_path = REFERENCE_DATA_PATH / "eng_train.json"
    ita_path = REFERENCE_DATA_PATH / "ita_train.json"

    if not eng_path.exists():
        pytest.skip(f"ENG dataset not found: {eng_path}. Run generate_data_with_original_dac_implementation.py first.")

    if not ita_path.exists():
        pytest.skip(f"ITA dataset not found: {ita_path}. Run generate_data_with_original_dac_implementation.py first.")

    with open(eng_path, encoding="utf-8") as f:
        eng_data = json.load(f)

    with open(ita_path, encoding="utf-8") as f:
        ita_data = json.load(f)

    return eng_data, ita_data


def _create_safety_contrastive_pairs(ensafe_data: List[Dict], enunsafe_data: List[Dict]) -> ContrastivePairSet:
    """Create contrastive pairs for safety steering (ENunsafe vs ENsafe).

    Since LLMs are trained to be safe by default, we train with ENunsafe as positive
    to learn the direction toward unsafe responses.
    """
    pairs = []
    max_pairs = min(len(ensafe_data), len(enunsafe_data), MAX_EXAMPLES)

    for i in range(max_pairs):
        safe_item = ensafe_data[i]
        unsafe_item = enunsafe_data[i]

        # Ensure inputs match (they should in the paired datasets)
        assert safe_item["input"] == unsafe_item["input"], (
            f"Input mismatch at index {i}: '{safe_item['input']}' != '{unsafe_item['input']}'"
        )

        prompt = safe_item["input"]
        safe_response = safe_item["output"]  # ENsafe response (negative)
        unsafe_response = unsafe_item["output"]  # ENunsafe response (positive)

        pos_response = PositiveResponse(text=unsafe_response)  # ENunsafe as positive
        neg_response = NegativeResponse(text=safe_response)  # ENsafe as negative

        pair = ContrastivePair(
            prompt=prompt, positive_response=pos_response, negative_response=neg_response, label="safety_steering"
        )
        pairs.append(pair)

    return ContrastivePairSet(name="enunsafe_ensafe_safety", pairs=pairs, task_type="safety_steering")


def _create_language_contrastive_pairs(eng_data: List[Dict], ita_data: List[Dict]) -> ContrastivePairSet:
    """Create contrastive pairs for language steering (ENG vs ITA).

    English as positive direction to learn "English steering" vector.
    When applied with positive weight, steers toward English.
    When applied with negative weight, steers toward Italian.
    """
    pairs = []
    max_pairs = min(len(eng_data), len(ita_data), MAX_EXAMPLES)

    for i in range(max_pairs):
        eng_item = eng_data[i]
        ita_item = ita_data[i]

        # Ensure inputs match
        assert eng_item["input"] == ita_item["input"], f"Mismatched inputs at index {i}"

        prompt = eng_item["input"]
        eng_response = eng_item["output"]  # English response (positive for ENG steering)
        ita_response = ita_item["output"]  # Italian response (negative for ENG steering)

        pos_response = PositiveResponse(text=eng_response)  # English as positive
        neg_response = NegativeResponse(text=ita_response)  # Italian as negative

        pair = ContrastivePair(
            prompt=prompt, positive_response=pos_response, negative_response=neg_response, label="language_steering"
        )
        pairs.append(pair)

    return ContrastivePairSet(name="eng_ita_language", pairs=pairs, task_type="language_steering")


def _count_italian_words(text: str) -> int:
    """Count Italian words in text."""
    text_lower = text.lower()
    return sum(1 for word in ITALIAN_WORDS if word in text_lower)


def _print_detailed_composition_results(
    test_prompts: List[str], high_eng_results: List[Dict], high_ita_results: List[Dict]
):
    """Print detailed, user-friendly composition test results."""
    print("\n" + "=" * 80)
    print("🎯 COMPOSITION BALANCE TEST RESULTS")
    print("=" * 80)
    print("Test Configuration:")
    print("  - High English Steering: ENG=+1.0, Unsafe=1.0")
    print("  - High Italian Steering: ENG=-1.0, Unsafe=1.0")
    print(f"  - Test prompts: {len(test_prompts)} examples from ENsafe dataset")
    print("  - Goal: Test if higher English steering reduces Italian words")
    print()

    total_high_eng_italian = 0
    total_high_ita_italian = 0
    effective_steering_count = 0

    for i, prompt in enumerate(test_prompts):
        high_eng_result = high_eng_results[i]
        high_ita_result = high_ita_results[i]

        high_eng_text = high_eng_result["generated_text"]
        high_ita_text = high_ita_result["generated_text"]

        high_eng_italian_words = _count_italian_words(high_eng_text)
        high_ita_italian_words = _count_italian_words(high_ita_text)

        total_high_eng_italian += high_eng_italian_words
        total_high_ita_italian += high_ita_italian_words

        steering_effective = high_eng_italian_words < high_ita_italian_words
        if steering_effective:
            effective_steering_count += 1

        print(f"Results for Prompt {i + 1}:")
        print(f'  Input: "{prompt[:60]}{"..." if len(prompt) > 60 else ""}"')
        print()
        print("  🇬🇧 High English Steering (+1.0 ENG, 1.0 Unsafe):")
        print(f'     Generated: "{high_eng_text[:70]}{"..." if len(high_eng_text) > 70 else ""}"')
        print(f"     Italian words found: {high_eng_italian_words}")
        print(f"     Word count: {len(high_eng_text.split())}")
        print()
        print("  🇮🇹 High Italian Steering (-1.0 ENG, 1.0 Unsafe):")
        print(f'     Generated: "{high_ita_text[:70]}{"..." if len(high_ita_text) > 70 else ""}"')
        print(f"     Italian words found: {high_ita_italian_words}")
        print(f"     Word count: {len(high_ita_text.split())}")
        print()

        status = "✅ EFFECTIVE" if steering_effective else "❌ NOT EFFECTIVE"
        print(f"  Steering Effect: {status} ({high_eng_italian_words} vs {high_ita_italian_words} Italian words)")
        print("  " + "-" * 70)
        print()

    # Calculate summary statistics
    avg_high_eng_italian = total_high_eng_italian / len(test_prompts)
    avg_high_ita_italian = total_high_ita_italian / len(test_prompts)
    composition_effectiveness = effective_steering_count / len(test_prompts)

    if total_high_ita_italian > 0:
        italian_reduction = (total_high_ita_italian - total_high_eng_italian) / total_high_ita_italian
    else:
        italian_reduction = 0.0

    print("📊 SUMMARY:")
    print(f"  - High English Steering: avg {avg_high_eng_italian:.1f} Italian words per response")
    print(f"  - High Italian Steering: avg {avg_high_ita_italian:.1f} Italian words per response")
    print(f"  - Italian word reduction: {italian_reduction:.1%}")
    print(
        f"  - Composition effectiveness: {composition_effectiveness:.1%} ({effective_steering_count}/{len(test_prompts)} prompts)"
    )
    print()

    if composition_effectiveness >= 0.66:  # 2 out of 3 prompts
        print("✅ COMPOSITION BALANCE TEST PASSED")
        print("   Higher English steering successfully reduced Italian words!")
    else:
        print("❌ COMPOSITION BALANCE TEST FAILED")
        print("   English steering did not consistently reduce Italian words")

    print("=" * 80)


@pytest.mark.slow
@pytest.mark.heavy
class TestDACCompositionSteering:
    """Test DAC multiproperty steering composition capabilities."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_multiproperty_steering_composition(self):
        """Test complete DAC multiproperty steering: training and composition balance with detailed output."""
        # Load datasets
        ensafe_data, enunsafe_data = _load_english_safety_datasets()
        eng_data, ita_data = _load_language_datasets()

        # Create contrastive pairs
        safety_pairs = _create_safety_contrastive_pairs(ensafe_data, enunsafe_data)
        language_pairs = _create_language_contrastive_pairs(eng_data, ita_data)

        print(f"\nCreated {len(safety_pairs.pairs)} safety pairs (ENunsafe vs ENsafe)")
        print(f"Created {len(language_pairs.pairs)} language pairs (ENG vs ITA)")

        # Initialize DAC
        dac = DAC(
            model_name=MODEL_NAME,
            max_examples=MAX_EXAMPLES,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=TORCH_DTYPE,
            icl_examples=ICL_EXAMPLES,
        )

        try:
            # PART 1: TRAINING
            # Train safety property (ENunsafe as positive direction)
            print("\n🔧 Training safety property (ENunsafe vs ENsafe)...")
            safety_stats = dac.train_property("safety_enunsafe_ensafe", safety_pairs)
            assert safety_stats["success"], f"Safety training failed: {safety_stats}"
            print("✅ Safety property trained successfully")

            # Train language property (English as positive direction)
            print("\n🔧 Training language property (ENG vs ITA)...")
            language_stats = dac.train_property("language_eng_ita", language_pairs)
            assert language_stats["success"], f"Language training failed: {language_stats}"
            print("✅ Language property trained successfully")

            # Verify both properties are available
            assert "safety_enunsafe_ensafe" in dac.property_tensors, "Safety property not found"
            assert "language_eng_ita" in dac.property_tensors, "Language property not found"

            print("\n📊 DAC Training Results:")
            print(f"  - Available properties: {list(dac.property_tensors.keys())}")
            print(f"  - Safety tensor shape: {dac.property_tensors['safety_enunsafe_ensafe'].shape}")
            print(f"  - Language tensor shape: {dac.property_tensors['language_eng_ita'].shape}")

            # PART 2: COMPOSITION BALANCE TESTING
            print("\n" + "=" * 60)
            print("🎯 STARTING COMPOSITION BALANCE TESTING")
            print("=" * 60)

            # Load test prompts from ENsafe dataset
            test_prompts = [item["input"] for item in ensafe_data[:TEST_PROMPTS_COUNT]]
            print(f"\n🎯 Testing composition balance with {len(test_prompts)} prompts from ENsafe...")

            high_eng_results = []
            high_ita_results = []

            # Test Case A: High English steering (1.0 ENG, 1.0 Unsafe)
            print("  Testing High English Steering (1.0 ENG, 1.0 Unsafe)...")
            for i, prompt in enumerate(test_prompts):
                result = dac.generate_with_multi_property_steering(
                    prompt=prompt,
                    property_weights={
                        "language_eng_ita": 1.0,
                        "safety_enunsafe_ensafe": 1.0,
                    },  # Full English + Unsafe steering
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                high_eng_results.append(result)
                print(f"    Prompt {i + 1}: {len(result['generated_text'].split())} words")

            # Test Case B: High Italian steering (-1.0 ENG, 1.0 Unsafe)
            print("  Testing High Italian Steering (-1.0 ENG, 1.0 Unsafe)...")
            for i, prompt in enumerate(test_prompts):
                result = dac.generate_with_multi_property_steering(
                    prompt=prompt,
                    property_weights={
                        "language_eng_ita": -1.0,
                        "safety_enunsafe_ensafe": 1.0,
                    },  # Full Italian + Unsafe steering
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                high_ita_results.append(result)
                print(f"    Prompt {i + 1}: {len(result['generated_text'].split())} words")

            # Print detailed results for users
            _print_detailed_composition_results(test_prompts, high_eng_results, high_ita_results)

            # Calculate metrics for assertions
            total_high_eng_italian = sum(_count_italian_words(r["generated_text"]) for r in high_eng_results)
            total_high_ita_italian = sum(_count_italian_words(r["generated_text"]) for r in high_ita_results)

            avg_high_eng_italian = total_high_eng_italian / len(test_prompts)
            avg_high_ita_italian = total_high_ita_italian / len(test_prompts)

            # Main assertion: Higher English steering should reduce Italian words
            assert avg_high_eng_italian < avg_high_ita_italian, (
                f"Higher English steering should reduce Italian words. "
                f"High ENG: {avg_high_eng_italian:.1f} vs High ITA: {avg_high_ita_italian:.1f} avg Italian words"
            )

            # Effectiveness threshold: should see meaningful difference
            if total_high_ita_italian > 0:
                composition_effectiveness = (total_high_ita_italian - total_high_eng_italian) / total_high_ita_italian
                assert composition_effectiveness > 0.2, (
                    f"Composition effectiveness {composition_effectiveness:.1%} too low. "
                    f"Expected at least 20% reduction in Italian words."
                )

            print("\n✅ Complete multiproperty steering composition test passed!")

        except Exception as e:
            print(f"\n❌ Multiproperty steering test failed: {e}")
            raise

        finally:
            # Cleanup
            if hasattr(dac, "_model") and dac._model:
                del dac._model
                del dac._tokenizer
            aggressive_memory_cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
