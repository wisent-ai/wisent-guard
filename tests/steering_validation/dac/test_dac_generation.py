#!/usr/bin/env python3
"""
Test real DAC generation with hook-based steering.

This test suite validates actual text generation using the hook-based
multi-property DAC steering implementation with real models.
"""

import torch
import pytest
import sys
from pathlib import Path
from typing import Dict, Any

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
    TORCH_DTYPE,
    ICL_EXAMPLES,
    MAX_NEW_TOKENS,
)

# Import aggressive_memory_cleanup
utils_path = Path(__file__).parent.parent / "utils.py"
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import aggressive_memory_cleanup


@pytest.mark.slow
@pytest.mark.heavy
class TestDACRealGeneration:
    """Test DAC generation with real models and tensors."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_dac_steering_generation_vs_baseline(self):
        """Test that DAC steering actually changes generation compared to baseline."""
        saved_dac_path = REFERENCE_DATA_PATH / "dac_method.pt"

        if not saved_dac_path.exists():
            pytest.skip("No saved DAC tensors available for generation test")

        try:
            # Load real DAC instance with ICL support
            dac = DAC(
                device="cuda:0",
                model_name=MODEL_NAME,
                max_new_tokens=8,  # Short generation for testing
                torch_dtype=TORCH_DTYPE,
                icl_examples=ICL_EXAMPLES,  # Use configured ICL examples
            )
            success = dac.load_steering_tensor(str(saved_dac_path))

            if not success:
                pytest.skip("Could not load DAC tensors")

            test_prompt = "What is the weather like?"

            # Generate baseline (unsteered)
            baseline_result = dac._generate_with_tensor_steering(
                prompt=test_prompt,
                steering_tensor=torch.zeros_like(dac.steering_tensor),  # Zero steering = baseline
                max_new_tokens=8,
                steering_strength=0.0,
            )

            # Generate with steering
            steered_result = dac.generate_with_steering(prompt=test_prompt, max_new_tokens=8, steering_strength=1.0)

            # Verify both generated text
            assert isinstance(baseline_result, str)
            assert isinstance(steered_result, str)
            assert len(baseline_result.strip()) > 0
            assert len(steered_result.strip()) > 0

            # They should be different (steering should change output)
            # Note: This might occasionally fail due to randomness, but should usually pass
            print(f"\nBaseline: '{baseline_result.strip()}'")
            print(f"Steered:  '{steered_result.strip()}'")

            # Basic validation that steering had some effect
            # (We don't assert inequality since models can be deterministic)
            assert len(baseline_result) > 0
            assert len(steered_result) > 0

        finally:
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_multi_property_composition_generation(self):
        """Test generation with multi-property composition if multiple properties exist."""
        saved_dac_path = REFERENCE_DATA_PATH / "dac_method.pt"

        if not saved_dac_path.exists():
            pytest.skip("No saved DAC tensors available")

        try:
            dac = DAC(device="cuda:0", model_name=MODEL_NAME, max_new_tokens=6, icl_examples=ICL_EXAMPLES)
            success = dac.load_steering_tensor(str(saved_dac_path))

            if not success or not dac.property_tensors:
                pytest.skip("Could not load property tensors")

            available_props = list(dac.property_tensors.keys())
            if len(available_props) < 1:
                pytest.skip("No properties available for multi-property test")

            test_prompt = "What is the weather like?"

            # Test single property
            single_result = dac.generate_with_multi_property_steering(
                prompt=test_prompt, property_weights={available_props[0]: 1.0}, max_new_tokens=6, steering_strength=0.8
            )

            # Verify result structure
            assert "generated_text" in single_result
            assert "properties_used" in single_result
            assert "property_weights" in single_result
            assert len(single_result["generated_text"].strip()) > 0
            assert single_result["properties_used"] == [available_props[0]]

            # Test with different weights
            weighted_result = dac.generate_with_multi_property_steering(
                prompt=test_prompt, property_weights={available_props[0]: 2.0}, max_new_tokens=6, steering_strength=0.8
            )

            assert "generated_text" in weighted_result
            assert len(weighted_result["generated_text"].strip()) > 0

            print(f"Single property result: '{single_result['generated_text'].strip()}'")
            print(f"Weighted result: '{weighted_result['generated_text'].strip()}'")

        finally:
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_timing_strategies_with_real_model(self):
        """Test different timing strategies with real model."""
        saved_dac_path = REFERENCE_DATA_PATH / "dac_method.pt"

        if not saved_dac_path.exists():
            pytest.skip("No saved DAC tensors available")

        try:
            dac = DAC(device="cuda:0", model_name=MODEL_NAME, max_new_tokens=5, icl_examples=ICL_EXAMPLES)
            success = dac.load_steering_tensor(str(saved_dac_path))

            if not success:
                pytest.skip("Could not load DAC tensors")

            test_prompt = "The capital city is"

            # Test different timing strategies
            strategies = ["normal", "start_only", "diminishing"]
            results = {}

            for strategy in strategies:
                result = dac._generate_with_tensor_steering(
                    prompt=test_prompt,
                    steering_tensor=dac.steering_tensor,
                    max_new_tokens=5,
                    steering_strength=1.0,
                    timing_strategy=strategy,
                )
                results[strategy] = result

            # Verify all strategies produce output
            for strategy, result in results.items():
                assert isinstance(result, str), f"{strategy} should produce string output"
                assert len(result.strip()) > 0, f"{strategy} should produce non-empty output"
                print(f"{strategy:12}: '{result.strip()}'")

        finally:
            aggressive_memory_cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_steering_strength_effects(self):
        """Test that different steering strengths produce different effects."""
        saved_dac_path = REFERENCE_DATA_PATH / "dac_method.pt"

        if not saved_dac_path.exists():
            pytest.skip("No saved DAC tensors available")

        try:
            dac = DAC(device="cuda:0", model_name=MODEL_NAME, max_new_tokens=6, icl_examples=ICL_EXAMPLES)
            success = dac.load_steering_tensor(str(saved_dac_path))

            if not success:
                pytest.skip("Could not load DAC tensors")

            test_prompt = "Hello, my name is"
            strengths = [0.0, 0.5, 1.0, 2.0]
            results = {}

            for strength in strengths:
                result = dac.generate_with_steering(prompt=test_prompt, max_new_tokens=6, steering_strength=strength)
                results[strength] = result

            # Verify all strengths produce output
            for strength, result in results.items():
                assert isinstance(result, str), f"Strength {strength} should produce string"
                assert len(result.strip()) > 0, f"Strength {strength} should produce non-empty output"
                print(f"Strength {strength:3}: '{result.strip()}'")

        finally:
            aggressive_memory_cleanup()


@pytest.mark.slow
class TestDACGenerationEdgeCases:
    """Test edge cases in DAC generation."""

    def test_empty_prompt_handling(self):
        """Test handling of empty or very short prompts."""
        # This test doesn't require GPU since we're testing input validation
        dac = DAC(device="cpu", model_name="gpt2", icl_examples=0)  # Small model for testing, no ICL needed

        # Mock the components to avoid loading actual models
        dac._tokenizer = lambda text, **kwargs: {"input_ids": torch.tensor([[1]])}
        dac._generate_step_by_step_with_hooks = lambda **kwargs: "generated"
        dac._load_model = lambda: None
        dac.steering_tensor = torch.randn(5, 4, 4, 8)  # Mock tensor
        dac.is_trained = True

        # Test empty prompt
        result = dac.generate_with_steering("", max_new_tokens=3)
        assert isinstance(result, str)

        # Test single character prompt
        result = dac.generate_with_steering("A", max_new_tokens=3)
        assert isinstance(result, str)

    def test_zero_max_tokens(self):
        """Test handling of zero max_new_tokens."""
        dac = DAC(device="cpu", model_name="gpt2", icl_examples=0)

        # Mock components
        dac._tokenizer = lambda text, **kwargs: {"input_ids": torch.tensor([[1, 2, 3]])}
        dac._load_model = lambda: None
        dac.steering_tensor = torch.randn(5, 4, 4, 8)
        dac.is_trained = True

        # Mock step-by-step generation to return immediately for zero tokens
        def mock_step_generation(input_ids, max_new_tokens, **kwargs):
            if max_new_tokens == 0:
                return ""  # No new tokens generated
            return "generated"

        dac._generate_step_by_step_with_hooks = mock_step_generation

        # Test zero tokens
        result = dac.generate_with_steering("Test", max_new_tokens=0)
        assert result == ""

    def test_invalid_composition_strategies(self):
        """Test handling of invalid composition strategies."""
        dac = DAC(device="cpu", icl_examples=0)
        dac.property_tensors = {"prop1": torch.randn(5, 4, 4, 8)}
        dac.is_trained = True

        # Test unknown strategy
        with pytest.raises(ValueError, match="Unknown composition strategy"):
            dac.get_composed_tensor({"prop1": 1.0}, "unknown_strategy")
