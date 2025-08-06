#!/usr/bin/env python3
"""
Generate A/B probability reference data for validation tests.

This generates the A/B probability data that tests need for steering validation
without requiring CAA repo dependencies.
"""

import sys
import json
import torch
from pathlib import Path

# Constants
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
LAYER_INDEX = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add CAA repo to path
CAA_PATH = Path(__file__).parent.parent.parent.parent / "CAA"
sys.path.insert(0, str(CAA_PATH))

# Import CAA utilities
from llama_wrapper import LlamaWrapper
from utils.helpers import get_a_b_probs


def generate_ab_probability_data():
    """Generate A/B probability reference data."""
    print("🔄 Generating A/B probability reference data...")

    # Load existing steered/unsteered outputs
    steered_path = (
        Path(__file__).parent.parent / "reference_data" / "generations" / "caa" / "hallucination_steered_outputs.json"
    )
    unsteered_path = (
        Path(__file__).parent.parent / "reference_data" / "generations" / "caa" / "hallucination_unsteered_outputs.json"
    )

    with open(steered_path, "r") as f:
        steered_data = json.load(f)
    with open(unsteered_path, "r") as f:
        unsteered_data = json.load(f)

    # Create combined reference data for A/B probability tests
    ab_reference_data = {
        "steered_a_probs": [item["a_prob"] for item in steered_data],
        "steered_b_probs": [item["b_prob"] for item in steered_data],
        "unsteered_a_probs": [item["a_prob"] for item in unsteered_data],
        "unsteered_b_probs": [item["b_prob"] for item in unsteered_data],
        "questions": [item["question"] for item in steered_data],
        "metadata": {
            "model": MODEL_NAME,
            "layer": LAYER_INDEX,
            "num_examples": len(steered_data),
            "steering_effect": sum(
                steered_data[i]["b_prob"] - unsteered_data[i]["b_prob"] for i in range(len(steered_data))
            )
            / len(steered_data),
        },
    }

    # Save combined A/B probability data
    ab_path = Path(__file__).parent.parent / "reference_data" / "generations" / "caa" / "ab_probabilities.json"
    with open(ab_path, "w") as f:
        json.dump(ab_reference_data, f, indent=2)

    print(f"✅ Saved A/B probability reference data to: {ab_path}")
    print(f"   Steering effect: {ab_reference_data['metadata']['steering_effect']:.3f}")

    return ab_reference_data


if __name__ == "__main__":
    print("🚀 Generating A/B probability reference data...")
    generate_ab_probability_data()
    print("✅ A/B probability generation complete!")
