#!/usr/bin/env python3
"""
Generate full text outputs for generation consistency testing.

This extends the CAA reference data by generating actual text completions
that we can compare token-by-token.
"""

import json
import sys
from pathlib import Path

import torch

# Constants
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
LAYER_INDEX = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add CAA repo to path
CAA_PATH = Path(__file__).parent.parent.parent.parent / "CAA"
sys.path.insert(0, str(CAA_PATH))

# Import CAA utilities
from llama_wrapper import LlamaWrapper
from utils.helpers import find_instruction_end_postion
from utils.tokenize import tokenize_llama_base


def generate_text_completions(max_examples=5):
    """Generate full text completions using CAA reference implementation."""
    print("🔄 Generating reference text completions...")

    # Load test dataset
    dataset_path = Path(__file__).parent.parent / "reference_data" / "datasets" / "hallucination.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    test_subset = dataset[:max_examples]
    print(f"Using {len(test_subset)} examples for text generation")

    # Load steering vector
    vector_path = Path(__file__).parent.parent / "reference_data" / "vectors" / "caa" / "hallucination_layer14.pt"
    vector_data = torch.load(vector_path, map_location="cpu", weights_only=False)
    steering_vector = vector_data["vector"]
    print(f"Loaded steering vector with norm: {torch.norm(steering_vector).item():.4f}")

    # Initialize model
    print("Loading Llama-2-7B model for text generation...")
    model = LlamaWrapper(hf_token=None, size="7b", use_chat=False)
    steering_vector = steering_vector.to(model.device)

    # Generate steered outputs
    print("Generating STEERED text completions...")
    steered_results = []

    for i, item in enumerate(test_subset):
        print(f"  Processing steered {i + 1}/{len(test_subset)}")

        model.reset_all()

        # Apply steering
        model.set_add_activations(LAYER_INDEX, 1.0 * steering_vector)

        # Tokenize exactly like CAA
        user_input = item["question"]
        prompt_tokens = tokenize_llama_base(model.tokenizer, user_input)
        tokens = torch.tensor(prompt_tokens).unsqueeze(0).to(model.device)

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Find instruction position and set from_positions
        instr_pos = find_instruction_end_postion(tokens[0], model.END_STR)
        model.set_from_positions(instr_pos)

        # Generate text
        with torch.no_grad():
            generated_ids = model.model.generate(
                inputs=tokens,
                max_new_tokens=50,
                top_k=1,  # Greedy decoding
            )

            # Decode generated text
            generated_text = model.tokenizer.batch_decode(generated_ids)[0]

        # Calculate prompt and response parts
        prompt_str = f"Input: {user_input}\nResponse:"
        response_part = (
            generated_text[len(prompt_str) :].strip() if prompt_str in generated_text else generated_text.strip()
        )

        result = {
            "prompt": prompt_str,
            "generated_full": generated_text,
            "generated_response": response_part,
            "tokens": generated_ids[0].tolist(),
            "question": user_input,
            "steered": True,
            "strength": 1.0,
        }
        steered_results.append(result)

    # Generate unsteered outputs
    print("Generating UNSTEERED text completions...")
    unsteered_results = []

    for i, item in enumerate(test_subset):
        print(f"  Processing unsteered {i + 1}/{len(test_subset)}")

        model.reset_all()
        # NO steering applied

        # Tokenize exactly like CAA
        user_input = item["question"]
        prompt_tokens = tokenize_llama_base(model.tokenizer, user_input)
        tokens = torch.tensor(prompt_tokens).unsqueeze(0).to(model.device)

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Generate text (no steering)
        with torch.no_grad():
            generated_ids = model.model.generate(
                inputs=tokens,
                max_new_tokens=50,
                top_k=1,  # Greedy decoding
            )

            # Decode generated text
            generated_text = model.tokenizer.batch_decode(generated_ids)[0]

        # Calculate prompt and response parts
        prompt_str = f"Input: {user_input}\nResponse:"
        response_part = (
            generated_text[len(prompt_str) :].strip() if prompt_str in generated_text else generated_text.strip()
        )

        result = {
            "prompt": prompt_str,
            "generated_full": generated_text,
            "generated_response": response_part,
            "tokens": generated_ids[0].tolist(),
            "question": user_input,
            "steered": False,
            "strength": 0.0,
        }
        unsteered_results.append(result)

    # Save results
    steered_path = (
        Path(__file__).parent.parent / "reference_data" / "generations" / "caa" / "text_completions_steered.json"
    )
    with open(steered_path, "w") as f:
        json.dump(steered_results, f, indent=2)
    print(f"✅ Saved steered text completions to: {steered_path}")

    unsteered_path = (
        Path(__file__).parent.parent / "reference_data" / "generations" / "caa" / "text_completions_unsteered.json"
    )
    with open(unsteered_path, "w") as f:
        json.dump(unsteered_results, f, indent=2)
    print(f"✅ Saved unsteered text completions to: {unsteered_path}")

    return steered_results, unsteered_results


if __name__ == "__main__":
    print("🚀 Generating reference text completions...")
    generate_text_completions()
    print("✅ Text completion generation complete!")
