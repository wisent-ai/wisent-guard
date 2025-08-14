#!/usr/bin/env python3
"""
Generate reference data using the ORIGINAL CAA implementation.

This script uses the original CAA repository's functions directly to ensure 100%
compatibility and eliminate any potential implementation differences.

Vector Generation: Uses CAA's original generate_save_vectors_for_behavior()
Text Generation: Uses CAA's LlamaWrapper with original tokenization

Dataset Configuration:
    - For fast testing, we limit the CAA dataset to 20 examples
    - Modified: /workspace/wisent-guard/CAA/datasets/generate/hallucination/generate_dataset.json
    - This dataset is used by generate_save_vectors_for_behavior()
    - Full dataset backup: /workspace/wisent-guard/CAA/datasets/raw/hallucination/dataset_full_backup.json

Prerequisites:
    1. Clone the CAA repository: git clone https://github.com/nrimsky/CAA
    2. Place it adjacent to this repo (or update CAA_PATH below)
    3. Login to Hugging Face: huggingface-cli login
    4. Ensure CAA dataset is limited to 20 examples for fast testing

Usage:
    python generate_data_with_original_caa_implementation.py

Output: ./reference_data/
- hallucination_layer14.pt (steering vector)
- text_completions_steered.json (steered text outputs)
- text_completions_unsteered.json (unsteered text outputs)
"""

import json
import sys
from pathlib import Path

import torch
from const import (
    BEHAVIOR,
    CAA_PATH,
    HALLUCINATION_VECTOR_PATH,
    LAYER_INDEX,
    MAX_EXAMPLES,
    MAX_NEW_TOKENS,
    MAX_TEXT_EXAMPLES,
    MODEL_SIZE,
    RANDOM_SEED,
    STEERING_STRENGTH,
    TOP_K,
    WISENT_PATH,
)

# Add CAA repo to path
sys.path.insert(0, str(CAA_PATH))

# Add our repo to path for accessing our data structures
sys.path.insert(0, str(WISENT_PATH))

try:
    import shutil

    from behaviors import get_ab_data_path, get_vector_path
    from generate_vectors import generate_save_vectors_for_behavior
    from llama_wrapper import LlamaWrapper
    from utils.helpers import find_instruction_end_postion
    from utils.tokenize import tokenize_llama_base
except ImportError:
    print("You have to clone original CAA repo https://github.com/nrimsky/CAA first")


def generate_reference_vector(max_examples=MAX_EXAMPLES):
    """Generate CAA steering vector using original CAA implementation.

    Args:
        max_examples: Maximum number of dataset examples to use (ignored - uses full dataset)

    Returns:
        tuple: (steering_vector, save_data) containing the generated vector and metadata
    """
    print("🔄 Generating reference CAA vector using ORIGINAL CAA implementation...")
    print(f"Behavior: {BEHAVIOR}, Layer: {LAYER_INDEX}, Model: {MODEL_SIZE}")

    # Initialize model using CAA's approach
    print("Loading Llama-2-7B model with CAA wrapper...")
    model = LlamaWrapper(
        hf_token=None,  # Use logged in credentials
        size=MODEL_SIZE,
        use_chat=False,  # Use base model
    )

    # Use CAA's original vector generation function
    print("🚀 Using original CAA generate_save_vectors_for_behavior()...")
    generate_save_vectors_for_behavior(
        layers=[LAYER_INDEX],
        save_activations=True,  # Save activations for our metadata
        behavior=BEHAVIOR,
        model=model,
    )

    # Get CAA's output path and copy to our desired location
    caa_vector_path = get_vector_path(BEHAVIOR, LAYER_INDEX, model.model_name_path)
    print(f"CAA generated vector at: {caa_vector_path}")

    # Load the generated vector
    if not Path(caa_vector_path).exists():
        raise FileNotFoundError(f"CAA failed to generate vector at {caa_vector_path}")

    steering_vector = torch.load(caa_vector_path, map_location="cpu", weights_only=False)
    print(f"Loaded CAA vector with shape: {steering_vector.shape}")
    print(f"Vector norm: {torch.norm(steering_vector).item():.4f}")

    # Create our save data structure for compatibility
    save_data = {
        "vector": steering_vector,
        "layer": LAYER_INDEX,
        "behavior": BEHAVIOR,
        "model_size": MODEL_SIZE,
        "use_base_model": True,
        "method": "CAA_original_implementation",
        "source_path": str(caa_vector_path),
        "metadata": {
            "vector_norm": torch.norm(steering_vector).item(),
        },
    }

    # Copy to our target location
    HALLUCINATION_VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(caa_vector_path, HALLUCINATION_VECTOR_PATH)
    print(f"✅ Copied CAA vector to: {HALLUCINATION_VECTOR_PATH}")

    # Also save our metadata
    torch.save(save_data, HALLUCINATION_VECTOR_PATH)
    print("✅ Saved reference vector with metadata")

    return steering_vector, save_data


def generate_text_completions(steering_vector, max_examples=MAX_TEXT_EXAMPLES):
    """Generate full text completions using CAA reference implementation.

    Args:
        steering_vector: The reference steering vector from generate_reference_vector()
        max_examples: Maximum number of examples to process for text generation

    Returns:
        tuple: (steered_results, unsteered_results) containing full text completions
    """
    print("🔄 Generating reference text completions...")

    # Load test dataset - use the same hallucination dataset
    dataset_path = Path(__file__).parent / "reference_data" / "hallucination.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    test_subset = dataset[:max_examples]
    print(f"Using {len(test_subset)} examples for text generation")

    # Initialize model with same settings as vector generation
    print("Loading Llama-2-7B model for text generation...")
    model = LlamaWrapper(hf_token=None, size=MODEL_SIZE, use_chat=False)
    steering_vector = steering_vector.to(model.device)

    # Generate steered outputs
    print("Generating STEERED text completions...")
    steered_results = []

    for i, item in enumerate(test_subset):
        print(f"  Processing steered {i + 1}/{len(test_subset)}")

        model.reset_all()

        # Apply steering
        model.set_add_activations(LAYER_INDEX, STEERING_STRENGTH * steering_vector)

        # Tokenize exactly like CAA
        user_input = item["question"]
        prompt_tokens = tokenize_llama_base(model.tokenizer, user_input)
        tokens = torch.tensor(prompt_tokens).unsqueeze(0).to(model.device)

        # Set random seed for reproducibility
        torch.manual_seed(RANDOM_SEED)

        # Find instruction position and set from_positions
        instr_pos = find_instruction_end_postion(tokens[0], model.END_STR)
        model.set_from_positions(instr_pos)

        # Generate text
        with torch.no_grad():
            generated_ids = model.model.generate(
                inputs=tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                top_k=TOP_K,  # Greedy decoding
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
            "strength": STEERING_STRENGTH,
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
        torch.manual_seed(RANDOM_SEED)

        # Generate text (no steering)
        with torch.no_grad():
            generated_ids = model.model.generate(
                inputs=tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                top_k=TOP_K,  # Greedy decoding
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
    steered_path = Path(__file__).parent / "reference_data" / "text_completions_steered.json"
    steered_path.parent.mkdir(parents=True, exist_ok=True)

    with open(steered_path, "w") as f:
        json.dump(steered_results, f, indent=2)
    print(f"✅ Saved steered text completions to: {steered_path}")

    unsteered_path = Path(__file__).parent / "reference_data" / "text_completions_unsteered.json"

    with open(unsteered_path, "w") as f:
        json.dump(unsteered_results, f, indent=2)
    print(f"✅ Saved unsteered text completions to: {unsteered_path}")

    return steered_results, unsteered_results


def main():
    """Main function to generate all reference data."""
    print("🚀 Starting CAA reference data generation...")

    # User should be logged in via huggingface-cli
    print("ℹ️ Using logged-in Hugging Face credentials")

    try:
        # Generate reference vector
        steering_vector, vector_data = generate_reference_vector()

        print("\n" + "=" * 60)
        print("📝 PHASE 2: Generating Reference Text Completions")
        print("=" * 60)
        text_steered_results, text_unsteered_results = generate_text_completions(
            steering_vector, max_examples=MAX_TEXT_EXAMPLES
        )

        print("\n✅ Reference data generation complete!")
        print("📁 Generated files:")
        print(f"  - {HALLUCINATION_VECTOR_PATH}")
        print("  - tests/steering_validation/caa/reference_data/text_completions_steered.json")
        print("  - tests/steering_validation/caa/reference_data/text_completions_unsteered.json")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
