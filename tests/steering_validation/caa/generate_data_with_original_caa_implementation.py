#!/usr/bin/env python3
"""
Generate reference data using the CAA implementation.

This script generates steering vectors and text outputs using the reference CAA
implementation, which we'll use as ground truth for validating our implementation.

Prerequisites:
    1. Clone the CAA repository: git clone https://github.com/nrimsky/CAA
    2. Place it adjacent to this repo (or update CAA_PATH below)
    3. Login to Hugging Face: huggingface-cli login

Usage:
    python generate_data_with_original_caa_implementation.py

The output you can change in ./reference_data
There is only one difference - the created data were used with model loaded with dtype=torch.float16
"""

import sys
import json
import torch
from pathlib import Path

from const import (
    MODEL_NAME,
    MODEL_SIZE,
    LAYER_INDEX,
    MAX_EXAMPLES,
    DEVICE,
    BEHAVIOR,
    STEERING_STRENGTH,
    MAX_NEW_TOKENS,
    TOP_K,
    RANDOM_SEED,
    MAX_TEXT_EXAMPLES,
    CAA_PATH,
    WISENT_PATH,
    HALLUCINATION_VECTOR_PATH,
)

# Add CAA repo to path
sys.path.insert(0, str(CAA_PATH))

# Add our repo to path for accessing our data structures
sys.path.insert(0, str(WISENT_PATH))

try:
    from llama_wrapper import LlamaWrapper
    from behaviors import get_ab_data_path
    from utils.tokenize import tokenize_llama_base
    from utils.helpers import find_instruction_end_postion
except ImportError:
    print("You have to clone original CAA repo https://github.com/nrimsky/CAA first")


def generate_reference_vector(max_examples=MAX_EXAMPLES):
    """Generate CAA steering vector using reference implementation.

    Args:
        max_examples: Maximum number of dataset examples to use

    Returns:
        tuple: (steering_vector, save_data) containing the generated vector and metadata
    """
    print(f"🔄 Generating reference CAA vector for {BEHAVIOR} behavior...")

    # Parameters matching our test setup
    behavior = BEHAVIOR
    layer = LAYER_INDEX
    model_size = MODEL_SIZE
    use_base_model = True  # Use base model (non-chat)

    # Initialize model (user is logged in via huggingface-cli)
    print("Loading Llama-2-7B model...")
    model = LlamaWrapper(
        hf_token=None,  # Use logged in credentials
        size=model_size,
        use_chat=not use_base_model,
    )

    # Load dataset
    data_path = get_ab_data_path(behavior)
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Use limited examples to fit in memory
    dataset = dataset[:max_examples]
    print(f"Using {len(dataset)} examples from {data_path} (limited for memory)")

    # Generate activations
    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = []
    neg_activations = []

    print(f"Extracting activations at layer {layer}...")
    for i, item in enumerate(dataset):
        if i % 10 == 0:
            print(f"Processing {i}/{len(dataset)}")

        # Get positive and negative texts
        question = item["question"]
        pos_text = item["answer_matching_behavior"]
        neg_text = item["answer_not_matching_behavior"]

        # Tokenize
        tokenizer = model.tokenizer
        pos_tokens = tokenize_llama_base(tokenizer, question, pos_text)
        neg_tokens = tokenize_llama_base(tokenizer, question, neg_text)

        pos_tokens = torch.tensor(pos_tokens).unsqueeze(0).to(model.device)
        neg_tokens = torch.tensor(neg_tokens).unsqueeze(0).to(model.device)

        # Extract positive activations
        model.reset_all()
        with torch.no_grad():
            model.get_logits(pos_tokens)
            pos_acts = model.get_last_activations(layer)
            pos_acts = pos_acts[0, -2, :].detach().cpu()  # Position -2
            pos_activations.append(pos_acts)

        # Extract negative activations
        model.reset_all()
        with torch.no_grad():
            model.get_logits(neg_tokens)
            neg_acts = model.get_last_activations(layer)
            neg_acts = neg_acts[0, -2, :].detach().cpu()  # Position -2
            neg_activations.append(neg_acts)

    # Compute steering vector (reference method)
    print("Computing steering vector...")
    pos_stack = torch.stack(pos_activations)
    neg_stack = torch.stack(neg_activations)
    steering_vector = (pos_stack - neg_stack).mean(dim=0)

    print(f"Generated steering vector with shape: {steering_vector.shape}")
    print(f"Vector norm: {torch.norm(steering_vector).item():.4f}")

    # Save vector
    output_path = HALLUCINATION_VECTOR_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "vector": steering_vector,
        "layer": layer,
        "behavior": behavior,
        "model_size": model_size,
        "use_base_model": use_base_model,
        "num_examples": len(dataset),
        "method": "CAA_reference_real",
        "pos_activations": pos_stack,
        "neg_activations": neg_stack,
        "metadata": {
            "vector_norm": torch.norm(steering_vector).item(),
            "pos_mean_norm": torch.norm(pos_stack, dim=-1).mean().item(),
            "neg_mean_norm": torch.norm(neg_stack, dim=-1).mean().item(),
        },
    }

    torch.save(save_data, output_path)
    print(f"✅ Saved reference vector to: {output_path}")

    return steering_vector, save_data


def generate_text_completions(steering_vector, max_examples=MAX_TEXT_EXAMPLES):
    """Generate full text completions using CAA reference implementation.

    Args:
        steering_vector: The reference steering vector from generate_reference_vector()
        max_examples: Maximum number of examples to process for text generation

    Returns:
        tuple: (steered_results, unsteered_results) containing full text completions
    """
    print(f"🔄 Generating reference text completions...")

    # Load test dataset - use the same hallucination dataset
    dataset_path = Path(__file__).parent / "reference_data" / "hallucination.json"
    with open(dataset_path, "r") as f:
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
        print(f"  - tests/steering_validation/caa/reference_data/text_completions_steered.json")
        print(f"  - tests/steering_validation/caa/reference_data/text_completions_unsteered.json")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
