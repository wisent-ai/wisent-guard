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
    python caa_reference.py
"""

import sys
import json
import torch
from pathlib import Path

# Constants
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
LAYER_INDEX = 14
BEHAVIOR = "hallucination"
MAX_EXAMPLES = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add CAA repo to path
CAA_PATH = Path(__file__).parent.parent.parent.parent / "CAA"
sys.path.insert(0, str(CAA_PATH))

# Add our repo to path for accessing our data structures
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

# Now we can import from CAA
from llama_wrapper import LlamaWrapper
from behaviors import get_ab_data_path, get_ab_test_data
from utils.tokenize import tokenize_llama_base
from utils.helpers import get_a_b_probs


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
    model_size = "7b"
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
    output_path = (
        Path(__file__).parent.parent / "reference_data" / "vectors" / "caa" / f"{BEHAVIOR}_layer{LAYER_INDEX}.pt"
    )
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


def generate_reference_text_outputs(steering_vector, max_examples=20):
    """Generate steered and unsteered text outputs using CAA reference implementation.

    Args:
        steering_vector: The reference steering vector from generate_reference_vector()
        max_examples: Maximum number of examples to process for text generation

    Returns:
        tuple: (steered_results, unsteered_results) containing A/B probability results
    """
    print(f"🔄 Generating reference text outputs with CAA implementation...")

    # Load test dataset (different from training dataset)
    test_data = get_ab_test_data(BEHAVIOR)
    test_subset = test_data[:max_examples]
    print(f"Using {len(test_subset)} test examples")

    # Initialize model with same settings as vector generation
    print("Loading Llama-2-7B model for text generation...")
    model = LlamaWrapper(
        hf_token=None,  # Use logged in credentials
        size="7b",
        use_chat=False,  # Use base model
    )

    # Get A and B token IDs for probability extraction
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")

    # Configure model for deterministic generation
    model.set_save_internal_decodings(False)

    # Prepare steering vector
    vector = steering_vector.to(model.device)
    if "7b" != "7b":  # Keep as float16 for 7B model
        vector = vector.half()

    print("Generating STEERED outputs...")
    steered_results = []
    for i, item in enumerate(test_subset):
        if i % 10 == 0:
            print(f"  Steered: {i}/{len(test_subset)}")

        model.reset_all()

        # Apply steering vector (multiplier = 1.0)
        model.set_add_activations(LAYER_INDEX, 1.0 * vector)

        # Generate response probabilities for A vs B
        question = item["question"]
        model_output = model.get_logits_from_text(
            user_input=question,
            model_output="(",  # Generate probabilities for "(A)" vs "(B)"
            system_prompt=None,
        )

        # Extract A and B probabilities
        a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)

        result = {
            "question": question,
            "answer_matching_behavior": item["answer_matching_behavior"],
            "answer_not_matching_behavior": item["answer_not_matching_behavior"],
            "a_prob": a_prob,
            "b_prob": b_prob,
            "steered": True,
            "layer": LAYER_INDEX,
            "multiplier": 1.0,
            "model": MODEL_NAME,
        }
        steered_results.append(result)

    print("Generating UNSTEERED (baseline) outputs...")
    unsteered_results = []
    for i, item in enumerate(test_subset):
        if i % 10 == 0:
            print(f"  Unsteered: {i}/{len(test_subset)}")

        model.reset_all()

        # NO steering applied - baseline behavior

        # Generate response probabilities for A vs B
        question = item["question"]
        model_output = model.get_logits_from_text(
            user_input=question,
            model_output="(",  # Generate probabilities for "(A)" vs "(B)"
            system_prompt=None,
        )

        # Extract A and B probabilities
        a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)

        result = {
            "question": question,
            "answer_matching_behavior": item["answer_matching_behavior"],
            "answer_not_matching_behavior": item["answer_not_matching_behavior"],
            "a_prob": a_prob,
            "b_prob": b_prob,
            "steered": False,
            "model": MODEL_NAME,
        }
        unsteered_results.append(result)

    # Save steered results
    steered_path = (
        Path(__file__).parent.parent / "reference_data" / "generations" / "caa" / f"{BEHAVIOR}_steered_outputs.json"
    )
    steered_path.parent.mkdir(parents=True, exist_ok=True)

    with open(steered_path, "w") as f:
        json.dump(steered_results, f, indent=2)
    print(f"✅ Saved steered outputs to: {steered_path}")

    # Save unsteered results
    unsteered_path = (
        Path(__file__).parent.parent / "reference_data" / "generations" / "caa" / f"{BEHAVIOR}_unsteered_outputs.json"
    )

    with open(unsteered_path, "w") as f:
        json.dump(unsteered_results, f, indent=2)
    print(f"✅ Saved unsteered outputs to: {unsteered_path}")

    # Calculate and display summary statistics
    steered_b_avg = sum(r["b_prob"] for r in steered_results) / len(steered_results)
    unsteered_b_avg = sum(r["b_prob"] for r in unsteered_results) / len(unsteered_results)
    steering_effect = steered_b_avg - unsteered_b_avg

    print(f"\n📊 Summary Statistics:")
    print(f"  Steered B probability (avg):   {steered_b_avg:.3f}")
    print(f"  Unsteered B probability (avg): {unsteered_b_avg:.3f}")
    print(f"  Steering effect:               {steering_effect:.3f}")
    print(f"  Behavior: {BEHAVIOR} - B should be {'HIGHER' if steering_effect > 0 else 'LOWER'} when steered")

    return steered_results, unsteered_results


def main():
    """Main function to generate all reference data."""
    print("🚀 Starting CAA reference data generation...")

    # User should be logged in via huggingface-cli
    print("ℹ️ Using logged-in Hugging Face credentials")

    try:
        # Generate reference vector
        steering_vector, vector_data = generate_reference_vector()

        # Generate reference text outputs (Phase 2.2)
        print("\n" + "=" * 60)
        print("📝 PHASE 2.2: Generating Reference Text Outputs")
        print("=" * 60)
        steered_results, unsteered_results = generate_reference_text_outputs(steering_vector, max_examples=20)

        print("\n✅ Reference data generation complete!")
        print("📁 Generated files:")
        print(f"  - tests/steering_validation/reference_data/vectors/caa/{BEHAVIOR}_layer{LAYER_INDEX}.pt")
        print(f"  - tests/steering_validation/reference_data/generations/caa/{BEHAVIOR}_steered_outputs.json")
        print(f"  - tests/steering_validation/reference_data/generations/caa/{BEHAVIOR}_unsteered_outputs.json")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
