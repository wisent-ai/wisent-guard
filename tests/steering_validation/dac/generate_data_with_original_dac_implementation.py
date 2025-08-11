#!/usr/bin/env python3
"""
Generate reference DAC steering vectors using the original DAC implementation.

This script creates reference steering vectors by:
1. Loading ITA (Italian) and ENG (English) datasets
2. Computing mean activations for each dataset using the original DAC implementation
3. Computing difference vectors (ITA - ENG) for language steering
4. Saving the vectors for validation testing

Dataset Configuration:
    - For fast testing, we limit both ITA and ENG datasets to 20 examples each
    - Datasets used: ITA_train.json (Italian responses) and ENG_train.json (English responses)
    - Original datasets: /workspace/wisent-guard/Dynamic-Activation-Composition/data/dataset/
    - Reference data: /workspace/wisent-guard/tests/steering_validation/dac/reference_data/

Usage:
    python generate_data_with_original_dac_implementation.py

Expected Output:
    - mean_activations_ita.pt: Mean activations for Italian responses (all layers)
    - mean_activations_eng.pt: Mean activations for English responses (all layers)
    - diff_activations_ita_eng.pt: Difference vector (ITA - ENG) for steering across all layers
"""

import sys
import time
from pathlib import Path

import torch

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from const import (
    MODEL_NAME,
    ACTIVATIONS_A_PATH,
    ACTIVATIONS_B_PATH,
    DIFF_ACTIVATIONS_PATH,
    DATASET_A_NAME,
    DATASET_B_NAME,
    MAX_EXAMPLES,
    REFERENCE_DATA_PATH,
)


import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from const import (
    DATASET_A_NAME,
    DATASET_B_NAME,
    DATASET_A_PATH,
    DATASET_B_PATH,
    DATASET_A_INSTR_PATH,
    DATASET_B_INSTR_PATH,
    MAX_EXAMPLES,
    ICL_EXAMPLES,
    MAX_NEW_TOKENS,
    MODEL_CONFIG,
)

# Add the Dynamic-Activation-Composition directory to the path so we can import from it
DAC_DIR = Path(__file__).parent.parent.parent.parent / "Dynamic-Activation-Composition"
sys.path.insert(0, str(DAC_DIR))

try:
    from src.utils.model_utils import load_model_and_tokenizer, set_seed
    from src.extraction import get_mean_activations
    from src.utils.prompt_helper import tokenize_ICL
except ImportError as e:
    print(f"Error importing DAC modules: {e}")
    print(f"Make sure the Dynamic-Activation-Composition directory exists at: {DAC_DIR}")
    raise


def load_dac_dataset(dataset_name: str, dataset_path: Path, instr_path: Path) -> Tuple[List[Dict], str]:
    """
    Load DAC dataset and instruction.

    Args:
        dataset_name: Name of the dataset (e.g., "ITA", "ENG")
        dataset_path: Path to the dataset JSON file
        instr_path: Path to the instruction text file

    Returns:
        Tuple of (dataset_list, instruction_string)
    """
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Load instruction
    with open(instr_path, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    print(f"Loaded {dataset_name} dataset: {len(dataset)} examples")
    return dataset, instruction


def create_dac_datasets() -> Tuple[List[Tuple], List[Tuple], str, str]:
    """
    Create DAC datasets for both languages, limited to MAX_EXAMPLES.

    Returns:
        Tuple of (dataset_a, dataset_b, instruction_a, instruction_b)
        where datasets are lists of (input, output) tuples
    """
    # Load datasets
    dataset_a_dict, instruction_a = load_dac_dataset(DATASET_A_NAME, DATASET_A_PATH, DATASET_A_INSTR_PATH)
    dataset_b_dict, instruction_b = load_dac_dataset(DATASET_B_NAME, DATASET_B_PATH, DATASET_B_INSTR_PATH)

    # Convert to tuples and limit size
    dataset_a = [(item["input"], item["output"]) for item in dataset_a_dict[:MAX_EXAMPLES]]
    dataset_b = [(item["input"], item["output"]) for item in dataset_b_dict[:MAX_EXAMPLES]]

    print(f"Using {len(dataset_a)} examples from {DATASET_A_NAME} dataset")
    print(f"Using {len(dataset_b)} examples from {DATASET_B_NAME} dataset")

    return dataset_a, dataset_b, instruction_a, instruction_b


def create_dac_original_mean_activations(
    model, tokenizer, config, device, dataset_a, dataset_b, instruction_a, instruction_b
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create mean activations using the original DAC implementation.

    Args:
        model: The language model
        tokenizer: The tokenizer
        config: Model configuration dict
        device: Device to run on
        dataset_a: Dataset A (e.g., Italian responses)
        dataset_b: Dataset B (e.g., English responses)
        instruction_a: Instruction for dataset A
        instruction_b: Instruction for dataset B

    Returns:
        Tuple of (mean_activations_a, mean_activations_b, diff_activations)
    """
    print(f"Processing {len(dataset_a)} examples for mean activations (limited for fast testing)...")

    # Set seed for reproducibility
    set_seed(32)

    # Generate prompts from both datasets using DAC's tokenize_ICL
    print("[1/4] Tokenizing dataset A...")
    tokenized_dict_a = tokenize_ICL(
        tokenizer,
        ICL_examples=ICL_EXAMPLES,
        dataset=dataset_a,
        pre_append_instruction=instruction_a,
    )
    icl_tokens_a = tokenized_dict_a["tokenized_prompts"]

    print("[2/4] Tokenizing dataset B...")
    tokenized_dict_b = tokenize_ICL(
        tokenizer,
        ICL_examples=ICL_EXAMPLES,
        dataset=dataset_b,
        pre_append_instruction=instruction_b,
    )
    icl_tokens_b = tokenized_dict_b["tokenized_prompts"]

    # Use random indices to ensure we're getting comparable examples
    num_examples = min(len(icl_tokens_a), len(icl_tokens_b))
    random_indexes = [random.randint(0, num_examples - 1) for _ in range(num_examples)]

    random_icl_tokens_a = [icl_tokens_a[i] for i in random_indexes]
    random_icl_tokens_b = [icl_tokens_b[i] for i in random_indexes]

    print("[3/4] Computing mean activations for dataset A...")
    mean_activations_a = get_mean_activations(
        tokenized_prompts=random_icl_tokens_a,
        tokenizer=tokenizer,
        model=model,
        config=config,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print("[4/4] Computing mean activations for dataset B...")
    mean_activations_b = get_mean_activations(
        tokenized_prompts=random_icl_tokens_b,
        tokenizer=tokenizer,
        model=model,
        config=config,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # Compute difference (A - B)
    diff_activations = mean_activations_a - mean_activations_b
    print(f"Computed difference activations with shape: {diff_activations.shape}")

    return mean_activations_a, mean_activations_b, diff_activations


def setup_dac_model(model_name: str, device: str = None) -> Tuple:
    """
    Set up the DAC model using the original implementation.

    Args:
        model_name: Name of the model to load
        device: Device to load model on (optional)

    Returns:
        Tuple of (model, tokenizer, config, device)
    """
    print(f"Loading model: {model_name}")
    model, tokenizer, config, device = load_model_and_tokenizer(
        model_name=model_name,
        load_in_8bit=True,  # Use 8-bit loading to reduce memory usage
    )

    # Disable gradients for inference
    torch.set_grad_enabled(False)

    print(f"Model loaded on device: {device}")
    print(f"Model config: {config}")

    return model, tokenizer, config, device


def validate_vector_shapes(mean_a: torch.Tensor, mean_b: torch.Tensor, diff: torch.Tensor):
    """
    Validate that the computed vectors have expected shapes.

    Args:
        mean_a: Mean activations for dataset A
        mean_b: Mean activations for dataset B
        diff: Difference activations (A - B)
    """
    expected_shape = (MAX_NEW_TOKENS, MODEL_CONFIG["n_layers"], MODEL_CONFIG["n_heads"], MODEL_CONFIG["d_head"])

    assert mean_a.shape == expected_shape, f"Mean A shape {mean_a.shape} != expected {expected_shape}"
    assert mean_b.shape == expected_shape, f"Mean B shape {mean_b.shape} != expected {expected_shape}"
    assert diff.shape == expected_shape, f"Diff shape {diff.shape} != expected {expected_shape}"

    print(f"✓ All vectors have correct shape: {expected_shape}")


def main():
    """
    Generate DAC reference vectors using the original implementation.
    """
    start_time = time.time()

    print("=" * 70)
    print("DAC Reference Vector Generation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Datasets: {DATASET_A_NAME} vs {DATASET_B_NAME}")
    print(f"Examples per dataset: {MAX_EXAMPLES}")
    print(f"DAC operates on: ALL layers simultaneously")
    print(f"Output directory: {REFERENCE_DATA_PATH}")
    print("=" * 70)

    # Check if reference data already exists
    if ACTIVATIONS_A_PATH.exists() and ACTIVATIONS_B_PATH.exists() and DIFF_ACTIVATIONS_PATH.exists():
        print("\n⚠️  Reference vectors already exist!")
        print(f"   - {ACTIVATIONS_A_PATH.name}")
        print(f"   - {ACTIVATIONS_B_PATH.name}")
        print(f"   - {DIFF_ACTIVATIONS_PATH.name}")

        response = input("\nDo you want to regenerate them? (y/N): ").strip().lower()
        if response != "y":
            print("Using existing reference vectors.")
            return

    try:
        # Step 1: Load datasets
        print("\n[1/5] Loading datasets...")
        dataset_a, dataset_b, instruction_a, instruction_b = create_dac_datasets()
        print(f"   Dataset A ({DATASET_A_NAME}): {len(dataset_a)} examples")
        print(f"   Dataset B ({DATASET_B_NAME}): {len(dataset_b)} examples")
        print(f"   Instruction A: '{instruction_a}'")
        print(f"   Instruction B: '{instruction_b}'")

        # Step 2: Setup model
        print(f"\n[2/5] Setting up model...")
        model, tokenizer, config, device = setup_dac_model(MODEL_NAME)

        # Step 3: Generate activations using original DAC implementation
        print(f"\n[3/5] Computing mean activations with original DAC implementation...")
        mean_activations_a, mean_activations_b, diff_activations = create_dac_original_mean_activations(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            instruction_a=instruction_a,
            instruction_b=instruction_b,
        )

        # Step 4: Validate shapes
        print(f"\n[4/5] Validating vector shapes...")
        validate_vector_shapes(mean_activations_a, mean_activations_b, diff_activations)

        # Step 5: Save vectors
        print(f"\n[5/5] Saving reference vectors...")

        # Ensure output directory exists
        REFERENCE_DATA_PATH.mkdir(parents=True, exist_ok=True)

        # Save all three vectors
        torch.save(mean_activations_a, ACTIVATIONS_A_PATH)
        print(f"   ✓ Saved mean activations A: {ACTIVATIONS_A_PATH}")

        torch.save(mean_activations_b, ACTIVATIONS_B_PATH)
        print(f"   ✓ Saved mean activations B: {ACTIVATIONS_B_PATH}")

        torch.save(diff_activations, DIFF_ACTIVATIONS_PATH)
        print(f"   ✓ Saved difference activations: {DIFF_ACTIVATIONS_PATH}")

        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("SUCCESS: DAC reference vectors generated!")
        print("=" * 70)
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Mean activations A shape: {mean_activations_a.shape}")
        print(f"Mean activations B shape: {mean_activations_b.shape}")
        print(f"Difference activations shape: {diff_activations.shape}")
        print(f"Vector norm (A): {torch.norm(mean_activations_a).item():.4f}")
        print(f"Vector norm (B): {torch.norm(mean_activations_b).item():.4f}")
        print(f"Difference norm: {torch.norm(diff_activations).item():.4f}")
        print("\nFiles saved:")
        print(f"   - {ACTIVATIONS_A_PATH}")
        print(f"   - {ACTIVATIONS_B_PATH}")
        print(f"   - {DIFF_ACTIVATIONS_PATH}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during vector generation: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
