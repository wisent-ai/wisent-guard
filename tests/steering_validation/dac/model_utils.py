"""
Model utilities for DAC steering validation.

This module provides utilities for working with the original DAC implementation,
including dataset loading, tokenization, and activation extraction.
"""

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
