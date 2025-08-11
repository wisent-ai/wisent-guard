"""
Model utilities for DAC steering validation tests.

This module provides utilities for working with DAC steering vectors
using our wisent-guard implementation and comparing with reference vectors.
"""

import json
import random
import sys
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nnsight import LanguageModel

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

# No longer using wisent-guard Response classes - using original DAC approach directly

from const import (
    MODEL_NAME,
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
    TORCH_DTYPE,
)


def load_dac_datasets_for_testing():
    """Load ITA and ENG datasets for DAC testing."""
    # Load datasets
    with open(DATASET_A_PATH, "r", encoding="utf-8") as f:
        dataset_a = json.load(f)

    with open(DATASET_B_PATH, "r", encoding="utf-8") as f:
        dataset_b = json.load(f)

    # Load instructions
    with open(DATASET_A_INSTR_PATH, "r", encoding="utf-8") as f:
        instruction_a = f.read().strip()

    with open(DATASET_B_INSTR_PATH, "r", encoding="utf-8") as f:
        instruction_b = f.read().strip()

    # Convert to (input, output) tuples and limit size
    dataset_a_tuples = [(item["input"], item["output"]) for item in dataset_a[:MAX_EXAMPLES]]
    dataset_b_tuples = [(item["input"], item["output"]) for item in dataset_b[:MAX_EXAMPLES]]

    return dataset_a_tuples, dataset_b_tuples, instruction_a, instruction_b


def tokenize_dac_prompts(tokenizer, dataset, instruction, icl_examples=ICL_EXAMPLES):
    """
    Tokenize DAC prompts with ICL format.

    This mimics the original DAC tokenization approach.
    """
    tokenized_prompts = []

    # Simple approach: use first few examples as ICL, then process the rest
    for i in range(icl_examples, len(dataset), icl_examples + 1):
        # Create ICL context from previous examples
        icl_context = f"{instruction}\n\n"

        # Add ICL examples
        for j in range(max(0, i - icl_examples), i):
            if j < len(dataset):
                query, answer = dataset[j]
                icl_context += f"Q: {query}\nA: {answer}\n\n"

        # Add current query without answer
        if i < len(dataset):
            current_query, _ = dataset[i]
            full_prompt = icl_context + f"Q: {current_query}\nA:"

            # Tokenize
            tokens = tokenizer(full_prompt, return_tensors="pt")["input_ids"].squeeze(0)
            tokenized_prompts.append(tokens)

    return tokenized_prompts
