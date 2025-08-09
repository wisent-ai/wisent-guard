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
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

from wisent_guard.core.steering_methods.dac import DAC
from wisent_guard.core.aggregation import ControlVectorAggregationMethod
from wisent_guard.core.contrastive_pairs import ContrastivePairSet, ContrastivePair
from wisent_guard.core.response import PositiveResponse, NegativeResponse

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


class DACModelWrapper:
    """Wrapper for real model to generate DAC steering vectors using wisent-guard implementation."""

    def __init__(self, model_name=MODEL_NAME, device="auto"):
        self.model_name = model_name
        self.device = device

        print(f"Loading {model_name} for DAC vector generation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=TORCH_DTYPE, device_map=device, trust_remote_code=True
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ DAC Model loaded on device: {self.model.device}")

    def get_activations_all_layers(self, prompts_list, max_new_tokens=MAX_NEW_TOKENS, position=-1):
        """
        Extract activations from ALL layers for DAC (not just one layer like CAA).

        Args:
            prompts_list: List of tokenized prompts
            max_new_tokens: Maximum tokens to generate
            position: Position to extract from each generation step (-1 = last token)

        Returns:
            Dict mapping layer_idx to list of activations for that layer
        """
        all_layer_activations = {i: [] for i in range(MODEL_CONFIG["n_layers"])}

        for prompt_tokens in prompts_list:
            # Convert to tensor if needed
            if isinstance(prompt_tokens, list):
                input_ids = torch.tensor(prompt_tokens).unsqueeze(0).to(self.model.device)
            else:
                input_ids = prompt_tokens.to(self.model.device)

            # Generate tokens and collect activations at each step
            step_activations = {i: [] for i in range(MODEL_CONFIG["n_layers"])}

            for step in range(max_new_tokens):
                layer_outputs = {}

                def create_hook(layer_idx):
                    def hook_fn(module, input, output):
                        layer_outputs[layer_idx] = output[0]  # Hidden states

                    return hook_fn

                # Register hooks for all layers
                hooks = []
                for layer_idx in range(MODEL_CONFIG["n_layers"]):
                    layer = self.model.model.layers[layer_idx]
                    hook = layer.register_forward_hook(create_hook(layer_idx))
                    hooks.append(hook)

                try:
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids)
                        logits = outputs.logits

                    # Get next token
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                    # Extract activations from all layers at the last token position
                    for layer_idx in range(MODEL_CONFIG["n_layers"]):
                        if layer_idx in layer_outputs:
                            hidden_states = layer_outputs[layer_idx]  # [batch, seq, hidden]
                            # Take activation from last token position
                            activation = hidden_states[0, -1, :].cpu()  # [hidden_dim]
                            step_activations[layer_idx].append(activation)

                    # Update input for next step
                    input_ids = torch.cat([input_ids, next_token], dim=-1)

                    # Stop on EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                finally:
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()

            # Convert step activations to proper format [steps, hidden_dim] for each layer
            for layer_idx in range(MODEL_CONFIG["n_layers"]):
                if step_activations[layer_idx]:
                    layer_tensor = torch.stack(step_activations[layer_idx])  # [steps, hidden_dim]
                    all_layer_activations[layer_idx].append(layer_tensor)

        return all_layer_activations


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


def create_dac_vectors_with_wisent_guard(model_wrapper):
    """
    Create DAC vectors using our wisent-guard implementation.

    This creates contrastive pairs from ITA vs ENG datasets and trains DAC.
    """
    # Load datasets
    dataset_a, dataset_b, instruction_a, instruction_b = load_dac_datasets_for_testing()

    # Tokenize prompts for both datasets
    prompts_a = tokenize_dac_prompts(model_wrapper.tokenizer, dataset_a, instruction_a)
    prompts_b = tokenize_dac_prompts(model_wrapper.tokenizer, dataset_b, instruction_b)

    # Take same number of prompts from both datasets
    min_len = min(len(prompts_a), len(prompts_b))
    prompts_a = prompts_a[:min_len]
    prompts_b = prompts_b[:min_len]

    print(f"Processing {min_len} prompt pairs for DAC vector generation...")

    # Get activations for both datasets from all layers
    print("Extracting activations for dataset A (ITA)...")
    activations_a = model_wrapper.get_activations_all_layers(prompts_a)

    print("Extracting activations for dataset B (ENG)...")
    activations_b = model_wrapper.get_activations_all_layers(prompts_b)

    # Create contrastive pairs by treating ITA as positive, ENG as negative
    pairs = []

    for i in range(min_len):
        # For each layer, create activations dict
        pos_activations = {}
        neg_activations = {}

        for layer_idx in range(MODEL_CONFIG["n_layers"]):
            if (
                layer_idx in activations_a
                and i < len(activations_a[layer_idx])
                and layer_idx in activations_b
                and i < len(activations_b[layer_idx])
            ):
                pos_activations[layer_idx] = activations_a[layer_idx][i]  # ITA response (positive)
                neg_activations[layer_idx] = activations_b[layer_idx][i]  # ENG response (negative)

        if pos_activations and neg_activations:
            pos_response = PositiveResponse(
                prompt=f"Example {i + 1}", response="ITA response", activations=pos_activations
            )
            neg_response = NegativeResponse(
                prompt=f"Example {i + 1}", response="ENG response", activations=neg_activations
            )

            pair = ContrastivePair(positive_response=pos_response, negative_response=neg_response)
            pairs.append(pair)

    # Create contrastive pair set
    pair_set = ContrastivePairSet(name=f"{DATASET_A_NAME}_vs_{DATASET_B_NAME}", pairs=pairs)

    print(f"Created {len(pairs)} contrastive pairs")

    # Train DAC using our implementation
    dac = DAC(
        device=model_wrapper.model.device,
        aggregation_method=ControlVectorAggregationMethod.CAA,  # Use CAA for aggregation
        dynamic_control=True,
    )

    # Train on all layers (DAC doesn't use single layer)
    training_stats = dac.train(pair_set, layer_index=None)

    return dac, training_stats


def compute_dac_vector_similarity(our_vectors_dict, reference_vectors):
    """
    Compare our multi-layer DAC vectors with reference vectors.

    Args:
        our_vectors_dict: Dict mapping layer_idx to our computed vectors
        reference_vectors: Reference tensor of shape [steps, layers, heads, d_head]

    Returns:
        Dict with similarity metrics
    """
    similarities = {}
    total_cosine_sim = 0.0
    total_layers = 0

    # Reference vectors have shape [steps, layers, heads, d_head]
    # We need to compare layer by layer

    for layer_idx in our_vectors_dict.keys():
        if layer_idx < reference_vectors.shape[1]:  # Check layer exists in reference
            our_vector = our_vectors_dict[layer_idx]
            ref_layer_vector = reference_vectors[:, layer_idx, :, :]  # [steps, heads, d_head]

            # Flatten both for comparison
            our_flat = our_vector.flatten()
            ref_flat = ref_layer_vector.flatten()

            # Compute cosine similarity
            cosine_sim = torch.nn.functional.cosine_similarity(our_flat, ref_flat, dim=0).item()

            similarities[f"layer_{layer_idx}"] = {
                "cosine_similarity": cosine_sim,
                "our_norm": torch.norm(our_flat).item(),
                "ref_norm": torch.norm(ref_flat).item(),
                "norm_ratio": torch.norm(our_flat).item() / torch.norm(ref_flat).item(),
            }

            total_cosine_sim += cosine_sim
            total_layers += 1

    # Overall similarity
    similarities["overall"] = {
        "average_cosine_similarity": total_cosine_sim / total_layers if total_layers > 0 else 0.0,
        "layers_compared": total_layers,
    }

    return similarities
