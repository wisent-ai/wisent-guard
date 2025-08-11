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


class DACModelWrapper:
    """Wrapper for real model to generate DAC steering vectors using wisent-guard implementation."""

    def __init__(self, model_name=MODEL_NAME, device="auto", use_nnsight=True, load_in_8bit=True):
        self.model_name = model_name
        self.device = device
        self.use_nnsight = use_nnsight
        self.load_in_8bit = load_in_8bit

        print(f"Loading {model_name} for DAC vector generation...")
        print(f"  Using nnsight: {use_nnsight}")
        print(f"  8-bit quantization: {load_in_8bit}")

        if use_nnsight:
            # EXACT match to original DAC setup
            print("  Using nnsight LanguageModel with bfloat16 + 8-bit quantization...")
            self.model = LanguageModel(
                model_name,
                device_map={"": 0} if load_in_8bit else device,  # Force GPU 0 for 8-bit
                quantization_config=(BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None),
                low_cpu_mem_usage=True if load_in_8bit else None,
                torch_dtype=torch.bfloat16,  # ← BFLOAT16 like original
                trust_remote_code=True,
            )
            self.tokenizer = self.model.tokenizer

            # Set pad token for nnsight tokenizer
            if not self.tokenizer.pad_token_id:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            print(f"✅ nnsight model loaded with dtype: bfloat16, 8-bit: {load_in_8bit}")

        else:
            # Fallback to transformers (for comparison)
            print("  Using transformers AutoModelForCausalLM...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            if load_in_8bit:
                # Use 8-bit quantization with proper device mapping
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 to match original
                    device_map="auto",  # Let transformers handle device mapping
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Optimize memory usage
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 to match original DAC
                    device_map=device,
                    trust_remote_code=True,
                )

            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✅ Transformers model loaded on device: {self.model.device}")

    def get_activations_all_layers(self, prompts_list, max_new_tokens=MAX_NEW_TOKENS, position=-1):
        """
        Extract activations from ALL layers for DAC (not just one layer like CAA).

        Uses PyTorch hooks to extract activations from attention output projections during generation.

        Args:
            prompts_list: List of tokenized prompts
            max_new_tokens: Maximum tokens to generate
            position: Position to extract from each generation step (-1 = last token)

        Returns:
            Dict mapping layer_idx to list of activations for that layer
        """
        all_layer_activations = {i: [] for i in range(MODEL_CONFIG["n_layers"])}

        print("  Using PyTorch hooks activation extraction...")

        for prompt_tokens in prompts_list:
            # Convert to tensor if needed
            if isinstance(prompt_tokens, list):
                input_ids = torch.tensor(prompt_tokens).unsqueeze(0).to(self.model.device)
            else:
                input_ids = prompt_tokens.unsqueeze(0).to(self.model.device)

            # Generate tokens and collect activations at each step (like original DAC)
            step_activations = []

            for step in range(max_new_tokens):
                # Store activations for this step [layers, heads, d_head]
                step_layer_activations = []

                # Hook attention output projections like original DAC
                layer_outputs = {}

                def create_hook(layer_idx):
                    def hook_fn(module, input, output):
                        # Store the input to the attention output projection (like original DAC)
                        # Original: rgetattr(model, config["attn_hook_names"][layer_i]).input[0][0]
                        # FIXED: Use input[0][0] to remove batch dimension like original
                        layer_outputs[layer_idx] = input[0][0]  # [seq, hidden] - batch dimension removed

                    return hook_fn

                # Register hooks for attention output projections (o_proj)
                hooks = []
                for layer_idx in range(MODEL_CONFIG["n_layers"]):
                    attn_o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
                    hook = attn_o_proj.register_forward_hook(create_hook(layer_idx))
                    hooks.append(hook)

                try:
                    with torch.no_grad():
                        # Forward pass with hooks active
                        outputs = self.model(input_ids=input_ids)
                        logits = outputs.logits

                    # Extract activations from the hooks (like original DAC)
                    for layer_idx in range(MODEL_CONFIG["n_layers"]):
                        if layer_idx in layer_outputs:
                            # Get the input to attention output projection
                            attn_input = layer_outputs[layer_idx]  # [seq, hidden] - batch dim already removed

                            # Take last token activation
                            last_token_hidden = attn_input[-1, :]  # [hidden_dim] - no batch indexing needed

                            # Reshape to match DAC head structure: [n_heads, d_head]
                            n_heads = MODEL_CONFIG["n_heads"]
                            d_head = MODEL_CONFIG["d_model"] // n_heads
                            head_activations = last_token_hidden.view(n_heads, d_head).cpu()

                            step_layer_activations.append(head_activations)
                        else:
                            # If hook didn't capture this layer, create zeros
                            n_heads = MODEL_CONFIG["n_heads"]
                            d_head = MODEL_CONFIG["d_model"] // n_heads
                            head_activations = torch.zeros(n_heads, d_head)
                            step_layer_activations.append(head_activations)

                finally:
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()

                # Stack to get [layers, heads, d_head] for this step
                step_tensor = torch.stack(step_layer_activations)  # [n_layers, n_heads, d_head]
                step_activations.append(step_tensor)

                # Get next token
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                # Update input for next step
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Stop on EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            # Convert to format expected by DAC: [steps, layers, heads, d_head]
            if step_activations:
                prompt_activations = torch.stack(step_activations)  # [steps, layers, heads, d_head]

                # Store by layer for compatibility with our training code
                for layer_idx in range(MODEL_CONFIG["n_layers"]):
                    layer_acts = prompt_activations[:, layer_idx, :, :]  # [steps, heads, d_head]
                    all_layer_activations[layer_idx].append(layer_acts)

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
    Create DAC vectors using the exact original DAC approach:
    1. Extract activations from all prompts for each dataset
    2. Compute mean activations for each dataset
    3. Compute difference vector (A - B)

    This matches the original DAC implementation in diff_main.py
    """
    import random

    # Load datasets
    dataset_a, dataset_b, instruction_a, instruction_b = load_dac_datasets_for_testing()

    # Tokenize prompts for both datasets (creates ~4 unique prompts each)
    prompts_a = tokenize_dac_prompts(model_wrapper.tokenizer, dataset_a, instruction_a)
    prompts_b = tokenize_dac_prompts(model_wrapper.tokenizer, dataset_b, instruction_b)

    # 🆕 ADD RANDOM SAMPLING STEP (like original DAC diff_main.py line 109)
    # Set seed for reproducibility (same as original DAC)
    random.seed(32)

    # Use support parameter like original (up to 20 examples)
    support = min(len(dataset_a), len(dataset_b), MAX_EXAMPLES)
    num_prompts = min(len(prompts_a), len(prompts_b))

    print(f"Available unique prompts: {num_prompts} each")
    print(f"Sampling {support} examples with replacement (like original DAC)...")

    # Random sampling with replacement (exact match to original DAC implementation)
    random_indexes_a = [random.randint(0, num_prompts - 1) for _ in range(support)]
    random_indexes_b = [random.randint(0, num_prompts - 1) for _ in range(support)]

    # Create sampled prompt lists (allows duplicates like original)
    sampled_prompts_a = [prompts_a[i] for i in random_indexes_a]
    sampled_prompts_b = [prompts_b[i] for i in random_indexes_b]

    print(f"Sampled prompts A: {len(sampled_prompts_a)} examples")
    print(f"Sampled prompts B: {len(sampled_prompts_b)} examples")
    print(f"Processing {len(sampled_prompts_a)} prompts for DAC vector generation (original approach)...")

    # Get activations for both datasets from all layers
    print("Extracting activations for dataset A (ITA)...")
    activations_a = model_wrapper.get_activations_all_layers(sampled_prompts_a)

    print("Extracting activations for dataset B (ENG)...")
    activations_b = model_wrapper.get_activations_all_layers(sampled_prompts_b)

    print("Computing mean activations (like original DAC)...")

    # Compute mean activations for each dataset (original DAC approach)
    # Expected shape: [steps, layers, heads, d_head] like the original
    mean_activations_a = compute_mean_activations_original_format(activations_a)
    mean_activations_b = compute_mean_activations_original_format(activations_b)

    print(f"Mean activations A shape: {mean_activations_a.shape}")
    print(f"Mean activations B shape: {mean_activations_b.shape}")

    # Compute difference vector (ITA - ENG) like original DAC
    diff_activations = mean_activations_a - mean_activations_b

    print(f"Difference activations shape: {diff_activations.shape}")
    print(f"Difference norm: {torch.norm(diff_activations).item():.4f}")

    # Create a simple stats dict to match the expected interface
    training_stats = {
        "method": "DAC_original_approach",
        "dataset_a_name": DATASET_A_NAME,
        "dataset_b_name": DATASET_B_NAME,
        "num_prompts": support,  # Now using the correct number (up to 20)
        "unique_prompts": num_prompts,  # Number of unique prompts (4)
        "sampling_with_replacement": True,
        "seed": 32,
        "mean_a_shape": mean_activations_a.shape,
        "mean_b_shape": mean_activations_b.shape,
        "diff_shape": diff_activations.shape,
        "diff_norm": torch.norm(diff_activations).item(),
        "success": True,
    }

    # Create a simple container for the difference vector
    class DACVectorContainer:
        def __init__(self, diff_vector, mean_a, mean_b):
            self.diff_activations = diff_vector
            self.mean_activations_a = mean_a
            self.mean_activations_b = mean_b

        def get_property_vectors(self):
            # Return the difference vector in a format compatible with our test
            return {
                "ita_vs_eng": {
                    "vectors": {
                        layer_idx: self.diff_activations[:, layer_idx, :, :]
                        for layer_idx in range(self.diff_activations.shape[1])
                    }
                }
            }

    dac_container = DACVectorContainer(diff_activations, mean_activations_a, mean_activations_b)

    return dac_container, training_stats


def compute_mean_activations_original_format(activations_dict):
    """
    Compute mean activations in the original DAC format.

    Input: activations_dict[layer_idx] = list of [steps, heads, d_head] tensors for each prompt
    Output: tensor of shape [steps, layers, heads, d_head] - mean across all prompts
    """
    n_layers = MODEL_CONFIG["n_layers"]
    n_heads = MODEL_CONFIG["n_heads"]
    d_head = MODEL_CONFIG["d_head"]
    max_steps = MAX_NEW_TOKENS

    # Initialize mean activations tensor
    mean_activations = torch.zeros(max_steps, n_layers, n_heads, d_head)

    for layer_idx in range(n_layers):
        if layer_idx in activations_dict and activations_dict[layer_idx]:
            # Get all prompt activations for this layer
            layer_activations = activations_dict[layer_idx]  # List of [steps, heads, d_head]

            # Stack them: [num_prompts, steps, heads, d_head]
            if layer_activations:
                # Pad or truncate to max_steps
                padded_activations = []
                for prompt_acts in layer_activations:
                    if prompt_acts.shape[0] < max_steps:
                        # Pad with zeros
                        pad_size = max_steps - prompt_acts.shape[0]
                        padding = torch.zeros(pad_size, n_heads, d_head)
                        padded = torch.cat([prompt_acts, padding], dim=0)
                    else:
                        # Truncate
                        padded = prompt_acts[:max_steps]
                    padded_activations.append(padded)

                # Stack and compute mean across prompts
                stacked = torch.stack(padded_activations)  # [num_prompts, steps, heads, d_head]
                layer_mean = stacked.mean(dim=0)  # [steps, heads, d_head]
                mean_activations[:, layer_idx, :, :] = layer_mean

    return mean_activations


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
