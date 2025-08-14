"""
Utilities for loading and using real models in validation tests.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

import sys

from wisent_guard.core.contrastive_pairs import ContrastivePair, ContrastivePairSet
from wisent_guard.core.response import NegativeResponse, PositiveResponse

from .caa_utils import tokenize_llama_base_format
from .const import LAYER_INDEX, MODEL_NAME, TORCH_DTYPE


class RealModelWrapper:
    """Wrapper for real Llama2 model for validation testing."""

    def __init__(self, model_name=MODEL_NAME, device="auto"):
        self.model_name = model_name
        self.device = device

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=TORCH_DTYPE, device_map=device, trust_remote_code=True
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ Model loaded on device: {self.model.device}")

    def get_activations(self, prompts_and_responses, layer_idx, position=-2):
        """
        Extract activations from specific layer and position using CAA tokenization format.

        Args:
            prompts_and_responses: List of tuples (user_input, model_output) or list of formatted strings
            layer_idx: Layer index to extract from
            position: Position in sequence (typically -2 for CAA)
        """
        activations = []

        for item in prompts_and_responses:
            if isinstance(item, tuple):
                # Tokenize using CAA format
                user_input, model_output = item
                tokens = tokenize_llama_base_format(self.tokenizer, user_input, model_output)
                inputs = {"input_ids": torch.tensor(tokens).unsqueeze(0).to(self.model.device)}
            else:
                # Assume it's already formatted text (for backward compatibility)
                inputs = self.tokenizer(item, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Forward pass with hooks to extract activations
            layer_activations = None

            def hook_fn(_, __, output):
                """Hook function to extract layer activations."""
                nonlocal layer_activations
                layer_activations = output[0]  # Hidden states

            # Register hook
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(hook_fn)

            try:
                with torch.no_grad():
                    _ = self.model(**inputs)

                # Extract activation at specified position
                if layer_activations is not None:
                    # Use position relative to sequence length
                    seq_len = layer_activations.shape[1]
                    if position < 0:
                        pos = seq_len + position
                    else:
                        pos = position
                    pos = max(0, min(pos, seq_len - 1))  # Clamp to valid range

                    activation = layer_activations[0, pos, :].cpu()  # [hidden_dim]
                    activations.append(activation)
                else:
                    raise RuntimeError("Failed to extract activations")

            finally:
                handle.remove()

        return torch.stack(activations)  # [batch_size, hidden_dim]


def create_real_contrastive_pairs(dataset, model, layer_idx=LAYER_INDEX, max_pairs=50):
    """Create ContrastivePairSet using real model activations with exact CAA tokenization."""

    print("Creating contrastive pairs with real model activations (CAA format)...")
    print(f"Processing {min(len(dataset), max_pairs)} examples...")

    pairs = []

    for i, item in enumerate(dataset[:max_pairs]):
        if i % 10 == 0:
            print(f"Processing {i}/{min(len(dataset), max_pairs)}")

        question = item["question"]
        pos_answer = item["answer_matching_behavior"]  # (B) - exhibits behavior
        neg_answer = item["answer_not_matching_behavior"]  # (A) - correct answer

        # Use CAA tokenization format: pass (user_input, model_output) tuples
        pos_input = (question, pos_answer)
        neg_input = (question, neg_answer)

        # Extract activations using real model with CAA tokenization
        pos_activations = model.get_activations([pos_input], layer_idx, position=-2)
        neg_activations = model.get_activations([neg_input], layer_idx, position=-2)

        # Create Response objects
        pos_response = PositiveResponse(
            text=pos_answer,
            activations=pos_activations[0],  # Remove batch dimension
        )
        neg_response = NegativeResponse(
            text=neg_answer,
            activations=neg_activations[0],  # Remove batch dimension
        )

        # Create ContrastivePair
        pair = ContrastivePair(prompt=question, positive_response=pos_response, negative_response=neg_response)

        pairs.append(pair)

    # Create ContrastivePairSet
    pair_set = ContrastivePairSet(name="hallucination_real_model", pairs=pairs)

    print(f"✅ Created {len(pair_set.pairs)} contrastive pairs with real activations")

    return pair_set


def create_caa_original_contrastive_pairs(layer_idx=LAYER_INDEX, max_pairs=None):
    """Create ContrastivePairSet replicating CAA's exact approach using wisent-guard modules.

    Uses MAX_EXAMPLES (default 20) for fast testing from the full dataset.
    Now uses RealModelWrapper + HuggingFace instead of external CAA dependencies.

    Args:
        layer_idx: Layer index to extract activations from
        max_pairs: Maximum number of pairs to process (None uses MAX_EXAMPLES)

    Returns:
        ContrastivePairSet with CAA-compatible activations
    """
    print("Creating contrastive pairs with CAA-compatible method (wisent-guard)...")

    # Use RealModelWrapper instead of CAA's LlamaWrapper
    model = RealModelWrapper(MODEL_NAME, device="auto")

    # Load dataset and limit to MAX_EXAMPLES for fast testing
    import json

    from .const import HALLUCINATION_DATASET_PATH, MAX_EXAMPLES

    print(f"Loading dataset from: {HALLUCINATION_DATASET_PATH}")

    with open(HALLUCINATION_DATASET_PATH) as f:
        full_dataset = json.load(f)

    # Use only first MAX_EXAMPLES for fast testing
    caa_dataset = full_dataset[:MAX_EXAMPLES] if max_pairs is None else full_dataset[:max_pairs]
    print(f"Processing {len(caa_dataset)} examples (limited for fast testing)...")

    pairs = []

    for i, item in enumerate(caa_dataset):
        if i % 10 == 0:
            print(f"Processing {i}/{len(caa_dataset)}")

        question = item["question"]
        pos_answer = item["answer_matching_behavior"]  # (B) - exhibits behavior
        neg_answer = item["answer_not_matching_behavior"]  # (A) - correct answer

        # Use CAA tokenization format: pass (user_input, model_output) tuples
        pos_input = (question, pos_answer)
        neg_input = (question, neg_answer)

        # Extract activations using RealModelWrapper with CAA tokenization
        pos_activations = model.get_activations([pos_input], layer_idx, position=-2)
        neg_activations = model.get_activations([neg_input], layer_idx, position=-2)

        # Create Response objects
        pos_response = PositiveResponse(
            text=pos_answer,
            activations=pos_activations[0],  # Remove batch dimension
        )
        neg_response = NegativeResponse(
            text=neg_answer,
            activations=neg_activations[0],  # Remove batch dimension
        )

        # Create ContrastivePair
        pair = ContrastivePair(prompt=question, positive_response=pos_response, negative_response=neg_response)
        pairs.append(pair)

    # Create ContrastivePairSet
    pair_set = ContrastivePairSet(name="hallucination_caa_compatible", pairs=pairs)

    print(f"✅ Created {len(pair_set.pairs)} contrastive pairs with CAA-compatible method")

    return pair_set


def compare_with_reference_activations(our_activations, ref_activations):
    """Compare our extracted activations with reference ones."""

    # Convert to same device and dtype
    if isinstance(our_activations, list):
        our_acts = torch.stack([act.cpu().float() for act in our_activations])
    else:
        our_acts = our_activations.cpu().float()

    ref_acts = ref_activations.cpu().float()

    # Calculate comparison metrics
    cosine_sim = torch.nn.functional.cosine_similarity(our_acts, ref_acts, dim=-1).mean()
    mse = torch.nn.functional.mse_loss(our_acts, ref_acts)
    norm_ratio = torch.norm(our_acts, dim=-1).mean() / torch.norm(ref_acts, dim=-1).mean()

    return {
        "cosine_similarity": cosine_sim.item(),
        "mse": mse.item(),
        "norm_ratio": norm_ratio.item(),
        "our_shape": our_acts.shape,
        "ref_shape": ref_acts.shape,
    }
