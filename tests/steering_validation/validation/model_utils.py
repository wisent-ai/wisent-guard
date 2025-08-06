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

from wisent_guard.core.contrastive_pairs import ContrastivePair, ContrastivePairSet
from wisent_guard.core.response import NegativeResponse, PositiveResponse

# Constants
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DEFAULT_LAYER_INDEX = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RealModelWrapper:
    """Wrapper for real Llama2 model for validation testing."""

    def __init__(self, model_name=MODEL_NAME, device="auto"):
        self.model_name = model_name
        self.device = device

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ Model loaded on device: {self.model.device}")

    def get_activations(self, texts, layer_idx, position=-2):
        """Extract activations from specific layer and position."""
        activations = []

        for text in texts:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Forward pass with hooks to extract activations
            layer_activations = None

            def hook_fn(module, input, output):
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


def create_real_contrastive_pairs(dataset, model, layer_idx=DEFAULT_LAYER_INDEX, max_pairs=50):
    """Create ContrastivePairSet using real model activations."""

    print("Creating contrastive pairs with real model activations...")
    print(f"Processing {min(len(dataset), max_pairs)} examples...")

    pairs = []

    for i, item in enumerate(dataset[:max_pairs]):
        if i % 10 == 0:
            print(f"Processing {i}/{min(len(dataset), max_pairs)}")

        question = item["question"]
        pos_answer = item["answer_matching_behavior"]  # (B) - exhibits behavior
        neg_answer = item["answer_not_matching_behavior"]  # (A) - correct answer

        # Create full prompts (matching exact CAA tokenization format)
        pos_prompt = f"Input: {question}\nResponse: {pos_answer}"
        neg_prompt = f"Input: {question}\nResponse: {neg_answer}"

        # Extract activations using real model
        pos_activations = model.get_activations([pos_prompt], layer_idx, position=-2)
        neg_activations = model.get_activations([neg_prompt], layer_idx, position=-2)

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
