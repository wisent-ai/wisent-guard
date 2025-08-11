#!/usr/bin/env python3
"""
DAC Vector Generation with wisent-guard Integration

This script creates DAC steering vectors by:
1. Loading ITA (Italian) and ENG (English) datasets
2. Creating ContrastivePair objects using wisent-guard modules
3. Computing mean activations for each dataset using step-by-step generation
4. Computing difference vectors (ITA - ENG) for language steering
5. Comparing results with reference vectors for validation

Key differences from original:
- Uses wisent-guard's ContrastivePairSet and ContrastivePair modules
- Uses wisent-guard's Model class for model loading
- Integrates with PositiveResponse/NegativeResponse classes
- Maintains compatibility with original DAC vector format

Usage:
    python diff_vector_generation.py
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add wisent-guard to path
WISENT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(WISENT_PATH))

# Import wisent-guard modules
from wisent_guard.core.contrastive_pairs import ContrastivePairSet, ContrastivePair
from wisent_guard.core.response import PositiveResponse, NegativeResponse

# Import local constants
from const import (
    MODEL_NAME,
    ACTIVATIONS_A_PATH,
    ACTIVATIONS_B_PATH,
    DIFF_ACTIVATIONS_PATH,
    DATASET_A_NAME,
    DATASET_B_NAME,
    MAX_EXAMPLES,
    MAX_NEW_TOKENS,
    ICL_EXAMPLES,
    REFERENCE_DATA_PATH,
    MODEL_CONFIG,
    TORCH_DTYPE,
    COSINE_SIMILARITY_THRESHOLD,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DACActivationExtractor:
    """
    Activation extractor compatible with original DAC format.

    Extracts activations step-by-step during generation and formats them
    to match the original DAC implementation: [steps, n_layers, n_heads, d_head]
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        # Use cuda:0 as primary device for computations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def split_activation(self, activations: List[torch.Tensor]) -> torch.Tensor:
        """
        Split the residual stream (d_model) into n_heads activations for each layer.

        Args:
            activations: List of residual streams for each layer [batch, seq, d_model]

        Returns:
            Reshaped activation in [batch, n_layers, n_heads, seq, d_head]
        """
        # Ensure all activations are on the same device
        target_device = self.device
        activations = [act.to(target_device) for act in activations]

        new_shape = torch.Size(
            [
                activations[0].shape[0],  # batch_size
                activations[0].shape[1],  # seq_len
                self.config["n_heads"],  # n_heads
                self.config["d_model"] // self.config["n_heads"],  # d_head
            ]
        )

        attn_activations = torch.stack([act.view(*new_shape) for act in activations])
        # layers batch seq heads dhead -> batch layers heads seq dhead
        attn_activations = torch.einsum("lbshd -> blhsd", attn_activations)
        return attn_activations

    def extract_activations_with_generation(
        self, input_tokens: torch.Tensor, max_new_tokens: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations during step-by-step generation.

        Args:
            input_tokens: Input token IDs [1, seq_len]
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with 'activations' and 'output' tensors
        """
        all_activations = []
        current_tokens = input_tokens.clone()

        # Hook storage
        layer_activations = {}

        def create_hook(layer_idx):
            def hook_fn(module, input_tensors, output):
                # Store the input to the attention layer (before attention computation)
                del module, output  # Unused parameters
                layer_activations[layer_idx] = input_tensors[0].detach().to(self.device)  # [batch, seq, hidden_dim]

            return hook_fn

        # Register hooks for all layers
        hooks = []
        for layer_idx in range(self.config["n_layers"]):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)

        try:
            for _ in range(max_new_tokens):
                layer_activations.clear()

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(current_tokens)
                    logits = outputs.logits

                # Get activations from all layers
                step_layer_activations = []
                for layer_idx in range(self.config["n_layers"]):
                    if layer_idx in layer_activations:
                        step_layer_activations.append(layer_activations[layer_idx])
                    else:
                        # Fallback: create zero tensor if hook didn't fire
                        batch_size, seq_len = current_tokens.shape
                        zero_act = torch.zeros(
                            batch_size, seq_len, self.config["d_model"], device=self.device, dtype=logits.dtype
                        )
                        step_layer_activations.append(zero_act)

                # Split heads and extract last token: [batch, n_layers, n_heads, d_head]
                attn_activations = self.split_activation(step_layer_activations)
                last_token_activations = attn_activations[0, :, :, -1, :].detach().cpu()
                all_activations.append(last_token_activations)

                # Get next token
                next_token = torch.argmax(logits[0, -1, :], dim=-1)
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Stack activations: [n_steps, n_layers, n_heads, d_head]
        all_activations = torch.stack(all_activations)

        return {"activations": all_activations, "output": current_tokens}

    def compute_mean_activations(
        self, contrastive_pairs: ContrastivePairSet, response_type: str = "positive", max_examples: int = 20
    ) -> torch.Tensor:
        """
        Compute mean activations for a set of responses.

        Args:
            contrastive_pairs: ContrastivePairSet containing the pairs
            response_type: "positive" or "negative"
            max_examples: Maximum number of examples to process

        Returns:
            Mean activations tensor [steps, n_layers, n_heads, d_head]
        """
        all_activations = []
        processed_count = 0

        logger.info(f"Computing mean activations for {response_type} responses...")

        for i, pair in enumerate(contrastive_pairs.pairs[:max_examples]):
            logger.info(f"Processing example {i + 1}/{min(len(contrastive_pairs.pairs), max_examples)}")

            # Get the appropriate response
            if response_type == "positive":
                response_text = pair.positive_response.text
                full_text = f"{pair.prompt}\n{response_text}"
            else:
                response_text = pair.negative_response.text
                full_text = f"{pair.prompt}\n{response_text}"

            # Tokenize the full conversation
            tokens = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(self.device)

            # Extract activations
            activations_dict = self.extract_activations_with_generation(input_ids, max_new_tokens=MAX_NEW_TOKENS)

            activations = activations_dict["activations"]  # [steps, layers, heads, d_head]
            all_activations.append(activations)
            processed_count += 1

        logger.info(f"Processed {processed_count} examples for {response_type} responses")

        # Compute mean across examples and steps
        # Handle different numbers of steps per example
        mean_activations = torch.zeros(
            MAX_NEW_TOKENS,
            self.config["n_layers"],
            self.config["n_heads"],
            self.config["d_model"] // self.config["n_heads"],
        )

        for step in range(MAX_NEW_TOKENS):
            step_activations = []
            for example_activations in all_activations:
                if step < example_activations.shape[0]:
                    step_activations.append(example_activations[step, :, :, :])

            if step_activations:
                mean_activations[step, :, :, :] = torch.stack(step_activations).mean(dim=0)

        return mean_activations


def load_datasets() -> Tuple[List[Dict], List[Dict]]:
    """Load ITA and ENG datasets from reference_data directory."""
    logger.info("Loading ITA and ENG datasets...")

    # Load ITA dataset
    ita_path = REFERENCE_DATA_PATH / "ita_train.json"
    with open(ita_path, "r", encoding="utf-8") as f:
        ita_data = json.load(f)

    # Load ENG dataset
    eng_path = REFERENCE_DATA_PATH / "eng_train.json"
    with open(eng_path, "r", encoding="utf-8") as f:
        eng_data = json.load(f)

    logger.info(f"Loaded {len(ita_data)} ITA examples and {len(eng_data)} ENG examples")

    return ita_data, eng_data


def create_contrastive_pairs(ita_data: List[Dict], eng_data: List[Dict]) -> ContrastivePairSet:
    """
    Create ContrastivePairSet where ITA responses are positive and ENG responses are negative.

    Args:
        ita_data: Italian dataset
        eng_data: English dataset

    Returns:
        ContrastivePairSet with language pairs
    """
    logger.info("Creating contrastive pairs...")

    pairs = []
    max_pairs = min(len(ita_data), len(eng_data), MAX_EXAMPLES)

    for i in range(max_pairs):
        ita_item = ita_data[i]
        eng_item = eng_data[i]

        # Ensure both have the same input
        assert ita_item["input"] == eng_item["input"], f"Mismatched inputs at index {i}"

        prompt = ita_item["input"]
        ita_response = ita_item["output"]
        eng_response = eng_item["output"]

        # Create responses (ITA as positive, ENG as negative for steering toward Italian)
        pos_response = PositiveResponse(text=ita_response)
        neg_response = NegativeResponse(text=eng_response)

        # Create contrastive pair
        pair = ContrastivePair(
            prompt=prompt, positive_response=pos_response, negative_response=neg_response, label="language_steering"
        )

        pairs.append(pair)

    # Create ContrastivePairSet
    pair_set = ContrastivePairSet(name="ita_eng_language_steering", pairs=pairs, task_type="language_steering")

    logger.info(f"Created {len(pair_set.pairs)} contrastive pairs")
    return pair_set


def load_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer."""
    logger.info(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=TORCH_DTYPE, device_map="auto", trust_remote_code=True
    )

    logger.info(f"Model loaded on device: {model.device}")
    return model, tokenizer


def compare_with_reference(
    our_mean_ita: torch.Tensor, our_mean_eng: torch.Tensor, our_diff: torch.Tensor
) -> Dict[str, Any]:
    """Compare our generated vectors with reference vectors."""
    logger.info("Comparing with reference vectors...")

    results = {}

    # Load reference vectors
    if ACTIVATIONS_A_PATH.exists():
        ref_mean_ita = torch.load(ACTIVATIONS_A_PATH, map_location="cpu")
        cosine_sim = F.cosine_similarity(our_mean_ita.flatten(), ref_mean_ita.flatten(), dim=0).item()
        mse = F.mse_loss(our_mean_ita, ref_mean_ita).item()

        results["mean_ita"] = {
            "cosine_similarity": cosine_sim,
            "mse": mse,
            "our_shape": our_mean_ita.shape,
            "ref_shape": ref_mean_ita.shape,
            "match": cosine_sim >= COSINE_SIMILARITY_THRESHOLD,
        }

    if ACTIVATIONS_B_PATH.exists():
        ref_mean_eng = torch.load(ACTIVATIONS_B_PATH, map_location="cpu")
        cosine_sim = F.cosine_similarity(our_mean_eng.flatten(), ref_mean_eng.flatten(), dim=0).item()
        mse = F.mse_loss(our_mean_eng, ref_mean_eng).item()

        results["mean_eng"] = {
            "cosine_similarity": cosine_sim,
            "mse": mse,
            "our_shape": our_mean_eng.shape,
            "ref_shape": ref_mean_eng.shape,
            "match": cosine_sim >= COSINE_SIMILARITY_THRESHOLD,
        }

    if DIFF_ACTIVATIONS_PATH.exists():
        ref_diff = torch.load(DIFF_ACTIVATIONS_PATH, map_location="cpu")
        cosine_sim = F.cosine_similarity(our_diff.flatten(), ref_diff.flatten(), dim=0).item()
        mse = F.mse_loss(our_diff, ref_diff).item()

        results["diff"] = {
            "cosine_similarity": cosine_sim,
            "mse": mse,
            "our_shape": our_diff.shape,
            "ref_shape": ref_diff.shape,
            "match": cosine_sim >= COSINE_SIMILARITY_THRESHOLD,
        }

    return results


def main():
    """Main execution function."""
    start_time = time.time()

    print("=" * 70)
    print("DAC Vector Generation with wisent-guard Integration")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Datasets: {DATASET_A_NAME} vs {DATASET_B_NAME}")
    print(f"Examples per dataset: {MAX_EXAMPLES}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"ICL examples: {ICL_EXAMPLES}")
    print(f"Output directory: {REFERENCE_DATA_PATH}")
    print("=" * 70)

    try:
        # Step 1: Load datasets
        print("\n[1/6] Loading datasets...")
        ita_data, eng_data = load_datasets()

        # Step 2: Create contrastive pairs
        print("\n[2/6] Creating contrastive pairs...")
        contrastive_pairs = create_contrastive_pairs(ita_data, eng_data)

        # Step 3: Load model and tokenizer
        print("\n[3/6] Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer()

        # Step 4: Create activation extractor
        print("\n[4/6] Creating activation extractor...")
        extractor = DACActivationExtractor(model, tokenizer, MODEL_CONFIG)

        # Step 5: Extract activations
        print("\n[5/6] Extracting activations...")

        # Compute mean activations for ITA (positive) responses
        mean_activations_ita = extractor.compute_mean_activations(
            contrastive_pairs, response_type="positive", max_examples=MAX_EXAMPLES
        )

        # Compute mean activations for ENG (negative) responses
        mean_activations_eng = extractor.compute_mean_activations(
            contrastive_pairs, response_type="negative", max_examples=MAX_EXAMPLES
        )

        # Compute difference vector (ITA - ENG)
        diff_activations = mean_activations_ita - mean_activations_eng

        # Step 6: Save vectors
        print("\n[6/6] Saving vectors...")
        torch.save(mean_activations_ita, ACTIVATIONS_A_PATH)
        torch.save(mean_activations_eng, ACTIVATIONS_B_PATH)
        torch.save(diff_activations, DIFF_ACTIVATIONS_PATH)

        # Compare with reference vectors
        comparison_results = compare_with_reference(mean_activations_ita, mean_activations_eng, diff_activations)

        # Save comparison results
        results_path = REFERENCE_DATA_PATH / "comparison_results_wisent.json"
        with open(results_path, "w") as f:
            json.dump(comparison_results, f, indent=2)

        # Display results
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("SUCCESS: DAC vectors generated with wisent-guard integration!")
        print("=" * 70)
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Mean activations ITA shape: {mean_activations_ita.shape}")
        print(f"Mean activations ENG shape: {mean_activations_eng.shape}")
        print(f"Difference activations shape: {diff_activations.shape}")
        print(f"Vector norm (ITA): {torch.norm(mean_activations_ita).item():.4f}")
        print(f"Vector norm (ENG): {torch.norm(mean_activations_eng).item():.4f}")
        print(f"Difference norm: {torch.norm(diff_activations).item():.4f}")

        # Display comparison results
        if comparison_results:
            print("\nComparison with reference vectors:")
            for key, metrics in comparison_results.items():
                print(
                    f"  {key}: cosine_sim={metrics['cosine_similarity']:.4f}, "
                    f"mse={metrics['mse']:.6f}, match={metrics['match']}"
                )

        print("\nFiles saved in reference_data/:")
        print(f"   - {ACTIVATIONS_A_PATH.name}")
        print(f"   - {ACTIVATIONS_B_PATH.name}")
        print(f"   - {DIFF_ACTIVATIONS_PATH.name}")
        print(f"   - {results_path.name}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during vector generation: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
