#!/usr/bin/env python3
"""
Test HEAD SPLITTING vs CONCATENATION hypothesis.

This tests if our head splitting is causing the similarity issues.
Original DAC might keep activations concatenated as [d_model] instead of [n_heads, d_head].
"""

import sys
import torch
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from const import MODEL_NAME, TORCH_DTYPE, MODEL_CONFIG, MAX_NEW_TOKENS
from dac_model_utils import DACModelWrapper, load_dac_datasets_for_testing, tokenize_dac_prompts


def test_head_splitting_vs_concatenation():
    """Test both head splitting and concatenated approaches."""

    print("🧪 Testing HEAD SPLITTING vs CONCATENATION...")
    print("=" * 70)

    # Create model wrapper
    model_wrapper = DACModelWrapper(MODEL_NAME, device="auto")

    # Load test data
    dataset_a, _, instruction_a, _ = load_dac_datasets_for_testing()
    prompts_a = tokenize_dac_prompts(model_wrapper.tokenizer, dataset_a, instruction_a)

    # Use first few prompts for testing
    test_prompts = prompts_a[:2]  # Just 2 prompts for speed

    print(f"Testing with {len(test_prompts)} prompts...")

    # Test both approaches
    approaches = {
        "head_split": "Split into [n_heads, d_head] like current implementation",
        "concatenated": "Keep concatenated as [d_model] like original might do",
    }

    results = {}

    for approach_name, description in approaches.items():
        print(f"\n🔬 Testing {approach_name}: {description}")

        # Create modified activation extraction
        def extract_with_approach(prompts_list, approach):
            all_layer_activations = {i: [] for i in range(MODEL_CONFIG["n_layers"])}

            for prompt_tokens in prompts_list:
                input_ids = prompt_tokens.unsqueeze(0).to(model_wrapper.model.device)
                step_activations = []

                for step in range(min(3, MAX_NEW_TOKENS)):  # Just 3 steps for speed
                    layer_outputs = {}

                    def create_hook(layer_idx):
                        def hook_fn(module, input, output):
                            layer_outputs[layer_idx] = input[0][0]  # [seq, hidden]

                        return hook_fn

                    # Register hooks for first few layers
                    hooks = []
                    for layer_idx in range(min(3, MODEL_CONFIG["n_layers"])):
                        attn_o_proj = model_wrapper.model.model.layers[layer_idx].self_attn.o_proj
                        hook = attn_o_proj.register_forward_hook(create_hook(layer_idx))
                        hooks.append(hook)

                    try:
                        with torch.no_grad():
                            outputs = model_wrapper.model(input_ids=input_ids)
                            logits = outputs.logits

                        step_layer_activations = []
                        for layer_idx in range(min(3, MODEL_CONFIG["n_layers"])):
                            if layer_idx in layer_outputs:
                                attn_input = layer_outputs[layer_idx]  # [seq, hidden]
                                last_token_hidden = attn_input[-1, :]  # [hidden_dim = 4096]

                                if approach == "head_split":
                                    # Current approach: split into heads
                                    n_heads = MODEL_CONFIG["n_heads"]
                                    d_head = MODEL_CONFIG["d_model"] // n_heads
                                    head_activations = last_token_hidden.view(n_heads, d_head).cpu()
                                    step_layer_activations.append(head_activations)
                                else:  # concatenated
                                    # NEW: Keep concatenated as full d_model
                                    full_activations = last_token_hidden.cpu()  # [4096]
                                    step_layer_activations.append(full_activations)

                        if step_layer_activations:
                            if approach == "head_split":
                                step_tensor = torch.stack(step_layer_activations)  # [n_layers, n_heads, d_head]
                            else:  # concatenated
                                step_tensor = torch.stack(step_layer_activations)  # [n_layers, d_model]

                            step_activations.append(step_tensor)

                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_token], dim=-1)

                        if next_token.item() == model_wrapper.tokenizer.eos_token_id:
                            break

                    finally:
                        for hook in hooks:
                            hook.remove()

                # Store activations for each layer
                if step_activations:
                    if approach == "head_split":
                        prompt_activations = torch.stack(step_activations)  # [steps, layers, heads, d_head]
                        for layer_idx in range(min(3, MODEL_CONFIG["n_layers"])):
                            layer_acts = prompt_activations[:, layer_idx, :, :]  # [steps, heads, d_head]
                            all_layer_activations[layer_idx].append(layer_acts)
                    else:  # concatenated
                        prompt_activations = torch.stack(step_activations)  # [steps, layers, d_model]
                        for layer_idx in range(min(3, MODEL_CONFIG["n_layers"])):
                            layer_acts = prompt_activations[:, layer_idx, :]  # [steps, d_model]
                            all_layer_activations[layer_idx].append(layer_acts)

            return all_layer_activations

        # Extract activations with this approach
        activations = extract_with_approach(test_prompts, approach_name)

        # Compute mean activations (simplified)
        layer_0_acts = activations[0] if activations[0] else None
        if layer_0_acts:
            # Stack all prompts and compute mean
            stacked = torch.stack(layer_0_acts)  # [num_prompts, steps, ...]
            mean_acts = stacked.mean(dim=0)  # [steps, ...]

            summary = {
                "shape": mean_acts.shape,
                "norm": torch.norm(mean_acts).item(),
                "mean": mean_acts.mean().item(),
                "std": mean_acts.std().item(),
                "num_elements": mean_acts.numel(),
            }
            results[approach_name] = {"success": True, "summary": summary, "tensor": mean_acts}
            print(f"   ✅ Success! Shape: {mean_acts.shape}")
            print(f"   📊 Norm: {summary['norm']:.6f}")
            print(f"   📊 Elements: {summary['num_elements']}")
        else:
            results[approach_name] = {"success": False}
            print(f"   ❌ Failed to extract activations")

    # Compare approaches
    print(f"\n📊 COMPARISON:")
    print("=" * 70)

    if all(r["success"] for r in results.values()):
        head_split = results["head_split"]
        concatenated = results["concatenated"]

        print(f"Head Split Shape:    {head_split['summary']['shape']}")
        print(f"Concatenated Shape:  {concatenated['summary']['shape']}")
        print(f"")
        print(f"Head Split Norm:     {head_split['summary']['norm']:.6f}")
        print(f"Concatenated Norm:   {concatenated['summary']['norm']:.6f}")
        print(f"")
        print(f"Head Split Elements: {head_split['summary']['num_elements']}")
        print(f"Concatenated Elements: {concatenated['summary']['num_elements']}")

        # Key insight: check if concatenated has same total elements but different shape
        hs_elements = head_split["summary"]["num_elements"]
        concat_elements = concatenated["summary"]["num_elements"]

        if hs_elements == concat_elements:
            print(f"\n🔍 SAME TOTAL ELEMENTS - just different reshaping!")
            print(f"   This suggests original DAC might use concatenated format")
        else:
            print(f"\n🔍 DIFFERENT TOTAL ELEMENTS")
            print(f"   Elements ratio: {hs_elements / concat_elements:.2f}")

        # If shapes are compatible, try direct comparison
        hs_tensor = head_split["tensor"]
        concat_tensor = concatenated["tensor"]

        # Try to reshape for comparison
        try:
            if len(hs_tensor.shape) == 4 and len(concat_tensor.shape) == 3:
                # head_split: [steps, layers, heads, d_head]
                # concatenated: [steps, layers, d_model]
                # Reshape head_split to concatenated format
                steps, layers, heads, d_head = hs_tensor.shape
                hs_reshaped = hs_tensor.view(steps, layers, heads * d_head)

                if hs_reshaped.shape == concat_tensor.shape:
                    cosine_sim = torch.cosine_similarity(hs_reshaped.flatten(), concat_tensor.flatten(), dim=0).item()
                    l2_dist = torch.norm(hs_reshaped - concat_tensor).item()

                    print(f"\n🎯 DIRECT TENSOR COMPARISON:")
                    print(f"   Cosine similarity: {cosine_sim:.8f}")
                    print(f"   L2 distance: {l2_dist:.8f}")

                    if cosine_sim > 0.9999:
                        print(f"   🎉 NEARLY IDENTICAL - just different reshaping!")
                    else:
                        print(f"   🤔 Different values - this could be the issue!")
        except Exception as e:
            print(f"   ⚠️  Could not reshape for comparison: {e}")

    else:
        print("❌ One or both approaches failed")

    return results


if __name__ == "__main__":
    results = test_head_splitting_vs_concatenation()
    print("\n🔬 Head splitting test completed!")
