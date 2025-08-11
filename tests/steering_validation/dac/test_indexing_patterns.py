#!/usr/bin/env python3
"""
Test different indexing patterns in PyTorch hooks to match original DAC.

This tests the hypothesis that .input[0][0] vs input[0] is causing our similarity issues.
"""

import sys
import torch
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from const import MODEL_NAME, TORCH_DTYPE, MODEL_CONFIG
from dac_model_utils import load_dac_datasets_for_testing, tokenize_dac_prompts


def test_hook_indexing_patterns():
    """Test different indexing patterns in hooks to see which matches original DAC."""

    print("🔍 Testing different hook indexing patterns...")
    print("=" * 60)

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=TORCH_DTYPE, device_map="auto", trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load a simple test prompt
    dataset_a, _, instruction_a, _ = load_dac_datasets_for_testing()
    prompts_a = tokenize_dac_prompts(tokenizer, dataset_a, instruction_a)

    # Use first prompt for testing
    test_prompt = prompts_a[0].unsqueeze(0).to(model.device)
    print(f"Test prompt shape: {test_prompt.shape}")

    # Test different indexing patterns
    indexing_patterns = {
        "input[0]": lambda inp, out: inp[0],  # Our current approach
        "input[0][0]": lambda inp, out: inp[0][0],  # Original DAC pattern
        "input": lambda inp, out: inp,  # No indexing
        "output": lambda inp, out: out,  # Use output instead
        "input[0]_batch0": lambda inp, out: inp[0][0, :, :] if len(inp[0].shape) >= 3 else inp[0],  # Explicit batch=0
    }

    results = {}

    for pattern_name, indexing_func in indexing_patterns.items():
        print(f"\n🧪 Testing pattern: {pattern_name}")

        try:
            # Test on layer 14 (middle layer)
            layer_idx = 14
            layer_outputs = {}

            def create_hook(layer_idx, pattern_func):
                def hook_fn(module, input, output):
                    try:
                        layer_outputs[layer_idx] = pattern_func(input, output)
                    except Exception as e:
                        layer_outputs[layer_idx] = f"ERROR: {e}"

                return hook_fn

            # Hook the attention output projection
            attn_o_proj = model.model.layers[layer_idx].self_attn.o_proj
            hook = attn_o_proj.register_forward_hook(create_hook(layer_idx, indexing_func))

            try:
                with torch.no_grad():
                    outputs = model(input_ids=test_prompt)

                if layer_idx in layer_outputs:
                    result = layer_outputs[layer_idx]
                    if isinstance(result, str):  # Error case
                        print(f"   ❌ {result}")
                        results[pattern_name] = {"error": result}
                    else:
                        print(f"   ✅ Shape: {result.shape}")
                        print(f"   ✅ Dtype: {result.dtype}")
                        print(f"   ✅ Device: {result.device}")
                        print(f"   ✅ Norm: {torch.norm(result).item():.4f}")

                        # Store for comparison
                        results[pattern_name] = {
                            "tensor": result.detach().cpu(),
                            "shape": result.shape,
                            "norm": torch.norm(result).item(),
                        }
                else:
                    print("   ❌ No activation captured")
                    results[pattern_name] = {"error": "No activation captured"}

            finally:
                hook.remove()

        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results[pattern_name] = {"error": str(e)}

    # Compare successful results
    print(f"\n📊 Comparing results...")
    print("=" * 60)

    successful_results = {k: v for k, v in results.items() if "tensor" in v}

    if len(successful_results) < 2:
        print("❌ Need at least 2 successful extractions to compare")
        return results

    # Compare all pairs
    pattern_names = list(successful_results.keys())
    for i, pattern1 in enumerate(pattern_names):
        for j, pattern2 in enumerate(pattern_names[i + 1 :], i + 1):
            tensor1 = successful_results[pattern1]["tensor"]
            tensor2 = successful_results[pattern2]["tensor"]

            # Handle different shapes
            if tensor1.shape == tensor2.shape:
                cosine_sim = torch.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0).item()
                l2_dist = torch.norm(tensor1 - tensor2).item()

                print(f"\n🔍 {pattern1} vs {pattern2}:")
                print(f"   Cosine similarity: {cosine_sim:.6f}")
                print(f"   L2 distance: {l2_dist:.6f}")
                print(f"   Shapes match: ✅ {tensor1.shape}")

                if cosine_sim > 0.99:
                    print(f"   🎯 VERY HIGH SIMILARITY! These might be equivalent")
                elif cosine_sim > 0.9:
                    print(f"   ✅ High similarity")
                elif cosine_sim > 0.5:
                    print(f"   ⚠️  Moderate similarity")
                else:
                    print(f"   ❌ Low similarity")
            else:
                print(f"\n🔍 {pattern1} vs {pattern2}:")
                print(f"   Shapes differ: {tensor1.shape} vs {tensor2.shape}")
                print(f"   Cannot compute similarity")

    print(f"\n📝 Summary:")
    print("=" * 60)
    for pattern, result in results.items():
        if "tensor" in result:
            print(f"✅ {pattern}: Shape {result['shape']}, Norm {result['norm']:.4f}")
        else:
            print(f"❌ {pattern}: {result.get('error', 'Failed')}")

    return results


if __name__ == "__main__":
    results = test_hook_indexing_patterns()
    print("\n🎉 Test completed!")
