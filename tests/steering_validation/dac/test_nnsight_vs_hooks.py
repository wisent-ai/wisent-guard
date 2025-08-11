#!/usr/bin/env python3
"""
Direct comparison: nnsight vs PyTorch hooks activation extraction.

This tests the fundamental hypothesis that different extraction libraries
cause the ~0.18 similarity ceiling in our DAC implementation.
"""

import sys
import torch
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Add the Dynamic-Activation-Composition directory to the path for nnsight
DAC_DIR = Path(__file__).parent.parent.parent.parent / "Dynamic-Activation-Composition"
sys.path.insert(0, str(DAC_DIR))

from const import MODEL_NAME, TORCH_DTYPE, MODEL_CONFIG


def test_nnsight_vs_hooks_extraction():
    """Direct comparison of nnsight vs PyTorch hooks activation extraction."""

    print("🔬 Testing nnsight vs PyTorch hooks extraction...")
    print("=" * 70)

    # Load same model with transformers (for hooks)
    print("📥 Loading model with transformers...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hooks_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,  # Match original DAC
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load same model with nnsight
    print("📥 Loading model with nnsight...")
    try:
        from src.utils.model_utils import load_model_and_tokenizer

        nnsight_model, nnsight_tokenizer, nnsight_config, device = load_model_and_tokenizer(
            model_name=MODEL_NAME,
            load_in_8bit=True,  # Match original DAC
        )
        print(f"✅ nnsight model loaded on: {device}")
    except ImportError as e:
        print(f"❌ Failed to import nnsight: {e}")
        return None

    # Create identical test input
    test_text = "What is the capital of France?"
    test_tokens = tokenizer(test_text, return_tensors="pt")["input_ids"]
    print(f"📝 Test input: '{test_text}'")
    print(f"📝 Token shape: {test_tokens.shape}")

    # Test on multiple layers for comprehensive comparison
    test_layers = [0, 7, 14, 23, 31]  # Early, middle, late layers
    results = {}

    for layer_idx in test_layers:
        print(f"\n🧪 Testing layer {layer_idx}...")
        layer_results = {}

        # ===== METHOD 1: PyTorch Hooks =====
        print("   [1/2] Extracting with PyTorch hooks...")
        try:
            hooks_model.eval()
            layer_outputs = {}

            def create_hook(layer_idx):
                def hook_fn(module, input, output):
                    # Match our current implementation
                    layer_outputs[layer_idx] = input[0][0].detach().clone()  # [seq, hidden]

                return hook_fn

            # Hook the attention output projection
            target_module = hooks_model.model.layers[layer_idx].self_attn.o_proj
            hook = target_module.register_forward_hook(create_hook(layer_idx))

            try:
                with torch.no_grad():
                    input_ids = test_tokens.to(hooks_model.device)
                    _ = hooks_model(input_ids=input_ids)

                if layer_idx in layer_outputs:
                    hooks_tensor = layer_outputs[layer_idx]
                    layer_results["hooks"] = {
                        "tensor": hooks_tensor.cpu(),
                        "shape": hooks_tensor.shape,
                        "dtype": hooks_tensor.dtype,
                        "device": str(hooks_tensor.device),
                        "requires_grad": hooks_tensor.requires_grad,
                        "norm": torch.norm(hooks_tensor).item(),
                        "mean": hooks_tensor.mean().item(),
                        "std": hooks_tensor.std().item(),
                        "min": hooks_tensor.min().item(),
                        "max": hooks_tensor.max().item(),
                    }
                    print(f"     ✅ Shape: {hooks_tensor.shape}")
                    print(f"     ✅ Dtype: {hooks_tensor.dtype}")
                    print(f"     ✅ Requires grad: {hooks_tensor.requires_grad}")
                    print(f"     ✅ Norm: {torch.norm(hooks_tensor).item():.6f}")
                else:
                    layer_results["hooks"] = {"error": "No activation captured"}
                    print("     ❌ No activation captured")

            finally:
                hook.remove()

        except Exception as e:
            layer_results["hooks"] = {"error": str(e)}
            print(f"     ❌ Error: {e}")

        # ===== METHOD 2: nnsight =====
        print("   [2/2] Extracting with nnsight...")
        try:
            torch.set_grad_enabled(False)  # Match original DAC

            with nnsight_model.trace(validate=False) as tracer:
                with tracer.invoke(test_tokens, scan=False) as _:
                    # Extract using original DAC method
                    attn_hook_name = nnsight_config["attn_hook_names"][layer_idx]

                    # Use rgetattr to access the module like original DAC
                    from src.utils.model_utils import rgetattr

                    activation_saved = rgetattr(nnsight_model, attn_hook_name).input[0][0].save()

            nnsight_raw = activation_saved.value.detach().clone()
            # Apply same indexing as hooks: remove batch dimension
            nnsight_tensor = nnsight_raw[0] if nnsight_raw.dim() == 3 else nnsight_raw
            # Convert to same dtype for fair comparison
            nnsight_tensor_converted = nnsight_tensor.to(torch.float16)

            layer_results["nnsight"] = {
                "tensor": nnsight_tensor_converted.cpu(),  # Use converted for comparison
                "tensor_original": nnsight_tensor.cpu(),  # Keep original for reference
                "shape": nnsight_tensor.shape,
                "dtype": nnsight_tensor.dtype,
                "dtype_converted": nnsight_tensor_converted.dtype,
                "device": str(nnsight_tensor.device),
                "requires_grad": nnsight_tensor.requires_grad,
                "norm": torch.norm(nnsight_tensor_converted).item(),
                "mean": nnsight_tensor_converted.mean().item(),
                "std": nnsight_tensor_converted.std().item(),
                "min": nnsight_tensor_converted.min().item(),
                "max": nnsight_tensor_converted.max().item(),
            }
            print(f"     ✅ Shape: {nnsight_tensor.shape}")
            print(f"     ✅ Dtype: {nnsight_tensor.dtype} -> {nnsight_tensor_converted.dtype} (converted)")
            print(f"     ✅ Requires grad: {nnsight_tensor.requires_grad}")
            print(f"     ✅ Norm: {torch.norm(nnsight_tensor_converted).item():.6f} (converted)")

        except Exception as e:
            layer_results["nnsight"] = {"error": str(e)}
            print(f"     ❌ Error: {e}")

        # ===== COMPARISON =====
        if "hooks" in layer_results and "nnsight" in layer_results:
            hooks_data = layer_results["hooks"]
            nnsight_data = layer_results["nnsight"]

            if "tensor" in hooks_data and "tensor" in nnsight_data:
                hooks_tensor = hooks_data["tensor"]
                nnsight_tensor = nnsight_data["tensor"]

                print(f"   🔍 COMPARISON:")

                # Shape comparison
                if hooks_tensor.shape == nnsight_tensor.shape:
                    print(f"     ✅ Shapes match: {hooks_tensor.shape}")

                    # Direct tensor comparison
                    cosine_sim = torch.cosine_similarity(hooks_tensor.flatten(), nnsight_tensor.flatten(), dim=0).item()
                    l2_dist = torch.norm(hooks_tensor - nnsight_tensor).item()

                    print(f"     📊 Cosine similarity: {cosine_sim:.8f}")
                    print(f"     📊 L2 distance: {l2_dist:.8f}")

                    # Metadata comparison
                    print(f"     📊 Dtype match: {hooks_data['dtype'] == nnsight_data['dtype']}")
                    print(
                        f"     📊 Requires grad match: {hooks_data['requires_grad'] == nnsight_data['requires_grad']}"
                    )
                    print(f"     📊 Norm diff: {abs(hooks_data['norm'] - nnsight_data['norm']):.8f}")
                    print(f"     📊 Mean diff: {abs(hooks_data['mean'] - nnsight_data['mean']):.8f}")

                    # Similarity assessment
                    if cosine_sim > 0.99999:
                        print(f"     🎯 NEARLY IDENTICAL!")
                    elif cosine_sim > 0.99:
                        print(f"     ✅ Very high similarity")
                    elif cosine_sim > 0.9:
                        print(f"     ⚠️  High similarity")
                    elif cosine_sim > 0.5:
                        print(f"     ❓ Moderate similarity")
                    else:
                        print(f"     ❌ Low similarity - fundamental difference!")

                    layer_results["comparison"] = {
                        "cosine_similarity": cosine_sim,
                        "l2_distance": l2_dist,
                        "shapes_match": True,
                    }

                else:
                    print(f"     ❌ Shape mismatch: {hooks_tensor.shape} vs {nnsight_tensor.shape}")
                    layer_results["comparison"] = {
                        "shapes_match": False,
                        "hooks_shape": hooks_tensor.shape,
                        "nnsight_shape": nnsight_tensor.shape,
                    }
            else:
                print(f"     ❌ One or both extractions failed")

        results[f"layer_{layer_idx}"] = layer_results

    # ===== SUMMARY =====
    print(f"\n📊 SUMMARY ACROSS ALL LAYERS:")
    print("=" * 70)

    similarities = []
    for layer_name, layer_data in results.items():
        if "comparison" in layer_data and "cosine_similarity" in layer_data["comparison"]:
            sim = layer_data["comparison"]["cosine_similarity"]
            similarities.append(sim)
            print(f"{layer_name}: cosine similarity = {sim:.6f}")
        else:
            print(f"{layer_name}: comparison failed")

    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)

        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Average similarity: {avg_sim:.6f}")
        print(f"   Range: {min_sim:.6f} - {max_sim:.6f}")
        print(f"   Layers tested: {len(similarities)}")

        if avg_sim > 0.99:
            print(f"   🎉 CONCLUSION: nnsight and hooks are nearly identical!")
            print(f"   📝 The ~0.18 DAC similarity must be due to other factors")
        elif avg_sim > 0.9:
            print(f"   ✅ CONCLUSION: nnsight and hooks are very similar")
            print(f"   📝 Small differences might contribute to DAC similarity gap")
        else:
            print(f"   🎯 CONCLUSION: Fundamental extraction difference found!")
            print(f"   📝 This explains the ~0.18 DAC similarity ceiling")

    return results


if __name__ == "__main__":
    results = test_nnsight_vs_hooks_extraction()
    print("\n🔬 Direct extraction comparison completed!")
