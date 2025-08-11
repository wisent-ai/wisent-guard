#!/usr/bin/env python3
"""
Simple test to debug the model loading and forward pass issues.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Test model loading and basic forward pass
def test_model_forward():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Model loaded on device: {model.device}")
    print(f"Model config - layers: {model.config.num_hidden_layers}, heads: {model.config.num_attention_heads}")
    print(f"Model config - hidden_size: {model.config.hidden_size}")

    # Test simple forward pass
    test_text = "Hello world"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    print(f"Input shape: {input_ids.shape}")

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            print(f"✅ Forward pass successful, logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    # Test forward pass with output_hidden_states
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            print(f"✅ Forward pass with hidden states successful")
            print(f"   Logits shape: {outputs.logits.shape}")
            print(f"   Number of hidden states: {len(outputs.hidden_states)}")
            print(f"   Hidden state shapes: {[h.shape for h in outputs.hidden_states[:3]]}")  # First 3
    except Exception as e:
        print(f"❌ Forward pass with hidden states failed: {e}")
        return False

    # Test generation
    try:
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=5, do_sample=False)
            print(f"✅ Generation successful, output shape: {generated.shape}")
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"   Generated: '{decoded}'")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_model_forward()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")
