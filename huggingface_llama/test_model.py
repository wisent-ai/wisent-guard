#!/usr/bin/env python3
"""
Test script for Wisent-Guarded Llama model.
"""

def test_local_model():
    """Test the model when loaded locally."""
    print("🧪 Testing Wisent-Guarded Llama Model")
    print("=" * 50)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("📦 Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            ".",  # Current directory
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu"  # Use CPU for testing
        )
        
        tokenizer = AutoTokenizer.from_pretrained(".")
        
        # Set tokenizer for wisent-guard
        if hasattr(model, 'set_tokenizer'):
            model.set_tokenizer(tokenizer)
        
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        print(f"✅ Wisent-guard enabled: {getattr(model.config, 'wisent_enabled', False)}")
        
        # Test basic functionality
        print("\n🔬 Testing basic functionality...")
        
        # Test safety checking
        if hasattr(model, 'is_harmful'):
            safe_text = "Hello, how are you?"
            is_harmful = model.is_harmful(safe_text)
            print(f"Safety check for '{safe_text}': {'❌ Harmful' if is_harmful else '✅ Safe'}")
        
        # Test generation
        print("\n🚀 Testing text generation...")
        prompt = "Tell me about renewable energy."
        
        inputs = tokenizer([prompt], return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Response: {response[len(prompt):].strip()}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import torch
    test_local_model() 