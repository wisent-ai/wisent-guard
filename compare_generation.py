#!/usr/bin/env python
"""
Compare the generation process between baseline and guard.

This script demonstrates the exact prompting and generation process used by both
the baseline evaluation and the guarded evaluation to ensure they are consistent.
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent_guard import ActivationGuard

def format_prompt(message, system_prompt=None):
    """Format prompt for Llama 3.1 Instruct models"""
    if system_prompt:
        formatted = f"<|system|>\n{system_prompt}\n<|user|>\n{message}\n<|assistant|>"
    else:
        formatted = f"<|user|>\n{message}\n<|assistant|>"
    return formatted

def main(args):
    # Device configuration
    if args.cpu_only:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available() and not args.no_mps:
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Apply MPS patches if needed
    if device == "mps":
        print("Applying MPS compatibility patches...")
        try:
            from patches.fix_mps_hooks import apply_patches
            apply_patches()
            print("✅ Applied MPS compatibility patches")
        except ImportError:
            print("No MPS patches found or applied")
    
    # Model configuration
    print(f"Loading model: {args.model_name}")
    load_kwargs = {
        "torch_dtype": torch.float16 if args.half_precision else torch.float32,
    }
    
    # Configure device mapping
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    elif device == "mps":
        # For MPS, load on CPU first then move to MPS
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = device
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **load_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Move model to MPS if needed
    if device == "mps" and load_kwargs["device_map"] == "cpu":
        model = model.to(device)
    
    # Initialize guard
    print("\nInitializing wisent-guard")
    guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=[args.layer_number],
        threshold=args.threshold,
        save_dir=args.save_dir,
        device=device
    )
    
    # Train guard on a simple example if not using existing vectors
    if not args.use_existing_vectors:
        print("Training guard on a single example pair")
        guard.train_on_phrase_pairs([
            {
                "harmful": "The Earth is flat and scientific evidence has been covered up",
                "harmless": "The Earth is approximately spherical, as confirmed by scientific evidence"
            }
        ], category="hallucination")
    
    # Prepare test question
    question = "How many states are in the United States?"
    prompt_text = f"Answer the following question accurately: {question}"
    
    print("\n----- BASELINE GENERATION PROCESS -----")
    # Format prompt exactly as in evaluate_model
    formatted_prompt = f"<|user|>\n{prompt_text}\n<|assistant|>"
    print(f"Formatted prompt:\n{formatted_prompt}")
    
    # Generate response as in evaluate_model
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    
    # Set up generation parameters
    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Only add temperature if not default
    if args.temperature != 0.0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["do_sample"] = True
    
    print("\nGenerating baseline response...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **gen_kwargs
        )
    
    # Decode and extract response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in generated_text:
        baseline_response = generated_text.split("<|assistant|>")[-1].strip()
    else:
        baseline_response = generated_text
    
    print(f"\nBaseline response:\n{baseline_response}")
    
    print("\n----- GUARD GENERATION PROCESS -----")
    # Use the same prompt text
    print(f"Using the same prompt text: {prompt_text}")
    
    # Check if guard detects hallucination
    is_harmful = guard.is_harmful(formatted_prompt)
    similarity_score = guard.get_similarity(formatted_prompt) if is_harmful else None
    
    print(f"\nGuard detected harmful content: {is_harmful}")
    if similarity_score:
        print(f"Similarity score: {similarity_score:.4f}")
    
    # Generate response with guard
    print("\nGenerating response with guard...")
    result = guard.generate_safe_response(
        prompt_text,
        max_new_tokens=args.max_tokens
    )
    
    guard_response = result.get('response', '')
    blocked = result.get('blocked', False)
    reason = result.get('reason', '') if blocked else ''
    
    print(f"\nGuard response:\n{guard_response}")
    if blocked:
        print(f"\nResponse was blocked: {reason}")
    
    # Compare the responses
    print("\n----- COMPARISON -----")
    if baseline_response == guard_response:
        print("✓ The responses are IDENTICAL")
    else:
        print("✗ The responses are DIFFERENT")
        
        # Check if they start the same
        min_len = min(len(baseline_response), len(guard_response))
        same_prefix_len = 0
        for i in range(min_len):
            if baseline_response[i] == guard_response[i]:
                same_prefix_len += 1
            else:
                break
        
        if same_prefix_len > 0:
            print(f"Responses match for the first {same_prefix_len} characters")
            
        # Print exact differences for better analysis
        print("\nFirst point of divergence:")
        print(f"Baseline: ...{baseline_response[max(0, same_prefix_len-10):same_prefix_len+30]}...")
        print(f"Guard:    ...{guard_response[max(0, same_prefix_len-10):same_prefix_len+30]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare generation processes")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="Name of the model to use")
    parser.add_argument("--cpu-only", action="store_true", 
                        help="Force CPU usage even if CUDA or MPS is available")
    parser.add_argument("--no-mps", action="store_true",
                        help="Disable MPS (Apple Silicon GPU) even if available")
    parser.add_argument("--half-precision", action="store_true", 
                        help="Load model in half precision (fp16)")
    
    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=100, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Sampling temperature")
    
    # Guard parameters
    parser.add_argument("--layer-number", type=int, default=15, 
                        help="Layer number to monitor")
    parser.add_argument("--threshold", type=float, default=0.2, 
                        help="Similarity threshold for hallucination detection")
    parser.add_argument("--save-dir", type=str, default="./hallucination_guard_data", 
                        help="Directory to save/load vectors")
    parser.add_argument("--use-existing-vectors", action="store_true",
                        help="Use existing vectors instead of training new ones")
    
    args = parser.parse_args()
    main(args) 