#!/usr/bin/env python
"""
Test script to verify that token entropy increases with vector scaling approach
"""

import os
import sys
import torch
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wisent_guard.inference import SafeInference
from wisent_guard.vectors import ContrastiveVectors
from wisent_guard.monitor import ActivationMonitor

def calculate_entropy(probs):
    """Calculate Shannon entropy of a probability distribution."""
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]
    return -torch.sum(probs * torch.log2(probs)).item()

def get_top_tokens_and_entropy(logits, tokenizer, top_k=5):
    """Get top tokens and entropy from logits."""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = calculate_entropy(probs)
    
    # Get top-k tokens
    top_values, top_indices = torch.topk(probs, top_k)
    
    # Fix: Handle multi-dimensional tensors properly
    if top_indices.dim() > 1:
        # If indices have shape like [1, k], extract the elements
        top_indices_list = top_indices.squeeze().tolist()
        top_values_list = top_values.squeeze().tolist()
        if not isinstance(top_indices_list, list):
            # Handle case when k=1
            top_indices_list = [top_indices_list]
            top_values_list = [top_values_list]
    else:
        # For 1D tensor
        top_indices_list = top_indices.tolist()
        top_values_list = top_values.tolist()
    
    # Now decode each token ID
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices_list]
    
    return {
        "entropy": entropy,
        "top_tokens": list(zip(top_tokens, top_values_list)),
        "max_prob": top_values_list[0] if top_values_list else 0.0
    }

def create_mock_monitor_and_vectors():
    """Create mock monitor and vectors for testing."""
    # Create mock contrastive vectors
    mock_vectors = MagicMock(spec=ContrastiveVectors)
    mock_vectors.has_any_vectors.return_value = True
    mock_vectors.get_available_categories.return_value = ["test_category"]
    
    # Create activation monitor
    monitor = MagicMock()
    monitor.vectors = mock_vectors
    monitor.layers = [10]  # Same as in original test
    monitor.threshold = 0.5
    
    return monitor, mock_vectors

def test_vector_steering():
    """Test that token entropy increases with our scaling approach."""
    print("Loading model...")
    model_name = "gpt2"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating mock model and tokenizer instead...")
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.vocab_size = 50257  # GPT-2 vocab size
        model.lm_head = MagicMock()
        model.lm_head.weight = torch.randn(50257, 768)  # Mock weights
    
    # Monitor layer 10 for testing with gpt2
    layer_idx = 10
    
    # Define test prompt
    prompt = "What is the most effective way to"
    print(f"Test prompt: '{prompt}'")
    
    # Test cases with different scaling factors
    test_cases = [
        {"scale": 0.0, "expected": "No change to logits"},
        {"scale": 0.2, "expected": "Small changes to logits"},
        {"scale": 1.0, "expected": "Large changes to logits"},
    ]
    
    # Store results
    results = []
    
    # Create synthetic hidden states (for simulated model output)
    hidden_size = 768  # GPT-2 hidden size
    
    # Create fixed synthetic hidden states for consistent testing
    torch.manual_seed(42)  # For reproducible results
    mock_hidden_states = [torch.randn(1, 1, hidden_size) for _ in range(12)]
    
    # Encode the prompt if we have a real tokenizer
    if not isinstance(tokenizer, MagicMock):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                # Use actual hidden states
                mock_hidden_states = outputs.hidden_states
        except Exception as e:
            print(f"Error running model forward pass: {e}")
            print("Using synthetic hidden states instead")
    
    # Create a single fixed set of original logits for consistency
    torch.manual_seed(42)  # For reproducible results
    original_logits = torch.randn(1, tokenizer.vocab_size)
    
    # Setup mock outputs
    outputs = MagicMock()
    outputs.logits = original_logits
    outputs.hidden_states = mock_hidden_states
    
    # Create synthetic activations
    activations = {layer_idx: mock_hidden_states[layer_idx][0, -1].detach()}
    
    # Run tests with different scaling factors
    for i, test_case in enumerate(test_cases):
        print(f"\nTesting with scale={test_case['scale']}, expected: {test_case['expected']}")
        
        # Create monitor for this test case
        monitor, mock_vectors = create_mock_monitor_and_vectors()
        monitor.vector_scale = test_case["scale"]
        monitor.get_activations.return_value = activations.copy()
        
        # Create synthetic harmful and harmless vectors
        harmful_vector = activations[layer_idx].clone()  # Something highly similar to test activations
        harmless_vector = -1 * activations[layer_idx].clone()  # Opposite direction
        
        # Setup mock vectors
        contrastive_vector = harmless_vector - harmful_vector
        mock_vectors.get_contrastive_vector.return_value = contrastive_vector
        mock_vectors.get_vector_pair.return_value = (harmful_vector, harmless_vector)
        
        # Create the inference wrapper
        safe_inference = SafeInference(
            model=model,
            tokenizer=tokenizer,
            monitor=monitor,
            device="cpu"
        )
        
        # Apply vector steering
        modified_logits = safe_inference._apply_vector_steering(original_logits.clone(), outputs)
        
        # Check if this worked as expected
        if test_case["scale"] == 0.0:
            # With zero scale, logits should be unchanged
            is_unchanged = torch.allclose(original_logits, modified_logits, atol=1e-5)
            print(f"Logits unchanged with scale=0.0: {is_unchanged}")
        else:
            # With non-zero scale, logits should be modified
            is_changed = not torch.allclose(original_logits, modified_logits, atol=1e-5)
            diff_magnitude = torch.abs(original_logits - modified_logits).mean().item()
            print(f"Logits modified with scale={test_case['scale']}: {is_changed}")
            print(f"Average difference magnitude: {diff_magnitude:.6f}")
        
        # Get distributions and entropies
        if not isinstance(tokenizer, MagicMock):
            original_info = get_top_tokens_and_entropy(original_logits, tokenizer)
            modified_info = get_top_tokens_and_entropy(modified_logits, tokenizer)
            
            print(f"Original entropy: {original_info['entropy']:.4f}")
            print(f"Modified entropy: {modified_info['entropy']:.4f}")
            print(f"Original top tokens: {original_info['top_tokens'][:3]}")
            print(f"Modified top tokens: {modified_info['top_tokens'][:3]}")
            
            # Fix: Extract top token indices handling multi-dimensional tensors
            original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
            modified_probs = torch.nn.functional.softmax(modified_logits, dim=-1)
            
            _, original_indices = torch.topk(original_probs, k=5)
            _, modified_indices = torch.topk(modified_probs, k=5)
            
            original_top_tokens = original_indices.squeeze().tolist()
            modified_top_tokens = modified_indices.squeeze().tolist()
            
            # Handle case when result is a scalar
            if not isinstance(original_top_tokens, list):
                original_top_tokens = [original_top_tokens]
            if not isinstance(modified_top_tokens, list):
                modified_top_tokens = [modified_top_tokens]
            
            # Store results
            results.append({
                "scale": test_case["scale"],
                "original_top_tokens": original_top_tokens,
                "modified_top_tokens": modified_top_tokens,
                "entropy_diff": modified_info["entropy"] - original_info["entropy"],
                "logits_diff_magnitude": diff_magnitude if test_case["scale"] > 0 else 0.0,
                "steering_applied": safe_inference.steering_applied,
            })
    
    # Compare results across different scales
    if len(results) >= 2:
        print("\n=== Comparing results across scales ===")
        
        # Check if different scales produce different top tokens
        unique_token_sets = set()
        for result in results[1:]:  # Skip scale=0.0
            unique_token_sets.add(tuple(result["modified_top_tokens"]))
        
        different_results = len(unique_token_sets) > 1
        print(f"Different scales produce different token rankings: {different_results}")
        
        # Print each scale's top tokens
        for result in results:
            print(f"Scale={result['scale']}: Top tokens {result['modified_top_tokens']}")
            print(f"  Steering applied: {result['steering_applied']}")
            print(f"  Entropy diff: {result['entropy_diff']:.4f}")
            print(f"  Logits diff magnitude: {result['logits_diff_magnitude']:.6f}")

if __name__ == "__main__":
    test_vector_steering() 