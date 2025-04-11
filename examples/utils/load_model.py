"""
Model loading utilities for Wisent Guard examples
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_for_example(model_name, device=None):
    """
    Load a model and tokenizer for use in examples.
    
    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on (auto-detected if None)
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon GPU via MPS")
        else:
            device = "cpu"
            print("Using CPU")
    
    # Load model with optimizations
    print(f"Loading model on {device} device...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision for better performance
        device_map=device,
        trust_remote_code=True
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set padding token to EOS token")
    
    return model, tokenizer 