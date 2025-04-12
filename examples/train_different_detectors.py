#!/usr/bin/env python
"""
Example script that demonstrates how to train threshold-based and
classifier-based detectors using the wisent-guard library.

This script:
1. Loads a language model (defaults to Llama-3.1-8B-Instruct)
2. Creates training data with good/bad code pairs
3. Trains a threshold-based detector
4. Trains a classifier-based detector
5. Saves the trained models for later use

Usage:
python train_detectors.py [--model MODEL_NAME] [--cpu-only]
"""

import os
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import wisent-guard
from wisent_guard import ActivationGuard

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train threshold-based and classifier-based detectors for code quality")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Name of the model to use (default: Llama-3.1-8B-Instruct)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU usage even if CUDA or MPS is available")
    parser.add_argument("--layer", type=int, default=15,
                        help="Layer to monitor for activations (default: 15)")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Similarity threshold for bad code detection (default: 0.2)")
    parser.add_argument("--threshold-save-dir", type=str, default="./models/code_threshold_detector",
                        help="Directory to save threshold detector data")
    parser.add_argument("--classifier-path", type=str, default="./models/code_quality_classifier.pkl",
                        help="Path to save classifier model")
    parser.add_argument("--half-precision", action="store_true", 
                        help="Load model in half precision (fp16)")
    parser.add_argument("--load-in-8bit", action="store_true", 
                        help="Load model in 8-bit quantization (not supported on MPS)")
    return parser.parse_args()

def get_device(cpu_only=False):
    """Determine the appropriate device to use."""
    if cpu_only:
        return "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def format_prompt(message, system_prompt=None):
    """Format prompt for Llama 3.1 Instruct models using the special tokens"""
    # Llama 3.1 special tokens
    BEGIN_TEXT = "<|begin_of_text|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    EOT = "<|eot_id|>"
    
    if system_prompt:
        # Format with system prompt
        formatted = f"{BEGIN_TEXT}{START_HEADER}system{END_HEADER}\n{system_prompt}{EOT}{START_HEADER}user{END_HEADER}\n{message}{EOT}{START_HEADER}assistant{END_HEADER}"
    else:
        # Format without system prompt
        formatted = f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{message}{EOT}{START_HEADER}assistant{END_HEADER}"
    
    return formatted

def load_model(model_name, device, half_precision=False, load_in_8bit=False):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Define loading parameters based on device
    load_kwargs = {
        "torch_dtype": torch.float16 if (device != "cpu" and half_precision) else torch.float32,
    }
    
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["device_map"] = "cpu"  # Load on CPU first for MPS
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move model to MPS if needed
    if device == "mps" and load_kwargs["device_map"] == "cpu":
        model = model.to(device)
    
    print(f"Model loaded successfully: {model.__class__.__name__}")
    print(f"Model has {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9):.2f} billion parameters")
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(f"Number of layers: {len(model.model.layers)}")
    
    return model, tokenizer

def create_training_pairs():
    """Create training pairs for good/bad code examples."""
    pairs = []
    
    # Example 1: Bad code - SQL injection vulnerability
    pairs.append({
        "bad_code": """
def login(username, password):
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    return database.execute(query)
        """,
        "good_code": """
def login(username, password):
    query = "SELECT * FROM users WHERE username = %s AND password = %s"
    return database.execute(query, (username, password))
        """
    })
    
    # Example 2: Bad code - memory leak
    pairs.append({
        "bad_code": """
def process_large_data(data):
    results = []
    for item in data:
        results.append(process_item(item))
    # No cleanup or resource management
    return results
        """,
        "good_code": """
def process_large_data(data):
    results = []
    try:
        for item in data:
            results.append(process_item(item))
        return results
    finally:
        # Ensure cleanup happens
        gc.collect()
        """
    })
    
    # Example 3: Bad code - inefficient algorithm
    pairs.append({
        "bad_code": """
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates
        """,
        "good_code": """
def find_duplicates(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)
        """
    })
    
    print(f"Created {len(pairs)} training pairs for code quality detection")
    return pairs

def train_threshold_detector(model, tokenizer, training_pairs, args):
    """Train a threshold-based detector for code quality."""
    print("\n==== Training Threshold-Based Code Quality Detector ====")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.threshold_save_dir, exist_ok=True)
    
    # Initialize the ActivationGuard
    threshold_guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=[args.layer],
        threshold=args.threshold,
        save_dir=args.threshold_save_dir,
        device=model.device,
        force_format="llama"
    )
    
    # Rename dictionary keys to match what ActivationGuard expects
    formatted_pairs = []
    for pair in training_pairs:
        formatted_pairs.append({
            "harmful": pair["bad_code"],
            "harmless": pair["good_code"]
        })
    
    # Train on phrase pairs
    start_time = time.time()
    threshold_guard.train_on_phrase_pairs(formatted_pairs, category="bad_code")
    train_time = time.time() - start_time
    
    print(f"Threshold-based code quality detector trained in {train_time:.2f} seconds")
    print(f"Using threshold: {args.threshold}")
    print(f"Threshold detector data saved to: {args.threshold_save_dir}")
    
    return threshold_guard

def train_classifier_detector(model, tokenizer, training_pairs, args):
    """Train a classifier-based detector for code quality."""
    print("\n==== Training Classifier-Based Code Quality Detector ====")
    
    # First, initialize a regular guard to collect activations
    temp_guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=[args.layer],
        threshold=args.threshold,
        save_dir="./temp_data",
        device=model.device,
        force_format="llama"
    )
    
    # Collect activations for training
    print("Collecting activations for training classifier...")
    
    bad_code_activations = []
    good_code_activations = []
    
    # Process training pairs with progress tracking
    for i, pair in enumerate(training_pairs):
        print(f"Processing pair {i+1}/{len(training_pairs)}...")
        
        # Process bad code example
        print(f"  - Processing bad code example...")
        temp_guard.monitor.reset()
        success = temp_guard._prepare_activations(pair["bad_code"])
        if success:
            activations = temp_guard.monitor.get_activations()
            layer = args.layer
            if layer in activations:
                activation_tensor = activations[layer]
                # Ensure tensor is detached and on CPU
                bad_code_activations.append({
                    "activations": activation_tensor.detach().cpu().flatten(),
                    "layer": layer,
                    "token_text": "bad_code",
                    "is_harmful": True  # Keep this as True for compatibility with API
                })
                print(f"    ✓ Successfully collected bad code activation")
            else:
                print(f"    ✗ Layer {layer} not found in activations")
        else:
            print(f"    ✗ Failed to prepare activations for bad code example")
        
        # Process good code example
        print(f"  - Processing good code example...")
        temp_guard.monitor.reset()
        success = temp_guard._prepare_activations(pair["good_code"])
        if success:
            activations = temp_guard.monitor.get_activations()
            layer = args.layer
            if layer in activations:
                activation_tensor = activations[layer]
                # Ensure tensor is detached and on CPU
                good_code_activations.append({
                    "activations": activation_tensor.detach().cpu().flatten(),
                    "layer": layer,
                    "token_text": "good_code",
                    "is_harmful": False  # Keep this as False for compatibility with API
                })
                print(f"    ✓ Successfully collected good code activation")
            else:
                print(f"    ✗ Layer {layer} not found in activations")
        else:
            print(f"    ✗ Failed to prepare activations for good code example")
    
    print(f"Collected {len(bad_code_activations)} bad code and {len(good_code_activations)} good code activations")
    
    # Ensure path exists
    os.makedirs(os.path.dirname(args.classifier_path), exist_ok=True)
    
    # Train classifier
    from wisent_guard.classifier import ActivationClassifier
    
    print("Training classifier (this may take a few minutes)...")
    start_time = time.time()
    try:
        classifier = ActivationClassifier.create_from_activations(
            harmful_activations=bad_code_activations,  # Using existing API
            harmless_activations=good_code_activations,  # Using existing API
            model_type="logistic",  # Use logistic regression
            save_path=args.classifier_path,
            threshold=0.5,
            positive_class_label="bad_code",  # Change label to reflect code quality
            test_size=0.2,
            random_state=42,
            device=model.device  # Use model's device (GPU/MPS) for training
        )
        train_time = time.time() - start_time
        
        print(f"Classifier trained in {train_time:.2f} seconds and saved to {args.classifier_path}")
    except Exception as e:
        print(f"Error training classifier: {e}")
        print(f"This may be due to device compatibility issues or tensor size problems.")
        print(f"Try running with --cpu-only flag or reducing the size of the examples.")

def main():
    """Main function."""
    args = parse_args()
    
    # Set up device
    device = get_device(args.cpu_only)
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(args.model, device, args.half_precision, args.load_in_8bit)
    
    # Create training data
    training_pairs = create_training_pairs()
    
    # Train threshold-based detector
    train_threshold_detector(model, tokenizer, training_pairs, args)
    
    # Train classifier-based detector
    train_classifier_detector(model, tokenizer, training_pairs, args)
    
    print("\n==== Training Complete ====")
    print(f"Threshold detector data saved to: {args.threshold_save_dir}")
    print(f"Classifier model saved to: {args.classifier_path}")
    print("\nYou can now use these trained code quality detectors in your applications.")

if __name__ == "__main__":
    main() 