#!/usr/bin/env python
"""
Script to create hallucination detection vectors for all model layers.
Based on save_activations_for_one_layer.py.
"""

import argparse
import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import wisent_guard
from wisent_guard import ActivationGuard

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

def load_truthfulqa_data(filepath):
    """Load TruthfulQA dataset"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples from {filepath}")
    return df

def create_training_pairs(train_df, max_pairs=0):
    """
    Create training pairs for the activation guard.
    Each pair consists of:
    - harmful: A question-incorrect answer pair (hallucination)
    - harmless: The same question with a correct answer
    
    Using proper Llama chat formatting with user/assistant tags.
    
    Args:
        train_df: DataFrame containing TruthfulQA training data
        max_pairs: Maximum number of pairs to create (0 for all)
    """
    training_pairs = []
    
    # If max_pairs is set, sample the dataframe
    if max_pairs > 0 and max_pairs < len(train_df):
        train_df = train_df.sample(max_pairs, random_state=42)
        print(f"Sampling {max_pairs} examples from training data")
    
    for _, row in train_df.iterrows():
        question = row['Question']
        
        # Get correct answers
        correct_answers = row['Correct Answers'].split(';')
        best_answer = row['Best Answer']
        
        # Get incorrect answers
        incorrect_answers = row['Incorrect Answers'].split(';')
        
        # Use the best answer if available, otherwise use the first correct answer
        harmless_answer = best_answer if best_answer else correct_answers[0]
        
        # Use the first incorrect answer
        harmful_answer = incorrect_answers[0] if incorrect_answers else None
        
        if harmful_answer and harmless_answer:
            # Format with proper user/assistant tags to match Llama 3.1 chat format
            training_pairs.append({
                "harmful": f"{question} {harmful_answer}",
                "harmless": f"{question} {harmless_answer}"
            })
    
    print(f"Created {len(training_pairs)} training pairs")
    return training_pairs

def train_guard_for_layer(model, tokenizer, training_pairs, layer_number, save_dir, device, threshold, force_llama_format=True):
    """
    Train the guard for a specific layer.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        training_pairs: List of training pairs
        layer_number: Layer to monitor
        save_dir: Base directory to save vectors
        device: Device to run model on
        threshold: Similarity threshold
        force_llama_format: Force Llama 3.1 format (default: True)
        
    Returns:
        Path to the saved vectors
    """
    print(f"\nProcessing layer {layer_number}...")
    
    # Create layer-specific output directory
    model_name_short = model.config._name_or_path.split('/')[-1] if hasattr(model.config, "_name_or_path") else "model"
    output_dir = f"{save_dir}/{model_name_short}_layer_{layer_number}"
    
    # Initialize wisent-guard for this layer
    print(f"Initializing wisent-guard for hallucination detection with layer {layer_number}")
    guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=[layer_number],
        threshold=threshold,
        save_dir=output_dir,
        device=device,
        force_llama_format=force_llama_format
    )
    
    # Train the guard on TruthfulQA pairs
    print(f"Training wisent-guard on TruthfulQA hallucination pairs for layer {layer_number}...")
    guard.train_on_phrase_pairs(training_pairs, category="hallucination")
    
    print(f"Layer {layer_number} - Available categories: {guard.get_available_categories()}")
    return output_dir

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
    
    # Model configuration
    print(f"Loading model: {args.model_name}")
    load_kwargs = {
        "torch_dtype": torch.float16 if args.half_precision else torch.float32,
    }
    
    # Configure device mapping based on device type
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    elif device == "mps":
        # For MPS, we'll load on CPU first then move to MPS
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = device
    
    if args.load_in_8bit and device != "mps":  # 8-bit quantization not supported on MPS
        load_kwargs["load_in_8bit"] = True
    elif args.load_in_8bit and device == "mps":
        print("Warning: 8-bit quantization not supported on MPS. Falling back to standard precision.")
        
    # Initialize model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **load_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Move model to MPS if needed (after loading)
        if device == "mps" and load_kwargs["device_map"] == "cpu":
            model = model.to(device)
            
        print(f"Model loaded successfully: {model.__class__.__name__}")
        
        # Calculate model parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {(param_count / 1e9):.2f} billion parameters")
        
        # Get number of layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            print(f"Number of layers: {num_layers}")
        else:
            print("Could not determine number of layers")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load TruthfulQA data
    train_df = load_truthfulqa_data(args.train_data)
    
    # Create training pairs
    print("\nCreating training pairs from TruthfulQA...")
    training_pairs = create_training_pairs(train_df, args.max_pairs)
    
    # Determine which layers to process
    if args.start_layer is not None and args.end_layer is not None:
        # Process specified range of layers
        layers_to_process = range(args.start_layer, args.end_layer + 1)
    elif args.layers:
        # Process specific layers
        layers_to_process = [int(layer) for layer in args.layers.split(',')]
    else:
        # Process all layers
        layers_to_process = range(num_layers)
    
    print(f"\nProcessing {len(layers_to_process)} layers: {list(layers_to_process)}")
    
    # Process each layer
    output_dirs = []
    for layer in layers_to_process:
        try:
            # Skip if layer is in skip_layers
            if args.skip_layers and str(layer) in args.skip_layers.split(','):
                print(f"Skipping layer {layer} as requested")
                continue
                
            output_dir = train_guard_for_layer(
                model=model,
                tokenizer=tokenizer,
                training_pairs=training_pairs,
                layer_number=layer,
                save_dir=args.save_dir,
                device=device,
                threshold=args.threshold,
                force_llama_format=args.force_llama_format
            )
            
            output_dirs.append(output_dir)
            print(f"Successfully processed layer {layer}")
            
        except Exception as e:
            print(f"Error processing layer {layer}: {e}")
            if args.continue_on_error:
                print(f"Continuing with next layer...")
            else:
                print(f"Stopping due to error. Use --continue-on-error to continue despite errors.")
                break
    
    print(f"\nProcessed {len(output_dirs)} layers successfully")
    print(f"Output directories: {output_dirs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create hallucination vectors for all layers")
    
    # Data paths
    parser.add_argument("--train-data", type=str, default="evaluation/data/TruthfulQA_en_train.csv", 
                        help="Path to TruthfulQA training data")
    parser.add_argument("--save-dir", type=str, default="./hallucination_guard_data", 
                        help="Base directory to save vectors")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="Name of the model to use")
    parser.add_argument("--cpu-only", action="store_true", 
                        help="Force CPU usage even if CUDA or MPS is available")
    parser.add_argument("--no-mps", action="store_true",
                        help="Disable MPS (Apple Silicon GPU) even if available")
    parser.add_argument("--half-precision", action="store_true", 
                        help="Load model in half precision (fp16)")
    parser.add_argument("--load-in-8bit", action="store_true", 
                        help="Load model in 8-bit quantization (not supported on MPS)")
    
    # Vector creation parameters
    parser.add_argument("--threshold", type=float, default=0.2, 
                        help="Similarity threshold (lower is more sensitive)")
    parser.add_argument("--max-pairs", type=int, default=0, 
                        help="Maximum number of training pairs to create (0 for all)")
    
    # Layer selection options
    parser.add_argument("--start-layer", type=int, default=None,
                        help="First layer to process (inclusive)")
    parser.add_argument("--end-layer", type=int, default=None,
                        help="Last layer to process (inclusive)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of specific layers to process (e.g., '0,5,10,15')")
    parser.add_argument("--skip-layers", type=str, default=None,
                        help="Comma-separated list of layers to skip (e.g., '4,7,12')")
    
    # Error handling
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue processing other layers if one layer fails")
    
    # Llama format settings
    parser.add_argument("--force-llama-format", action="store_true", default=True,
                        help="Force Llama 3.1 token format (default: True)")
    
    args = parser.parse_args()
    main(args)
