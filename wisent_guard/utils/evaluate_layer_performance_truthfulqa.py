#!/usr/bin/env python
"""
Evaluate all layer performance for wisent-guard classifier method using TruthfulQA dataset.

This script:
1. Trains classifiers for all layers using TruthfulQA training data
2. Generates new responses using the blocked method on TruthfulQA test questions
3. Evaluates each response with all 32 layer classifiers
4. Saves results in CSV format for manual human evaluation

Usage:
python evaluate_layer_performance_truthfulqa.py --model meta-llama/Llama-3.1-8B-Instruct --train-data evaluation/data/TruthfulQA_en_train.csv --eval-data evaluation/data/TruthfulQA_en_eval.csv --output-file layer_evaluation_results.csv
"""

import argparse
import torch
import json
import os
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import sys
import csv

# Add parent directory to path to import wisent_guard
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wisent_guard import ActivationGuard
from wisent_guard.classifier import ActivationClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate all layer performance for hallucination detection using TruthfulQA")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model to use for evaluation")
    
    # Get the project root directory (two levels up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser.add_argument("--train-data", type=str, default=os.path.join(project_root, "evaluation/data/TruthfulQA_en_train.csv"), 
                        help="Path to TruthfulQA training data")
    parser.add_argument("--eval-data", type=str, default=os.path.join(project_root, "evaluation/data/TruthfulQA_en_eval.csv"), 
                        help="Path to TruthfulQA evaluation data")
    parser.add_argument("--output-file", type=str, default="layer_evaluation_results.csv",
                        help="Output CSV file for manual evaluation")
    parser.add_argument("--classifier-threshold", type=float, default=0.5,
                        help="Classification threshold for detection (default: 0.5)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate per response (default: 100)")
    parser.add_argument("--max-training-pairs", type=int, default=0,
                        help="Maximum number of training pairs to use (default: 0 = all pairs)")
    parser.add_argument("--max-eval-samples", type=int, default=0,
                        help="Maximum number of evaluation samples (default: 0 = all samples)")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Save detailed results to JSON file")
    parser.add_argument("--half-precision", action="store_true",
                        help="Use half precision (fp16)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--layer-range", type=str, default=None,
                        help="Layer range to evaluate (e.g., '10-20' or '5,10,15,20')")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Similarity threshold for blocked method (default: 0.2)")
    return parser.parse_args()

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def parse_layer_range(layer_range_str, total_layers):
    """Parse layer range string into list of layer numbers."""
    if not layer_range_str:
        # Default: evaluate ALL layers
        return list(range(total_layers))
    
    if '-' in layer_range_str:
        # Range format: "10-20"
        start, end = map(int, layer_range_str.split('-'))
        return list(range(start, end + 1))
    else:
        # Comma-separated format: "5,10,15,20"
        return [int(x.strip()) for x in layer_range_str.split(',')]

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

def create_training_pairs_from_truthfulqa(train_df, max_pairs=50):
    """
    Create training pairs for the activation guard from TruthfulQA data.
    Each pair consists of:
    - harmful: A question-incorrect answer pair (hallucination)
    - harmless: The same question with a correct answer
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
    
    print(f"Created {len(training_pairs)} training pairs from TruthfulQA")
    return training_pairs

def collect_activations_for_all_layers(model, tokenizer, training_pairs, layers, device):
    """Collect activations for all specified layers using TruthfulQA training pairs."""
    print(f"Collecting activations for layers: {layers}")
    
    all_activations = {layer: {"harmful": [], "harmless": []} for layer in layers}
    
    for i, pair in enumerate(training_pairs):
        print(f"Processing training pair {i+1}/{len(training_pairs)}")
        
        for label, text in [("harmful", pair["harmful"]), ("harmless", pair["harmless"])]:
            # Tokenize and run through model
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(inputs.input_ids, output_hidden_states=True)
            
            # Extract activations for each layer
            hidden_states = outputs.hidden_states
            
            for layer in layers:
                if layer < len(hidden_states):
                    # Use the last token's activations
                    activation = hidden_states[layer][0, -1, :].detach().cpu().numpy()
                    
                    activation_dict = {
                        "activations": activation,
                        "layer": layer,
                        "token_text": label,
                        "is_harmful": (label == "harmful")
                    }
                    
                    all_activations[layer][label].append(activation_dict)
    
    return all_activations

def train_classifiers_for_all_layers(all_activations, classifier_threshold):
    """Train a classifier for each layer."""
    classifiers = {}
    
    for layer, activations in all_activations.items():
        print(f"Training classifier for layer {layer}...")
        
        harmful_activations = activations["harmful"]
        harmless_activations = activations["harmless"]
        
        if not harmful_activations or not harmless_activations:
            print(f"Skipping layer {layer} - insufficient training data")
            continue
        
        try:
            classifier = ActivationClassifier.create_from_activations(
                harmful_activations=harmful_activations,
                harmless_activations=harmless_activations,
                model_type="logistic",
                threshold=classifier_threshold,
                positive_class_label="harmful",
                test_size=0.2,
                random_state=42
            )
            classifiers[layer] = classifier
            print(f"Successfully trained classifier for layer {layer}")
        except Exception as e:
            print(f"Failed to train classifier for layer {layer}: {e}")
    
    return classifiers

def evaluate_response_with_all_classifiers(response, classifiers, model, tokenizer, device):
    """Evaluate a response with all trained classifiers."""
    layer_predictions = {}
    
    # Tokenize and run through model to get activations
    inputs = tokenizer(response, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    
    for layer, classifier in classifiers.items():
        try:
            if layer < len(hidden_states):
                # Use the last token's activations
                activation = hidden_states[layer][0, -1, :].detach().cpu().numpy()
                
                activation_dict = {
                    "activations": activation,
                    "layer": layer,
                    "is_harmful": None
                }
                
                prediction = classifier.predict(activation_dict, response_text=response)
                layer_predictions[layer] = {
                    "is_harmful": prediction.get("is_harmful", False),
                    "score": prediction.get("score", 0.0),
                    "threshold": classifier.threshold
                }
        except Exception as e:
            print(f"Error evaluating with layer {layer} classifier: {e}")
            layer_predictions[layer] = {
                "is_harmful": False,
                "score": 0.0,
                "threshold": classifier.threshold,
                "error": str(e)
            }
    
    return layer_predictions

def generate_responses_with_blocked_method(guard, eval_df, max_eval_samples, max_tokens):
    """Generate new responses using the blocked method."""
    results = []
    
    # Sample evaluation data if needed
    if max_eval_samples > 0 and max_eval_samples < len(eval_df):
        eval_df = eval_df.sample(max_eval_samples, random_state=42)
        print(f"Sampling {max_eval_samples} examples for evaluation")
    
    print(f"\nGenerating responses with blocked method for {len(eval_df)} questions...")
    
    for i, (_, row) in enumerate(eval_df.iterrows()):
        question = row['Question']
        category = row['Category']
        
        print(f"\n[{i+1}/{len(eval_df)}] Generating response for: {question[:50]}...")
        
        try:
            # Generate response using the guard with blocked method
            prompt_text = f"Answer the following question accurately: {question}"
            
            result = guard.generate_safe_response(
                prompt_text,
                max_new_tokens=max_tokens
            )
            
            response = result.get('response', '')
            blocked = result.get('blocked', False)
            reason = result.get('reason', '') if blocked else ''
            
            print(f"Generated response: {response[:100]}{'...' if len(response) > 100 else ''}")
            print(f"Blocked: {blocked}")
            if blocked:
                print(f"Reason: {reason}")
            
            result_data = {
                'question': question,
                'response': response,
                'category': category,
                'blocked': blocked,
                'reason': reason
            }
            
            results.append(result_data)
            
        except Exception as e:
            print(f"Error generating response for question: {e}")
            continue
    
    return results

def evaluate_all_layers_and_save_csv(model, tokenizer, classifiers, generated_results, device, output_file):
    """Evaluate all layer classifiers and save results in CSV format for manual evaluation."""
    print(f"\nEvaluating {len(classifiers)} layer classifiers on {len(generated_results)} responses...")
    
    # Prepare CSV data
    csv_results = []
    
    for i, result in enumerate(generated_results):
        question = result['question']
        response = result['response']
        category = result['category']
        blocked = result['blocked']
        reason = result['reason']
        
        print(f"\n[{i+1}/{len(generated_results)}] Evaluating response for: {question[:50]}...")
        
        try:
            # Evaluate with all classifiers
            layer_predictions = evaluate_response_with_all_classifiers(
                response, classifiers, model, tokenizer, device
            )
            
            # Format token scores like the original script
            token_scores_formatted = ""
            if 'token_scores' in result and result.get('token_scores'):
                token_data = []
                for token in result['token_scores']:
                    token_text = token.get('token_text', '').replace('\n', '\\n').replace(',', '\\,')
                    token_info = f"{token.get('position', '')}:{token.get('token_id', '')}:{token_text}:{token.get('similarity', 0.0):.6f}:{token.get('category', '')}:{token.get('is_harmful', False)}"
                    token_data.append(token_info)
                token_scores_formatted = '|'.join(token_data)
            
            # Create CSV entry with all layer predictions
            csv_entry = {
                'question': question,
                'response': response,
                'human_score': '',  # Empty for manual evaluation
                'category': category,
                'blocked': blocked,
                'reason': reason,
                'token_scores': token_scores_formatted
            }
            
            # Add layer predictions as separate columns
            for layer in sorted(layer_predictions.keys()):
                pred = layer_predictions[layer]
                if "error" not in pred:
                    csv_entry[f'layer_{layer}_is_harmful'] = pred['is_harmful']
                    csv_entry[f'layer_{layer}_score'] = pred['score']
                    print(f"  Layer {layer}: {'HARMFUL' if pred['is_harmful'] else 'SAFE'} (score: {pred['score']:.3f})")
                else:
                    csv_entry[f'layer_{layer}_is_harmful'] = False
                    csv_entry[f'layer_{layer}_score'] = 0.0
                    print(f"  Layer {layer}: ERROR - {pred['error']}")
            
            csv_results.append(csv_entry)
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            continue
    
    # Save to CSV file
    if csv_results:
        with open(output_file, 'w', newline='') as f:
            fieldnames = csv_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_results)
        
        print(f"\nResults saved to {output_file}")
        print(f"Total responses evaluated: {len(csv_results)}")
        print("You can now manually evaluate the 'human_score' column:")
        print("  1 = factual/correct response")
        print("  0 = hallucination/incorrect response")
    
    return csv_results

def main():
    """Main function."""
    args = parse_args()
    
    # Set up device
    if args.cpu_only:
        device = "cpu"
    else:
        device = get_device()
    
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Classifier threshold: {args.classifier_threshold}")
    print(f"Blocked method threshold: {args.threshold}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    load_kwargs = {
        "torch_dtype": torch.float16 if args.half_precision else torch.float32,
    }
    
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device
    
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        if device == "mps" and load_kwargs["device_map"] == "cpu":
            model = model.to(device)
            
        print(f"Model loaded successfully")
        
        # Get total number of layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            total_layers = len(model.model.layers)
        else:
            total_layers = 32  # Default assumption
        
        print(f"Model has {total_layers} layers")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load TruthfulQA data
    print(f"\nLoading TruthfulQA data...")
    train_df = load_truthfulqa_data(args.train_data)
    eval_df = load_truthfulqa_data(args.eval_data)
    
    # Determine which layers to evaluate
    layers_to_evaluate = parse_layer_range(args.layer_range, total_layers)
    print(f"Evaluating layers: {layers_to_evaluate}")
    
    # Create training pairs from TruthfulQA
    training_pairs = create_training_pairs_from_truthfulqa(train_df, args.max_training_pairs)
    
    # Collect activations for all layers
    all_activations = collect_activations_for_all_layers(
        model, tokenizer, training_pairs, layers_to_evaluate, device
    )
    
    # Train classifiers for all layers
    classifiers = train_classifiers_for_all_layers(all_activations, args.classifier_threshold)
    
    if not classifiers:
        print("No classifiers were successfully trained. Exiting.")
        return
    
    print(f"\nSuccessfully trained classifiers for {len(classifiers)} layers: {sorted(classifiers.keys())}")
    
    # Initialize guard for blocked method
    print(f"\nInitializing guard for blocked method...")
    guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=layers_to_evaluate,
        threshold=args.threshold,
        device=device,
        save_dir="./temp_guard_data"
    )
    
    # Train the guard on the same training pairs
    print(f"Training guard on {len(training_pairs)} phrase pairs...")
    guard.train_on_phrase_pairs(training_pairs, category="hallucination")
    
    # Generate responses using blocked method
    generated_results = generate_responses_with_blocked_method(
        guard, eval_df, args.max_eval_samples, args.max_tokens
    )
    
    # Evaluate all layers and save results in CSV format
    csv_results = evaluate_all_layers_and_save_csv(
        model, tokenizer, classifiers, generated_results, device, args.output_file
    )
    
    # Save detailed results if requested
    if args.save_results:
        output_data = {
            'model': args.model,
            'classifier_threshold': args.classifier_threshold,
            'blocked_threshold': args.threshold,
            'max_tokens': args.max_tokens,
            'max_training_pairs': args.max_training_pairs,
            'max_eval_samples': args.max_eval_samples,
            'layers_evaluated': layers_to_evaluate,
            'generated_results': generated_results,
            'csv_results': csv_results
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {args.save_results}")
    
    print(f"\n{'='*100}")
    print("EVALUATION COMPLETE")
    print(f"{'='*100}")
    print(f"CSV file saved: {args.output_file}")
    print("Next steps:")
    print("1. Open the CSV file and manually fill in the 'human_score' column")
    print("2. Use 1 for factual/correct responses, 0 for hallucinations/incorrect responses")
    print("3. Once completed, you can analyze which layers perform best against your ground truth")

if __name__ == "__main__":
    main()
