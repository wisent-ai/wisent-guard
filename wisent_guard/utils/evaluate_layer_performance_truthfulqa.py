#!/usr/bin/env python
"""
Evaluate all layer performance for wisent-guard classifier method using TruthfulQA dataset.

This script:
1. Trains classifiers for all layers using TruthfulQA training data
2. Generates new responses using the blocked method on TruthfulQA test questions
3. Evaluates each response with all 32 layer classifiers
4. Uses human scores from guard_results.csv as ground truth to determine best performing layer

Usage:
python evaluate_layer_performance_truthfulqa.py --model meta-llama/Llama-3.1-8B-Instruct --train-data evaluation/data/TruthfulQA_en_train.csv --eval-data evaluation/data/TruthfulQA_en_eval.csv --guard-results wisent_guard/data/guard_results.csv
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

# Add parent directory to path to import wisent_guard
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wisent_guard import ActivationGuard
from wisent_guard.classifier import ActivationClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate all layer performance for hallucination detection using TruthfulQA and guard_results.csv")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model to use for evaluation")
    
    # Get the project root directory (two levels up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser.add_argument("--train-data", type=str, default=os.path.join(project_root, "evaluation/data/TruthfulQA_en_train.csv"), 
                        help="Path to TruthfulQA training data")
    parser.add_argument("--eval-data", type=str, default=os.path.join(project_root, "evaluation/data/TruthfulQA_en_eval.csv"), 
                        help="Path to TruthfulQA evaluation data")
    parser.add_argument("--guard-results", type=str, default=os.path.join(project_root, "wisent_guard/data/guard_results.csv"),
                        help="Path to guard_results.csv with human scores")
    parser.add_argument("--classifier-threshold", type=float, default=0.5,
                        help="Classification threshold for detection (default: 0.5)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate per response (default: 100)")
    parser.add_argument("--max-training-pairs", type=int, default=0,
                        help="Maximum number of training pairs to use (default: 0 = all pairs)")
    parser.add_argument("--max-eval-samples", type=int, default=0,
                        help="Maximum number of evaluation samples (default: 0 = all samples)")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Save results to JSON file")
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

def load_guard_results(filepath):
    """Load guard results with human scores"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} guard results from {filepath}")
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

def match_with_guard_results(generated_results, guard_results_df):
    """Match generated results with guard results by question to get human scores."""
    matched_results = []
    
    print(f"\nMatching {len(generated_results)} generated responses with guard results...")
    
    for result in generated_results:
        question = result['question']
        
        # Find matching question in guard results
        # Try exact match first
        matches = guard_results_df[guard_results_df['question'] == question]
        
        if len(matches) == 0:
            # Try partial match if exact match fails
            matches = guard_results_df[guard_results_df['question'].str.contains(question[:50], case=False, na=False)]
        
        if len(matches) > 0:
            # Use the first match
            guard_result = matches.iloc[0]
            human_score = guard_result['human_score']
            
            # Add human score to our result
            result['human_score'] = human_score
            matched_results.append(result)
            
            print(f"Matched question: {question[:50]}... -> Human score: {human_score}")
        else:
            print(f"No match found for question: {question[:50]}...")
    
    print(f"Successfully matched {len(matched_results)} out of {len(generated_results)} questions")
    return matched_results

def evaluate_all_layers_with_guard_results(model, tokenizer, classifiers, matched_results, device):
    """Evaluate all layer classifiers using the matched results with human scores."""
    evaluation_results = []
    
    print(f"\nEvaluating {len(classifiers)} layer classifiers on {len(matched_results)} matched responses...")
    
    for i, result in enumerate(matched_results):
        question = result['question']
        response = result['response']
        human_score = result['human_score']
        
        print(f"\n[{i+1}/{len(matched_results)}] Evaluating response for: {question[:50]}...")
        print(f"Human score: {human_score}")
        
        try:
            # Evaluate with all classifiers
            layer_predictions = evaluate_response_with_all_classifiers(
                response, classifiers, model, tokenizer, device
            )
            
            # Print layer predictions
            print("Layer predictions:")
            for layer in sorted(layer_predictions.keys()):
                pred = layer_predictions[layer]
                if "error" not in pred:
                    print(f"  Layer {layer}: {'HARMFUL' if pred['is_harmful'] else 'SAFE'} (score: {pred['score']:.3f})")
                else:
                    print(f"  Layer {layer}: ERROR - {pred['error']}")
            
            evaluation_result = {
                'question': question,
                'response': response,
                'human_score': human_score,
                'category': result['category'],
                'blocked': result['blocked'],
                'reason': result['reason'],
                'layer_predictions': layer_predictions
            }
            
            evaluation_results.append(evaluation_result)
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            continue
    
    return evaluation_results

def calculate_layer_metrics_with_human_scores(evaluation_results):
    """Calculate performance metrics for each layer using human scores as ground truth."""
    # Get all layers that were evaluated
    all_layers = set()
    for result in evaluation_results:
        all_layers.update(result['layer_predictions'].keys())
    
    layer_metrics = {}
    
    for layer in sorted(all_layers):
        # Filter results with human scores and this layer's predictions
        valid_results = []
        for result in evaluation_results:
            if (result.get('human_score') is not None and 
                layer in result['layer_predictions'] and
                'error' not in result['layer_predictions'][layer]):
                valid_results.append(result)
        
        if not valid_results:
            layer_metrics[layer] = {"error": "No valid evaluations"}
            continue
        
        # Calculate metrics for this layer
        # human_score: 1=factual, 0=hallucination
        # is_harmful: True=detected as hallucination, False=detected as safe
        tp = sum(1 for r in valid_results 
                if r['layer_predictions'][layer]['is_harmful'] and r['human_score'] == 0)  # Correctly detected hallucination
        fp = sum(1 for r in valid_results 
                if r['layer_predictions'][layer]['is_harmful'] and r['human_score'] == 1)  # Incorrectly flagged factual as harmful
        tn = sum(1 for r in valid_results 
                if not r['layer_predictions'][layer]['is_harmful'] and r['human_score'] == 1)  # Correctly allowed factual
        fn = sum(1 for r in valid_results 
                if not r['layer_predictions'][layer]['is_harmful'] and r['human_score'] == 0)  # Missed hallucination
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        
        # Additional statistics
        total_evaluated = len(valid_results)
        total_harmful_predictions = sum(1 for r in valid_results 
                                      if r['layer_predictions'][layer]['is_harmful'])
        total_human_hallucinations = sum(1 for r in valid_results if r['human_score'] == 0)
        total_human_factual = sum(1 for r in valid_results if r['human_score'] == 1)
        avg_score = sum(r['layer_predictions'][layer]['score'] for r in valid_results) / total_evaluated
        
        layer_metrics[layer] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'total_evaluated': total_evaluated,
            'total_harmful_predictions': total_harmful_predictions,
            'total_human_hallucinations': total_human_hallucinations,
            'total_human_factual': total_human_factual,
            'avg_score': avg_score
        }
    
    return layer_metrics

def print_all_layers_report_with_human_scores(layer_metrics, classifier_threshold):
    """Print a comprehensive report for all layers using human scores as ground truth."""
    print(f"\n{'='*100}")
    print("ALL LAYERS PERFORMANCE REPORT (Human Score Ground Truth)")
    print(f"{'='*100}")
    
    if not layer_metrics:
        print("No metrics available")
        return
    
    print(f"Classifier threshold used: {classifier_threshold}")
    
    # Find best performing layer
    valid_layers = {layer: metrics for layer, metrics in layer_metrics.items() 
                   if 'error' not in metrics}
    
    if valid_layers:
        best_layer = max(valid_layers.keys(), key=lambda l: valid_layers[l]['f1'])
        best_f1 = valid_layers[best_layer]['f1']
        
        print(f"Best performing layer: {best_layer} (F1: {best_f1:.3f})")
    
    # Summary table
    print(f"\n{'Layer':<6} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Acc':<6} {'TP':<3} {'FP':<3} {'TN':<3} {'FN':<3} {'H_Fact':<6} {'H_Hall':<6} {'Avg Score':<9}")
    print("-" * 100)
    
    for layer in sorted(layer_metrics.keys()):
        metrics = layer_metrics[layer]
        if 'error' in metrics:
            print(f"{layer:<6} ERROR: {metrics['error']}")
        else:
            print(f"{layer:<6} {metrics['f1']:<6.3f} {metrics['precision']:<6.3f} "
                  f"{metrics['recall']:<6.3f} {metrics['accuracy']:<6.3f} "
                  f"{metrics['tp']:<3} {metrics['fp']:<3} {metrics['tn']:<3} {metrics['fn']:<3} "
                  f"{metrics['total_human_factual']:<6} {metrics['total_human_hallucinations']:<6} "
                  f"{metrics['avg_score']:<9.3f}")
    
    # Detailed report for best layer
    if valid_layers:
        print(f"\n{'='*60}")
        print(f"DETAILED REPORT FOR BEST LAYER ({best_layer})")
        print(f"{'='*60}")
        
        best_metrics = valid_layers[best_layer]
        print(f"F1 Score: {best_metrics['f1']:.3f}")
        print(f"Precision: {best_metrics['precision']:.3f}")
        print(f"Recall: {best_metrics['recall']:.3f}")
        print(f"Accuracy: {best_metrics['accuracy']:.3f}")
        print(f"Total evaluated: {best_metrics['total_evaluated']}")
        print(f"Human-labeled hallucinations: {best_metrics['total_human_hallucinations']}")
        print(f"Human-labeled factual responses: {best_metrics['total_human_factual']}")
        print(f"Average prediction score: {best_metrics['avg_score']:.3f}")
        
        # Performance interpretation
        if best_metrics['f1'] >= 0.8:
            performance = "Excellent"
        elif best_metrics['f1'] >= 0.6:
            performance = "Good"
        elif best_metrics['f1'] >= 0.4:
            performance = "Fair"
        else:
            performance = "Poor"
        
        print(f"Overall performance: {performance}")
        
        if best_metrics['precision'] > best_metrics['recall']:
            print("This layer is conservative (high precision, lower recall)")
            print("→ Good at avoiding false positives, but may miss some hallucinations")
        elif best_metrics['recall'] > best_metrics['precision']:
            print("This layer is aggressive (high recall, lower precision)")
            print("→ Good at catching hallucinations, but may flag some factual content")
        else:
            print("This layer has balanced precision and recall")

def save_results_to_model_params(model_name, layer_metrics, classifier_threshold, max_tokens, layers_evaluated, dataset="truthfulqa_human_scores"):
    """Save evaluation results to model_params.json organized by model name."""
    model_params_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "constants",
        "model_params.json"
    )
    
    # Load existing data
    try:
        with open(model_params_path, 'r') as f:
            model_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        model_params = {}
    
    # Find best performing layer
    valid_layers = {layer: metrics for layer, metrics in layer_metrics.items() 
                   if 'error' not in metrics}
    
    best_layer = None
    best_f1 = 0.0
    if valid_layers:
        best_layer = max(valid_layers.keys(), key=lambda l: valid_layers[l]['f1'])
        best_f1 = valid_layers[best_layer]['f1']
    
    # Create evaluation summary
    evaluation_summary = {
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "dataset": dataset,
        "classifier_threshold": classifier_threshold,
        "max_tokens": max_tokens,
        "layers_evaluated": layers_evaluated,
        "total_layers_evaluated": len(layers_evaluated),
        "best_layer": best_layer,
        "best_f1_score": best_f1,
        "layer_performance": layer_metrics
    }
    
    # Update model params
    if model_name not in model_params:
        model_params[model_name] = {}
    
    model_params[model_name][f"layer_evaluation_{dataset}"] = evaluation_summary
    
    # Save back to file
    with open(model_params_path, 'w') as f:
        json.dump(model_params, f, indent=2)
    
    print(f"\nResults saved to {model_params_path} under model '{model_name}' (dataset: {dataset})")
    return model_params_path

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
    
    # Load guard results with human scores
    print(f"\nLoading guard results...")
    guard_results_df = load_guard_results(args.guard_results)
    
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
    
    # Match with guard results to get human scores
    matched_results = match_with_guard_results(generated_results, guard_results_df)
    
    if not matched_results:
        print("No matches found between generated responses and guard results. Exiting.")
        return
    
    # Evaluate all layers using human scores as ground truth
    evaluation_results = evaluate_all_layers_with_guard_results(
        model, tokenizer, classifiers, matched_results, device
    )
    
    # Calculate metrics for all layers
    layer_metrics = calculate_layer_metrics_with_human_scores(evaluation_results)
    
    # Print comprehensive report
    print_all_layers_report_with_human_scores(layer_metrics, args.classifier_threshold)
    
    # Save results if requested
    if args.save_results:
        output_data = {
            'model': args.model,
            'dataset': 'truthfulqa_human_scores',
            'classifier_threshold': args.classifier_threshold,
            'blocked_threshold': args.threshold,
            'max_tokens': args.max_tokens,
            'max_training_pairs': args.max_training_pairs,
            'max_eval_samples': args.max_eval_samples,
            'layers_evaluated': layers_to_evaluate,
            'generated_results': generated_results,
            'matched_results': matched_results,
            'evaluation_results': evaluation_results,
            'layer_metrics': layer_metrics
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.save_results}")
    
    # Save results to model_params.json
    model_params_path = save_results_to_model_params(
        args.model, layer_metrics, args.classifier_threshold, 
        args.max_tokens, layers_to_evaluate, "truthfulqa_human_scores"
    )
    
    print(f"\n{'='*100}")
    print("EVALUATION COMPLETE")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
