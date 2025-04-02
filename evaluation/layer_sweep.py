#!/usr/bin/env python3
"""
Perform a layer sweep analysis to determine which layer(s) in a model
are most effective for hallucination detection.

This script evaluates each layer's performance individually and identifies
the best-performing layer(s) for the guardrail system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
import argparse
import sys
import subprocess
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score
from collections import defaultdict
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_evaluation_with_layer(layer, model_name, output_dir, dataset="evaluation/data/TruthfulQA_en_eval.csv", 
                            sample_size=None, threshold=0.2, classifier_path=None, no_claude=True):
    """
    Run the evaluation script with a specific layer and capture the results.
    
    Args:
        layer: Layer number to evaluate
        model_name: Name of the model to evaluate
        output_dir: Directory to save results
        dataset: Path to evaluation dataset
        sample_size: Number of samples to evaluate (None for all)
        threshold: Similarity threshold for hallucination detection
        classifier_path: Path to classifier model (for classifier-based approach)
        no_claude: Whether to use human evaluation or Claude (usually True to skip Claude)
        
    Returns:
        Dictionary with evaluation metrics for the layer
    """
    print(f"\n===== Evaluating Layer {layer} =====")
    
    # Create the command to run the evaluation script
    cmd = [
        "python", "evaluation/evaluate_llama_truthfulqa_classifier.py",
        "--model-name", model_name,
        "--eval-data", dataset,
        "--layer-number", str(layer),
        "--threshold", str(threshold),
        "--output-dir", output_dir
    ]
    
    # Add optional arguments
    if sample_size:
        cmd.extend(["--sample-size", str(sample_size)])
    
    if classifier_path:
        cmd.extend(["--use-classifier", "--classifier-path", classifier_path])
    
    if no_claude:
        cmd.append("--no-claude")
    
    # Run the command and capture output
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # Check for errors
    if process.returncode != 0:
        print(f"Error running evaluation for layer {layer}:")
        print(stderr)
        return None
    
    # Parse the output to extract metrics
    metrics = parse_evaluation_output(stdout, layer)
    
    # Also load the saved metrics file if available
    metrics_file = os.path.join(output_dir, "guard_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                saved_metrics = json.load(f)
                # Merge saved metrics with parsed metrics
                metrics.update(saved_metrics)
        except Exception as e:
            print(f"Error loading saved metrics: {e}")
    
    # Add layer number to metrics
    metrics['layer'] = layer
    
    # Process the results CSV to extract additional metrics
    try:
        results_file = os.path.join(output_dir, "guard_results_classifier-based.csv" 
                                    if classifier_path else "guard_results.csv")
        if os.path.exists(results_file):
            results_metrics = process_results_file(results_file)
            metrics.update(results_metrics)
    except Exception as e:
        print(f"Error processing results file: {e}")
    
    print(f"Completed evaluation for layer {layer}")
    return metrics

def parse_evaluation_output(output, layer):
    """
    Parse the evaluation script's stdout to extract relevant metrics.
    
    Args:
        output: Stdout from evaluation script
        layer: Layer number being evaluated
        
    Returns:
        Dictionary with parsed metrics
    """
    metrics = {
        'layer': layer,
        'harmful_detected': 0,
        'harmful_rate': 0,
        'blocked_count': 0,
        'blocked_rate': 0,
        'human_correct': 0,
        'human_correct_rate': 0,
        'human_incorrect': 0,
        'human_incorrect_rate': 0
    }
    
    # Extract metrics using regular expressions
    harmful_match = re.search(r"Hallucinations detected: (\d+) \(([0-9.]+)%\)", output)
    if harmful_match:
        metrics['harmful_detected'] = int(harmful_match.group(1))
        metrics['harmful_rate'] = float(harmful_match.group(2)) / 100
    
    blocked_match = re.search(r"Responses blocked: (\d+) \(([0-9.]+)%\)", output)
    if blocked_match:
        metrics['blocked_count'] = int(blocked_match.group(1))
        metrics['blocked_rate'] = float(blocked_match.group(2)) / 100
    
    # Extract human evaluation metrics if available
    human_correct_match = re.search(r"Correct responses \(Human\): (\d+) \(([0-9.]+)%\)", output)
    if human_correct_match:
        metrics['human_correct'] = int(human_correct_match.group(1))
        metrics['human_correct_rate'] = float(human_correct_match.group(2)) / 100
    
    human_incorrect_match = re.search(r"Incorrect responses \(Human\): (\d+) \(([0-9.]+)%\)", output)
    if human_incorrect_match:
        metrics['human_incorrect'] = int(human_incorrect_match.group(1))
        metrics['human_incorrect_rate'] = float(human_incorrect_match.group(2)) / 100
    
    return metrics

def process_results_file(results_file):
    """
    Process the evaluation results CSV file to extract additional metrics.
    
    Args:
        results_file: Path to results CSV file
        
    Returns:
        Dictionary with additional metrics
    """
    try:
        # Load the results file
        df = pd.read_csv(results_file)
        
        # Skip if no valid data
        if len(df) == 0:
            return {}
            
        # Check if human_score column exists and has valid values
        has_human_scores = 'human_score' in df.columns and df['human_score'].isin([0, 1]).any()
        
        metrics = {}
        
        if has_human_scores:
            # Convert scores to boolean
            df['is_correct'] = df['human_score'] == 1
            df['is_hallucination'] = df['human_score'] == 0
            
            # Filter out ambiguous scores (-1)
            df_valid = df[df['human_score'].isin([0, 1])]
            
            if len(df_valid) > 0:
                # Calculate true/false positives/negatives
                tp = ((df_valid['blocked'] == True) & (df_valid['is_hallucination'] == True)).sum()
                fp = ((df_valid['blocked'] == True) & (df_valid['is_hallucination'] == False)).sum()
                tn = ((df_valid['blocked'] == False) & (df_valid['is_hallucination'] == False)).sum()
                fn = ((df_valid['blocked'] == False) & (df_valid['is_hallucination'] == True)).sum()
                
                # Calculate metrics
                total = tp + fp + tn + fn
                accuracy = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                balanced_acc = (recall + specificity) / 2
                
                # Add to metrics
                metrics.update({
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'specificity': specificity,
                    'balanced_accuracy': balanced_acc
                })
                
                # Calculate net impact
                net_impact = tp - fp
                net_impact_rate = net_impact / total if total > 0 else 0
                metrics.update({
                    'net_impact': int(net_impact),
                    'net_impact_rate': net_impact_rate
                })
        
        return metrics
        
    except Exception as e:
        print(f"Error processing results file: {e}")
        return {}

def evaluate_all_layers(model_name, num_layers, output_dir, 
                      dataset="evaluation/data/TruthfulQA_en_eval.csv",
                      sample_size=None, threshold=0.2, 
                      classifier_path=None, use_classifier=False,
                      no_claude=True):
    """
    Evaluate all layers of the model to find the best-performing layer.
    
    Args:
        model_name: Name of the model to evaluate
        num_layers: Number of layers in the model
        output_dir: Directory to save results
        dataset: Path to evaluation dataset
        sample_size: Number of samples to evaluate (None for all)
        threshold: Similarity threshold for hallucination detection
        classifier_path: Path to classifier model (for classifier-based approach)
        use_classifier: Whether to use classifier-based approach
        no_claude: Whether to skip Claude evaluation
        
    Returns:
        Dictionary with results for all layers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results for each layer
    layer_results = {}
    
    # Evaluate each layer
    for layer in range(num_layers):
        # Create layer-specific output directory
        layer_output_dir = os.path.join(output_dir, f"layer_{layer}")
        os.makedirs(layer_output_dir, exist_ok=True)
        
        # Skip if results already exist for this layer
        results_file = os.path.join(layer_output_dir, "guard_results_classifier-based.csv" 
                                   if use_classifier else "guard_results.csv")
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            print(f"Results already exist for layer {layer}. Loading existing results.")
            metrics = process_results_file(results_file)
            metrics['layer'] = layer
            layer_results[layer] = metrics
            continue
        
        # Run evaluation for this layer
        cp = classifier_path if use_classifier else None
        layer_metrics = run_evaluation_with_layer(
            layer, model_name, layer_output_dir, dataset,
            sample_size, threshold, cp, no_claude
        )
        
        if layer_metrics:
            layer_results[layer] = layer_metrics
    
    return layer_results

def find_best_layers(layer_results, metric='f1_score', top_k=3):
    """
    Find the best-performing layers based on a specified metric.
    
    Args:
        layer_results: Dictionary with results for all layers
        metric: Metric to use for ranking
        top_k: Number of top layers to return
        
    Returns:
        List of (layer, metric_value) tuples for top-k layers
    """
    # Sort layers by the specified metric
    sorted_layers = []
    for layer, metrics in layer_results.items():
        if metric in metrics:
            sorted_layers.append((layer, metrics[metric]))
    
    # Sort in descending order of metric value
    sorted_layers.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k layers
    return sorted_layers[:top_k]

def visualize_layer_results(layer_results, output_dir, num_layers=None):
    """
    Visualize the performance of different layers.
    
    Args:
        layer_results: Dictionary with results for all layers
        output_dir: Directory to save visualizations
        num_layers: Total number of layers in the model
    """
    print("Generating layer performance visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all evaluated layers
    layers = sorted(layer_results.keys())
    
    # If num_layers not provided, use the maximum layer number + 1
    if num_layers is None:
        num_layers = max(layers) + 1
    
    # 1. Plot multiple metrics across all layers
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']
    
    plt.figure(figsize=(12, 8))
    for metric in metrics_to_plot:
        metric_values = [layer_results[l].get(metric, 0) for l in layers]
        plt.plot(layers, metric_values, marker='o', linestyle='-', label=metric)
    
    plt.title('Classifier Performance Metrics Across Layers')
    plt.xlabel('Layer Number')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(0, num_layers, max(1, num_layers // 10)))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_performance_metrics.png'))
    
    # 2. Plot hallucination detection rates
    plt.figure(figsize=(12, 8))
    harmful_rates = [layer_results[l].get('harmful_rate', 0) for l in layers]
    blocked_rates = [layer_results[l].get('blocked_rate', 0) for l in layers]
    
    plt.plot(layers, harmful_rates, marker='o', linestyle='-', 
            label='Hallucination Detection Rate')
    plt.plot(layers, blocked_rates, marker='o', linestyle='-', 
            label='Response Blocking Rate')
    
    plt.title('Hallucination Detection and Blocking Rates Across Layers')
    plt.xlabel('Layer Number')
    plt.ylabel('Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(0, num_layers, max(1, num_layers // 10)))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_detection_rates.png'))
    
    # 3. Plot confusion matrix metrics
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
    
    for metric in metrics_to_plot:
        metric_values = [layer_results[l].get(metric, 0) for l in layers]
        plt.plot(layers, metric_values, marker='o', linestyle='-', label=metric)
    
    plt.title('Confusion Matrix Metrics Across Layers')
    plt.xlabel('Layer Number')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(0, num_layers, max(1, num_layers // 10)))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_confusion_matrix.png'))
    
    # 4. Plot net impact across layers
    plt.figure(figsize=(12, 8))
    net_impact_rates = [layer_results[l].get('net_impact_rate', 0) for l in layers]
    
    # Create a color map for positive and negative impact
    colors = ['g' if rate >= 0 else 'r' for rate in net_impact_rates]
    
    plt.bar(layers, net_impact_rates, color=colors)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    plt.title('Net Impact Rate Across Layers')
    plt.xlabel('Layer Number')
    plt.ylabel('Net Impact Rate')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(range(0, num_layers, max(1, num_layers // 10)))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_net_impact.png'))
    
    # 5. Heatmap of layer position vs performance
    plt.figure(figsize=(14, 10))
    
    # Calculate normalized positions (early, middle, late)
    positions = np.array(layers) / (num_layers - 1)  # 0 to 1 range
    
    # Divide into early, middle, and late layers
    position_bins = [0, 0.33, 0.67, 1.0]
    position_labels = ['Early', 'Middle', 'Late']
    
    # Convert positions to categorical
    positions_cat = np.digitize(positions, position_bins[1:]) - 1  # 0, 1, or 2
    
    # Metrics to display in heatmap
    metrics_for_heatmap = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']
    
    # Group metrics by position
    position_metrics = defaultdict(lambda: defaultdict(list))
    
    for layer, pos_cat in zip(layers, positions_cat):
        for metric in metrics_for_heatmap:
            if metric in layer_results[layer]:
                position_metrics[position_labels[pos_cat]][metric].append(layer_results[layer][metric])
    
    # Calculate average metrics for each position
    avg_metrics = {}
    for pos, metrics_dict in position_metrics.items():
        avg_metrics[pos] = {metric: np.mean(values) for metric, values in metrics_dict.items()}
    
    # Create heatmap data
    heatmap_data = []
    for pos in position_labels:
        if pos in avg_metrics:
            row = [avg_metrics[pos].get(metric, 0) for metric in metrics_for_heatmap]
            heatmap_data.append(row)
        else:
            heatmap_data.append([0] * len(metrics_for_heatmap))
    
    heatmap_data = np.array(heatmap_data)
    
    # Plot heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Add labels and annotations
    plt.yticks(range(len(position_labels)), position_labels)
    plt.xticks(range(len(metrics_for_heatmap)), [m.capitalize() for m in metrics_for_heatmap], rotation=45)
    
    # Add colorbar
    plt.colorbar(label='Score')
    
    # Add text annotations
    for i in range(len(position_labels)):
        for j in range(len(metrics_for_heatmap)):
            plt.text(j, i, f"{heatmap_data[i, j]:.2f}", 
                   ha="center", va="center", color="white" if heatmap_data[i, j] > 0.5 else "black")
    
    plt.title('Performance Metrics by Layer Position')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_position_heatmap.png'))
    
    # Close all plots to free memory
    plt.close('all')
    
    print(f"Layer visualizations saved to {output_dir}")

def count_model_layers(model_name):
    """
    Count the number of layers in the model by attempting to load it.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Number of layers in the model
    """
    try:
        print(f"Loading model {model_name} to count layers...")
        from transformers import AutoModelForCausalLM
        
        # Load model with minimal resources
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Determine the number of layers
        if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
        else:
            # Default to a reasonable number if we can't determine
            print("Warning: Could not determine number of layers from model. Using default value.")
            num_layers = 24
        
        print(f"Model has {num_layers} layers")
        return num_layers
        
    except Exception as e:
        print(f"Error counting model layers: {e}")
        print("Using default of 24 layers")
        return 24

def save_results(layer_results, output_dir, pretty=True):
    """Save layer sweep results to a JSON file"""
    
    # Convert numpy values to Python types for JSON
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert results to JSON-serializable format
    json_results = {}
    for layer, metrics in layer_results.items():
        json_results[str(layer)] = {k: convert_to_json_serializable(v) for k, v in metrics.items()}
    
    # Save to file
    output_file = os.path.join(output_dir, "layer_sweep_results.json")
    with open(output_file, 'w') as f:
        if pretty:
            json.dump(json_results, f, indent=2)
        else:
            json.dump(json_results, f)
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Perform a layer sweep to find the best layer for hallucination detection")
    parser.add_argument("--model-name", type=str, 
                        default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="Name of the model to evaluate")
    parser.add_argument("--output-dir", type=str, 
                        default="evaluation/results/layer_sweep",
                        help="Directory to save layer sweep results and visualizations")
    parser.add_argument("--dataset", type=str, 
                        default="evaluation/data/TruthfulQA_en_eval.csv",
                        help="Path to evaluation dataset")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Similarity threshold for hallucination detection")
    parser.add_argument("--use-classifier", action="store_true",
                        help="Use classifier-based approach instead of threshold-based")
    parser.add_argument("--classifier-path", type=str, 
                        help="Path to classifier model (required if use-classifier is set)")
    parser.add_argument("--layer-range", type=str, default=None,
                        help="Range of layers to evaluate (e.g., '0-10', '15,20,25')")
    parser.add_argument("--evaluation-metric", type=str, 
                        choices=["f1_score", "accuracy", "balanced_accuracy", "precision", "recall", "net_impact_rate"],
                        default="f1_score",
                        help="Metric to use for ranking layers")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top-performing layers to highlight")
    
    args = parser.parse_args()
    
    # Check if classifier-based but no path
    if args.use_classifier and not args.classifier_path:
        print("Error: --classifier-path is required when --use-classifier is set.")
        return
    
    # Count layers in the model
    num_layers = count_model_layers(args.model_name)
    
    # Process layer range argument
    if args.layer_range:
        if ',' in args.layer_range:
            # Comma-separated list of layers
            layers_to_evaluate = [int(l) for l in args.layer_range.split(',')]
        elif '-' in args.layer_range:
            # Range of layers
            start, end = map(int, args.layer_range.split('-'))
            layers_to_evaluate = list(range(start, end + 1))
        else:
            # Single layer
            layers_to_evaluate = [int(args.layer_range)]
    else:
        # Evaluate all layers
        layers_to_evaluate = list(range(num_layers))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store configuration
    config = {
        'model_name': args.model_name,
        'num_layers': num_layers,
        'layers_evaluated': layers_to_evaluate,
        'dataset': args.dataset,
        'sample_size': args.sample_size,
        'threshold': args.threshold,
        'use_classifier': args.use_classifier,
        'classifier_path': args.classifier_path,
        'evaluation_metric': args.evaluation_metric
    }
    
    config_file = os.path.join(args.output_dir, "layer_sweep_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Evaluate specified layers
    layer_results = {}
    for layer in layers_to_evaluate:
        if layer < 0 or layer >= num_layers:
            print(f"Warning: Layer {layer} is out of range (0-{num_layers-1}). Skipping.")
            continue
            
        # Create layer-specific output directory
        layer_output_dir = os.path.join(args.output_dir, f"layer_{layer}")
        os.makedirs(layer_output_dir, exist_ok=True)
        
        # Run evaluation for this layer
        cp = args.classifier_path if args.use_classifier else None
        layer_metrics = run_evaluation_with_layer(
            layer, args.model_name, layer_output_dir, args.dataset,
            args.sample_size, args.threshold, cp, no_claude=True
        )
        
        if layer_metrics:
            layer_results[layer] = layer_metrics
    
    # Find best layers based on evaluation metric
    top_layers = find_best_layers(layer_results, args.evaluation_metric, args.top_k)
    
    # Print results
    print("\n===== Layer Sweep Results =====")
    print(f"Evaluated {len(layer_results)} layers")
    print(f"Top {args.top_k} layers based on {args.evaluation_metric}:")
    
    for i, (layer, score) in enumerate(top_layers):
        print(f"{i+1}. Layer {layer}: {score:.4f}")
        
        # Print additional metrics for the top layers
        metrics = layer_results[layer]
        print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"   Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        print(f"   Precision: {metrics.get('precision', 0):.4f}")
        print(f"   Recall: {metrics.get('recall', 0):.4f}")
        print(f"   F1 Score: {metrics.get('f1_score', 0):.4f}")
        print(f"   Hallucinations Detected: {metrics.get('harmful_detected', 0)} ({metrics.get('harmful_rate', 0):.2%})")
        print(f"   Responses Blocked: {metrics.get('blocked_count', 0)} ({metrics.get('blocked_rate', 0):.2%})")
        print(f"   Net Impact: {metrics.get('net_impact', 0)} ({metrics.get('net_impact_rate', 0):.2%})")
    
    # Save best layers to a separate file
    best_layers = {
        'evaluation_metric': args.evaluation_metric,
        'top_layers': [{'layer': layer, 'score': float(score)} for layer, score in top_layers]
    }
    
    best_layers_file = os.path.join(args.output_dir, "best_layers.json")
    with open(best_layers_file, 'w') as f:
        json.dump(best_layers, f, indent=2)
    
    # Save all results
    save_results(layer_results, args.output_dir)
    
    # Visualize results
    visualize_layer_results(layer_results, args.output_dir, num_layers)
    
    print("\n===== Layer Sweep Complete =====")
    print(f"Best layer for {args.evaluation_metric}: Layer {top_layers[0][0]} with score {top_layers[0][1]:.4f}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 