#!/usr/bin/env python
"""
Script to optimize the threshold value for hallucination detection.

This script:
1. Loads evaluation results with Claude scores and token scores
2. Tests different threshold values to find the optimal value that:
   - Maximizes hallucination detection
   - Minimizes false positives
3. Outputs the optimal threshold and metrics for different threshold values
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple

def load_evaluation_results(filepath: str) -> pd.DataFrame:
    """
    Load evaluation results from CSV file.
    
    Args:
        filepath: Path to CSV file with evaluation results
        
    Returns:
        DataFrame with evaluation results
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} evaluation results from {filepath}")
    return df

def parse_token_scores(token_scores_str: str) -> List[Dict]:
    """
    Parse token scores from the formatted string in the CSV.
    
    Format: position:token_id:token_text:similarity:category:is_harmful|...
    
    Args:
        token_scores_str: String containing token scores
        
    Returns:
        List of dictionaries with token information
    """
    if not token_scores_str or pd.isna(token_scores_str):
        return []
    
    tokens = []
    for token_str in token_scores_str.split('|'):
        if not token_str:
            continue
            
        parts = token_str.split(':')
        if len(parts) < 6:
            continue
            
        try:
            tokens.append({
                'position': int(parts[0]),
                'token_id': int(parts[1]),
                'token_text': parts[2],
                'similarity': float(parts[3]),
                'category': parts[4],
                'is_harmful': parts[5].lower() == 'true'
            })
        except (ValueError, IndexError):
            continue
            
    return tokens

def get_max_similarity_score(token_data: List[Dict]) -> float:
    """
    Get the maximum similarity score from token data.
    
    Args:
        token_data: List of token dictionaries
        
    Returns:
        Maximum similarity score, or 0 if no tokens
    """
    if not token_data:
        return 0.0
        
    return max([t.get('similarity', 0.0) for t in token_data])

def evaluate_threshold(df: pd.DataFrame, threshold: float) -> Dict:
    """
    Evaluate a specific threshold value.
    
    Args:
        df: DataFrame with evaluation results
        threshold: Threshold value to test
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Apply threshold to determine which responses would be flagged as harmful
    predictions = []
    for _, row in df.iterrows():
        token_data = parse_token_scores(row['token_scores'])
        max_similarity = get_max_similarity_score(token_data)
        is_predicted_harmful = max_similarity >= threshold
        
        # Use Claude score as ground truth (1 = good response, 0 = hallucination)
        is_actually_harmful = row['claude_score'] == 0
        
        predictions.append({
            'predicted_harmful': is_predicted_harmful,
            'actually_harmful': is_actually_harmful
        })
    
    # Compute metrics
    y_true = [p['actually_harmful'] for p in predictions]
    y_pred = [p['predicted_harmful'] for p in predictions]
    
    # If all predictions are the same, handle edge case
    if all(y_pred) or not any(y_pred):
        precision = 1.0 if all(y_pred) and all(y_true) else 0.0
        recall = 1.0 if all(y_pred) or not any(y_true) else 0.0
    else:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate detection rates
    true_positives = sum(1 for p in predictions if p['predicted_harmful'] and p['actually_harmful'])
    false_positives = sum(1 for p in predictions if p['predicted_harmful'] and not p['actually_harmful'])
    false_negatives = sum(1 for p in predictions if not p['predicted_harmful'] and p['actually_harmful'])
    true_negatives = sum(1 for p in predictions if not p['predicted_harmful'] and not p['actually_harmful'])
    
    total_harmful = sum(1 for p in predictions if p['actually_harmful'])
    total_harmless = sum(1 for p in predictions if not p['actually_harmful'])
    
    # Compute combined score that balances hallucination detection with false positives
    # Weight precision slightly more to avoid excessive false positives
    combined_score = (0.6 * precision) + (0.4 * recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'detection_rate': true_positives / total_harmful if total_harmful > 0 else 0.0,
        'false_positive_rate': false_positives / total_harmless if total_harmless > 0 else 0.0,
        'combined_score': combined_score
    }
    
    return metrics

def optimize_threshold(df: pd.DataFrame, start: float = 0.05, end: float = 1.0, step: float = 0.05) -> Tuple[float, List[Dict]]:
    """
    Find the optimal threshold value by testing a range of thresholds.
    
    Args:
        df: DataFrame with evaluation results
        start: Starting threshold value
        end: Ending threshold value
        step: Step size for threshold values
        
    Returns:
        Tuple of (optimal_threshold, all_metrics)
    """
    print(f"Optimizing threshold from {start} to {end} with step {step}...")
    
    # Generate threshold values to test
    thresholds = np.arange(start, end + step, step)
    
    # Evaluate each threshold
    all_metrics = []
    for threshold in thresholds:
        metrics = evaluate_threshold(df, threshold)
        all_metrics.append(metrics)
        print(f"Threshold {threshold:.2f}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}, Combined={metrics['combined_score']:.4f}")
    
    # Find optimal threshold based on combined score
    optimal_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['combined_score'])
    optimal_threshold = all_metrics[optimal_idx]['threshold']
    optimal_metrics = all_metrics[optimal_idx]
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"Precision: {optimal_metrics['precision']:.4f}")
    print(f"Recall: {optimal_metrics['recall']:.4f}")
    print(f"F1 score: {optimal_metrics['f1_score']:.4f}")
    print(f"Detection rate: {optimal_metrics['detection_rate']:.4f}")
    print(f"False positive rate: {optimal_metrics['false_positive_rate']:.4f}")
    
    return optimal_threshold, all_metrics

def plot_metrics(all_metrics: List[Dict], output_path: str = None):
    """
    Plot metrics for different threshold values.
    
    Args:
        all_metrics: List of metric dictionaries for different thresholds
        output_path: Path to save the plot, if None plot is displayed
    """
    thresholds = [m['threshold'] for m in all_metrics]
    precision = [m['precision'] for m in all_metrics]
    recall = [m['recall'] for m in all_metrics]
    f1_scores = [m['f1_score'] for m in all_metrics]
    combined_scores = [m['combined_score'] for m in all_metrics]
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(thresholds, precision, 'b-', label='Precision')
    plt.plot(thresholds, recall, 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    plt.plot(thresholds, combined_scores, 'k-', label='Combined Score')
    
    # Highlight the optimal threshold
    optimal_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['combined_score'])
    optimal_threshold = all_metrics[optimal_idx]['threshold']
    plt.axvline(x=optimal_threshold, color='purple', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.4f}')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics by Threshold Value')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def generate_threshold_table(all_metrics: List[Dict], output_path: str = None):
    """
    Generate a markdown table of threshold metrics.
    
    Args:
        all_metrics: List of metric dictionaries for different thresholds
        output_path: Path to save the table, if None string is returned
    """
    table = "| Threshold | Precision | Recall | F1 Score | Detection Rate | False Positive Rate | Combined Score |\n"
    table += "|-----------|-----------|--------|----------|----------------|-------------------|---------------|\n"
    
    for metrics in all_metrics:
        row = f"| {metrics['threshold']:.2f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['detection_rate']:.4f} | {metrics['false_positive_rate']:.4f} | {metrics['combined_score']:.4f} |\n"
        table += row
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"Table saved to {output_path}")
    
    return table

def main(args):
    # Load evaluation results
    df = load_evaluation_results(args.input_file)
    
    # Filter out rows without Claude scores or with ambiguous scores (-1)
    df = df[df['claude_score'] != -1]
    
    # Optimize threshold
    optimal_threshold, all_metrics = optimize_threshold(
        df, 
        start=args.min_threshold, 
        end=args.max_threshold, 
        step=args.threshold_step
    )
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_path = os.path.join(args.output_dir, "threshold_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save optimal threshold to JSON
        optimal_path = os.path.join(args.output_dir, "optimal_threshold.json")
        with open(optimal_path, 'w') as f:
            json.dump({
                'optimal_threshold': optimal_threshold,
                'metrics': all_metrics[max(range(len(all_metrics)), key=lambda i: all_metrics[i]['combined_score'])]
            }, f, indent=2)
        
        # Generate plot
        plot_path = os.path.join(args.output_dir, "threshold_metrics.png")
        plot_metrics(all_metrics, plot_path)
        
        # Generate table
        table_path = os.path.join(args.output_dir, "threshold_metrics.md")
        generate_threshold_table(all_metrics, table_path)
        
        print(f"\nResults saved to {args.output_dir}")
    else:
        # Just display the plot
        plot_metrics(all_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize threshold for hallucination detection")
    
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to CSV file with evaluation results (containing claude_score and token_scores)")
    parser.add_argument("--output-dir", type=str, default="evaluation/threshold_optimization",
                        help="Directory to save optimization results")
    
    # Threshold optimization parameters
    parser.add_argument("--min-threshold", type=float, default=0.05,
                        help="Minimum threshold value to test")
    parser.add_argument("--max-threshold", type=float, default=0.95,
                        help="Maximum threshold value to test")
    parser.add_argument("--threshold-step", type=float, default=0.05,
                        help="Step size for threshold values")
    
    # Optional plotting parameters
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots")
    
    args = parser.parse_args()
    main(args)
