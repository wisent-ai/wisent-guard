#!/usr/bin/env python
"""
Script to analyze token-level results from TruthfulQA evaluation and determine optimal thresholds.

This script:
1. Reads combined evaluation results containing token-level scores
2. Extracts maximum token scores for each response
3. Categorizes by Claude evaluation (correct/incorrect)
4. Finds optimal threshold for hallucination detection
5. Generates ROC curve and other performance metrics
"""

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def parse_token_scores(token_scores_str):
    """
    Parse token scores from the formatted string stored in CSV.
    Format: "position:token_id:token_text:similarity:category:is_harmful|..."
    
    Args:
        token_scores_str: String containing token scores
        
    Returns:
        List of dictionaries with token information
    """
    if not token_scores_str or pd.isna(token_scores_str):
        return []
        
    tokens = []
    for token_info in token_scores_str.split('|'):
        if not token_info:
            continue
            
        parts = token_info.split(':')
        
        # Skip malformed token data
        if len(parts) < 6:
            continue
            
        try:
            # Try to parse values with robust error handling
            position = int(parts[0]) if parts[0].isdigit() else 0
            token_id = int(parts[1]) if parts[1].isdigit() else 0
            token_text = parts[2]
            try:
                similarity = float(parts[3])
            except ValueError:
                # Skip tokens with invalid similarity scores
                continue
                
            category = parts[4]
            is_harmful = parts[5].lower() == 'true'
            
            token = {
                'position': position,
                'token_id': token_id,
                'token_text': token_text,
                'similarity': similarity,
                'category': category,
                'is_harmful': is_harmful
            }
            tokens.append(token)
        except Exception as e:
            # Skip any token that causes errors
            print(f"Warning: Skipping malformed token data: {e}")
            continue
    
    return tokens

def get_max_similarity(token_scores):
    """
    Get the maximum similarity score from token scores.
    
    Args:
        token_scores: List of token dictionaries
        
    Returns:
        Maximum similarity score (0.0 if no tokens)
    """
    if not token_scores:
        return 0.0
    
    return max(token.get('similarity', 0.0) for token in token_scores)

def find_optimal_threshold(max_scores, is_hallucinatory):
    """
    Find the optimal threshold that maximizes correct classification of hallucinatory responses.
    
    Args:
        max_scores: List of maximum similarity scores
        is_hallucinatory: Boolean list where True means the response is hallucinatory
        
    Returns:
        optimal_threshold: The threshold that maximizes hallucinatory detection
        optimal_metrics: Dictionary with metrics at the optimal threshold
    """
    # Convert to numpy arrays
    scores = np.array(max_scores)
    labels = np.array(is_hallucinatory)
    
    # Get all unique scores as potential thresholds
    potential_thresholds = sorted(set(scores))
    
    # Metrics for each threshold
    metrics = []
    
    # Try each threshold
    for threshold in potential_thresholds:
        # Predict hallucinatory if score >= threshold
        predictions = scores >= threshold
        
        # Calculate metrics
        true_positives = np.sum((predictions == True) & (labels == True))
        true_negatives = np.sum((predictions == False) & (labels == False))
        false_positives = np.sum((predictions == True) & (labels == False))
        false_negatives = np.sum((predictions == False) & (labels == True))
        
        # Calculate rates and F1 score
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
            
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0
            
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        # Calculate accuracy
        accuracy = (true_positives + true_negatives) / len(labels)
        
        # Store metrics
        metrics.append({
            'threshold': threshold,
            'true_positives': int(true_positives),
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
    
    # Find threshold with highest F1 score
    # If multiple thresholds have the same F1, choose the one with better recall
    max_f1 = max(m['f1'] for m in metrics) if metrics else 0
    max_f1_metrics = [m for m in metrics if m['f1'] == max_f1]
    
    # If multiple thresholds have the same F1, choose the one with better recall
    optimal_metric = max(max_f1_metrics, key=lambda x: x['recall']) if max_f1_metrics else None
    
    return optimal_metric

def plot_score_distributions(correct_scores, incorrect_scores, optimal_threshold, output_dir):
    """
    Plot the distribution of maximum token similarity scores for correct and incorrect responses.
    
    Args:
        correct_scores: List of maximum similarity scores for correct responses
        incorrect_scores: List of maximum similarity scores for incorrect responses
        optimal_threshold: The optimal threshold value
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    bins = np.linspace(0, 1, 30)
    plt.hist(correct_scores, bins=bins, alpha=0.5, label='Correct Responses', color='green')
    plt.hist(incorrect_scores, bins=bins, alpha=0.5, label='Incorrect Responses', color='red')
    
    # Plot optimal threshold line
    plt.axvline(x=optimal_threshold, color='black', linestyle='--', 
                label=f'Optimal Threshold: {optimal_threshold:.4f}')
    
    plt.title('Distribution of Maximum Token Similarity Scores')
    plt.xlabel('Maximum Token Similarity Score')
    plt.ylabel('Number of Responses')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'token_score_distribution.png')
    plt.savefig(plot_path)
    print(f"Score distribution plot saved to: {plot_path}")
    
    # Close the plot
    plt.close()

def plot_roc_curve(max_scores, is_hallucinatory, output_dir):
    """
    Plot ROC curve for token similarity scores.
    
    Args:
        max_scores: List of maximum similarity scores
        is_hallucinatory: Boolean list where True means the response is hallucinatory
        output_dir: Directory to save the plot
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(is_hallucinatory, max_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'token_score_roc_curve.png')
    plt.savefig(plot_path)
    print(f"ROC curve plot saved to: {plot_path}")
    
    # Close the plot
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze token-level results and find optimal thresholds")
    parser.add_argument("--results-file", type=str, default=None,
                       help="Path to the results CSV file (if not specified, will auto-detect)")
    parser.add_argument("--output-dir", type=str, default="evaluation/results",
                       help="Directory to save analysis results and plots")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-detect results file if not specified
    if args.results_file is None:
        possible_files = [
            os.path.join(args.output_dir, "combined_results.csv"),
            os.path.join(args.output_dir, "guard_results.csv"),
            os.path.join(args.output_dir, "baseline_results.csv")
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                args.results_file = file_path
                print(f"Auto-detected results file: {file_path}")
                break
        
        if args.results_file is None:
            print("Error: Could not find any results files. Please run evaluate_llama_truthfulqa.py first.")
            return
    
    # Read the results
    try:
        results_df = pd.read_csv(args.results_file)
        print(f"Loaded {len(results_df)} results from {args.results_file}")
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    # Prepare data for analysis
    max_scores = []
    is_hallucinatory = []
    correct_scores = []
    incorrect_scores = []
    
    # Process each result
    skipped = 0
    for _, row in results_df.iterrows():
        # Parse token scores
        token_scores = parse_token_scores(row.get('token_scores', ''))
        
        # Skip if no token scores (these would be baseline results without guard)
        if not token_scores:
            skipped += 1
            continue
        
        # Get maximum similarity score
        max_sim = get_max_similarity(token_scores)
        
        # Determine if the response is hallucinatory based on Claude score
        # A Claude score of 0 means the response is incorrect/hallucinatory
        claude_score = row.get('guard_claude_score', row.get('claude_score', -1))
        
        # Skip ambiguous responses (claude_score = -1)
        if claude_score == -1:
            skipped += 1
            continue
            
        # Add to datasets
        max_scores.append(max_sim)
        is_hallucinatory.append(claude_score == 0)
        
        # Add to separate lists for distribution plotting
        if claude_score == 1:
            correct_scores.append(max_sim)
        elif claude_score == 0:
            incorrect_scores.append(max_sim)
    
    print(f"Processed {len(max_scores)} results with token scores (skipped {skipped})")
    print(f"Correct responses: {len(correct_scores)}")
    print(f"Incorrect/hallucinatory responses: {len(incorrect_scores)}")
    
    # Check if we have enough data
    if len(max_scores) < 2 or len(correct_scores) == 0 or len(incorrect_scores) == 0:
        print("Not enough data for analysis. Need both correct and incorrect responses with token scores.")
        return
    
    # Find optimal threshold
    optimal_metric = find_optimal_threshold(max_scores, is_hallucinatory)
    
    if not optimal_metric:
        print("Could not determine optimal threshold. Check your data.")
        return
        
    optimal_threshold = optimal_metric['threshold']
    
    # Print optimal threshold and metrics
    print("\n===== OPTIMAL THRESHOLD ANALYSIS =====")
    print(f"Optimal threshold: {optimal_threshold:.6f}")
    print(f"Precision: {optimal_metric['precision']:.4f}")
    print(f"Recall: {optimal_metric['recall']:.4f}")
    print(f"F1 Score: {optimal_metric['f1']:.4f}")
    print(f"Accuracy: {optimal_metric['accuracy']:.4f}")
    print(f"True Positives: {optimal_metric['true_positives']}")
    print(f"True Negatives: {optimal_metric['true_negatives']}")
    print(f"False Positives: {optimal_metric['false_positives']}")
    print(f"False Negatives: {optimal_metric['false_negatives']}")
    
    # Save optimal threshold and metrics
    metrics_path = os.path.join(args.output_dir, 'optimal_threshold_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(optimal_metric, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Plot score distributions
    plot_score_distributions(correct_scores, incorrect_scores, optimal_threshold, args.output_dir)
    
    # Plot ROC curve
    plot_roc_curve(max_scores, is_hallucinatory, args.output_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
