#!/usr/bin/env python3
"""
Optimize the threshold value for the hallucination classifier.
This script analyzes performance across different threshold values and
identifies the optimal threshold based on various metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
import argparse
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def extract_scores_from_csv(results_file):
    """
    Extract probability scores and evaluation results from the CSV file.
    
    Args:
        results_file: Path to the CSV file with evaluation results
        
    Returns:
        Dictionary with extracted scores and ground truth labels
    """
    print(f"Loading data from {results_file}...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(results_file)
        
        # Check if we have the necessary columns
        if 'human_score' not in df.columns or 'reason' not in df.columns:
            print("Error: Required columns not found in the data")
            return None
        
        # Extract ground truth (human/claude scores)
        # Convert to boolean: 1 = correct (not harmful), 0 = incorrect (harmful)
        ground_truth = df['human_score'] == 1
        ground_truth_valid = df['human_score'].isin([0, 1])  # Filter out ambiguous (-1) scores
        
        # Extract probability scores from reason field
        prob_scores = []
        pattern = r"probability ([0-9.]+)"
        
        for idx, row in df.iterrows():
            if pd.notna(row['reason']):
                match = re.search(pattern, row['reason'])
                if match:
                    prob_score = float(match.group(1))
                    prob_scores.append(prob_score)
                else:
                    prob_scores.append(np.nan)
            else:
                prob_scores.append(np.nan)
        
        # Add probability scores to dataframe
        df['probability_score'] = prob_scores
        
        # Filter to rows where we have both valid ground truth and probability scores
        valid_data = df[ground_truth_valid & df['probability_score'].notna()]
        
        if len(valid_data) == 0:
            print("Error: No valid data points found with both scores and ground truth")
            return None
            
        print(f"Extracted {len(valid_data)} valid data points for threshold optimization")
        
        # Prepare final data
        data = {
            'probability_scores': valid_data['probability_score'].values,
            'is_correct': valid_data['human_score'].values == 1,  # True if correct (not harmful)
            'is_hallucination': valid_data['human_score'].values == 0,  # True if incorrect (harmful)
            'categories': valid_data['category'].values if 'category' in valid_data.columns else None
        }
        
        return data
        
    except Exception as e:
        print(f"Error loading or processing results file: {e}")
        return None

def evaluate_threshold(data, threshold):
    """
    Evaluate performance metrics for a specific threshold value.
    
    Args:
        data: Dictionary with probability scores and ground truth
        threshold: Threshold value to evaluate
        
    Returns:
        Dictionary with performance metrics
    """
    # True = "This is a hallucination" prediction when probability >= threshold
    predictions = data['probability_scores'] >= threshold
    
    # For classifier metrics, we consider "hallucination detected" as positive class
    # So True Positive = predicted hallucination (True) and actual hallucination (True)
    true_positives = np.sum((predictions == True) & (data['is_hallucination'] == True))
    false_positives = np.sum((predictions == True) & (data['is_hallucination'] == False))
    true_negatives = np.sum((predictions == False) & (data['is_hallucination'] == False))
    false_negatives = np.sum((predictions == False) & (data['is_hallucination'] == True))
    
    # Calculate metrics
    total = len(data['probability_scores'])
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate balanced accuracy - average of sensitivity and specificity
    sensitivity = recall  # Same as recall/TPR
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Calculate net impact (hallucinations caught minus correct responses blocked)
    hallucinations_caught = true_positives
    correct_blocked = false_positives
    net_impact = hallucinations_caught - correct_blocked
    net_impact_rate = net_impact / total if total > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'net_impact': net_impact,
        'net_impact_rate': net_impact_rate
    }

def optimize_threshold(data, metric='f1', threshold_range=None, num_thresholds=50):
    """
    Evaluate multiple thresholds and find the optimal one based on the specified metric.
    
    Args:
        data: Dictionary with probability scores and ground truth
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'net_impact_rate', etc.)
        threshold_range: Tuple with (min, max) threshold values, or None for auto
        num_thresholds: Number of threshold values to test
        
    Returns:
        Dictionary with optimal threshold and performance metrics
    """
    if threshold_range is None:
        # Auto-determine threshold range from data
        min_threshold = max(0.01, np.min(data['probability_scores']))
        max_threshold = min(0.99, np.max(data['probability_scores']))
        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
    else:
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    
    # Evaluate each threshold
    results = []
    for threshold in thresholds:
        metrics = evaluate_threshold(data, threshold)
        results.append(metrics)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Find the optimal threshold based on the specified metric
    if metric == 'f1':
        best_idx = results_df['f1'].idxmax()
    elif metric == 'accuracy':
        best_idx = results_df['accuracy'].idxmax()
    elif metric == 'balanced_accuracy':
        best_idx = results_df['balanced_accuracy'].idxmax()
    elif metric == 'precision':
        best_idx = results_df['precision'].idxmax()
    elif metric == 'recall':
        best_idx = results_df['recall'].idxmax()
    elif metric == 'net_impact_rate':
        best_idx = results_df['net_impact_rate'].idxmax()
    else:
        print(f"Warning: Unknown metric '{metric}', defaulting to f1")
        best_idx = results_df['f1'].idxmax()
    
    optimal_result = results_df.iloc[best_idx].to_dict()
    
    # Calculate precision-recall and ROC curves for all possible thresholds
    precision, recall, pr_thresholds = precision_recall_curve(
        data['is_hallucination'], 
        data['probability_scores']
    )
    fpr, tpr, roc_thresholds = roc_curve(
        data['is_hallucination'], 
        data['probability_scores']
    )
    
    # Calculate AUC metrics
    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)
    
    # Add to optimal result
    optimal_result['pr_auc'] = pr_auc
    optimal_result['roc_auc'] = roc_auc
    optimal_result['all_results'] = results_df
    optimal_result['pr_curve'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
    }
    optimal_result['roc_curve'] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': roc_thresholds.tolist() if len(roc_thresholds) > 0 else []
    }
    
    return optimal_result

def visualize_optimization_results(results, output_dir):
    """
    Visualize the threshold optimization results.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save visualization files
    """
    print("Generating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the results dataframe
    df = results['all_results']
    
    # 1. Performance metrics by threshold
    plt.figure(figsize=(12, 7))
    plt.plot(df['threshold'], df['accuracy'], label='Accuracy', marker='.')
    plt.plot(df['threshold'], df['balanced_accuracy'], label='Balanced Accuracy', marker='.')
    plt.plot(df['threshold'], df['precision'], label='Precision', marker='.')
    plt.plot(df['threshold'], df['recall'], label='Recall', marker='.')
    plt.plot(df['threshold'], df['f1'], label='F1 Score', marker='.')
    plt.plot(df['threshold'], df['specificity'], label='Specificity', marker='.')
    
    # Mark the optimal threshold
    plt.axvline(x=results['threshold'], color='r', linestyle='--', alpha=0.5, 
                label=f'Optimal Threshold = {results["threshold"]:.4f}')
    
    plt.title('Performance Metrics by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_metrics.png'))
    
    # 2. Precision-Recall curve
    plt.figure(figsize=(10, 7))
    plt.plot(results['pr_curve']['recall'], results['pr_curve']['precision'], 
             label=f'PR Curve (AUC = {results["pr_auc"]:.4f})', marker='', linestyle='-')
    
    # If thresholds are available, mark some points
    if results['pr_curve']['thresholds']:
        # Select a few threshold points to annotate
        thresholds = np.array(results['pr_curve']['thresholds'])
        precision = np.array(results['pr_curve']['precision'][:-1])  # Precision has one more point
        recall = np.array(results['pr_curve']['recall'][:-1])
        
        # Choose threshold values to annotate (e.g., 0.2, 0.5, 0.8)
        for t in [0.2, 0.5, 0.8]:
            # Find closest threshold
            idx = np.abs(thresholds - t).argmin()
            plt.plot(recall[idx], precision[idx], 'ro')
            plt.annotate(f'T={t:.1f}', 
                         (recall[idx], precision[idx]),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center')
    
    # Add the optimal point
    best_recall = df.loc[df['threshold'] == results['threshold'], 'recall'].values[0]
    best_precision = df.loc[df['threshold'] == results['threshold'], 'precision'].values[0]
    plt.plot(best_recall, best_precision, 'go', markersize=8)
    plt.annotate(f'Optimal (T={results["threshold"]:.2f})', 
                 (best_recall, best_precision),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center',
                 fontweight='bold')
    
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    
    # 3. ROC curve
    plt.figure(figsize=(10, 7))
    plt.plot(results['roc_curve']['fpr'], results['roc_curve']['tpr'], 
             label=f'ROC Curve (AUC = {results["roc_auc"]:.4f})', marker='', linestyle='-')
    
    # If thresholds are available, mark some points
    if results['roc_curve']['thresholds']:
        # Select a few threshold points to annotate
        thresholds = np.array(results['roc_curve']['thresholds'])
        fpr = np.array(results['roc_curve']['fpr'])
        tpr = np.array(results['roc_curve']['tpr'])
        
        # Choose threshold values to annotate (e.g., 0.2, 0.5, 0.8)
        for t in [0.2, 0.5, 0.8]:
            # Find closest threshold
            idx = np.abs(thresholds - t).argmin()
            plt.plot(fpr[idx], tpr[idx], 'ro')
            plt.annotate(f'T={t:.1f}', 
                         (fpr[idx], tpr[idx]),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center')
    
    # Add diagonal random line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    
    # Add the optimal point
    best_fpr = 1 - df.loc[df['threshold'] == results['threshold'], 'specificity'].values[0]
    best_tpr = df.loc[df['threshold'] == results['threshold'], 'recall'].values[0]
    plt.plot(best_fpr, best_tpr, 'go', markersize=8)
    plt.annotate(f'Optimal (T={results["threshold"]:.2f})', 
                 (best_fpr, best_tpr),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center',
                 fontweight='bold')
    
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # 4. Net impact by threshold
    plt.figure(figsize=(10, 7))
    plt.plot(df['threshold'], df['net_impact_rate'], label='Net Impact Rate', marker='.')
    
    # Mark the optimal threshold
    plt.axvline(x=results['threshold'], color='r', linestyle='--', alpha=0.5, 
                label=f'Optimal Threshold = {results["threshold"]:.4f}')
    
    # Add zero line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add shaded areas for positive and negative impact
    plt.fill_between(df['threshold'], df['net_impact_rate'], 0, 
                     where=(df['net_impact_rate'] > 0), 
                     interpolate=True, color='green', alpha=0.2, label='Positive Net Impact')
    plt.fill_between(df['threshold'], df['net_impact_rate'], 0, 
                     where=(df['net_impact_rate'] <= 0), 
                     interpolate=True, color='red', alpha=0.2, label='Negative Net Impact')
    
    plt.title('Net Impact Rate by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Net Impact Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'net_impact_rate.png'))
    
    # 5. Confusion matrix counts
    plt.figure(figsize=(10, 7))
    plt.plot(df['threshold'], df['true_positives'], label='True Positives (Hallucinations Caught)', marker='.')
    plt.plot(df['threshold'], df['false_positives'], label='False Positives (Correct Responses Blocked)', marker='.')
    plt.plot(df['threshold'], df['false_negatives'], label='False Negatives (Hallucinations Missed)', marker='.')
    plt.plot(df['threshold'], df['true_negatives'], label='True Negatives (Correct Responses Allowed)', marker='.')
    
    # Mark the optimal threshold
    plt.axvline(x=results['threshold'], color='r', linestyle='--', alpha=0.5, 
                label=f'Optimal Threshold = {results["threshold"]:.4f}')
    
    plt.title('Confusion Matrix Counts by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_counts.png'))
    
    # Close all plots to free memory
    plt.close('all')
    
    print(f"Visualizations saved to {output_dir}")

def save_results(results, output_file, pretty=True):
    """Save optimization results to a JSON file"""
    
    # Convert numpy values to Python types for JSON
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        return obj
    
    # Deep copy and convert results
    json_results = {}
    for k, v in results.items():
        if k != 'all_results':  # Skip the full DataFrame to keep file smaller
            json_results[k] = convert_to_json_serializable(v)
    
    # Save to file
    with open(output_file, 'w') as f:
        if pretty:
            json.dump(json_results, f, indent=2)
        else:
            json.dump(json_results, f)
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Optimize hallucination classifier threshold")
    parser.add_argument("--input-file", type=str, 
                      default="evaluation/results/guard_results_classifier-based.csv",
                      help="Path to the evaluation results CSV file")
    parser.add_argument("--output-dir", type=str, 
                      default="evaluation/results/threshold_optimization",
                      help="Directory to save optimization results and visualizations")
    parser.add_argument("--metric", type=str, 
                      choices=["f1", "accuracy", "balanced_accuracy", "precision", "recall", "net_impact_rate"],
                      default="f1",
                      help="Metric to optimize for threshold selection")
    parser.add_argument("--min-threshold", type=float, default=0.01,
                      help="Minimum threshold value to test")
    parser.add_argument("--max-threshold", type=float, default=0.99,
                      help="Maximum threshold value to test")
    parser.add_argument("--num-thresholds", type=int, default=50,
                      help="Number of threshold values to test")
    
    args = parser.parse_args()
    
    # Extract scores from CSV file
    data = extract_scores_from_csv(args.input_file)
    if data is None:
        return
    
    # Optimize threshold
    print(f"Optimizing threshold using {args.metric} metric...")
    threshold_range = (args.min_threshold, args.max_threshold)
    results = optimize_threshold(
        data, 
        metric=args.metric,
        threshold_range=threshold_range,
        num_thresholds=args.num_thresholds
    )
    
    # Print optimal results
    print("\n===== Optimal Threshold Results =====")
    print(f"Optimal threshold: {results['threshold']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"PR AUC: {results['pr_auc']:.4f}")
    print(f"Net Impact: {results['net_impact']} ({results['net_impact_rate']:.4f})")
    
    # Visualize results
    visualize_optimization_results(results, args.output_dir)
    
    # Save results
    output_file = os.path.join(args.output_dir, "optimal_threshold_results.json")
    save_results(results, output_file)
    
    print("\n===== Optimization Complete =====")
    print(f"Optimal threshold {results['threshold']:.4f} selected based on {args.metric}.")
    print(f"This threshold achieves:")
    print(f"- {results['true_positives']} hallucinations caught")
    print(f"- {results['false_positives']} correct responses incorrectly blocked")
    print(f"- {results['false_negatives']} hallucinations missed")
    print(f"- {results['true_negatives']} correct responses allowed")
    print(f"For a net impact of {results['net_impact']} ({results['net_impact_rate']:.2%})")

if __name__ == "__main__":
    main() 