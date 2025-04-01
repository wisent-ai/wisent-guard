#!/usr/bin/env python
"""
Comprehensive optimization script for hallucination detection parameters.

This script:
1. Explores multiple parameters simultaneously:
   - Threshold values
   - Token aggregation strategies (max, mean, first-n, etc.)
   - Position weighting schemes
   - Number of tokens to consider
2. Uses efficient optimization strategies
3. Provides detailed metrics and visualizations
4. Performs cross-validation for robust results
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Any, Callable
import itertools
from functools import partial

# Token aggregation strategies
TOKEN_STRATEGIES = [
    'max',           # Maximum similarity of any token
    'mean',          # Mean similarity across all tokens
    'first_token',   # Only consider the first token
    'first_n',       # Mean of first N tokens
    'last_n',        # Mean of last N tokens
    'weighted_pos',  # Position-weighted mean (earlier tokens weighted more)
    'weighted_neg',  # Position-weighted mean (later tokens weighted more)
]

def load_evaluation_results(filepath: str) -> pd.DataFrame:
    """Load evaluation results from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} evaluation results from {filepath}")
    return df

def parse_token_scores(token_scores_str: str) -> List[Dict]:
    """
    Parse token scores from the formatted string in the CSV.
    
    Format: position:token_id:token_text:similarity:category:is_harmful|...
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

def aggregate_token_scores(tokens: List[Dict], strategy: str, n_tokens: int = 5) -> float:
    """
    Aggregate token similarity scores based on the specified strategy.
    
    Args:
        tokens: List of token dictionaries with similarity scores
        strategy: Aggregation strategy from TOKEN_STRATEGIES
        n_tokens: Number of tokens to consider for first_n and last_n strategies
        
    Returns:
        Aggregated similarity score
    """
    if not tokens:
        return 0.0
    
    # Extract similarity scores
    similarities = [t.get('similarity', 0.0) for t in tokens]
    
    if strategy == 'max':
        return max(similarities)
    
    elif strategy == 'mean':
        return sum(similarities) / len(similarities)
    
    elif strategy == 'first_token':
        return similarities[0] if similarities else 0.0
    
    elif strategy == 'first_n':
        n = min(n_tokens, len(similarities))
        return sum(similarities[:n]) / n if n > 0 else 0.0
    
    elif strategy == 'last_n':
        n = min(n_tokens, len(similarities))
        return sum(similarities[-n:]) / n if n > 0 else 0.0
    
    elif strategy == 'weighted_pos':
        # Earlier tokens weighted more (exponential decay)
        weights = [np.exp(-0.1 * i) for i in range(len(similarities))]
        return sum(w * s for w, s in zip(weights, similarities)) / sum(weights)
    
    elif strategy == 'weighted_neg':
        # Later tokens weighted more (exponential growth)
        weights = [np.exp(0.1 * i) for i in range(len(similarities))]
        return sum(w * s for w, s in zip(weights, similarities)) / sum(weights)
    
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")

def evaluate_parameters(
    df: pd.DataFrame, 
    threshold: float, 
    strategy: str, 
    n_tokens: int
) -> Dict:
    """
    Evaluate a specific parameter configuration.
    
    Args:
        df: DataFrame with evaluation results
        threshold: Threshold value for classification
        strategy: Token aggregation strategy
        n_tokens: Number of tokens to consider (for strategies that use it)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Apply configuration to determine which responses would be flagged as harmful
    predictions = []
    
    for _, row in df.iterrows():
        token_data = parse_token_scores(row['token_scores'])
        agg_similarity = aggregate_token_scores(token_data, strategy, n_tokens)
        is_predicted_harmful = agg_similarity >= threshold
        
        # Use Claude score as ground truth (1 = good response, 0 = hallucination)
        is_actually_harmful = row['claude_score'] == 0
        
        predictions.append({
            'predicted_harmful': is_predicted_harmful,
            'actually_harmful': is_actually_harmful,
            'agg_similarity': agg_similarity,
            'question': row['question'],
            'category': row.get('category', 'Unknown')
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
    
    # Compute balanced accuracy
    sensitivity = true_positives / total_harmful if total_harmful > 0 else 0
    specificity = true_negatives / total_harmless if total_harmless > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    # Compute combined score that balances hallucination detection with false positives
    # Weight precision slightly more to avoid excessive false positives
    combined_score = (0.6 * precision) + (0.4 * recall) if (precision + recall) > 0 else 0.0
    
    # Calculate AUC if possible
    auc_score = 0.0
    try:
        y_scores = [p['agg_similarity'] for p in predictions]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
    except:
        pass
    
    metrics = {
        'params': {
            'threshold': threshold,
            'strategy': strategy,
            'n_tokens': n_tokens
        },
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'detection_rate': true_positives / total_harmful if total_harmful > 0 else 0.0,
        'false_positive_rate': false_positives / total_harmless if total_harmless > 0 else 0.0,
        'balanced_accuracy': balanced_acc,
        'combined_score': combined_score,
        'auc': auc_score,
        'predictions': predictions
    }
    
    return metrics

def bayesian_optimization(df: pd.DataFrame, max_evals: int = 100) -> Dict:
    """
    Find optimal parameters using Bayesian optimization.
    
    Args:
        df: DataFrame with evaluation results
        max_evals: Maximum number of evaluations
        
    Returns:
        Dictionary with best parameters and metrics
    """
    # For simplicity, we'll implement a guided search instead of full Bayesian optimization
    # First, do a coarse grid search to find promising regions
    print("Phase 1: Coarse grid search to find promising parameter regions...")
    
    # Define parameter ranges for coarse search
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    strategies = ['max', 'mean', 'first_token', 'weighted_pos']
    n_tokens_values = [3, 5, 10]
    
    best_score = -1
    best_params = None
    all_results = []
    
    # Evaluate all combinations in the coarse grid
    total_combos = len(thresholds) * len(strategies) * len(n_tokens_values)
    for threshold, strategy, n_tokens in tqdm(itertools.product(thresholds, strategies, n_tokens_values), 
                                            total=total_combos, 
                                            desc="Coarse grid search"):
        metrics = evaluate_parameters(df, threshold, strategy, n_tokens)
        all_results.append(metrics)
        
        # Track best parameters based on combined score
        if metrics['combined_score'] > best_score:
            best_score = metrics['combined_score']
            best_params = metrics['params']
    
    print(f"Phase 1 complete. Best parameters: {best_params}, Score: {best_score:.4f}")
    
    # Phase 2: Fine-grained search around best parameters
    print("Phase 2: Fine-grained search around best parameters...")
    
    # Generate refined search space based on best parameters
    best_threshold = best_params['threshold']
    best_strategy = best_params['strategy']
    best_n_tokens = best_params['n_tokens']
    
    # Refined threshold search space
    refined_thresholds = [
        max(0.01, best_threshold - 0.04),
        max(0.01, best_threshold - 0.02),
        best_threshold,
        best_threshold + 0.02,
        best_threshold + 0.04
    ]
    
    # Refined strategy search space - include best strategy and nearby alternatives
    if best_strategy in ['first_n', 'last_n', 'weighted_pos', 'weighted_neg']:
        refined_strategies = [best_strategy]
    else:
        refined_strategies = [best_strategy]
    
    # Refined n_tokens search space (if applicable)
    if best_strategy in ['first_n', 'last_n']:
        refined_n_tokens = [
            max(1, best_n_tokens - 2),
            max(1, best_n_tokens - 1),
            best_n_tokens,
            best_n_tokens + 1,
            best_n_tokens + 2
        ]
    else:
        refined_n_tokens = [best_n_tokens]
    
    # Run fine-grained search
    total_refined = len(refined_thresholds) * len(refined_strategies) * len(refined_n_tokens)
    for threshold, strategy, n_tokens in tqdm(
        itertools.product(refined_thresholds, refined_strategies, refined_n_tokens),
        total=total_refined,
        desc="Fine-grained search"
    ):
        metrics = evaluate_parameters(df, threshold, strategy, n_tokens)
        all_results.append(metrics)
        
        # Track best parameters based on combined score
        if metrics['combined_score'] > best_score:
            best_score = metrics['combined_score']
            best_params = metrics['params']
    
    print(f"Phase 2 complete. Best parameters: {best_params}, Score: {best_score:.4f}")
    
    # Extract best result
    best_result = next(r for r in all_results 
                      if r['params']['threshold'] == best_params['threshold'] 
                      and r['params']['strategy'] == best_params['strategy']
                      and r['params']['n_tokens'] == best_params['n_tokens'])
    
    return {
        'best_params': best_params,
        'best_metrics': {k: v for k, v in best_result.items() if k != 'predictions'},
        'all_results': [{k: v for k, v in r.items() if k != 'predictions'} for r in all_results]
    }

def cross_validate_parameters(
    df: pd.DataFrame, 
    best_params: Dict[str, Any], 
    n_splits: int = 5
) -> Dict:
    """
    Perform cross-validation for the best parameters to ensure robustness.
    
    Args:
        df: DataFrame with evaluation results
        best_params: Best parameters from optimization
        n_splits: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation results
    """
    print(f"Performing {n_splits}-fold cross-validation...")
    
    # Extract parameters
    threshold = best_params['threshold']
    strategy = best_params['strategy']
    n_tokens = best_params['n_tokens']
    
    # Prepare data for cross-validation
    token_scores = [parse_token_scores(row['token_scores']) for _, row in df.iterrows()]
    agg_scores = [aggregate_token_scores(tokens, strategy, n_tokens) for tokens in token_scores]
    y_true = [row['claude_score'] == 0 for _, row in df.iterrows()]
    
    # Create cross-validation folds
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Track metrics across folds
    fold_metrics = []
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(cv.split(agg_scores, y_true)):
        # Get test set
        y_test = [y_true[i] for i in test_idx]
        scores_test = [agg_scores[i] for i in test_idx]
        
        # Apply threshold to classify
        y_pred = [score >= threshold for score in scores_test]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate AUC
        try:
            fpr, tpr, _ = roc_curve(y_test, scores_test)
            auc_score = auc(fpr, tpr)
        except:
            auc_score = 0.0
        
        fold_metrics.append({
            'fold': i + 1,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score
        })
    
    # Calculate average metrics
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'precision_std': np.std([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'recall_std': np.std([m['recall'] for m in fold_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in fold_metrics]),
        'f1_score_std': np.std([m['f1_score'] for m in fold_metrics]),
        'auc': np.mean([m['auc'] for m in fold_metrics]),
        'auc_std': np.std([m['auc'] for m in fold_metrics])
    }
    
    return {
        'fold_metrics': fold_metrics,
        'avg_metrics': avg_metrics
    }

def plot_precision_recall_curves(df: pd.DataFrame, top_results: List[Dict], output_path: str = None):
    """
    Plot precision-recall curves for top parameter configurations.
    
    Args:
        df: DataFrame with evaluation results
        top_results: List of top parameter configurations
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for result in top_results:
        params = result['params']
        threshold = params['threshold']
        strategy = params['strategy']
        n_tokens = params['n_tokens']
        
        # Get predictions for this configuration
        token_scores = [parse_token_scores(row['token_scores']) for _, row in df.iterrows()]
        agg_scores = [aggregate_token_scores(tokens, strategy, n_tokens) for tokens in token_scores]
        y_true = [row['claude_score'] == 0 for _, row in df.iterrows()]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, agg_scores)
        
        # Calculate area under curve
        auc_score = auc(recall, precision)
        
        # Plot curve
        label = f"Strategy: {strategy}, Threshold: {threshold:.3f}, n={n_tokens}, AUC: {auc_score:.3f}"
        plt.plot(recall, precision, lw=2, label=label)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Top Parameter Configurations')
    plt.legend(loc="best")
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Precision-recall curves saved to {output_path}")
    else:
        plt.show()

def analyze_per_category(df: pd.DataFrame, best_params: Dict, output_path: str = None):
    """
    Analyze performance per category.
    
    Args:
        df: DataFrame with evaluation results
        best_params: Best parameters from optimization
        output_path: Path to save the results
    """
    # Extract parameters
    threshold = best_params['threshold']
    strategy = best_params['strategy']
    n_tokens = best_params['n_tokens']
    
    # Get category information
    categories = df['category'].unique()
    
    # Calculate metrics per category
    category_metrics = []
    
    for category in categories:
        # Filter dataframe to this category
        cat_df = df[df['category'] == category]
        
        # Skip if too few samples
        if len(cat_df) < 5:
            continue
        
        # Evaluate on this category
        metrics = evaluate_parameters(cat_df, threshold, strategy, n_tokens)
        
        category_metrics.append({
            'category': category,
            'count': len(cat_df),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'false_positive_rate': metrics['false_positive_rate']
        })
    
    # Sort by count, descending
    category_metrics = sorted(category_metrics, key=lambda x: x['count'], reverse=True)
    
    # Create table
    table = "| Category | Count | Precision | Recall | F1 Score | False Positive Rate |\n"
    table += "|----------|-------|-----------|--------|----------|-----------------|\n"
    
    for metrics in category_metrics:
        row = f"| {metrics['category']} | {metrics['count']} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['false_positive_rate']:.4f} |\n"
        table += row
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"Category analysis saved to {output_path}")
    
    return category_metrics, table

def main(args):
    # Load evaluation results
    df = load_evaluation_results(args.input_file)
    
    # Filter out rows without Claude scores or with ambiguous scores (-1)
    df = df[df['claude_score'] != -1]
    
    # Optimize parameters
    results = bayesian_optimization(df, max_evals=args.max_evaluations)
    best_params = results['best_params']
    best_metrics = results['best_metrics']
    
    # Print best parameters and metrics
    print("\n===== Best Parameters =====")
    print(f"Token aggregation strategy: {best_params['strategy']}")
    print(f"Threshold: {best_params['threshold']:.4f}")
    print(f"Number of tokens (if applicable): {best_params['n_tokens']}")
    
    print("\n===== Best Metrics =====")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 score: {best_metrics['f1_score']:.4f}")
    print(f"Balanced accuracy: {best_metrics['balanced_accuracy']:.4f}")
    print(f"AUC: {best_metrics['auc']:.4f}")
    print(f"Detection rate: {best_metrics['detection_rate']:.4f}")
    print(f"False positive rate: {best_metrics['false_positive_rate']:.4f}")
    
    # Cross-validate best parameters
    cv_results = cross_validate_parameters(df, best_params)
    
    print("\n===== Cross-Validation Results =====")
    print(f"Average precision: {cv_results['avg_metrics']['precision']:.4f} ± {cv_results['avg_metrics']['precision_std']:.4f}")
    print(f"Average recall: {cv_results['avg_metrics']['recall']:.4f} ± {cv_results['avg_metrics']['recall_std']:.4f}")
    print(f"Average F1 score: {cv_results['avg_metrics']['f1_score']:.4f} ± {cv_results['avg_metrics']['f1_score_std']:.4f}")
    print(f"Average AUC: {cv_results['avg_metrics']['auc']:.4f} ± {cv_results['avg_metrics']['auc_std']:.4f}")
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save results to JSON
        results_path = os.path.join(args.output_dir, "optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_metrics': best_metrics,
                'cross_validation': cv_results,
                'all_results': results['all_results']
            }, f, indent=2)
        
        # Get top 5 parameter configurations
        top_results = sorted(results['all_results'], key=lambda x: x['combined_score'], reverse=True)[:5]
        
        # Generate precision-recall curves
        pr_curves_path = os.path.join(args.output_dir, "precision_recall_curves.png")
        plot_precision_recall_curves(df, top_results, pr_curves_path)
        
        # Analyze per category
        category_path = os.path.join(args.output_dir, "category_analysis.md")
        analyze_per_category(df, best_params, category_path)
        
        print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hallucination detection parameters")
    
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to CSV file with evaluation results (containing claude_score and token_scores)")
    parser.add_argument("--output-dir", type=str, default="evaluation/parameter_optimization",
                        help="Directory to save optimization results")
    
    # Optimization parameters
    parser.add_argument("--max-evaluations", type=int, default=100,
                        help="Maximum number of parameter combinations to evaluate")
    
    args = parser.parse_args()
    main(args) 