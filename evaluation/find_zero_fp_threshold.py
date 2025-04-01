#!/usr/bin/env python
"""
Enhanced script to find the optimal zero false positive configuration.

This script:
1. Loads results from guard_results.csv
2. Tests multiple token aggregation strategies (max, mean, first_n, etc.)
3. For each strategy, finds the threshold that produces zero false positives
4. Compares strategies to find which one catches the most hallucinations
   while maintaining zero false positives
5. Creates visualization and saves detailed results
"""

import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Token aggregation strategies to evaluate
TOKEN_STRATEGIES = [
    'max',           # Maximum similarity of any token
    'mean',          # Mean similarity across all tokens
    'first_token',   # Only consider the first token
    'first_n',       # Mean of first N tokens
    'last_n',        # Mean of last N tokens
    'weighted_pos',  # Position-weighted mean (earlier tokens weighted more)
    'weighted_neg',  # Position-weighted mean (later tokens weighted more)
]

def parse_token_scores(token_scores_str):
    """
    Parse token scores from the formatted string stored in CSV.
    
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
        # Default to max if unknown strategy
        return max(similarities)

def find_zero_fp_threshold(correct_scores: List[float]) -> Tuple[float, float]:
    """
    Find the threshold that produces zero false positives.
    
    Args:
        correct_scores: List of similarity scores from correct responses
        
    Returns:
        Tuple of (threshold, minimum correct score)
    """
    if not correct_scores:
        return 0.0, 0.0
        
    min_correct_score = min(correct_scores)
    
    # Set threshold just below the minimum correct score
    epsilon = 0.000001
    threshold = min_correct_score - epsilon
    
    return threshold, min_correct_score

def evaluate_strategy(
    correct_tokens: List[List[Dict]], 
    incorrect_tokens: List[List[Dict]],
    strategy: str,
    n_tokens_values: List[int] = [3, 5, 10]
) -> List[Dict]:
    """
    Evaluate a strategy with different n_tokens values.
    
    Args:
        correct_tokens: List of token lists from correct responses
        incorrect_tokens: List of token lists from incorrect responses
        strategy: Token aggregation strategy
        n_tokens_values: List of n_tokens values to try
        
    Returns:
        List of results dictionaries
    """
    results = []
    
    for n_tokens in n_tokens_values:
        # Aggregate scores using the strategy and n_tokens
        correct_scores = [aggregate_token_scores(tokens, strategy, n_tokens) for tokens in correct_tokens]
        incorrect_scores = [aggregate_token_scores(tokens, strategy, n_tokens) for tokens in incorrect_tokens]
        
        # Find zero FP threshold
        threshold, min_correct_score = find_zero_fp_threshold(correct_scores)
        
        # Count caught hallucinations
        caught_hallucinations = sum(1 for score in incorrect_scores if score < threshold)
        total_hallucinations = len(incorrect_scores)
        
        # Calculate recall
        recall = caught_hallucinations / total_hallucinations if total_hallucinations > 0 else 0.0
        
        # Calculate ROC curve for visualization (using thresholds below min_correct)
        thresholds = np.linspace(0, min_correct_score * 1.2, 100)
        recalls = []
        fps = []
        
        for t in thresholds:
            rec = sum(1 for score in incorrect_scores if score < t) / total_hallucinations
            fp = sum(1 for score in correct_scores if score < t) / len(correct_scores)
            recalls.append(rec)
            fps.append(fp)
        
        # Calculate AUC (area under ROC curve)
        auc = 0.0
        for i in range(1, len(fps)):
            auc += (recalls[i] - recalls[i-1]) * (1 - (fps[i] + fps[i-1]) / 2)
        
        result = {
            "strategy": strategy,
            "n_tokens": n_tokens,
            "threshold": threshold,
            "min_correct_score": min_correct_score,
            "caught_hallucinations": caught_hallucinations,
            "total_hallucinations": total_hallucinations,
            "recall": recall,
            "auc": auc,
            "roc_data": {
                "thresholds": thresholds.tolist(),
                "recalls": recalls,
                "false_positives": fps
            }
        }
        
        results.append(result)
    
    return results

def main():
    results_file = "evaluation/results/guard_results.csv"
    
    # Check if file exists
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Please run evaluate_llama_truthfulqa.py first.")
        return
    
    # Load results
    try:
        results_df = pd.read_csv(results_file)
        print(f"Loaded {len(results_df)} results from {results_file}")
        
        # Print column names to verify data structure
        print("\nColumns in the dataset:")
        for col in results_df.columns:
            print(f"  - {col}")
            
        # Check for missing values
        print("\nMissing values in key columns:")
        print(f"  claude_score: {results_df['claude_score'].isna().sum()} missing")
        print(f"  token_scores: {results_df['token_scores'].isna().sum()} missing")
        
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    # Prepare data for analysis
    correct_tokens = []
    incorrect_tokens = []
    
    # Store example data for reporting
    correct_examples = []
    incorrect_examples = []
    
    # Track parsing failures
    parsing_failures = 0
    empty_tokens = 0
    
    # Process each result
    skipped = 0
    for idx, row in results_df.iterrows():
        # Parse token scores
        token_scores_str = row.get('token_scores', '')
        token_scores = parse_token_scores(token_scores_str)
        
        # Track parsing issues
        if token_scores_str and not token_scores:
            parsing_failures += 1
            if parsing_failures < 3:  # Show a few examples of parsing failures
                print(f"\nParsing failure example {parsing_failures}:")
                print(f"  Token string (first 100 chars): {token_scores_str[:100]}...")
        
        # Skip if no token scores
        if not token_scores:
            empty_tokens += 1
            skipped += 1
            continue
        
        # Get Claude score (1 = correct, 0 = incorrect/hallucinatory)
        claude_score = row.get('claude_score', -1)
        
        # Skip ambiguous responses
        if claude_score == -1:
            skipped += 1
            continue
            
        # Add to correct or incorrect list
        question = row.get('question', '')
        if claude_score == 1:
            correct_tokens.append(token_scores)
            # Store example for reporting
            correct_examples.append({
                'index': idx,
                'question': question,
                'first_token': token_scores[0].get('token_text', '') if token_scores else ''
            })
        else:
            incorrect_tokens.append(token_scores)
            # Store example for reporting
            incorrect_examples.append({
                'index': idx,
                'question': question,
                'first_token': token_scores[0].get('token_text', '') if token_scores else ''
            })
    
    print(f"\nProcessed {len(correct_tokens) + len(incorrect_tokens)} results (skipped {skipped})")
    print(f"Correct responses: {len(correct_tokens)}")
    print(f"Incorrect/hallucinatory responses: {len(incorrect_tokens)}")
    print(f"Parsing failures: {parsing_failures}")
    print(f"Rows with empty tokens: {empty_tokens}")
    
    # Try all strategies and find the best one
    print("\nEvaluating token aggregation strategies...")
    
    # N tokens values to try
    n_tokens_values = [3, 5, 10, 15]
    
    # Evaluate all strategies
    all_results = []
    
    for strategy in tqdm(TOKEN_STRATEGIES, desc="Strategies"):
        strategy_results = evaluate_strategy(correct_tokens, incorrect_tokens, strategy, n_tokens_values)
        all_results.extend(strategy_results)
    
    # Find best result (highest recall)
    best_result = max(all_results, key=lambda x: x['recall'])
    
    # Display overall results
    print("\n===== OVERALL RESULTS =====")
    print(f"Best strategy: {best_result['strategy']} with n_tokens={best_result['n_tokens']}")
    print(f"Zero FP threshold: {best_result['threshold']:.6f}")
    print(f"Hallucinations caught: {best_result['caught_hallucinations']} out of {best_result['total_hallucinations']} ({best_result['recall']:.2%})")
    
    # Show results for all strategies
    print("\n===== RESULTS BY STRATEGY =====")
    
    # Group by strategy and select best n_tokens for each
    strategy_best = {}
    for result in all_results:
        strategy = result['strategy']
        if strategy not in strategy_best or result['recall'] > strategy_best[strategy]['recall']:
            strategy_best[strategy] = result
    
    # Print sorted by recall (descending)
    for strategy, result in sorted(strategy_best.items(), key=lambda x: x[1]['recall'], reverse=True):
        print(f"{strategy} (n={result['n_tokens']}): Threshold={result['threshold']:.6f}, Recall={result['recall']:.2%}, Caught={result['caught_hallucinations']}")
    
    # Create directory for results if needed
    os.makedirs("evaluation/results", exist_ok=True)
    
    # Plot ROC curves for all strategies (best n_tokens for each)
    plt.figure(figsize=(12, 8))
    
    for strategy, result in strategy_best.items():
        roc = result['roc_data']
        plt.plot(roc['false_positives'], roc['recalls'], label=f"{strategy} (n={result['n_tokens']}, AUC={result['auc']:.3f})")
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall (Hallucination Detection Rate)')
    plt.title('ROC Curves by Token Aggregation Strategy')
    plt.legend()
    plt.grid(True)
    plt.savefig('evaluation/results/zero_fp_strategy_roc.png')
    print(f"\nROC curves saved to evaluation/results/zero_fp_strategy_roc.png")
    
    # Plot bar chart of recall by strategy
    plt.figure(figsize=(12, 6))
    strategies = [r['strategy'] for r in sorted(strategy_best.values(), key=lambda x: x['recall'], reverse=True)]
    recalls = [strategy_best[s]['recall'] for s in strategies]
    
    plt.bar(strategies, recalls)
    plt.xlabel('Token Aggregation Strategy')
    plt.ylabel('Recall (% of Hallucinations Caught)')
    plt.title('Hallucination Detection Rate by Strategy (Zero False Positives)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('evaluation/results/zero_fp_strategy_comparison.png')
    print(f"Strategy comparison saved to evaluation/results/zero_fp_strategy_comparison.png")
    
    # Save detailed results to JSON
    output_file = "evaluation/results/zero_fp_optimization_results.json"
    full_results = {
        "best_result": best_result,
        "strategy_best": {k: v for k, v in strategy_best.items()},
        "correct_count": len(correct_tokens),
        "incorrect_count": len(incorrect_tokens),
        "skipped_count": skipped
    }
    
        with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main() 