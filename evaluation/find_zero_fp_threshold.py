#!/usr/bin/env python
"""
Script to find the threshold that produces zero false positives.

This script:
1. Loads results from guard_results.csv
2. Finds the minimum similarity score among incorrect responses (hallucinations)
3. Calculates the improvement with this threshold
"""

import os
import csv
import json
import pandas as pd
import numpy as np

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
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    # Prepare data for analysis
    correct_max_scores = []
    incorrect_max_scores = []
    
    # Process each result
    skipped = 0
    for _, row in results_df.iterrows():
        # Parse token scores
        token_scores = parse_token_scores(row.get('token_scores', ''))
        
        # Skip if no token scores
        if not token_scores:
            skipped += 1
            continue
        
        # Get maximum similarity score for this response
        max_sim = get_max_similarity(token_scores)
        
        # Get Claude score (1 = correct, 0 = incorrect/hallucinatory)
        claude_score = row.get('claude_score', -1)
        
        # Skip ambiguous responses
        if claude_score == -1:
            skipped += 1
            continue
            
        # Add to correct or incorrect list
        if claude_score == 1:
            correct_max_scores.append(max_sim)
        else:
            incorrect_max_scores.append(max_sim)
    
    print(f"Processed {len(correct_max_scores) + len(incorrect_max_scores)} results (skipped {skipped})")
    print(f"Correct responses: {len(correct_max_scores)}")
    print(f"Incorrect/hallucinatory responses: {len(incorrect_max_scores)}")
    
    # Find minimum score among correct responses
    # Any threshold lower than this would cause false positives
    if correct_max_scores:
        min_correct_score = min(correct_max_scores)
        print(f"\nMinimum score among correct responses: {min_correct_score:.6f}")
        
        # This is our threshold for zero false positives
        zero_fp_threshold = min_correct_score
        
        # Count how many hallucinations we'd catch with this threshold
        caught_hallucinations = sum(1 for score in incorrect_max_scores if score >= zero_fp_threshold)
        
        # Calculate performance metrics
        total_responses = len(correct_max_scores) + len(incorrect_max_scores)
        baseline_accuracy = len(correct_max_scores) / total_responses
        
        # If we replace caught hallucinations with correct responses
        improved_correct = len(correct_max_scores) + caught_hallucinations
        improved_accuracy = improved_correct / total_responses
        improvement = improved_accuracy - baseline_accuracy
        
        print("\n===== ZERO FALSE POSITIVE THRESHOLD ANALYSIS =====")
        print(f"Threshold for zero false positives: {zero_fp_threshold:.6f}")
        print(f"Hallucinations caught: {caught_hallucinations} out of {len(incorrect_max_scores)} ({caught_hallucinations/len(incorrect_max_scores):.2%})")
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        print(f"Improved accuracy with perfect fallback: {improved_accuracy:.2%}")
        print(f"Net improvement: {improvement:.2%} ({improvement*100:.2f} percentage points)")
        
        # Save results to JSON
        results = {
            "zero_fp_threshold": zero_fp_threshold,
            "caught_hallucinations": caught_hallucinations,
            "total_hallucinations": len(incorrect_max_scores),
            "correct_responses": len(correct_max_scores),
            "total_responses": total_responses,
            "baseline_accuracy": baseline_accuracy,
            "improved_accuracy": improved_accuracy,
            "net_improvement": improvement
        }
        
        output_file = "evaluation/results/zero_fp_threshold_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    else:
        print("No correct responses found. Cannot determine threshold.")

if __name__ == "__main__":
    main() 