#!/usr/bin/env python
"""
Script to find the threshold that produces zero false positives.

This script:
1. Loads results from guard_results.csv
2. Finds the minimum similarity score among correct responses
3. Sets threshold just below this value to achieve zero false positives
4. Calculates the recall and other metrics with this threshold
"""

import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    correct_max_scores = []
    incorrect_max_scores = []
    
    # Store full row data for debug purposes
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
            # Store full example for debugging
            correct_examples.append({
                'index': idx,
                'question': row.get('question', ''),
                'max_sim': max_sim,
                'num_tokens': len(token_scores),
                'token_text': token_scores[0].get('token_text', '') if token_scores else ''
            })
        else:
            incorrect_max_scores.append(max_sim)
            # Store full example for debugging
            incorrect_examples.append({
                'index': idx,
                'question': row.get('question', ''),
                'max_sim': max_sim,
                'num_tokens': len(token_scores),
                'token_text': token_scores[0].get('token_text', '') if token_scores else ''
            })
    
    print(f"\nProcessed {len(correct_max_scores) + len(incorrect_max_scores)} results (skipped {skipped})")
    print(f"Correct responses: {len(correct_max_scores)}")
    print(f"Incorrect/hallucinatory responses: {len(incorrect_max_scores)}")
    print(f"Parsing failures: {parsing_failures}")
    print(f"Rows with empty tokens: {empty_tokens}")
    
    # Show basic statistics about the scores
    if correct_max_scores:
        print("\n===== CORRECT RESPONSES SIMILARITY SCORES =====")
        print(f"Min: {min(correct_max_scores):.6f}")
        print(f"Max: {max(correct_max_scores):.6f}")
        print(f"Mean: {sum(correct_max_scores)/len(correct_max_scores):.6f}")
        
    if incorrect_max_scores:
        print("\n===== INCORRECT RESPONSES SIMILARITY SCORES =====")
        print(f"Min: {min(incorrect_max_scores):.6f}")
        print(f"Max: {max(incorrect_max_scores):.6f}")
        print(f"Mean: {sum(incorrect_max_scores)/len(incorrect_max_scores):.6f}")
    
    # Plot histogram of scores
    if correct_max_scores and incorrect_max_scores:
        plt.figure(figsize=(12, 6))
        plt.hist(correct_max_scores, bins=30, alpha=0.5, label='Correct Responses')
        plt.hist(incorrect_max_scores, bins=30, alpha=0.5, label='Hallucinations')
        plt.xlabel('Maximum Similarity Score')
        plt.ylabel('Count')
        plt.title('Distribution of Maximum Similarity Scores')
        plt.legend()
        plt.savefig('evaluation/results/similarity_distribution.png')
        print(f"\nSaved distribution histogram to evaluation/results/similarity_distribution.png")
    
    # Find minimum score among correct responses
    # Any threshold below this won't cause false positives
    if correct_max_scores:
        min_correct_score = min(correct_max_scores)
        # Subtract a small epsilon to ensure we're below the minimum correct score
        epsilon = 0.000001
        zero_fp_threshold = min_correct_score - epsilon
        
        print(f"\nMinimum score among correct responses: {min_correct_score:.6f}")
        print(f"Setting zero false positive threshold to: {zero_fp_threshold:.6f}")
        
        # Show examples near the threshold
        print("\n===== LOWEST-SCORING CORRECT RESPONSES =====")
        sorted_correct = sorted(correct_examples, key=lambda x: x['max_sim'])
        for i, example in enumerate(sorted_correct[:5]):  # Show 5 lowest scoring correct responses
            print(f"{i+1}. Score: {example['max_sim']:.6f}, Question: {example['question'][:50]}...")
            print(f"   First token: '{example['token_text']}'")
        
        # Check how many correct responses have scores below threshold
        # These would be falsely classified as hallucinations (false positives)
        false_positives = [s for s in correct_max_scores if s < zero_fp_threshold]
        print(f"\nCorrect responses below threshold (false positives): {len(false_positives)}")
        print(f"False positive rate: {len(false_positives)/len(correct_max_scores):.2%}")
            
        # Count how many hallucinations would be caught with this threshold
        # Hallucinations with scores below the threshold are detected
        caught_hallucinations = sum(1 for score in incorrect_max_scores if score < zero_fp_threshold)
        
        # Calculate recall (percentage of hallucinations caught)
        recall = caught_hallucinations / len(incorrect_max_scores) if len(incorrect_max_scores) > 0 else 0
        
        # Show examples of hallucinations near threshold
        print("\n===== LOWEST-SCORING HALLUCINATIONS =====")
        sorted_incorrect = sorted(incorrect_examples, key=lambda x: x['max_sim'])
        for i, example in enumerate(sorted_incorrect[:5]):  # Show 5 lowest scoring hallucinations
            print(f"{i+1}. Score: {example['max_sim']:.6f}, Question: {example['question'][:50]}...")
            print(f"   First token: '{example['token_text']}'")
        
        # Show list of hallucinations that would be caught
        if caught_hallucinations > 0:
            print(f"\n===== HALLUCINATIONS CAUGHT ({caught_hallucinations}) =====")
            caught = [ex for ex in incorrect_examples if ex['max_sim'] < zero_fp_threshold]
            for i, example in enumerate(caught[:3]):  # Show up to 3 examples
                print(f"{i+1}. Score: {example['max_sim']:.6f}, Question: {example['question'][:50]}...")
        
        # Calculate performance metrics
        total_responses = len(correct_max_scores) + len(incorrect_max_scores)
        baseline_accuracy = len(correct_max_scores) / total_responses
        
        # If we replace caught hallucinations with correct responses
        improved_correct = len(correct_max_scores) + caught_hallucinations
        improved_accuracy = improved_correct / total_responses
        improvement = improved_accuracy - baseline_accuracy
        
        print("\n===== ZERO FALSE POSITIVE THRESHOLD ANALYSIS =====")
        print(f"Threshold for zero false positives: {zero_fp_threshold:.6f}")
        print(f"Hallucinations caught: {caught_hallucinations} out of {len(incorrect_max_scores)} ({recall:.2%})")
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        print(f"Improved accuracy with perfect fallback: {improved_accuracy:.2%}")
        print(f"Net improvement: {improvement:.2%} ({improvement*100:.2f} percentage points)")
        
        # Save results to JSON
        results = {
            "zero_fp_threshold": zero_fp_threshold,
            "caught_hallucinations": caught_hallucinations,
            "total_hallucinations": len(incorrect_max_scores),
            "false_positives": len(false_positives),
            "correct_responses": len(correct_max_scores),
            "total_responses": total_responses,
            "baseline_accuracy": baseline_accuracy,
            "improved_accuracy": improved_accuracy,
            "net_improvement": improvement,
            "recall": recall
        }
        
        output_file = "evaluation/results/zero_fp_threshold_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    else:
        print("No correct responses found. Cannot determine threshold.")

if __name__ == "__main__":
    main() 