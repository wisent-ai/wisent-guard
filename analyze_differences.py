#!/usr/bin/env python
"""
Analyze differences between baseline model responses and guard-enhanced responses
from the TruthfulQA evaluation.

This script identifies cases where Claude's evaluation of the responses differed,
helping us understand where and how the guard may be improving results.
"""

import pandas as pd
import argparse
import os
from collections import Counter

def analyze_differences(results_file):
    """Analyze differences between baseline and guard responses."""
    # Load the combined results file
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found.")
        return
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} evaluation samples from {results_file}")
    
    # Find cases where Claude's evaluation differed
    improved_cases = df[df['baseline_claude_score'] < df['guard_claude_score']]
    worsened_cases = df[df['baseline_claude_score'] > df['guard_claude_score']]
    
    # Calculate statistics
    total_samples = len(df)
    total_different = len(improved_cases) + len(worsened_cases)
    improved_count = len(improved_cases)
    worsened_count = len(worsened_cases)
    
    # Print summary statistics
    print("\n===== DIFFERENCE ANALYSIS =====")
    print(f"Total samples: {total_samples}")
    print(f"Samples with different outcomes: {total_different} ({total_different/total_samples:.2%})")
    print(f"Samples where guard improved response: {improved_count} ({improved_count/total_samples:.2%})")
    print(f"Samples where guard worsened response: {worsened_count} ({worsened_count/total_samples:.2%})")
    print(f"Net improvement: {improved_count - worsened_count} samples ({(improved_count - worsened_count)/total_samples:.2%})")
    
    # Analyze by category
    print("\n===== CATEGORY ANALYSIS =====")
    improved_by_category = Counter(improved_cases['category'])
    worsened_by_category = Counter(worsened_cases['category'])
    
    # Get all unique categories
    all_categories = sorted(set(df['category']))
    
    print(f"{'Category':<25} {'Improved':<10} {'Worsened':<10} {'Net':<10}")
    print("-" * 55)
    
    for category in all_categories:
        improved = improved_by_category.get(category, 0)
        worsened = worsened_by_category.get(category, 0)
        net = improved - worsened
        print(f"{category:<25} {improved:<10} {worsened:<10} {net:<10}")
    
    # Print detailed information about differences
    if improved_count > 0:
        print("\n===== IMPROVED RESPONSES =====")
        for i, row in improved_cases.iterrows():
            print(f"\nQuestion {i+1}: {row['question']}")
            print(f"Category: {row['category']}")
            print(f"Baseline response (Claude score: {row['baseline_claude_score']}):")
            print_truncated(row['baseline_response'])
            print(f"Guard response (Claude score: {row['guard_claude_score']}):")
            print_truncated(row['guard_response'])
            print("-" * 80)
    
    if worsened_count > 0:
        print("\n===== WORSENED RESPONSES =====")
        for i, row in worsened_cases.iterrows():
            print(f"\nQuestion {i+1}: {row['question']}")
            print(f"Category: {row['category']}")
            print(f"Baseline response (Claude score: {row['baseline_claude_score']}):")
            print_truncated(row['baseline_response'])
            print(f"Guard response (Claude score: {row['guard_claude_score']}):")
            print_truncated(row['guard_response'])
            print("-" * 80)
    
    # Look for patterns in detection
    if 'is_harmful' in df.columns:
        print("\n===== HALLUCINATION DETECTION ANALYSIS =====")
        detected_harmful = df[df['is_harmful'] == True]
        print(f"Total detected as harmful: {len(detected_harmful)} ({len(detected_harmful)/total_samples:.2%})")
        
        # Check correlation with Claude scores
        harmful_correct = detected_harmful[detected_harmful['guard_claude_score'] == 1]
        harmful_incorrect = detected_harmful[detected_harmful['guard_claude_score'] == 0]
        
        print(f"Detected as harmful but Claude scored as correct: {len(harmful_correct)}")
        print(f"Detected as harmful and Claude scored as incorrect: {len(harmful_incorrect)}")
        
        if len(detected_harmful) > 0:
            print("\nExamples detected as harmful:")
            for i, row in detected_harmful.head(5).iterrows():
                print(f"\nQuestion: {row['question']}")
                print(f"Category: {row['category']}")
                print(f"Similarity score: {row['similarity_score']}")
                print(f"Claude score: {row['guard_claude_score']}")
                print(f"Response: {row['guard_response'][:100]}...")

def print_truncated(text, max_length=200):
    """Print a truncated version of a long text."""
    if len(text) > max_length:
        print(f"{text[:max_length]}... [truncated]")
    else:
        print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze differences between baseline and guard responses")
    parser.add_argument("--results-file", type=str, 
                        default="evaluation/results/combined_results.csv",
                        help="Path to the combined results CSV file")
    
    args = parser.parse_args()
    analyze_differences(args.results_file) 