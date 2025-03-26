#!/usr/bin/env python
"""
Compare baseline and guard results from their individual CSV files.

This script loads and compares the baseline_results.csv and guard_results.csv
to analyze the differences in Claude scores between baseline and guard models.
"""

import pandas as pd
import argparse
import os
from collections import Counter

def compare_results(baseline_file, guard_file):
    """Compare baseline and guard results from separate files."""
    # Check if files exist
    if not os.path.exists(baseline_file):
        print(f"Error: Baseline results file {baseline_file} not found.")
        return
    
    if not os.path.exists(guard_file):
        print(f"Error: Guard results file {guard_file} not found.")
        return
    
    # Load the result files
    baseline_df = pd.read_csv(baseline_file)
    guard_df = pd.read_csv(guard_file)
    
    print(f"Loaded {len(baseline_df)} baseline samples from {baseline_file}")
    print(f"Loaded {len(guard_df)} guard samples from {guard_file}")
    
    # Ensure the question order matches
    baseline_questions = baseline_df['question'].tolist()
    guard_questions = guard_df['question'].tolist()
    
    # Check if the datasets have the same questions
    if len(baseline_questions) != len(guard_questions) or set(baseline_questions) != set(guard_questions):
        print("Warning: Baseline and guard datasets have different questions.")
        # Find common questions
        common_questions = list(set(baseline_questions) & set(guard_questions))
        print(f"Found {len(common_questions)} common questions for comparison.")
        
        # Filter to common questions
        baseline_df = baseline_df[baseline_df['question'].isin(common_questions)]
        guard_df = guard_df[guard_df['question'].isin(common_questions)]
    
    # Create a combined dataset
    combined_df = pd.DataFrame()
    combined_df['question'] = baseline_df['question']
    combined_df['category'] = baseline_df['category']
    combined_df['baseline_response'] = baseline_df['response']
    combined_df['guard_response'] = guard_df.set_index('question').loc[baseline_df['question']]['response'].values
    combined_df['baseline_claude_score'] = baseline_df['claude_score']
    combined_df['guard_claude_score'] = guard_df.set_index('question').loc[baseline_df['question']]['claude_score'].values
    
    if 'is_harmful' in guard_df.columns:
        combined_df['is_harmful'] = guard_df.set_index('question').loc[baseline_df['question']]['is_harmful'].values
        combined_df['similarity_score'] = guard_df.set_index('question').loc[baseline_df['question']]['similarity_score'].values
        combined_df['blocked'] = guard_df.set_index('question').loc[baseline_df['question']]['blocked'].values
    
    # Find cases where Claude's evaluation differed
    improved_cases = combined_df[combined_df['baseline_claude_score'] < combined_df['guard_claude_score']]
    worsened_cases = combined_df[combined_df['baseline_claude_score'] > combined_df['guard_claude_score']]
    
    # Calculate statistics
    total_samples = len(combined_df)
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
    all_categories = sorted(set(combined_df['category']))
    
    print(f"{'Category':<25} {'Improved':<10} {'Worsened':<10} {'Net':<10}")
    print("-" * 55)
    
    for category in all_categories:
        improved = improved_by_category.get(category, 0)
        worsened = worsened_by_category.get(category, 0)
        net = improved - worsened
        print(f"{category:<25} {improved:<10} {worsened:<10} {net:<10}")
    
    # Print detailed information about differences (limit to first 10 of each for brevity)
    if improved_count > 0:
        print(f"\n===== IMPROVED RESPONSES (showing first {min(10, improved_count)}) =====")
        for i, (_, row) in enumerate(improved_cases.iterrows()):
            if i >= 10:
                break
            print(f"\nQuestion: {row['question']}")
            print(f"Category: {row['category']}")
            print(f"Baseline response (Claude score: {row['baseline_claude_score']}):")
            print_truncated(row['baseline_response'])
            print(f"Guard response (Claude score: {row['guard_claude_score']}):")
            print_truncated(row['guard_response'])
            print("-" * 80)
    
    if worsened_count > 0:
        print(f"\n===== WORSENED RESPONSES (showing first {min(10, worsened_count)}) =====")
        for i, (_, row) in enumerate(worsened_cases.iterrows()):
            if i >= 10:
                break
            print(f"\nQuestion: {row['question']}")
            print(f"Category: {row['category']}")
            print(f"Baseline response (Claude score: {row['baseline_claude_score']}):")
            print_truncated(row['baseline_response'])
            print(f"Guard response (Claude score: {row['guard_claude_score']}):")
            print_truncated(row['guard_response'])
            print("-" * 80)
    
    # Look for patterns in detection
    if 'is_harmful' in combined_df.columns:
        print("\n===== HALLUCINATION DETECTION ANALYSIS =====")
        detected_harmful = combined_df[combined_df['is_harmful'] == True]
        print(f"Total detected as harmful: {len(detected_harmful)} ({len(detected_harmful)/total_samples:.2%})")
        
        # Check correlation with Claude scores
        harmful_correct = detected_harmful[detected_harmful['guard_claude_score'] == 1]
        harmful_incorrect = detected_harmful[detected_harmful['guard_claude_score'] == 0]
        
        print(f"Detected as harmful but Claude scored as correct: {len(harmful_correct)}")
        print(f"Detected as harmful and Claude scored as incorrect: {len(harmful_incorrect)}")
        
        if len(detected_harmful) > 0:
            print(f"\nExamples detected as harmful (showing first {min(5, len(detected_harmful))}):")
            for i, (_, row) in enumerate(detected_harmful.iterrows()):
                if i >= 5:
                    break
                print(f"\nQuestion: {row['question']}")
                print(f"Category: {row['category']}")
                print(f"Similarity score: {row['similarity_score']}")
                print(f"Claude score: {row['guard_claude_score']}")
                print(f"Response: {row['guard_response'][:100]}...")
    
    # Save combined dataset to file
    output_file = "complete_combined_results.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined results to {output_file}")

def print_truncated(text, max_length=200):
    """Print a truncated version of a long text."""
    if text is None:
        print("[None]")
        return
    
    if isinstance(text, float) and pd.isna(text):
        print("[NaN]")
        return
    
    if len(str(text)) > max_length:
        print(f"{str(text)[:max_length]}... [truncated]")
    else:
        print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline and guard results")
    parser.add_argument("--baseline-file", type=str, 
                        default="evaluation/results/baseline_results.csv",
                        help="Path to the baseline results CSV file")
    parser.add_argument("--guard-file", type=str, 
                        default="evaluation/results/guard_results.csv",
                        help="Path to the guard results CSV file")
    
    args = parser.parse_args()
    compare_results(args.baseline_file, args.guard_file) 