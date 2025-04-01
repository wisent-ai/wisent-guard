#!/usr/bin/env python
"""
Visualization script for token similarity scores from TruthfulQA evaluation.

This script:
1. Loads results from guard_results.csv
2. Creates visualizations of similarity scores
3. Shows separation between correct and hallucinatory responses
4. Marks key thresholds (zero-FP and F1-optimal)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

def parse_token_scores(token_scores_str):
    """Parse token scores from the formatted string stored in CSV."""
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
            position = int(parts[0]) if parts[0].isdigit() else 0
            token_id = int(parts[1]) if parts[1].isdigit() else 0
            token_text = parts[2]
            try:
                similarity = float(parts[3])
            except ValueError:
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
        except Exception:
            continue
    
    return tokens

def get_max_similarity(token_scores):
    """Get the maximum similarity score from token scores."""
    if not token_scores:
        return 0.0
    
    return max(token.get('similarity', 0.0) for token in token_scores)

def load_thresholds():
    """Load threshold values from JSON files."""
    thresholds = {}
    
    # Load zero FP threshold
    zero_fp_file = "evaluation/results/zero_fp_threshold_results.json"
    if os.path.exists(zero_fp_file):
        with open(zero_fp_file, 'r') as f:
            data = json.load(f)
            thresholds['zero_fp'] = data.get('zero_fp_threshold', 0.05)
    
    # Load optimal F1 threshold
    optimal_file = "evaluation/results/optimal_threshold_metrics.json"
    if os.path.exists(optimal_file):
        with open(optimal_file, 'r') as f:
            data = json.load(f)
            thresholds['optimal_f1'] = data.get('threshold', 0.056)
    
    return thresholds

def create_scatter_plot(df, thresholds, output_dir):
    """Create scatter plot of similarity scores."""
    plt.figure(figsize=(14, 8))
    
    # Scatter plot with jitter for better visibility
    correct = df[df['is_correct']]
    incorrect = df[~df['is_correct']]
    
    plt.scatter(correct.index, correct['max_similarity'], 
                alpha=0.7, color='green', label='Correct Responses', s=60)
    plt.scatter(incorrect.index, incorrect['max_similarity'], 
                alpha=0.7, color='red', label='Hallucinations', s=60)
    
    # Add threshold lines
    if 'zero_fp' in thresholds:
        plt.axhline(y=thresholds['zero_fp'], color='black', linestyle='--', 
                    label=f'Zero FP Threshold ({thresholds["zero_fp"]:.4f})')
    
    if 'optimal_f1' in thresholds:
        plt.axhline(y=thresholds['optimal_f1'], color='blue', linestyle='--', 
                    label=f'Optimal F1 Threshold ({thresholds["optimal_f1"]:.4f})')
    
    plt.xlabel('Response Index')
    plt.ylabel('Maximum Similarity Score')
    plt.title('Maximum Similarity Scores by Response')
    plt.legend()
    plt.ylim(0, df['max_similarity'].max() * 1.1)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'similarity_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {output_path}")

def create_histogram(df, thresholds, output_dir):
    """Create histogram of similarity scores by class."""
    plt.figure(figsize=(14, 8))
    
    # Create separate histograms for correct and incorrect responses
    sns.histplot(df[df['is_correct']]['max_similarity'], 
                 color='green', alpha=0.5, label='Correct Responses', bins=30)
    sns.histplot(df[~df['is_correct']]['max_similarity'], 
                 color='red', alpha=0.5, label='Hallucinations', bins=30)
    
    # Add threshold lines
    if 'zero_fp' in thresholds:
        plt.axvline(x=thresholds['zero_fp'], color='black', linestyle='--', 
                    label=f'Zero FP Threshold ({thresholds["zero_fp"]:.4f})')
    
    if 'optimal_f1' in thresholds:
        plt.axvline(x=thresholds['optimal_f1'], color='blue', linestyle='--', 
                    label=f'Optimal F1 Threshold ({thresholds["optimal_f1"]:.4f})')
    
    plt.xlabel('Maximum Similarity Score')
    plt.ylabel('Count')
    plt.title('Distribution of Maximum Similarity Scores')
    plt.legend()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'similarity_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved to {output_path}")

def create_density_plot(df, thresholds, output_dir):
    """Create density plot of similarity scores by class."""
    plt.figure(figsize=(14, 8))
    
    # Create separate density plots for correct and incorrect responses
    sns.kdeplot(df[df['is_correct']]['max_similarity'], 
                color='green', label='Correct Responses', fill=True, alpha=0.3)
    sns.kdeplot(df[~df['is_correct']]['max_similarity'], 
                color='red', label='Hallucinations', fill=True, alpha=0.3)
    
    # Add threshold lines
    if 'zero_fp' in thresholds:
        plt.axvline(x=thresholds['zero_fp'], color='black', linestyle='--', 
                    label=f'Zero FP Threshold ({thresholds["zero_fp"]:.4f})')
    
    if 'optimal_f1' in thresholds:
        plt.axvline(x=thresholds['optimal_f1'], color='blue', linestyle='--', 
                    label=f'Optimal F1 Threshold ({thresholds["optimal_f1"]:.4f})')
    
    plt.xlabel('Maximum Similarity Score')
    plt.ylabel('Density')
    plt.title('Density Distribution of Maximum Similarity Scores')
    plt.legend()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'similarity_density.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Density plot saved to {output_path}")

def create_roc_curve(df, output_dir):
    """Create ROC curve for similarity scores."""
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(~df['is_correct'], df['max_similarity'])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # Save the plot
    output_path = os.path.join(output_dir, 'similarity_roc.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {output_path}")

def create_cumulative_distribution(df, thresholds, output_dir):
    """Create cumulative distribution plot of similarity scores by class."""
    plt.figure(figsize=(14, 8))
    
    # Sort values
    correct_scores = sorted(df[df['is_correct']]['max_similarity'])
    incorrect_scores = sorted(df[~df['is_correct']]['max_similarity'])
    
    # Create cumulative distributions
    x_correct = np.linspace(0, 1, len(correct_scores))
    x_incorrect = np.linspace(0, 1, len(incorrect_scores))
    
    plt.plot(correct_scores, x_correct, color='green', linewidth=3, 
             label='Correct Responses')
    plt.plot(incorrect_scores, x_incorrect, color='red', linewidth=3, 
             label='Hallucinations')
    
    # Add threshold lines
    if 'zero_fp' in thresholds:
        plt.axvline(x=thresholds['zero_fp'], color='black', linestyle='--', 
                    label=f'Zero FP Threshold ({thresholds["zero_fp"]:.4f})')
    
    if 'optimal_f1' in thresholds:
        plt.axvline(x=thresholds['optimal_f1'], color='blue', linestyle='--', 
                    label=f'Optimal F1 Threshold ({thresholds["optimal_f1"]:.4f})')
    
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Maximum Similarity Score')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Distribution of Maximum Similarity Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'similarity_cumulative.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cumulative distribution plot saved to {output_path}")

def main():
    # Set up paths
    results_file = "evaluation/results/guard_results.csv"
    output_dir = "evaluation/results/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Please run evaluate_llama_truthfulqa.py first.")
        return
    
    # Load threshold values
    thresholds = load_thresholds()
    print(f"Loaded thresholds: {thresholds}")
    
    # Load results
    try:
        results_df = pd.read_csv(results_file)
        print(f"Loaded {len(results_df)} results from {results_file}")
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    # Process each result
    data = []
    skipped = 0
    
    for idx, row in results_df.iterrows():
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
            
        # Add to dataset
        data.append({
            'index': idx,
            'question': row.get('question', f"Question {idx}"),
            'category': row.get('category', 'Unknown'),
            'max_similarity': max_sim,
            'is_correct': claude_score == 1,
            'claude_score': claude_score
        })
    
    # Create DataFrame
    if not data:
        print("No valid data found for visualization.")
        return
        
    df = pd.DataFrame(data)
    print(f"Processed {len(df)} results for visualization (skipped {skipped})")
    
    # Create visualizations
    create_scatter_plot(df, thresholds, output_dir)
    create_histogram(df, thresholds, output_dir)
    create_density_plot(df, thresholds, output_dir)
    create_roc_curve(df, output_dir)
    create_cumulative_distribution(df, thresholds, output_dir)
    
    # Generate summary statistics
    summary = {
        'num_responses': int(len(df)),
        'num_correct': int(df['is_correct'].sum()),
        'num_hallucinations': int((~df['is_correct']).sum()),
        'correct_mean_similarity': float(df[df['is_correct']]['max_similarity'].mean()),
        'hallucination_mean_similarity': float(df[~df['is_correct']]['max_similarity'].mean()),
        'correct_min_similarity': float(df[df['is_correct']]['max_similarity'].min()),
        'correct_max_similarity': float(df[df['is_correct']]['max_similarity'].max()),
        'hallucination_min_similarity': float(df[~df['is_correct']]['max_similarity'].min()),
        'hallucination_max_similarity': float(df[~df['is_correct']]['max_similarity'].max()),
        'thresholds': thresholds
    }
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, 'visualization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nVisualization summary saved to {summary_path}")
    print("\nAll visualizations complete!")

if __name__ == "__main__":
    main() 