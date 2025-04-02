#!/usr/bin/env python3
"""
Evaluate classifier performance using the guard_results_classifier-based.csv file.
This script analyzes how well the hallucination classifier performs by examining
the correlation between classification results and accuracy as determined by human evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def evaluate_classifier(results_file=None, score_source="human"):
    """
    Evaluate the performance of the hallucination classifier.
    
    Args:
        results_file: Path to CSV results file
        score_source: Source of the scores ('human' or 'claude')
    """
    print("Evaluating classifier performance...")
    
    # Handle terminology based on score source
    score_type = score_source.capitalize()
    
    # Set the score column based on source
    score_column = 'human_score'
    
    # Load the results CSV file
    try:
        if results_file is None:
            results_file = "evaluation/results/guard_results_classifier-based.csv"
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} records from {results_file}")
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # Check if the file contains the needed columns
    if score_column not in df.columns:
        print(f"Error: {score_column} column not found in the data")
        return
        
    # Clean up data - remove NaN values or replace with defaults
    df['is_harmful'] = df['is_harmful'].fillna(False)
    df['blocked'] = df['blocked'].fillna(False)
    
    # Convert score to boolean for analysis
    # Score of 1 = correct answer (not harmful)
    # Score of 0 = incorrect answer (harmful)
    df['is_correct'] = df[score_column] == 1
    
    # Analysis 1: How often is the classifier's prediction correct?
    
    # True positives: Classifier identified as harmful and evaluation says it's incorrect (0)
    true_positives = df[(df['blocked'] == True) & (df['is_correct'] == False)].shape[0]
    
    # True negatives: Classifier identified as not harmful and evaluation says it's correct (1)
    true_negatives = df[(df['blocked'] == False) & (df['is_correct'] == True)].shape[0]
    
    # False positives: Classifier identified as harmful but evaluation says it's correct (1)
    false_positives = df[(df['blocked'] == True) & (df['is_correct'] == True)].shape[0]
    
    # False negatives: Classifier identified as not harmful but evaluation says it's incorrect (0)
    false_negatives = df[(df['blocked'] == False) & (df['is_correct'] == False)].shape[0]
    
    total = len(df)
    
    # Calculate totals for correct and incorrect responses
    total_correct = df['is_correct'].sum()
    total_incorrect = total - total_correct
    
    # Calculate metrics
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print summary statistics
    print("\n===== Classifier Performance Summary =====")
    print(f"Total examples evaluated: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"               | {score_type}: Correct | {score_type}: Incorrect |")
    print(f"---------------|----------------|------------------|")
    print(f"Model: Blocked |      {false_positives:4d}       |        {true_positives:4d}        |")
    print(f"Model: Allowed |      {true_negatives:4d}       |        {false_negatives:4d}        |")
    
    # Add net impact analysis
    print(f"\n===== Net Impact of Using Classifier =====")
    
    # Calculate totals by evaluation scores
    correct_responses = total_correct
    incorrect_responses = total_incorrect
    
    # Calculate responses after classifier is applied
    correct_kept = true_negatives
    incorrect_caught = true_positives
    incorrect_allowed = false_negatives
    correct_blocked = false_positives
    
    print(f"Total responses: {total}")
    print(f"Responses marked correct by {score_type.lower()}: {correct_responses}")
    print(f"Responses marked incorrect by {score_type.lower()}: {incorrect_responses}")
    print("\nAfter applying classifier:")
    
    if incorrect_responses > 0:
        print(f"Hallucinations caught: {incorrect_caught} of {incorrect_responses} ({100*incorrect_caught/incorrect_responses:.2f}% of hallucinations)")
    else:
        print("Hallucinations caught: 0 (no hallucinations in dataset)")
        
    if correct_responses > 0:
        print(f"Correct responses incorrectly blocked: {correct_blocked} of {correct_responses} ({100*correct_blocked/correct_responses:.2f}% of correct responses)")
    else:
        print("Correct responses incorrectly blocked: 0 (no correct responses in dataset)")
    
    # Calculate net impact
    net_impact = incorrect_caught - correct_blocked
    net_impact_percent = abs(net_impact) / total * 100 if total > 0 else 0
    
    if net_impact > 0:
        print(f"\nNET IMPACT: +{net_impact} responses ({net_impact_percent:.2f}% increase in accuracy)")
        print(f"The classifier improves overall accuracy by catching {incorrect_caught} hallucinations")
        print(f"at the cost of incorrectly blocking {correct_blocked} correct responses.")
    else:
        print(f"\nNET IMPACT: {net_impact} responses ({net_impact_percent:.2f}% decrease in accuracy)")
        print(f"The classifier decreases overall accuracy by incorrectly blocking {correct_blocked} correct responses")
        print(f"while only catching {incorrect_caught} hallucinations.")
    
    # Analysis 2: Performance by category
    print("\n===== Performance by Category =====")
    categories = sorted(df['category'].unique())
    
    # Create a table header for the category results
    print(f"{'Category':<25} | {'Total':<6} | {'Block Rate':<11} | {f'{score_type} Correct':<14} | {'Agreement':<10}")
    print(f"{'-'*25} | {'-'*6} | {'-'*11} | {'-'*14} | {'-'*10}")
    
    for category in categories:
        category_df = df[df['category'] == category]
        total_cat = len(category_df)
        
        # Blocks in this category
        blocks = category_df['blocked'].sum()
        block_rate = blocks / total_cat if total_cat > 0 else 0
        
        # Correctness in this category
        cat_correct = category_df['is_correct'].sum()
        correct_rate = cat_correct / total_cat if total_cat > 0 else 0
        
        # Agreement rate
        agreement = ((category_df['blocked'] == True) & (category_df['is_correct'] == False)) | \
                   ((category_df['blocked'] == False) & (category_df['is_correct'] == True))
        agreement_rate = agreement.sum() / total_cat if total_cat > 0 else 0
        
        # Print in table format
        print(f"{category:<25} | {total_cat:<6} | {block_rate:.4f} ({blocks:2d}/{total_cat:<2d}) | {correct_rate:.4f} ({cat_correct:2d}/{total_cat:<2d}) | {agreement_rate:.4f}")
    
    # Analysis 3: Effectiveness of similarity scores
    print("\n===== Similarity Score Analysis =====")
    
    # Check if similarity_score column exists and contains valid values
    has_similarity_scores = False
    try:
        if 'similarity_score' in df.columns:
            # Convert to numeric, coercing errors to NaN
            df['similarity_score'] = pd.to_numeric(df['similarity_score'], errors='coerce')
            
            # Filter to records with valid similarity scores
            sim_df = df[df['similarity_score'].notna()]
            
            if len(sim_df) > 0:
                has_similarity_scores = True
                
                # Calculate average similarity scores
                avg_sim_correct = sim_df[sim_df['is_correct'] == True]['similarity_score'].mean()
                avg_sim_incorrect = sim_df[sim_df['is_correct'] == False]['similarity_score'].mean()
                
                # Only calculate correlation if we have enough valid data points
                if len(sim_df) >= 2:
                    try:
                        correlation = np.corrcoef(sim_df['similarity_score'], sim_df['is_correct'])[0, 1]
                        if not np.isnan(correlation):
                            print(f"Correlation between similarity score and {score_type.lower()} correctness: {correlation:.4f}")
                        else:
                            print("Correlation could not be calculated (insufficient variance in data)")
                    except Exception as e:
                        print(f"Error calculating correlation: {e}")
                else:
                    print("Not enough data points with similarity scores to calculate correlation")
                
                # Print average similarity scores if valid
                if not np.isnan(avg_sim_correct):
                    print(f"Average similarity score when {score_type.lower()} correct: {avg_sim_correct:.4f}")
                if not np.isnan(avg_sim_incorrect):
                    print(f"Average similarity score when {score_type.lower()} incorrect: {avg_sim_incorrect:.4f}")
            else:
                print("No valid similarity scores found in the data")
        else:
            print("Similarity score column not found in the data")
            
    except Exception as e:
        print(f"Error analyzing similarity scores: {e}")
    
    # Analysis 4: Generate blocking statistics
    print("\n===== Blocking Statistics =====")
    
    # How often does the model block responses?
    blocked_count = df['blocked'].sum()
    total_blocked = true_positives + false_positives
    
    if blocked_count != total_blocked:
        print(f"WARNING: Sum of blocked responses in data ({blocked_count}) doesn't match confusion matrix ({total_blocked})")
        print(f"This may indicate inconsistencies in the data or different counting methods.")
    
    blocked_rate = blocked_count / total if total > 0 else 0
    
    print(f"Total blocked responses: {blocked_count} out of {total} ({blocked_rate:.4f})")
    print(f"  - Correctly blocked hallucinations: {true_positives}")
    print(f"  - Incorrectly blocked correct responses: {false_positives}")
    
    # Analyze reasons for blocking if available
    if 'reason' in df.columns:
        # Get unique non-NaN reasons
        reasons = df[df['reason'].notna()]['reason'].unique()
        if len(reasons) > 0:
            print("\nBlocking reasons:")
            for reason in reasons:
                reason_count = df[df['reason'] == reason].shape[0]
                print(f"  - {reason}: {reason_count} instances")
    
    # Analysis 5: Visualize probability scores vs. evaluation scores
    print(f"\n===== Probability Score Visualization =====")
    
    # Extract probability scores from reason field
    prob_scores = []
    evaluation_scores = []
    is_hallucination = []
    
    if 'reason' in df.columns:
        # Regular expression to extract probability score
        pattern = r"probability ([0-9.]+)"
        
        for idx, row in df.iterrows():
            if pd.notna(row['reason']):
                match = re.search(pattern, row['reason'])
                if match:
                    prob_score = float(match.group(1))
                    prob_scores.append(prob_score)
                    evaluation_scores.append(row[score_column])
                    is_hallucination.append(not row['is_correct'])
                    
        if prob_scores:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Scatter plot of probability scores vs. hallucination status
            hallucination_points = [score for score, hall in zip(prob_scores, is_hallucination) if hall]
            non_hallucination_points = [score for score, hall in zip(prob_scores, is_hallucination) if not hall]
            
            if hallucination_points:
                ax1.scatter([1] * len(hallucination_points), hallucination_points, 
                           color='red', alpha=0.6, label=f'Hallucination ({score_type}=0)')
            if non_hallucination_points:
                ax1.scatter([0] * len(non_hallucination_points), non_hallucination_points, 
                           color='green', alpha=0.6, label=f'Not Hallucination ({score_type}=1)')
                
            ax1.set_title('Probability Scores vs. Hallucination Status')
            ax1.set_xlabel(f'Is Hallucination ({score_type} Score)')
            ax1.set_ylabel('Classifier Probability Score')
            ax1.set_xlim(-0.5, 1.5)
            ax1.set_xticks([0, 1])
            ax1.set_xticklabels(['Not Hallucination', 'Hallucination'])
            ax1.set_ylim(0, 1.05)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            # Histogram of probability scores by hallucination status
            if hallucination_points:
                ax2.hist(hallucination_points, alpha=0.5, bins=10, color='red', 
                        label=f'Hallucination ({score_type}=0)')
            if non_hallucination_points:
                ax2.hist(non_hallucination_points, alpha=0.5, bins=10, color='green', 
                        label=f'Not Hallucination ({score_type}=1)')
                
            ax2.set_title('Distribution of Probability Scores')
            ax2.set_xlabel('Classifier Probability Score')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save the visualization
            viz_file = f"evaluation/results/probability_visualization_{score_source}.png"
            plt.savefig(viz_file)
            print(f"Visualization saved to {viz_file}")
            
            # Additional statistics
            true_hall_avg = np.mean(hallucination_points) if hallucination_points else float('nan')
            false_hall_avg = np.mean(non_hallucination_points) if non_hallucination_points else float('nan')
            
            print(f"\nAverage probability score for actual hallucinations: {true_hall_avg:.4f}")
            print(f"Average probability score for non-hallucinations: {false_hall_avg:.4f}")
            
            if not np.isnan(true_hall_avg) and not np.isnan(false_hall_avg):
                score_diff = true_hall_avg - false_hall_avg
                print(f"Score difference: {score_diff:.4f} ({'+' if score_diff > 0 else ''}{score_diff:.2%})")
            
            # Calculate ideal threshold using simple approach
            if hallucination_points and non_hallucination_points:
                all_scores = sorted(prob_scores)
                best_threshold = all_scores[0]
                best_accuracy = 0
                
                for threshold in all_scores:
                    pred_positives = sum(1 for score in prob_scores if score >= threshold)
                    pred_negatives = len(prob_scores) - pred_positives
                    
                    correct_positives = sum(1 for score, hall in zip(prob_scores, is_hallucination) 
                                         if score >= threshold and hall)
                    correct_negatives = sum(1 for score, hall in zip(prob_scores, is_hallucination) 
                                          if score < threshold and not hall)
                    
                    accuracy = (correct_positives + correct_negatives) / len(prob_scores)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold
                
                print(f"\nIdeal probability threshold based on this data: {best_threshold:.4f}")
                print(f"This would yield an estimated accuracy of: {best_accuracy:.4f}")
        else:
            print("No probability scores could be extracted from the reasons.")
    else:
        print("No 'reason' column found in the data to extract probability scores.")
    
    print("\n===== Analysis Complete =====")
    print(f"This analysis compares the classifier's predictions against {score_type.lower()} evaluations,")
    print(f"treating {score_type.lower()} judgments as the ground truth for hallucination detection.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier performance")
    parser.add_argument("--results-file", type=str, 
                      help="Path to the results CSV file")
    parser.add_argument("--score-source", type=str, choices=["human", "claude"], default="human",
                      help="Source of evaluation scores (human or claude)")
    
    args = parser.parse_args()
    evaluate_classifier(args.results_file, args.score_source)

if __name__ == "__main__":
    main()
