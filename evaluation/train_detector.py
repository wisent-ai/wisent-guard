#!/usr/bin/env python
"""
Train a classifier to detect hallucinations from activation patterns.

This script:
1. Loads evaluation results from guard_results.csv
2. Extracts activation vectors and Claude scores (ground truth)
3. Trains both logistic regression and MLP classifiers
4. Evaluates performance with cross-validation
5. Reports effectiveness metrics and saves the model
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, auc, confusion_matrix,
    classification_report
)
import joblib
import argparse

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

def extract_features_and_labels(df, max_tokens=10):
    """
    Extract feature vectors and labels from the dataset.
    
    Args:
        df: DataFrame with evaluation results
        max_tokens: Maximum number of tokens to consider per example
        
    Returns:
        X_all: List of feature vectors for all tokens
        y_all: List of labels (1=hallucination, 0=correct) for all tokens
        token_to_example: Mapping from token index to example index
    """
    X_all = []
    y_all = []
    token_to_example = []
    example_idx = 0
    
    print("Extracting features and labels...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tokens = parse_token_scores(row.get('token_scores', ''))
        
        # Skip examples with no token scores
        if not tokens:
            continue
        
        # Get label from Claude score (0=hallucination, 1=correct)
        # We invert it for our classifier (1=hallucination, 0=correct)
        is_hallucination = 1 if row.get('claude_score', -1) == 0 else 0
        
        # Extract features from tokens (limited to max_tokens)
        for i, token in enumerate(tokens[:max_tokens]):
            # Feature vector: [position, similarity]
            # We could expand this with more features like token_id or category encoding
            feature_vector = [
                token.get('position', 0) / 100.0,  # Normalize position
                token.get('similarity', 0.0)      # Similarity score
            ]
            
            X_all.append(feature_vector)
            y_all.append(is_hallucination)
            token_to_example.append(example_idx)
        
        example_idx += 1
    
    return np.array(X_all), np.array(y_all), token_to_example

def train_and_evaluate_classifiers(X, y, token_to_example, output_dir):
    """
    Train and evaluate classifiers on the dataset.
    
    Args:
        X: Feature vectors
        y: Labels
        token_to_example: Mapping from token index to example index
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data into train and test sets (stratified by label)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define classifiers
    classifiers = {
        'logistic_regression': LogisticRegression(
            C=0.001,  # Strong regularization
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )
    }
    
    results = {}
    
    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        print(f"\nTraining {name} classifier...")
        
        # Train classifier
        clf.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = clf.predict(X_test)
        
        # For probabilities, we need the positive class probability (class 1)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = clf.decision_function(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve and AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"\n===== {name.upper()} CLASSIFIER RESULTS =====")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Example-level evaluation (maximum probability per example)
        example_max_prob = {}
        for i, (prob, ex_idx) in enumerate(zip(y_prob, [token_to_example[i] for i in range(len(X_test))])):
            example_max_prob[ex_idx] = max(prob, example_max_prob.get(ex_idx, -1))
        
        example_truth = {}
        for i, ex_idx in enumerate([token_to_example[i] for i in range(len(X_test))]):
            example_truth[ex_idx] = y_test[i]
        
        # Calculate performance at different thresholds for example-level decisions
        thresholds = np.linspace(0.01, 0.99, 20)
        threshold_results = []
        
        for threshold in thresholds:
            tp, fp, tn, fn = 0, 0, 0, 0
            
            for ex_idx, prob in example_max_prob.items():
                truth = example_truth[ex_idx]
                pred = 1 if prob >= threshold else 0
                
                if truth == 1 and pred == 1:
                    tp += 1
                elif truth == 0 and pred == 1:
                    fp += 1
                elif truth == 0 and pred == 0:
                    tn += 1
                elif truth == 1 and pred == 0:
                    fn += 1
            
            # Calculate metrics
            try:
                ex_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                ex_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                ex_f1 = 2 * (ex_precision * ex_recall) / (ex_precision + ex_recall) if (ex_precision + ex_recall) > 0 else 0
                ex_accuracy = (tp + tn) / (tp + tn + fp + fn)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            except ZeroDivisionError:
                ex_precision, ex_recall, ex_f1, ex_accuracy, fpr = 0, 0, 0, 0, 0
            
            threshold_results.append({
                'threshold': threshold,
                'precision': ex_precision,
                'recall': ex_recall,
                'f1': ex_f1,
                'accuracy': ex_accuracy,
                'fpr': fpr
            })
        
        # Find optimal threshold based on F1 score
        optimal_threshold = max(threshold_results, key=lambda x: x['f1'])
        print(f"\nOptimal threshold: {optimal_threshold['threshold']:.4f}")
        print(f"At optimal threshold - Precision: {optimal_threshold['precision']:.4f}, Recall: {optimal_threshold['recall']:.4f}, F1: {optimal_threshold['f1']:.4f}")
        print(f"False Positive Rate: {optimal_threshold['fpr']:.4f}")
        
        # Plot precision-recall curve for different thresholds
        plt.figure(figsize=(10, 6))
        plt.plot([x['threshold'] for x in threshold_results], 
                [x['precision'] for x in threshold_results], 'b-', label='Precision')
        plt.plot([x['threshold'] for x in threshold_results], 
                [x['recall'] for x in threshold_results], 'r-', label='Recall')
        plt.plot([x['threshold'] for x in threshold_results], 
                [x['f1'] for x in threshold_results], 'g-', label='F1 Score')
        plt.plot([x['threshold'] for x in threshold_results], 
                [x['fpr'] for x in threshold_results], 'y-', label='False Positive Rate')
        plt.axvline(x=optimal_threshold['threshold'], color='k', linestyle='--', 
                   label=f'Optimal Threshold ({optimal_threshold["threshold"]:.2f})')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Performance Metrics vs Threshold ({name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{name}_threshold_performance.png'))
        
        # Save cross-validation scores
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {
            'accuracy': cross_val_score(clf, X, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(clf, X, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(clf, X, y, cv=cv, scoring='recall'),
            'f1': cross_val_score(clf, X, y, cv=cv, scoring='f1')
        }
        
        print("\nCross-Validation Results:")
        for metric, scores in cv_scores.items():
            print(f"{metric.capitalize()}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Save model and results
        joblib.dump(clf, os.path.join(output_dir, f'{name}_classifier.joblib'))
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm.tolist(),
            'optimal_threshold': optimal_threshold['threshold'],
            'optimal_precision': optimal_threshold['precision'],
            'optimal_recall': optimal_threshold['recall'],
            'optimal_f1': optimal_threshold['f1'],
            'optimal_fpr': optimal_threshold['fpr'],
            'cv_scores': {k: v.tolist() for k, v in cv_scores.items()}
        }
    
    # Save overall results
    with open(os.path.join(output_dir, 'classifier_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return classifiers, results

def main():
    parser = argparse.ArgumentParser(description="Train a classifier to detect hallucinations from activation patterns")
    
    parser.add_argument("--input-file", type=str, default="evaluation/results/guard_results.csv",
                       help="Path to CSV file with evaluation results")
    parser.add_argument("--output-dir", type=str, default="evaluation/results/hallucination_detector",
                       help="Directory to save classifier and results")
    parser.add_argument("--max-tokens", type=int, default=10,
                       help="Maximum number of tokens to consider per example")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    # Load data
    try:
        df = pd.read_csv(args.input_file)
        print(f"Loaded {len(df)} examples from {args.input_file}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    # Extract features and labels
    X, y, token_to_example = extract_features_and_labels(df, args.max_tokens)
    print(f"Extracted {len(X)} feature vectors from {len(set(token_to_example))} examples")
    print(f"Label distribution: {np.sum(y)} hallucinations, {len(y) - np.sum(y)} correct responses")
    
    # Train and evaluate classifiers
    classifiers, results = train_and_evaluate_classifiers(X, y, token_to_example, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
