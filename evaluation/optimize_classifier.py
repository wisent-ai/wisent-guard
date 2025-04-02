#!/usr/bin/env python3
"""
Optimize the classifier for hallucination detection.
This script tunes the classifier parameters and token aggregation strategies
to find the optimal configuration for hallucination detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
import argparse
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def extract_token_scores_from_csv(results_file):
    """
    Extract token scores and evaluation results from the CSV file.
    
    Args:
        results_file: Path to the CSV file with evaluation results
        
    Returns:
        DataFrame with extracted token scores and ground truth labels
    """
    print(f"Loading data from {results_file}...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(results_file)
        
        # Check if we have the necessary columns
        if 'human_score' not in df.columns or 'token_scores' not in df.columns:
            print("Error: Required columns not found in the data")
            return None
        
        # Extract ground truth (human/claude scores)
        # Convert to boolean: 1 = correct (not harmful), 0 = incorrect (harmful)
        df['is_correct'] = df['human_score'] == 1
        df['is_hallucination'] = df['human_score'] == 0
        
        # Filter out ambiguous scores (-1)
        df = df[df['human_score'].isin([0, 1])]
        
        # Extract token scores
        print("Extracting and parsing token scores...")
        token_data = []
        
        for idx, row in df.iterrows():
            question = row['question']
            response = row['guard_response'] if 'guard_response' in df.columns else row['response']
            category = row['category'] if 'category' in df.columns else 'unknown'
            human_score = row['human_score']
            is_hallucination = row['is_hallucination']
            
            # Parse token scores if available
            if pd.notna(row['token_scores']) and row['token_scores']:
                tokens = []
                token_items = row['token_scores'].split('|')
                
                for token_item in token_items:
                    try:
                        parts = token_item.split(':')
                        if len(parts) >= 6:
                            position = int(parts[0])
                            token_id = int(parts[1])
                            token_text = parts[2]
                            similarity = float(parts[3])
                            token_category = parts[4]
                            is_harmful = parts[5].lower() == 'true'
                            
                            tokens.append({
                                'position': position,
                                'token_id': token_id,
                                'token_text': token_text,
                                'similarity': similarity,
                                'category': token_category,
                                'is_harmful': is_harmful
                            })
                    except Exception as e:
                        # Skip problematic token
                        continue
                
                if tokens:
                    # Sort tokens by position (just to be safe)
                    tokens = sorted(tokens, key=lambda x: x['position'])
                    
                    # Add to data
                    token_data.append({
                        'question': question,
                        'response': response,
                        'category': category,
                        'human_score': human_score,
                        'is_hallucination': is_hallucination,
                        'tokens': tokens
                    })
        
        print(f"Extracted token data for {len(token_data)} responses")
        if len(token_data) == 0:
            print("Error: No valid token scores found in the data")
            return None
            
        return token_data
        
    except Exception as e:
        print(f"Error loading or processing results file: {e}")
        return None

def aggregate_token_scores(token_data, strategy='max'):
    """
    Aggregate token scores using different strategies.
    
    Args:
        token_data: List of dictionaries with token data
        strategy: Aggregation strategy (max, mean, first_token, weighted_pos)
        
    Returns:
        List of dictionaries with aggregated scores
    """
    print(f"Aggregating token scores using '{strategy}' strategy...")
    
    aggregated_data = []
    
    for item in token_data:
        tokens = item['tokens']
        
        # Skip if no tokens
        if not tokens:
            continue
        
        # Get similarity scores
        similarity_scores = [token['similarity'] for token in tokens]
        
        # Aggregate based on strategy
        if strategy == 'max':
            # Use maximum similarity score
            aggregated_score = max(similarity_scores)
        elif strategy == 'mean':
            # Use mean of similarity scores
            aggregated_score = sum(similarity_scores) / len(similarity_scores)
        elif strategy == 'first_token':
            # Use first token's similarity score
            aggregated_score = similarity_scores[0]
        elif strategy == 'weighted_pos':
            # Weight by position (earlier tokens get higher weight)
            weights = [1.0 / (i + 1) for i in range(len(similarity_scores))]
            weight_sum = sum(weights)
            normalized_weights = [w / weight_sum for w in weights]
            aggregated_score = sum(s * w for s, w in zip(similarity_scores, normalized_weights))
        else:
            # Default to max if unknown strategy
            print(f"Warning: Unknown strategy '{strategy}', defaulting to max")
            aggregated_score = max(similarity_scores)
        
        # Add to aggregated data
        aggregated_data.append({
            'question': item['question'],
            'response': item['response'],
            'category': item['category'],
            'human_score': item['human_score'],
            'is_hallucination': item['is_hallucination'],
            'token_count': len(tokens),
            'aggregated_score': aggregated_score
        })
    
    print(f"Aggregated {len(aggregated_data)} items")
    return aggregated_data

def train_evaluate_classifier(train_data, test_data, classifier_type='logistic', params=None):
    """
    Train and evaluate a classifier on the aggregated token scores.
    
    Args:
        train_data: Training data with aggregated scores
        test_data: Test data with aggregated scores
        classifier_type: Type of classifier ('logistic' or 'mlp')
        params: Classifier parameters
        
    Returns:
        Dictionary with trained classifier and evaluation metrics
    """
    print(f"Training and evaluating {classifier_type} classifier...")
    
    # Prepare features and labels
    X_train = np.array([item['aggregated_score'] for item in train_data]).reshape(-1, 1)
    y_train = np.array([item['is_hallucination'] for item in train_data])
    
    X_test = np.array([item['aggregated_score'] for item in test_data]).reshape(-1, 1)
    y_test = np.array([item['is_hallucination'] for item in test_data])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize classifier with default or provided parameters
    if classifier_type == 'logistic':
        clf = LogisticRegression(
            C=params.get('C', 1.0),
            class_weight=params.get('class_weight', None),
            random_state=42,
            max_iter=1000
        )
    elif classifier_type == 'mlp':
        clf = MLPClassifier(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (10,)),
            alpha=params.get('alpha', 0.0001),
            learning_rate_init=params.get('learning_rate_init', 0.001),
            random_state=42,
            max_iter=1000
        )
    else:
        print(f"Warning: Unknown classifier type '{classifier_type}', defaulting to logistic")
        clf = LogisticRegression(random_state=42, max_iter=1000)
    
    # Train classifier
    clf.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None
    
    # Calculate threshold-dependent metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate threshold-independent metrics (if probabilities available)
    if y_pred_proba is not None:
        # Precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (maximize F1)
        f1_scores = []
        for t in pr_thresholds:
            y_pred_t = (y_pred_proba >= t).astype(int)
            f1 = f1_score(y_test, y_pred_t)
            f1_scores.append(f1)
        
        best_idx = np.argmax(f1_scores)
        optimal_threshold = pr_thresholds[best_idx]
        optimal_f1 = f1_scores[best_idx]
        
        # Re-evaluate with optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        optimal_accuracy = accuracy_score(y_test, y_pred_optimal)
    else:
        pr_auc = None
        roc_auc = None
        optimal_threshold = None
        optimal_f1 = None
        optimal_accuracy = None
    
    # Calculate confusion matrix
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))
    
    # Calculate metrics
    precision_val = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall_val = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    
    # Calculate balanced accuracy
    sensitivity = recall_val
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Return results
    return {
        'classifier': clf,
        'scaler': scaler,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision_val,
        'recall': recall_val,
        'f1': f1_val,
        'specificity': specificity,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives),
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'optimal_accuracy': optimal_accuracy,
        'pr_curve': {
            'precision': precision.tolist() if y_pred_proba is not None else None,
            'recall': recall.tolist() if y_pred_proba is not None else None,
            'thresholds': pr_thresholds.tolist() if y_pred_proba is not None else None
        },
        'roc_curve': {
            'fpr': fpr.tolist() if y_pred_proba is not None else None,
            'tpr': tpr.tolist() if y_pred_proba is not None else None,
            'thresholds': roc_thresholds.tolist() if y_pred_proba is not None else None
        }
    }

def optimize_classifier_parameters(train_data, test_data, classifier_type='logistic'):
    """
    Optimize classifier parameters using grid search.
    
    Args:
        train_data: Training data with aggregated scores
        test_data: Test data with aggregated scores
        classifier_type: Type of classifier ('logistic' or 'mlp')
        
    Returns:
        Dictionary with optimal parameters and performance metrics
    """
    print(f"Optimizing {classifier_type} classifier parameters...")
    
    # Prepare features and labels
    X_train = np.array([item['aggregated_score'] for item in train_data]).reshape(-1, 1)
    y_train = np.array([item['is_hallucination'] for item in train_data])
    
    X_test = np.array([item['aggregated_score'] for item in test_data]).reshape(-1, 1)
    y_test = np.array([item['is_hallucination'] for item in test_data])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid based on classifier type
    if classifier_type == 'logistic':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced']
        }
        clf = LogisticRegression(random_state=42, max_iter=1000)
    elif classifier_type == 'mlp':
        param_grid = {
            'hidden_layer_sizes': [(10,), (20,), (50,), (10, 10)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
        clf = MLPClassifier(random_state=42, max_iter=1000)
    else:
        print(f"Warning: Unknown classifier type '{classifier_type}', defaulting to logistic")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced']
        }
        clf = LogisticRegression(random_state=42, max_iter=1000)
    
    # Perform grid search
    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring='f1', return_train_score=True
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Train and evaluate classifier with best parameters
    best_result = train_evaluate_classifier(train_data, test_data, classifier_type, best_params)
    best_result['best_params'] = best_params
    
    # Add grid search results
    cv_results = {}
    for param_name in param_grid:
        param_values = []
        mean_test_scores = []
        
        for i, params in enumerate(grid_search.cv_results_['params']):
            if param_name in params:
                param_values.append(str(params[param_name]))
                mean_test_scores.append(grid_search.cv_results_['mean_test_score'][i])
        
        cv_results[param_name] = {
            'param_values': param_values,
            'mean_test_scores': mean_test_scores
        }
    
    best_result['cv_results'] = cv_results
    
    return best_result

def evaluate_all_strategies(token_data, classifier_type='logistic', params=None, output_dir=None):
    """
    Evaluate all aggregation strategies and find the best one.
    
    Args:
        token_data: List of dictionaries with token data
        classifier_type: Type of classifier ('logistic' or 'mlp')
        params: Classifier parameters for each strategy
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results for all strategies
    """
    print("Evaluating all token aggregation strategies...")
    
    # Define strategies to evaluate
    strategies = ['max', 'mean', 'first_token', 'weighted_pos']
    
    # Initialize results
    results = {}
    best_strategy = None
    best_f1 = -1
    
    # Process each strategy
    for strategy in strategies:
        print(f"\n----- Evaluating '{strategy}' aggregation strategy -----")
        
        # Aggregate token scores
        aggregated_data = aggregate_token_scores(token_data, strategy)
        
        if not aggregated_data:
            print(f"Error: Failed to aggregate token scores for '{strategy}' strategy")
            continue
        
        # Split data into train and test sets
        train_data, test_data = train_test_split(
            aggregated_data, test_size=0.3, random_state=42,
            stratify=[item['is_hallucination'] for item in aggregated_data]
        )
        
        # Determine parameters to use
        strategy_params = params.get(strategy, {}) if params else {}
        
        if not strategy_params:
            # Optimize parameters if not provided
            strategy_result = optimize_classifier_parameters(
                train_data, test_data, classifier_type
            )
        else:
            # Evaluate with provided parameters
            strategy_result = train_evaluate_classifier(
                train_data, test_data, classifier_type, strategy_params
            )
        
        # Add to results
        results[strategy] = strategy_result
        
        # Check if this is the best strategy so far
        if strategy_result['f1'] > best_f1:
            best_f1 = strategy_result['f1']
            best_strategy = strategy
        
        # Print summary stats
        print(f"\nStrategy: {strategy}")
        print(f"F1 Score: {strategy_result['f1']:.4f}")
        print(f"Accuracy: {strategy_result['accuracy']:.4f}")
        print(f"Balanced Accuracy: {strategy_result['balanced_accuracy']:.4f}")
        print(f"Precision: {strategy_result['precision']:.4f}")
        print(f"Recall: {strategy_result['recall']:.4f}")
        
        # Save strategy-specific classifier if output_dir provided
        if output_dir:
            save_strategy_results(strategy, strategy_result, output_dir)
    
    # Add best strategy to results
    if best_strategy:
        results['best_strategy'] = best_strategy
        print(f"\nBest strategy: {best_strategy} (F1: {best_f1:.4f})")
    
    return results

def save_strategy_results(strategy, results, output_dir):
    """Save strategy results to files"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save classifier and scaler
    model_dir = os.path.join(output_dir, f"{strategy}_model")
    os.makedirs(model_dir, exist_ok=True)
    
    if 'classifier' in results:
        with open(os.path.join(model_dir, "classifier.pkl"), 'wb') as f:
            pickle.dump(results['classifier'], f)
    
    if 'scaler' in results:
        with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(results['scaler'], f)
    
    # Save configuration
    config = {
        'strategy': strategy,
        'threshold': results.get('optimal_threshold', 0.5),
        'performance': {
            'f1': results.get('f1'),
            'accuracy': results.get('accuracy'),
            'balanced_accuracy': results.get('balanced_accuracy'),
            'precision': results.get('precision'),
            'recall': results.get('recall'),
            'pr_auc': results.get('pr_auc'),
            'roc_auc': results.get('roc_auc')
        }
    }
    
    with open(os.path.join(model_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

def visualize_strategy_results(results, output_dir):
    """
    Visualize the results for all aggregation strategies.
    
    Args:
        results: Dictionary with results for all strategies
        output_dir: Directory to save visualization files
    """
    print("Generating strategy comparison visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get strategies (excluding 'best_strategy')
    strategies = [s for s in results.keys() if s != 'best_strategy']
    
    # 1. Bar chart of performance metrics
    metrics = ['f1', 'precision', 'recall', 'accuracy', 'balanced_accuracy']
    metric_values = {metric: [] for metric in metrics}
    
    for strategy in strategies:
        for metric in metrics:
            metric_values[metric].append(results[strategy].get(metric, 0))
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.15
    x = np.arange(len(strategies))
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width - (len(metrics)-1)*width/2, metric_values[metric], width, label=metric.capitalize())
    
    ax.set_title('Performance Metrics by Aggregation Strategy')
    ax.set_xlabel('Aggregation Strategy')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, strategy in enumerate(strategies):
        if strategy == results.get('best_strategy'):
            ax.annotate('BEST', (i, 0.05), ha='center', va='bottom', 
                     xytext=(0, 10), textcoords='offset points',
                     color='green', weight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_performance_metrics.png'))
    
    # 2. ROC curves for all strategies
    plt.figure(figsize=(10, 8))
    
    for strategy in strategies:
        if 'roc_curve' in results[strategy] and results[strategy]['roc_curve']['fpr']:
            fpr = results[strategy]['roc_curve']['fpr']
            tpr = results[strategy]['roc_curve']['tpr']
            roc_auc = results[strategy]['roc_auc']
            
            plt.plot(fpr, tpr, lw=2, label=f'{strategy} (AUC = {roc_auc:.4f})')
    
    # Add random baseline
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    plt.title('ROC Curves for Different Aggregation Strategies')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_roc_curves.png'))
    
    # 3. Precision-Recall curves for all strategies
    plt.figure(figsize=(10, 8))
    
    for strategy in strategies:
        if 'pr_curve' in results[strategy] and results[strategy]['pr_curve']['precision']:
            precision = results[strategy]['pr_curve']['precision']
            recall = results[strategy]['pr_curve']['recall']
            pr_auc = results[strategy]['pr_auc']
            
            plt.plot(recall, precision, lw=2, label=f'{strategy} (AUC = {pr_auc:.4f})')
    
    plt.title('Precision-Recall Curves for Different Aggregation Strategies')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_pr_curves.png'))
    
    # Close all plots to free memory
    plt.close('all')
    
    print(f"Strategy visualizations saved to {output_dir}")

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
    
    # Deep copy and convert results while skipping non-serializable objects
    json_results = {}
    
    for strategy, strategy_results in results.items():
        if strategy == 'best_strategy':
            json_results[strategy] = strategy_results  # This is just a string
            continue
            
        # For each strategy, extract serializable data
        strategy_data = {}
        for k, v in strategy_results.items():
            if k not in ['classifier', 'scaler']:  # Skip non-serializable objects
                strategy_data[k] = convert_to_json_serializable(v)
        
        json_results[strategy] = strategy_data
    
    # Save to file
    with open(output_file, 'w') as f:
        if pretty:
            json.dump(json_results, f, indent=2)
        else:
            json.dump(json_results, f)
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Optimize hallucination classifier parameters and token aggregation strategies")
    parser.add_argument("--input-file", type=str, 
                      default="evaluation/results/guard_results_classifier-based.csv",
                      help="Path to the evaluation results CSV file")
    parser.add_argument("--output-dir", type=str, 
                      default="evaluation/results/classifier_optimization",
                      help="Directory to save optimization results and visualizations")
    parser.add_argument("--classifier-type", type=str, 
                      choices=["logistic", "mlp"],
                      default="logistic",
                      help="Type of classifier to use")
    parser.add_argument("--optimize-params", action="store_true",
                      help="Optimize classifier parameters for each strategy")
    parser.add_argument("--single-strategy", type=str,
                      choices=["max", "mean", "first_token", "weighted_pos"],
                      help="Evaluate a single token aggregation strategy")
    
    args = parser.parse_args()
    
    # Extract token scores from CSV file
    token_data = extract_token_scores_from_csv(args.input_file)
    if token_data is None or len(token_data) == 0:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process strategies
    if args.single_strategy:
        # Process a single strategy
        print(f"\nProcessing single strategy: {args.single_strategy}")
        strategy = args.single_strategy
        
        # Aggregate token scores
        aggregated_data = aggregate_token_scores(token_data, strategy)
        
        if not aggregated_data:
            print(f"Error: Failed to aggregate token scores for '{strategy}' strategy")
            return
        
        # Split data into train and test sets
        train_data, test_data = train_test_split(
            aggregated_data, test_size=0.3, random_state=42,
            stratify=[item['is_hallucination'] for item in aggregated_data]
        )
        
        if args.optimize_params:
            # Optimize parameters
            strategy_result = optimize_classifier_parameters(
                train_data, test_data, args.classifier_type
            )
        else:
            # Evaluate with default parameters
            strategy_result = train_evaluate_classifier(
                train_data, test_data, args.classifier_type, {}
            )
        
        # Print summary stats
        print(f"\nStrategy: {strategy}")
        print(f"F1 Score: {strategy_result['f1']:.4f}")
        print(f"Accuracy: {strategy_result['accuracy']:.4f}")
        print(f"Balanced Accuracy: {strategy_result['balanced_accuracy']:.4f}")
        print(f"Precision: {strategy_result['precision']:.4f}")
        print(f"Recall: {strategy_result['recall']:.4f}")
        
        # Save strategy-specific results
        save_strategy_results(strategy, strategy_result, args.output_dir)
        
        # Save detailed results
        results_file = os.path.join(args.output_dir, f"{strategy}_results.json")
        
        # Convert strategy_result to JSON-serializable format
        json_result = {}
        for k, v in strategy_result.items():
            if k not in ['classifier', 'scaler']:  # Skip non-serializable objects
                try:
                    # Try to convert the object to a JSON-serializable form
                    if isinstance(v, np.ndarray):
                        json_result[k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        json_result[k] = float(v)
                    else:
                        json_result[k] = v
                except:
                    # Skip if can't be converted
                    pass
        
        with open(results_file, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
    else:
        # Process all strategies
        print("\nEvaluating all token aggregation strategies...")
        
        # Initialize parameters dictionary
        params = {}
        
        # Evaluate all strategies
        results = evaluate_all_strategies(
            token_data,
            classifier_type=args.classifier_type,
            params=params,
            output_dir=args.output_dir
        )
        
        # Visualize results
        visualize_strategy_results(results, args.output_dir)
        
        # Save combined results
        results_file = os.path.join(args.output_dir, "all_strategies_results.json")
        save_results(results, results_file)
        
        # Print best strategy
        if 'best_strategy' in results:
            best_strategy = results['best_strategy']
            strategy_result = results[best_strategy]
            
            print("\n===== Best Strategy =====")
            print(f"Strategy: {best_strategy}")
            print(f"F1 Score: {strategy_result['f1']:.4f}")
            print(f"Accuracy: {strategy_result['accuracy']:.4f}")
            print(f"Balanced Accuracy: {strategy_result['balanced_accuracy']:.4f}")
            print(f"Precision: {strategy_result['precision']:.4f}")
            print(f"Recall: {strategy_result['recall']:.4f}")
            
            # Save best configuration to a separate file
            best_config = {
                'strategy': best_strategy,
                'classifier_type': args.classifier_type,
                'threshold': strategy_result.get('optimal_threshold', 0.5),
                'parameters': strategy_result.get('best_params', {})
            }
            
            best_config_file = os.path.join(args.output_dir, "best_strategy_config.json")
            with open(best_config_file, 'w') as f:
                json.dump(best_config, f, indent=2)
            
            print(f"Best strategy configuration saved to {best_config_file}")
    
    print("\n===== Optimization Complete =====")

if __name__ == "__main__":
    main() 