#!/usr/bin/env python3
"""
OPTIMIZE TRAINING SAMPLE SIZE EXPERIMENT

This script evaluates how classifier performance scales with training data size.

Process:
1. Load TruthfulQA dataset and split into train/test
2. Train classifiers with different training sample sizes (10%, 20%, 30%, etc.)
3. Evaluate all classifiers on the same test set
4. Create performance vs sample size curve

Configuration:
- Target layer (configurable)
- Aggregation method (configurable)
- Sample size percentages to test
- Model to use
"""

import pandas as pd
import numpy as np
import torch
import pickle
import os
import sys
import subprocess
import json
import tempfile
import argparse
from typing import List, Dict, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path so we can import wisent_guard modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wisent_guard.core.model import Model
from wisent_guard.core.layer import Layer
from wisent_guard.core.activations import Activations, ActivationAggregationMethod


def load_truthfulqa_data() -> pd.DataFrame:
    """Load TruthfulQA training data."""
    print("📊 Loading TruthfulQA training data...")
    
    # Try to load from multiple possible locations
    possible_paths = [
        "TruthfulQA.csv",
        "data/TruthfulQA.csv",
        "wisent_guard/data/TruthfulQA.csv",
        "../TruthfulQA.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"   ✅ Loaded from {path}: {len(df)} samples")
                break
            except Exception as e:
                print(f"   ⚠️ Could not load from {path}: {e}")
                continue
    
    if df is None:
        raise FileNotFoundError(f"Could not find TruthfulQA.csv in any of: {possible_paths}")
    
    # Ensure we have the right columns
    if 'Question' not in df.columns:
        raise ValueError("Expected 'Question' column not found in TruthfulQA data")
    
    print(f"   📊 Available columns: {list(df.columns)}")
    return df


def create_contrastive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Create contrastive pairs from TruthfulQA data."""
    print("🔄 Creating contrastive pairs...")
    
    pairs = []
    
    for _, row in df.iterrows():
        question = row['Question']
        
        # Get correct and incorrect answers
        correct_answers = []
        incorrect_answers = []
        
        # Check for different answer columns
        if 'Correct Answers' in row and pd.notna(row['Correct Answers']):
            correct_answers.extend(str(row['Correct Answers']).split(';'))
        if 'Best Answer' in row and pd.notna(row['Best Answer']):
            correct_answers.append(str(row['Best Answer']))
            
        if 'Incorrect Answers' in row and pd.notna(row['Incorrect Answers']):
            incorrect_answers.extend(str(row['Incorrect Answers']).split(';'))
        
        # Clean up answers
        correct_answers = [ans.strip() for ans in correct_answers if ans.strip()]
        incorrect_answers = [ans.strip() for ans in incorrect_answers if ans.strip()]
        
        # Create pairs
        for correct in correct_answers:
            pairs.append({
                'question': question,
                'response': correct,
                'label': 1  # Truthful
            })
            
        for incorrect in incorrect_answers:
            pairs.append({
                'question': question,
                'response': incorrect,
                'label': 0  # Hallucination
            })
    
    pairs_df = pd.DataFrame(pairs)
    print(f"   ✅ Created {len(pairs_df)} contrastive pairs")
    print(f"   📊 Truthful: {sum(pairs_df['label'] == 1)}")
    print(f"   📊 Hallucinations: {sum(pairs_df['label'] == 0)}")
    
    return pairs_df


def create_training_data_file(pairs_df: pd.DataFrame, sample_size: float, output_path: str) -> str:
    """Create a training data file with the specified sample size."""
    print(f"📝 Creating training data file with {sample_size*100:.1f}% of data...")
    
    # Sample the data
    sampled_df = pairs_df.sample(frac=sample_size, random_state=42).reset_index(drop=True)
    
    # Create the training data in the format expected by the CLI
    training_data = []
    for _, row in sampled_df.iterrows():
        training_data.append({
            'question': row['question'],
            'response': row['response'],
            'label': int(row['label'])
        })
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"   ✅ Saved {len(training_data)} samples to {output_path}")
    return output_path


def train_classifier_with_sample_size(layer: int, sample_size: float, training_data_path: str, 
                                     model_name: str) -> str:
    """Train a classifier with specific sample size."""
    print(f"🎓 Training classifier for layer {layer} with {sample_size*100:.1f}% sample size...")
    
    classifier_path = f"classifier_layer_{layer}_sample_{int(sample_size*100)}.pkl"
    
    # Use the CLI to train the classifier
    cmd = [
        "python", "-m", "wisent_guard.cli", "train",
        "--model", model_name,
        "--layer", str(layer),
        "--training-data", training_data_path,
        "--save-classifier", classifier_path,
        "--no-test"  # Don't test during training
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode != 0:
            print(f"   ❌ Training failed for layer {layer}, sample size {sample_size}")
            print(f"   STDERR: {result.stderr}")
            return None
        
        if os.path.exists(classifier_path):
            print(f"   ✅ Classifier saved to: {classifier_path}")
            return classifier_path
        else:
            print(f"   ❌ Classifier file not found: {classifier_path}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Training timed out for layer {layer}, sample size {sample_size}")
        return None
    except Exception as e:
        print(f"   ❌ Error training: {e}")
        return None


def extract_activation_scores(model: Model, text: str, layer: int, classifier: Any) -> List[float]:
    """Extract token-level activation scores from text using the trained classifier."""
    try:
        # Tokenize the text
        inputs = model.tokenizer(text, return_tensors="pt")
        tokens = model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Get model device and process inputs
        model_device = next(model.hf_model.parameters()).device
        inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Get hidden states
        with torch.no_grad():
            outputs = model.hf_model(**inputs_on_device, output_hidden_states=True)
        
        # Extract from specified layer
        if layer + 1 < len(outputs.hidden_states):
            hidden_states = outputs.hidden_states[layer + 1]
        else:
            hidden_states = outputs.hidden_states[-1]
        
        # Score each token
        token_scores = []
        layer_obj = Layer(index=layer, type="transformer")
        
        for token_idx in range(len(tokens)):
            if token_idx < hidden_states.shape[1]:
                # Extract activation
                token_activation = hidden_states[0, token_idx, :].cpu()
                
                # Create Activations object
                activation_obj = Activations(
                    tensor=token_activation.unsqueeze(0),
                    layer=layer_obj,
                    aggregation_method=ActivationAggregationMethod.LAST_TOKEN
                )
                
                # Get features
                features = activation_obj.extract_features_for_classifier()
                
                # Use classifier
                if hasattr(classifier, 'predict_proba'):
                    prob = classifier.predict_proba([features.numpy()])
                    if isinstance(prob, (list, tuple, np.ndarray)) and len(prob) > 0:
                        if hasattr(prob[0], '__len__') and len(prob[0]) > 1:
                            prob = float(prob[0][1])  # Binary classification - positive class
                        else:
                            prob = float(prob[0])
                    else:
                        prob = float(prob)
                    token_scores.append(prob)
                else:
                    token_scores.append(0.5)  # Neutral if no classifier
        
        return token_scores
        
    except Exception as e:
        print(f"      ⚠️ Error extracting scores: {e}")
        return []


def aggregate_scores(token_scores: List[float], method: str) -> float:
    """Aggregate token scores using specified method."""
    if not token_scores:
        return 0.5
    
    if method == "average":
        return np.mean(token_scores)
    elif method == "max":
        return np.max(token_scores)
    elif method == "min":
        return np.min(token_scores)
    elif method == "first":
        return token_scores[0]
    elif method == "last":
        return token_scores[-1]
    elif method == "median":
        return np.median(token_scores)
    else:
        return np.mean(token_scores)  # Default to average


def evaluate_classifier(model: Model, test_df: pd.DataFrame, classifier_path: str, 
                       layer: int, aggregation_method: str, threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate classifier on test set."""
    print(f"🧪 Evaluating classifier: {classifier_path}")
    
    # Load the saved classifier
    try:
        with open(classifier_path, "rb") as f:
            steering_method = pickle.load(f)
        print(f"   ✅ Loaded classifier from {classifier_path}")
    except Exception as e:
        print(f"   ❌ Error loading classifier: {e}")
        return {}
    
    # Handle classifier object
    if isinstance(steering_method, dict) and 'classifier' in steering_method:
        classifier = steering_method['classifier']
    elif hasattr(steering_method, 'classifier'):
        classifier = steering_method.classifier
    else:
        print(f"   ❌ No classifier found in steering method")
        return {}
    
    # Collect predictions
    aggregated_scores = []
    true_labels = []
    
    for idx, row in test_df.iterrows():
        # Create full response text
        if 'question' in row and 'response' in row:
            text = f"{row['question']} {row['response']}"
        else:
            text = row['response']
        true_label = row['label']
        
        # Extract activation scores
        token_scores = extract_activation_scores(model, text, layer, classifier)
        
        if token_scores:
            # Aggregate scores
            agg_score = aggregate_scores(token_scores, aggregation_method)
            aggregated_scores.append(agg_score)
            true_labels.append(true_label)
    
    if not aggregated_scores:
        print(f"   ❌ No valid predictions")
        return {}
    
    # Make predictions based on threshold
    predictions = [1 if score <= threshold else 0 for score in aggregated_scores]  # Lower score = more truthful
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_samples': len(predictions),
        'mean_score': np.mean(aggregated_scores),
        'std_score': np.std(aggregated_scores)
    }
    
    print(f"   ✅ Accuracy: {accuracy:.3f}, F1: {f1:.3f}, N: {len(predictions)}")
    
    return metrics


def create_performance_curve(results: Dict[float, Dict[str, float]], output_path: str = "sample_size_performance_curve.png"):
    """Create and save performance vs sample size curve."""
    print("📊 Creating performance curve...")
    
    sample_sizes = sorted(results.keys())
    accuracies = [results[size]['accuracy'] for size in sample_sizes]
    f1_scores = [results[size]['f1'] for size in sample_sizes]
    precisions = [results[size]['precision'] for size in sample_sizes]
    recalls = [results[size]['recall'] for size in sample_sizes]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot multiple metrics
    plt.subplot(2, 2, 1)
    plt.plot([s*100 for s in sample_sizes], accuracies, 'o-', label='Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Training Sample Size (%)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Training Sample Size')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot([s*100 for s in sample_sizes], f1_scores, 'o-', label='F1 Score', color='orange', linewidth=2, markersize=6)
    plt.xlabel('Training Sample Size (%)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Training Sample Size')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot([s*100 for s in sample_sizes], precisions, 'o-', label='Precision', color='green', linewidth=2, markersize=6)
    plt.xlabel('Training Sample Size (%)')
    plt.ylabel('Precision')
    plt.title('Precision vs Training Sample Size')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot([s*100 for s in sample_sizes], recalls, 'o-', label='Recall', color='red', linewidth=2, markersize=6)
    plt.xlabel('Training Sample Size (%)')
    plt.ylabel('Recall')
    plt.title('Recall vs Training Sample Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Performance curve saved to: {output_path}")
    
    # Also create a combined plot
    plt.figure(figsize=(10, 6))
    plt.plot([s*100 for s in sample_sizes], accuracies, 'o-', label='Accuracy', linewidth=2, markersize=6)
    plt.plot([s*100 for s in sample_sizes], f1_scores, 's-', label='F1 Score', linewidth=2, markersize=6)
    plt.plot([s*100 for s in sample_sizes], precisions, '^-', label='Precision', linewidth=2, markersize=6)
    plt.plot([s*100 for s in sample_sizes], recalls, 'v-', label='Recall', linewidth=2, markersize=6)
    
    plt.xlabel('Training Sample Size (%)')
    plt.ylabel('Performance Score')
    plt.title('Classifier Performance vs Training Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    combined_path = output_path.replace('.png', '_combined.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Combined performance curve saved to: {combined_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize training sample size experiment - analyze classifier performance vs training data size"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name to use (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    
    parser.add_argument(
        "--layer", 
        type=int, 
        default=15,
        help="Target layer to analyze (default: 15)"
    )
    
    parser.add_argument(
        "--aggregation", 
        type=str, 
        default="average",
        choices=["average", "max", "min", "first", "last", "median"],
        help="Aggregation method for token scores (default: average)"
    )
    
    parser.add_argument(
        "--sample-sizes", 
        type=str, 
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated list of sample sizes as fractions (default: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)"
    )
    
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    
    parser.add_argument(
        "--output-prefix", 
        type=str, 
        default="",
        help="Prefix for output files (default: empty)"
    )
    
    return parser.parse_args()


def main():
    """Main experiment function."""
    args = parse_args()
    
    print("🧪 OPTIMIZE TRAINING SAMPLE SIZE EXPERIMENT")
    print("=" * 70)
    
    # Parse sample sizes
    try:
        sample_sizes = [float(x.strip()) for x in args.sample_sizes.split(',')]
        # Validate sample sizes
        for size in sample_sizes:
            if size <= 0 or size > 1:
                raise ValueError(f"Sample size {size} must be between 0 and 1")
    except Exception as e:
        print(f"❌ Error parsing sample sizes: {e}")
        print("   Sample sizes should be comma-separated values between 0 and 1, e.g., '0.1,0.2,0.5,1.0'")
        return
    
    # Configuration from arguments
    MODEL_NAME = args.model
    TARGET_LAYER = args.layer
    AGGREGATION_METHOD = args.aggregation
    SAMPLE_SIZES = sample_sizes
    TEST_SIZE = args.test_size
    OUTPUT_PREFIX = args.output_prefix
    
    print(f"🎯 Model: {MODEL_NAME}")
    print(f"🎯 Target Layer: {TARGET_LAYER}")
    print(f"🎯 Aggregation Method: {AGGREGATION_METHOD}")
    print(f"🎯 Sample Sizes: {[f'{s*100:.0f}%' for s in SAMPLE_SIZES]}")
    print(f"🎯 Test Size: {TEST_SIZE*100:.0f}%")
    if OUTPUT_PREFIX:
        print(f"🎯 Output Prefix: {OUTPUT_PREFIX}")
    print()
    
    try:
        # Step 1: Load TruthfulQA data
        truthfulqa_df = load_truthfulqa_data()
        
        # Step 2: Create contrastive pairs
        pairs_df = create_contrastive_pairs(truthfulqa_df)
        
        # Step 3: Split into train/test
        train_df, test_df = train_test_split(pairs_df, test_size=TEST_SIZE, random_state=42, stratify=pairs_df['label'])
        print(f"📊 Train samples: {len(train_df)}")
        print(f"📊 Test samples: {len(test_df)}")
        
        # Step 4: Load model for evaluation
        print(f"🤖 Loading model: {MODEL_NAME}")
        model = Model(MODEL_NAME)
        print(f"   ✅ Model loaded on device: {model.device}")
        
        # Step 5: Train classifiers with different sample sizes
        results = {}
        
        for sample_size in SAMPLE_SIZES:
            print(f"\n🔄 Processing sample size: {sample_size*100:.1f}%")
            print("-" * 50)
            
            # Create training data file
            training_data_path = f"training_data_sample_{int(sample_size*100)}.json"
            create_training_data_file(train_df, sample_size, training_data_path)
            
            # Train classifier
            classifier_path = train_classifier_with_sample_size(
                TARGET_LAYER, sample_size, training_data_path, MODEL_NAME
            )
            
            if classifier_path and os.path.exists(classifier_path):
                # Evaluate classifier
                metrics = evaluate_classifier(
                    model, test_df, classifier_path, TARGET_LAYER, AGGREGATION_METHOD
                )
                
                if metrics:
                    results[sample_size] = metrics
                    print(f"   ✅ Sample size {sample_size*100:.1f}%: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
                else:
                    print(f"   ❌ Evaluation failed for sample size {sample_size*100:.1f}%")
            else:
                print(f"   ❌ Training failed for sample size {sample_size*100:.1f}%")
            
            # Clean up training data file
            if os.path.exists(training_data_path):
                os.remove(training_data_path)
        
        # Step 6: Create performance curve
        if results:
            # Create output filenames with parameters
            base_name = f"{OUTPUT_PREFIX}sample_size_layer_{TARGET_LAYER}_{AGGREGATION_METHOD}"
            if OUTPUT_PREFIX and not OUTPUT_PREFIX.endswith('_'):
                base_name = f"{OUTPUT_PREFIX}_sample_size_layer_{TARGET_LAYER}_{AGGREGATION_METHOD}"
            
            curve_file = f"{base_name}_curve.png"
            create_performance_curve(results, curve_file)
            
            # Save results
            results_file = f"{base_name}_results.pkl"
            with open(results_file, "wb") as f:
                pickle.dump({
                    'results': results,
                    'config': {
                        'model': MODEL_NAME,
                        'layer': TARGET_LAYER,
                        'aggregation': AGGREGATION_METHOD,
                        'sample_sizes': SAMPLE_SIZES,
                        'test_size': TEST_SIZE
                    }
                }, f)
            print(f"\n💾 Results saved to: {results_file}")
            
            # Print summary
            print("\n📊 SUMMARY")
            print("=" * 50)
            for sample_size in sorted(results.keys()):
                metrics = results[sample_size]
                print(f"Sample {sample_size*100:3.0f}%: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}")
        
        print("\n🎉 Experiment completed!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
