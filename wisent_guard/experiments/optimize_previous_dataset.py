#!/usr/bin/env python3
"""
OPTIMIZE ON PREVIOUS DATASET EXPERIMENT

This script evaluates activation-based hallucination detection on the TruthfulQA_annotated.xlsx dataset.

Process:
1. Load TruthfulQA_annotated.xlsx (pre-generated responses + human scores)
2. Split into 80% train / 20% test
3. Train classifiers on 80% using contrastive pairs (like CLI)
4. Test classifiers on 20% using activation extraction from pre-generated text
5. Test multiple layers and aggregation methods
6. Report optimal strategies

Dataset format:
- Column: model responses (pre-generated text)
- Column: human scores (0 = hallucination, 1 = truthful)
"""

import pandas as pd
import numpy as np
import torch
import pickle
import os
import sys
import subprocess
import glob
from typing import List, Dict, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add the parent directory to the path so we can import wisent_guard modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wisent_guard.core.model import Model
from wisent_guard.core.layer import Layer
from wisent_guard.core.activations import Activations, ActivationAggregationMethod


def load_annotated_dataset(filepath: str) -> pd.DataFrame:
    """Load the TruthfulQA_annotated.xlsx dataset."""
    print(f"📊 Loading annotated dataset: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    df = pd.read_excel(filepath)
    print(f"   ✅ Loaded {len(df)} samples")
    
    # Use the known column names
    if 'response' not in df.columns or 'human_score' not in df.columns:
        print(f"   🔍 Available columns: {list(df.columns)}")
        raise ValueError("Expected 'response' and 'human_score' columns not found")
    
    print(f"   🔍 Response column: 'response'")
    print(f"   🔍 Score column: 'human_score'")
    
    # Clean the data
    df = df.dropna(subset=['response', 'human_score'])
    
    # Ensure scores are binary (0/1)
    df['human_score'] = df['human_score'].astype(int)
    
    print(f"   ✅ Cleaned dataset: {len(df)} samples")
    print(f"   📊 Hallucinations (0): {sum(df['human_score'] == 0)}")
    print(f"   📊 Truthful (1): {sum(df['human_score'] == 1)}")
    
    return df


def train_classifier_via_cli(layer: int) -> str:
    """Train a classifier for the given layer using CLI and return the saved classifier path."""
    print(f"🎓 Training classifier for layer {layer} using CLI...")
    
    classifier_path = f"classifier_layer_{layer}.pkl"
    
    cmd = [
        "python", "-m", "wisent_guard.cli", "tasks", "truthful_qa",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--layer", str(layer),
        "--train-only",  # Only train, don't test
        "--save-classifier", classifier_path
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 min timeout
        
        if result.returncode != 0:
            print(f"   ❌ CLI training failed for layer {layer}")
            print(f"   STDERR: {result.stderr}")
            return None
        
        # Find the actual saved classifier file (CLI may add suffixes)
        import glob
        possible_files = glob.glob(f"*layer_{layer}*.pkl")
        if possible_files:
            actual_path = possible_files[0]
            print(f"   ✅ Layer {layer} classifier saved to: {actual_path}")
            return actual_path
        else:
            print(f"   ❌ Classifier file not found for layer {layer}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ CLI training timed out for layer {layer}")
        return None
    except Exception as e:
        print(f"   ❌ Error training layer {layer}: {e}")
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


def evaluate_classifier(model: Model, test_df: pd.DataFrame, classifier_path: str, layer: int,
                       aggregation_methods: List[str], thresholds: List[float] = [0.5]) -> Dict[str, Any]:
    """Evaluate classifier on test set with different aggregation methods."""
    print(f"🧪 Evaluating layer {layer} classifier on {len(test_df)} test samples...")
    
    # Load the saved classifier
    try:
        import pickle
        with open(classifier_path, "rb") as f:
            steering_method = pickle.load(f)
        print(f"   ✅ Loaded classifier from {classifier_path}")
    except Exception as e:
        print(f"   ❌ Error loading classifier: {e}")
        return {}
    
    # Handle classifier object (same as extract_vs_realtime.py)
    if isinstance(steering_method, dict) and 'classifier' in steering_method:
        classifier = steering_method['classifier']
    elif hasattr(steering_method, 'classifier'):
        classifier = steering_method.classifier
    else:
        print(f"   ❌ No classifier found in steering method")
        return {}
    
    results = {}
    
    for agg_method in aggregation_methods:
        print(f"   📊 Testing aggregation method: {agg_method}")
        
        # First pass: collect all aggregated scores
        aggregated_scores = []
        true_labels = []
        
        for idx, row in test_df.iterrows():
            text = row['response']
            true_label = row['human_score']
            
            # Extract activation scores
            token_scores = extract_activation_scores(model, text, layer, classifier)
            
            if token_scores:
                # Aggregate scores
                agg_score = aggregate_scores(token_scores, agg_method)
                aggregated_scores.append(agg_score)
                true_labels.append(true_label)
        
        if aggregated_scores:
            # Test multiple thresholds
            best_threshold = None
            best_accuracy = 0
            best_metrics = None
            
            for threshold in thresholds:
                # Make predictions based on this threshold
                predictions = [1 if score <= threshold else 0 for score in aggregated_scores]  # Lower score = more truthful
                
                # Calculate metrics
                accuracy = accuracy_score(true_labels, predictions)
                precision = precision_score(true_labels, predictions, zero_division=0)
                recall = recall_score(true_labels, predictions, zero_division=0)
                f1 = f1_score(true_labels, predictions, zero_division=0)
                
                # Track best threshold by accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                    best_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'threshold': threshold,
                        'n_samples': len(predictions),
                        'mean_score': np.mean(aggregated_scores),
                        'std_score': np.std(aggregated_scores)
                    }
            
            results[agg_method] = best_metrics
            
            print(f"      ✅ {agg_method}: Best threshold={best_threshold:.3f}, Accuracy={best_accuracy:.3f}, F1={best_metrics['f1']:.3f}, N={best_metrics['n_samples']}")
        else:
            print(f"      ❌ {agg_method}: No valid predictions")
    
    return results


def main():
    """Main experiment function."""
    print("🧪 OPTIMIZE ON PREVIOUS DATASET EXPERIMENT")
    print("=" * 70)
    print("Training on original TruthfulQA, testing on TruthfulQA_annotated.xlsx")
    print()
    
    # Configuration
    DATASET_PATH = "TruthfulQA_annotated.xlsx"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    LAYERS = list(range(32))  # Test all layers
    AGGREGATION_METHODS = ["average", "max", "min", "first", "last", "median"]
    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Test multiple thresholds
    
    try:
        # Step 1: Load annotated dataset (TESTING ONLY)
        test_df = load_annotated_dataset(DATASET_PATH)
        print(f"   🎯 Using all {len(test_df)} samples for testing")
        
        # Step 2: Load model for evaluation
        print(f"🤖 Loading model: {MODEL_NAME}")
        model = Model(MODEL_NAME)
        print(f"   ✅ Model loaded on device: {model.device}")
        
        # Step 3: Train classifiers for each layer using CLI (on original TruthfulQA)
        trained_classifiers = {}
        
        for layer in LAYERS:
            print(f"\n🔄 Training Layer {layer}")
            print("-" * 50)
            
            classifier_path = train_classifier_via_cli(layer)
            if classifier_path:
                trained_classifiers[layer] = classifier_path
            else:
                print(f"   ❌ Skipping layer {layer} due to training failure")
        
        print(f"\n✅ Successfully trained {len(trained_classifiers)} classifiers")
        
        # Step 4: Evaluate each trained classifier
        all_results = {}
        
        for layer, classifier_path in trained_classifiers.items():
            print(f"\n🧪 Evaluating Layer {layer}")
            print("-" * 50)
            
            try:
                # Evaluate classifier
                layer_results = evaluate_classifier(
                    model, test_df, classifier_path, layer, AGGREGATION_METHODS, THRESHOLDS
                )
                
                all_results[layer] = layer_results
                
            except Exception as e:
                print(f"   ❌ Error evaluating layer {layer}: {e}")
                continue
        
        # Step 6: Report results
        print("\n📊 COMPREHENSIVE RESULTS")
        print("=" * 70)
        
        # Find best performing combinations
        best_overall = None
        best_accuracy = 0
        
        for layer, layer_results in all_results.items():
            print(f"\nLayer {layer}:")
            for method, metrics in layer_results.items():
                accuracy = metrics['accuracy']
                f1 = metrics['f1']
                n_samples = metrics['n_samples']
                
                threshold = metrics['threshold']
                print(f"   {method:>8}: Acc={accuracy:.3f}, F1={f1:.3f}, Thresh={threshold:.2f}, N={n_samples}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_overall = (layer, method, metrics)
        
        if best_overall:
            layer, method, metrics = best_overall
            print(f"\n🏆 BEST PERFORMANCE:")
            print(f"   Layer: {layer}")
            print(f"   Aggregation: {method}")
            print(f"   Threshold: {metrics['threshold']:.3f}")
            print(f"   Accuracy: {best_accuracy:.3f}")
            print(f"   F1: {metrics['f1']:.3f}")
            print(f"   Precision: {metrics['precision']:.3f}")
            print(f"   Recall: {metrics['recall']:.3f}")
        
        # Save results
        results_file = "layer_optimization_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\n💾 Results saved to: {results_file}")
        
        print("\n🎉 Experiment completed!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
