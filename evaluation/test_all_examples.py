#!/usr/bin/env python
"""
Script to test all examples in the wisent-guard library and save accuracy results as JSON.

This script:
1. Tests each detector example using their defined test cases
2. Collects accuracy metrics for each detector
3. Saves the results as a JSON file

Usage:
python test_all_examples.py [--model MODEL_NAME] [--output OUTPUT_PATH]
"""

import os
import sys
import json
import argparse
import torch
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import detectors
from examples.detect_harmful_content import create_harmful_content_pairs, get_test_statements_and_ground_truth as get_harmful_statements
from examples.detect_bad_code import create_bad_code_pairs, get_test_statements_and_ground_truth as get_bad_code_statements
from examples.detect_bias import create_bias_pairs, get_test_statements_and_ground_truth as get_bias_statements
from examples.detect_gender_bias import create_gender_bias_pairs, get_test_statements_and_ground_truth as get_gender_bias_statements
from examples.detect_hallucinations import create_hallucination_pairs, get_test_statements_and_ground_truth as get_hallucination_statements
from examples.detect_personal_info import create_personal_info_pairs, get_test_statements_and_ground_truth as get_personal_info_statements
from examples.detect_scheming import create_scheming_pairs, get_test_statements_and_ground_truth as get_scheming_statements

# Import utility functions
from examples.utils.content_detector import (
    get_device,
    load_model,
    train_classifier_detector,
    test_classifier
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test all examples and save accuracy results as JSON")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                      help="Name of the model to use (default: Llama-3.1-8B-Instruct)")
    parser.add_argument("--output", type=str, default="./evaluation/results/detector_accuracy_results.json",
                      help="Path to save the JSON results")
    parser.add_argument("--layer", type=int, default=15,
                      help="Layer to monitor for activations (default: 15)")
    parser.add_argument("--half-precision", action="store_true",
                      help="Load model in half precision (fp16)")
    parser.add_argument("--load-in-8bit", action="store_true",
                      help="Load model in 8-bit quantization (not supported on MPS)")
    return parser.parse_args()

def test_detector(model, tokenizer, training_pairs, test_statements, ground_truth, 
                 content_type, layer=15, classifier_threshold=0.5):
    """
    Test a specific detector with given test statements.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        training_pairs: List of training pairs for the detector
        test_statements: List of statements to test
        ground_truth: List of ground truth values (True for harmful, False for harmless)
        content_type: Type of content being detected
        layer: Layer to monitor for activations
        classifier_threshold: Threshold for classification
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    print(f"\n===== Testing {content_type} Detector =====")
    
    # Create directory structure
    os.makedirs(f"./models", exist_ok=True)
    os.makedirs(f"./{content_type.replace(' ', '_')}_detector_data", exist_ok=True)
    
    # Create configuration for the detector
    detector_config = {
        'layer': layer,
        'classifier_path': f"./models/{content_type.replace(' ', '_')}_classifier.pkl",
        'classifier_threshold': classifier_threshold,
        'positive_class_label': content_type,
        'save_dir': f"./{content_type.replace(' ', '_')}_detector_data",
        'classifier_model': 'logistic'
    }
    
    try:
        # Train classifier-based detector
        print(f"Training {content_type} detector...")
        classifier_guard = train_classifier_detector(model, tokenizer, training_pairs, detector_config)
        
        # Test the classifier
        print(f"Testing {content_type} detector...")
        metrics = test_classifier(classifier_guard, test_statements, ground_truth, content_type=content_type)
        
        print(f"{content_type} detection testing complete with F1 score: {metrics['f1']:.4f}")
        return metrics
        
    except Exception as e:
        print(f"Error testing {content_type} detector: {e}")
        # Return empty metrics with error message
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "error": str(e)
        }

def main():
    """Main function to test all examples."""
    args = parse_args()
    
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(args.model, device, args.half_precision, args.load_in_8bit)
    
    # Dictionary to store all results
    results = {
        "metadata": {
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "layer": args.layer,
            "device": str(device)
        },
        "detectors": {}
    }
    
    # Test harmful content detector
    print("\nTesting harmful content detector...")
    training_pairs = create_harmful_content_pairs()
    test_statements, ground_truth = get_harmful_statements()
    harmful_metrics = test_detector(
        model, tokenizer, training_pairs, test_statements, ground_truth,
        "harmful content", args.layer
    )
    results["detectors"]["harmful_content"] = harmful_metrics
    
    # Test bad code detector
    print("\nTesting bad code detector...")
    training_pairs = create_bad_code_pairs()
    test_statements, ground_truth = get_bad_code_statements()
    bad_code_metrics = test_detector(
        model, tokenizer, training_pairs, test_statements, ground_truth,
        "bad code", args.layer
    )
    results["detectors"]["bad_code"] = bad_code_metrics
    
    # Test bias detector
    print("\nTesting bias detector...")
    training_pairs = create_bias_pairs()
    test_statements, ground_truth = get_bias_statements()
    bias_metrics = test_detector(
        model, tokenizer, training_pairs, test_statements, ground_truth,
        "bias", args.layer
    )
    results["detectors"]["bias"] = bias_metrics
    
    # Test gender bias detector
    print("\nTesting gender bias detector...")
    training_pairs = create_gender_bias_pairs()
    test_statements, ground_truth = get_gender_bias_statements()
    gender_bias_metrics = test_detector(
        model, tokenizer, training_pairs, test_statements, ground_truth,
        "gender bias", args.layer
    )
    results["detectors"]["gender_bias"] = gender_bias_metrics
    
    # Test hallucination detector
    print("\nTesting hallucination detector...")
    training_pairs = create_hallucination_pairs()
    test_statements, ground_truth = get_hallucination_statements()
    hallucination_metrics = test_detector(
        model, tokenizer, training_pairs, test_statements, ground_truth,
        "hallucination", args.layer
    )
    results["detectors"]["hallucination"] = hallucination_metrics
    
    # Test personal info detector
    print("\nTesting personal info detector...")
    training_pairs = create_personal_info_pairs()
    test_statements, ground_truth = get_personal_info_statements()
    personal_info_metrics = test_detector(
        model, tokenizer, training_pairs, test_statements, ground_truth,
        "personal info", args.layer
    )
    results["detectors"]["personal_info"] = personal_info_metrics
    
    # Test scheming detector
    print("\nTesting scheming detector...")
    training_pairs = create_scheming_pairs()
    test_statements, ground_truth = get_scheming_statements()
    scheming_metrics = test_detector(
        model, tokenizer, training_pairs, test_statements, ground_truth,
        "scheming", args.layer
    )
    results["detectors"]["scheming"] = scheming_metrics
    
    # Create summary metrics
    results["summary"] = {
        "average_accuracy": sum(d.get("accuracy", 0) for d in results["detectors"].values()) / len(results["detectors"]),
        "average_f1": sum(d.get("f1", 0) for d in results["detectors"].values()) / len(results["detectors"]),
        "best_detector": max(results["detectors"].items(), key=lambda x: x[1].get("f1", 0))[0],
        "worst_detector": min(results["detectors"].items(), key=lambda x: x[1].get("f1", 0))[0]
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save results as JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("\nSummary:")
    print(f"Average accuracy: {results['summary']['average_accuracy']:.4f}")
    print(f"Average F1 score: {results['summary']['average_f1']:.4f}")
    print(f"Best detector: {results['summary']['best_detector']}")
    print(f"Worst detector: {results['summary']['worst_detector']}")

if __name__ == "__main__":
    main() 