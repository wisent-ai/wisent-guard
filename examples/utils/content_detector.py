#!/usr/bin/env python
"""
General-purpose content detection utility for wisent-guard.

This module provides reusable functions for content detection with wisent-guard,
supporting various content types such as hallucinations, bias, toxicity, etc.
"""

import os
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from wisent_guard import ActivationGuard
from wisent_guard.classifier import ActivationClassifier

def get_device(cpu_only=False):
    """Determine the appropriate device to use."""
    if cpu_only:
        return "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def format_prompt(message, system_prompt=None):
    """Format prompt for Llama 3.1 Instruct models using the special tokens"""
    # Llama 3.1 special tokens
    BEGIN_TEXT = "<|begin_of_text|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    EOT = "<|eot_id|>"
    
    if system_prompt:
        # Format with system prompt
        formatted = f"{BEGIN_TEXT}{START_HEADER}system{END_HEADER}\n{system_prompt}{EOT}{START_HEADER}user{END_HEADER}\n{message}{EOT}{START_HEADER}assistant{END_HEADER}"
    else:
        # Format without system prompt
        formatted = f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{message}{EOT}{START_HEADER}assistant{END_HEADER}"
    
    return formatted

def load_model(model_name, device, half_precision=False, load_in_8bit=False):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Define loading parameters based on device
    load_kwargs = {
        "torch_dtype": torch.float16 if (device != "cpu" and half_precision) else torch.float32,
    }
    
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["device_map"] = "cpu"  # Load on CPU first for MPS
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move model to MPS if needed
    if device == "mps" and load_kwargs["device_map"] == "cpu":
        model = model.to(device)
    
    print(f"Model loaded successfully: {model.__class__.__name__}")
    print(f"Model has {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9):.2f} billion parameters")
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(f"Number of layers: {len(model.model.layers)}")
    
    return model, tokenizer

def train_classifier_detector(model, tokenizer, training_pairs, config):
    """
    Train a classifier-based detector for content detection.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        training_pairs: List of training pairs with "harmful" and "harmless" examples
        config: Configuration dictionary with:
            - layer: Layer to monitor
            - classifier_path: Path to save classifier model
            - classifier_threshold: Threshold for classification
            - positive_class_label: Label for the "harmful" class
            - save_dir: Directory to save temporary data
    
    Returns:
        classifier_guard: Trained ActivationGuard with classifier
    """
    print(f"\n==== Training Classifier-Based {config['positive_class_label'].capitalize()} Detector ====")
    
    # First, initialize a temporary guard to collect activations
    temp_guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=[config['layer']],
        threshold=0.2,  # Default threshold, not actually used for classification
        save_dir="./temp_data",
        device=model.device,
        force_format="llama"
    )
    
    # Explicitly initialize the monitor
    temp_guard._initialize_monitor_and_inference()
    
    # Collect activations for training
    print("Collecting activations for training classifier...")
    
    harmful_activations = []
    harmless_activations = []
    
    for pair in training_pairs:
        # Process harmful example
        temp_guard.monitor.reset()
        success = temp_guard._prepare_activations(pair["harmful"])
        if success:
            activations = temp_guard.monitor.get_activations()
            layer = config['layer']
            if layer in activations:
                activation_tensor = activations[layer]
                tensor_data = activation_tensor.detach().cpu().numpy().flatten()
                harmful_activations.append({
                    "activations": tensor_data,
                    "layer": layer,
                    "token_text": config['positive_class_label'],
                    "is_harmful": True  # Keep this as True for compatibility with API
                })
        
        # Process harmless example
        temp_guard.monitor.reset()
        success = temp_guard._prepare_activations(pair["harmless"])
        if success:
            activations = temp_guard.monitor.get_activations()
            layer = config['layer']
            if layer in activations:
                activation_tensor = activations[layer]
                tensor_data = activation_tensor.detach().cpu().numpy().flatten()
                harmless_activations.append({
                    "activations": tensor_data,
                    "layer": layer,
                    "token_text": "normal",
                    "is_harmful": False  # Keep this as False for compatibility with API
                })
    
    print(f"Collected {len(harmful_activations)} {config['positive_class_label']} and {len(harmless_activations)} normal activations")
    
    # Ensure path exists
    os.makedirs(os.path.dirname(config['classifier_path']), exist_ok=True)
    
    # Train classifier
    start_time = time.time()
    classifier = ActivationClassifier.create_from_activations(
        harmful_activations=harmful_activations,
        harmless_activations=harmless_activations,
        model_type=config.get('classifier_model', "logistic"),  # Default to logistic regression
        save_path=config['classifier_path'],
        threshold=config['classifier_threshold'],
        positive_class_label=config['positive_class_label'],
        test_size=0.2,
        random_state=42
    )
    train_time = time.time() - start_time
    
    print(f"Classifier trained in {train_time:.2f} seconds and saved to {config['classifier_path']}")
    
    # Initialize guard with classifier
    classifier_guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=[config['layer']],
        threshold=0.2,  # Default threshold, not actually used for classification
        save_dir=config.get('save_dir', "./classifier_detector_data"),
        device=model.device,
        use_classifier=True,
        classifier_path=config['classifier_path'],
        classifier_threshold=config['classifier_threshold'],
        force_format="llama"
    )
    
    # Explicitly initialize the monitor for the classifier guard as well
    classifier_guard._initialize_monitor_and_inference()
    
    return classifier_guard

def test_classifier(classifier_guard, test_statements, ground_truth, content_type="harmful content"):
    """
    Test the content detector on example statements and calculate metrics.
    
    Args:
        classifier_guard: Trained ActivationGuard with classifier
        test_statements: List of statements to test
        ground_truth: List of boolean values indicating true class for each statement
        content_type: Type of content being detected (for display purposes)
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    print(f"\n==== Testing {content_type.capitalize()} Detector ====")
    
    print("\nResults:")
    print("-" * 80)
    print(f"{'Statement':<65} | {'Classification':<15}")
    print("-" * 80)
    
    # Track metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    results = []
    
    for i, statement in enumerate(test_statements):
        # Test with classifier
        is_harmful = classifier_guard.is_harmful(statement)
        
        # Get classifier confidence if possible
        try:
            if classifier_guard.classifier and classifier_guard.monitor:
                classifier_guard.monitor.reset()
                success = classifier_guard._prepare_activations(statement)
                if success:
                    activations = classifier_guard.monitor.get_activations()
                    layer = classifier_guard.layers[0]
                    if layer in activations:
                        activation_tensor = activations[layer]
                        tensor_data = activation_tensor.detach().cpu().numpy().flatten()
                        activation_data = {
                            "activations": tensor_data,
                            "layer": layer,
                            "is_harmful": None
                        }
                        prediction = classifier_guard.classifier.predict(activation_data, response_text=statement)
                        confidence = prediction.get('confidence', 0)
                    else:
                        confidence = None
                else:
                    confidence = None
            else:
                confidence = None
        except Exception as e:
            print(f"Error getting classifier confidence: {e}")
            confidence = None
        
        # Print results in a table format
        classification = content_type.upper() if is_harmful else "NORMAL"
        
        truncated_statement = statement[:62] + "..." if len(statement) > 65 else statement.ljust(65)
        print(f"{truncated_statement} | {classification:<15}")
        if confidence is not None:
            print(f"  Confidence: {confidence:.4f}")
            
        # Update metrics
        if is_harmful and ground_truth[i]:
            true_positives += 1
        elif is_harmful and not ground_truth[i]:
            false_positives += 1
        elif not is_harmful and not ground_truth[i]:
            true_negatives += 1
        else:  # not is_harmful and ground_truth[i]
            false_negatives += 1
            
        # Store result for confusion matrix
        results.append({
            "statement": statement,
            "predicted": is_harmful,
            "actual": ground_truth[i],
            "confidence": confidence
        })
    
    # Calculate metrics
    total = len(test_statements)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display metrics
    print(f"\n==== {content_type.capitalize()} Detector Metrics ====")
    print(f"Total examples tested: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Display confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 40)
    print(f"{'':20} | {'Predicted':^18}")
    print(f"{'':20} | {content_type.capitalize():^9} | {'Normal':^8}")
    print("-" * 40)
    print(f"{'Actual: ' + content_type.capitalize():20} | {true_positives:^9} | {false_negatives:^8}")
    print(f"{'Actual: Normal':20} | {false_positives:^9} | {true_negatives:^8}")
    print("-" * 40)
    
    # Return metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "total": total,
        "results": results
    }
    
    return metrics 