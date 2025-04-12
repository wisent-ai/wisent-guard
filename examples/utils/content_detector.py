#!/usr/bin/env python
"""
General-purpose content detection utility for wisent-guard.

This module provides reusable functions for content detection with wisent-guard,
supporting various content types such as hallucinations, bias, toxicity, etc.
"""

import os
import time
import torch
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
    
    # Process training pairs with progress tracking
    for i, pair in enumerate(training_pairs):
        print(f"Processing pair {i+1}/{len(training_pairs)}...")
        
        # Process harmful example
        print(f"  - Processing harmful example ({len(pair['harmful'])} chars)...")
        temp_guard.monitor.reset()
        try:
            success = temp_guard._prepare_activations(pair["harmful"])
            if success:
                activations = temp_guard.monitor.get_activations()
                layer = config['layer']
                if layer in activations:
                    activation_tensor = activations[layer]
                    # Ensure tensor is on CPU and detached
                    harmful_activations.append({
                        "activations": activation_tensor.detach().cpu().flatten(),
                        "layer": layer,
                        "token_text": config['positive_class_label'],
                        "is_harmful": True  # Keep this as True for compatibility with API
                    })
                    print(f"    ✓ Successfully collected harmful activation")
                else:
                    print(f"    ✗ Layer {layer} not found in activations")
            else:
                print(f"    ✗ Failed to prepare activations for harmful example")
        except Exception as e:
            print(f"    ✗ Error processing harmful example: {e}")
        
        # Process harmless example
        print(f"  - Processing harmless example ({len(pair['harmless'])} chars)...")
        temp_guard.monitor.reset()
        try:
            success = temp_guard._prepare_activations(pair["harmless"])
            if success:
                activations = temp_guard.monitor.get_activations()
                layer = config['layer']
                if layer in activations:
                    activation_tensor = activations[layer]
                    # Ensure tensor is on CPU and detached
                    harmless_activations.append({
                        "activations": activation_tensor.detach().cpu().flatten(),
                        "layer": layer,
                        "token_text": "normal",
                        "is_harmful": False  # Keep this as False for compatibility with API
                    })
                    print(f"    ✓ Successfully collected harmless activation")
                else:
                    print(f"    ✗ Layer {layer} not found in activations")
            else:
                print(f"    ✗ Failed to prepare activations for harmless example")
        except Exception as e:
            print(f"    ✗ Error processing harmless example: {e}")
    
    print(f"Collected {len(harmful_activations)} {config['positive_class_label']} and {len(harmless_activations)} normal activations")
    
    # Ensure path exists
    os.makedirs(os.path.dirname(config['classifier_path']), exist_ok=True)
    
    # Train classifier
    print("Training classifier (this may take a few minutes)...")
    start_time = time.time()
    
    # Set a timeout for training to prevent hanging
    max_train_time = 600  # 10 minutes
    
    try:
        classifier = ActivationClassifier.create_from_activations(
            harmful_activations=harmful_activations,
            harmless_activations=harmless_activations,
            model_type=config.get('classifier_model', "logistic"),  # Default to logistic regression
            save_path=config['classifier_path'],
            threshold=config['classifier_threshold'],
            positive_class_label=config['positive_class_label'],
            test_size=0.2,
            random_state=42,
            device=model.device  # Use the model's device (GPU/MPS) instead of forcing CPU
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
    except Exception as e:
        print(f"Error training classifier: {e}")
        raise

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
        print(f"Testing statement {i+1}/{len(test_statements)}...")
        
        # Test with classifier
        try:
            is_harmful = classifier_guard.is_harmful(statement)
        except Exception as e:
            print(f"Error during classification: {e}")
            is_harmful = False
        
        # Get classifier confidence if possible
        confidence = None
        try:
            if classifier_guard.classifier and classifier_guard.monitor:
                classifier_guard.monitor.reset()
                success = classifier_guard._prepare_activations(statement)
                if success:
                    activations = classifier_guard.monitor.get_activations()
                    layer = classifier_guard.layers[0]
                    if layer in activations:
                        activation_tensor = activations[layer]
                        # Ensure tensor is on CPU and detached
                        activation_data = {
                            "activations": activation_tensor.detach().cpu().flatten(),
                            "layer": layer,
                            "is_harmful": None
                        }
                        prediction = classifier_guard.classifier.predict(activation_data, response_text=statement)
                        confidence = prediction.get('score', None)
        except Exception as e:
            print(f"Error getting classifier confidence: {e}")
            confidence = None
        
        # Update metrics
        true_class = ground_truth[i]
        if is_harmful and true_class:     # True positive
            true_positives += 1
            result_type = "TP"
        elif is_harmful and not true_class:  # False positive
            false_positives += 1
            result_type = "FP"
        elif not is_harmful and not true_class:  # True negative
            true_negatives += 1
            result_type = "TN"
        else:  # not is_harmful and true_class:  # False negative
            false_negatives += 1
            result_type = "FN"
        
        # Display truncated statement and classification
        truncated = statement[:60] + "..." if len(statement) > 60 else statement
        confidence_str = f" (conf: {confidence:.4f})" if confidence is not None else ""
        classification = f"{is_harmful} ({result_type}){confidence_str}"
        print(f"{truncated:<65} | {classification}")
        
        # Store result
        results.append({
            "statement": statement,
            "prediction": is_harmful,
            "true_class": true_class,
            "confidence": confidence,
            "result_type": result_type
        })
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(test_statements)
    
    # Print metrics
    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "results": results
    }
    
    return metrics

def evaluate_responses(model, tokenizer, guard, prompts, show_input=True):
    """
    Evaluate model responses with the content guard.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        guard: ActivationGuard instance
        prompts: List of prompts to evaluate
        show_input: Whether to show input prompts
    """
    print("\n==== Evaluating Model Responses ====")
    
    for i, prompt in enumerate(prompts):
        if show_input:
            print(f"\nPrompt {i+1}: {prompt}")
        else:
            print(f"\nPrompt {i+1}: <input hidden>")
            
        # Format prompt for Llama models
        formatted_prompt = format_prompt(prompt)
        
        # Generate response with the guard
        result = guard.generate_safe_response(
            formatted_prompt,
            max_new_tokens=200
        )
        
        # Get response and safety information
        response = result.get("response", "")
        blocked = result.get("blocked", False)
        reason = result.get("reason", None)
        classification_score = result.get("classification_score", None)
        classification_threshold = result.get("classification_threshold", None)
        
        # Display results
        print("\nResponse:")
        print("-" * 80)
        print(response[:500] + "..." if len(response) > 500 else response)
        print("-" * 80)
        
        # Display safety information
        print("Safety Analysis:")
        if blocked:
            print(f"⚠️ Content flagged as potentially {guard.classifier.positive_class_label}")
            print(f"Reason: {reason}")
        else:
            print(f"✅ Content classified as safe")
            
        if classification_score is not None:
            print(f"Classification score: {classification_score:.4f} (threshold: {classification_threshold:.4f})")
            
        print("-" * 80) 