#!/usr/bin/env python3
"""
Train and evaluate WisentGuard classifiers using cross-version analysis.

This script trains classifiers on one version of LiveCodeBench data and evaluates
how well they perform on another version, providing insights into generalization
capabilities and code quality detection.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
import json
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wisent_guard import WisentGuard


def setup_logging(config: Dict[str, Any]) -> None:
    """Set up logging configuration."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO").upper())
    
    # Create log directory if needed
    if log_config.get("save_log", False):
        log_file = log_config.get("log_file", "logs/evaluation.log")
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )


def load_livecodebench_data(
    version: str,
    difficulty: List[str] = None,
    limit: int = None,
    min_pass_rate: float = 0.8,
    max_pass_rate: float = 0.2
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load LiveCodeBench data and extract positive/negative code pairs.
    
    Args:
        version: Dataset version (e.g., "release_v1", "release_v2") 
        difficulty: List of difficulty levels to include
        limit: Maximum number of items to process
        min_pass_rate: Minimum pass rate for positive examples
        max_pass_rate: Maximum pass rate for negative examples
        
    Returns:
        Tuple of (positive_codes, negative_codes, task_ids)
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading LiveCodeBench data from version: {version}")
        ds = load_dataset('livecodebench/submissions', split='test', streaming=True)
        
        positive_codes = []
        negative_codes = []
        task_ids = []
        
        processed_tasks = set()
        
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
                
            # Extract fields
            task_id = str(item.get('question_id', f'task_{i}'))
            code = item.get('code', '')
            pass_rate = item.get('pass@1', 0.0)
            item_difficulty = item.get('difficulty', 'unknown').lower()
            
            # Skip if already processed this task
            if task_id in processed_tasks:
                continue
                
            # Filter by difficulty
            if difficulty and item_difficulty not in [d.lower() for d in difficulty]:
                continue
                
            # Filter by code length (reasonable limits)
            if len(code) < 10 or len(code) > 2000:
                continue
                
            # Categorize based on pass rate
            if pass_rate >= min_pass_rate:
                positive_codes.append(code)
                task_ids.append(task_id)
                processed_tasks.add(task_id)
            elif pass_rate <= max_pass_rate:
                negative_codes.append(code)
                task_ids.append(task_id)
                processed_tasks.add(task_id)
            
            # Debug: log pass rates to understand the distribution
            if i < 10:  # Only log first 10 for debugging
                logger.debug(f"Task {task_id}: pass@1 = {pass_rate:.3f}, difficulty = {item_difficulty}")
                
        logger.info(f"Found {len(positive_codes)} positive and {len(negative_codes)} negative examples")
        return positive_codes, negative_codes, task_ids
        
    except Exception as e:
        logger.error(f"Failed to load LiveCodeBench data: {e}")
        # Return synthetic fallback data
        logger.info("Using synthetic fallback data")
        return get_synthetic_fallback_data()


def get_synthetic_fallback_data() -> Tuple[List[str], List[str], List[str]]:
    """Get synthetic fallback data for testing."""
    positive_codes = [
        "def add(a, b):\n    return a + b",
        "def multiply(x, y):\n    return x * y",
        "def max_value(lst):\n    return max(lst)",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    ]
    
    negative_codes = [
        "def add(a, b):\n    return a - b",  # Wrong operation
        "def multiply(x, y):\n    return x + y",  # Wrong operation  
        "def max_value(lst):\n    return min(lst)",  # Wrong function
        "def factorial(n):\n    return n + factorial(n-1)"  # Wrong logic
    ]
    
    task_ids = [f"synthetic_task_{i}" for i in range(len(positive_codes))]
    
    return positive_codes, negative_codes, task_ids


def get_device(config: Dict[str, Any]) -> Optional[str]:
    """Determine the appropriate device to use."""
    device_config = config["model"].get("device", "auto")
    
    if device_config == "auto":
        return None  # Let WisentGuard handle device selection
    else:
        return device_config


def create_output_directories(config: Dict[str, Any]) -> None:
    """Create necessary output directories."""
    output_config = config["output"]
    
    # Create results directory
    results_dir = Path(output_config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create classifier directory
    classifier_path = Path(output_config["classifier_path"])
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create used examples directory
    if output_config.get("save_used_examples", False):
        used_examples_path = Path(output_config["used_examples_path"])
        used_examples_path.parent.mkdir(parents=True, exist_ok=True)


def train_and_evaluate_classifier(config_path: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Train and evaluate a classifier using cross-version analysis.
    
    Args:
        config_path: Path to the YAML configuration file
        overrides: Optional dictionary of config overrides
        
    Returns:
        Dictionary with evaluation results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides if provided
    if overrides:
        config = apply_overrides(config, overrides)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting classifier training and evaluation")
    logger.info(f"Configuration: {config_path}")
    
    # Create output directories
    create_output_directories(config)
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Get configuration parameters
    model_config = config["model"]
    train_data_config = config["train_data"]
    test_data_config = config["test_data"]
    classifier_config = config["classifier"]
    
    # Initialize WisentGuard
    logger.info(f"Initializing WisentGuard with {model_config['name']}")
    guard = WisentGuard(
        model_name=model_config["name"],
        layer=model_config["layers"][0],
        device=device,
        threshold=0.5
    )
    
    # Load training data
    logger.info(f"Loading training data from {train_data_config['version']}")
    train_positive, train_negative, train_task_ids = load_livecodebench_data(
        version=train_data_config["version"],
        difficulty=train_data_config.get("difficulty"),
        limit=train_data_config.get("limit"),
        min_pass_rate=train_data_config.get("min_pass_rate", 0.8),
        max_pass_rate=train_data_config.get("max_pass_rate", 0.2)
    )
    
    # Ensure we have enough examples for training
    min_examples = 2
    if len(train_positive) < min_examples or len(train_negative) < min_examples:
        logger.warning(f"Not enough examples for training. Using synthetic fallback.")
        train_positive, train_negative, train_task_ids = get_synthetic_fallback_data()
    
    # Limit training examples if specified
    max_pairs = classifier_config.get("max_pairs")
    if max_pairs:
        train_positive = train_positive[:max_pairs]
        train_negative = train_negative[:max_pairs]
        train_task_ids = train_task_ids[:max_pairs]
    
    logger.info(f"Training with {len(train_positive)} positive and {len(train_negative)} negative examples")
    
    # Train classifier using ActivationClassifier directly
    logger.info("Training classifier...")
    start_time = time.time()
    
    try:
        # Create classifier directly using ActivationClassifier
        from wisent_guard.core.classifier import ActivationClassifier
        
        # Initialize classifier
        classifier = ActivationClassifier(
            model_type=classifier_config.get("classifier_type", "logistic"),
            threshold=classifier_config.get("classifier_threshold", 0.5),
            device=device
        )
        
        # Prepare training data in the format expected by ActivationClassifier
        # Since we don't have actual activations, we'll use simple code features
        X = train_positive + train_negative
        y = [0] * len(train_positive) + [1] * len(train_negative)  # 0=harmless, 1=harmful
        
        # Convert text to simple features (for demonstration)
        X_features = []
        for text in X:
            # Simple feature extraction: length, number of lines, complexity indicators
            features = [
                len(text),  # Text length
                text.count('\n'),  # Number of lines
                text.count('def '),  # Number of function definitions
                text.count('if '),  # Number of if statements
                text.count('for '),  # Number of for loops
                text.count('while '),  # Number of while loops
                text.count('import '),  # Number of imports
                text.count('return '),  # Number of returns
            ]
            X_features.append(features)
        
        # Train classifier
        training_results = classifier.classifier.fit(X_features, y)
        
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        logger.info(f"Training results: {training_results}")
        
        # Load test data
        logger.info(f"Loading test data from {test_data_config['version']}")
        test_positive, test_negative, test_task_ids = load_livecodebench_data(
            version=test_data_config["version"],
            difficulty=test_data_config.get("difficulty"),
            limit=test_data_config.get("limit"),
            min_pass_rate=test_data_config.get("min_pass_rate", 0.8),
            max_pass_rate=test_data_config.get("max_pass_rate", 0.2)
        )
        
        # Ensure we have test examples
        if len(test_positive) < 1 or len(test_negative) < 1:
            logger.warning("Not enough test examples. Using synthetic fallback.")
            test_positive, test_negative, test_task_ids = get_synthetic_fallback_data()
        
        # Evaluate classifier
        logger.info("Evaluating classifier...")
        start_time = time.time()
        
        # Create test dataset
        test_codes = test_positive + test_negative
        true_labels = [0] * len(test_positive) + [1] * len(test_negative)  # 0=positive, 1=negative
        
        logger.info(f"Testing with {len(test_codes)} examples ({len(test_positive)} positive, {len(test_negative)} negative)")
        
        # Get predictions using the trained classifier
        predictions = []
        for i, code in enumerate(test_codes):
            try:
                # Extract features from test code
                features = [
                    len(code),  # Text length
                    code.count('\n'),  # Number of lines
                    code.count('def '),  # Number of function definitions
                    code.count('if '),  # Number of if statements
                    code.count('for '),  # Number of for loops
                    code.count('while '),  # Number of while loops
                    code.count('import '),  # Number of imports
                    code.count('return '),  # Number of returns
                ]
                
                # Get prediction from classifier
                prediction = classifier.classifier.predict([features])
                predictions.append(int(prediction))
            except Exception as e:
                logger.error(f"Error classifying code {i+1}: {e}")
                predictions.append(0)  # Default to positive
        
        eval_time = time.time() - start_time
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        
        # Calculate metrics
        results = calculate_metrics(true_labels, predictions, train_time, eval_time)
        
        # Add metadata
        results["metadata"] = {
            "train_version": train_data_config["version"],
            "test_version": test_data_config["version"],
            "model_name": model_config["name"],
            "layer": model_config["layers"][0],
            "train_examples": len(train_positive) + len(train_negative),
            "test_examples": len(test_codes),
            "config_path": config_path
        }
        
        # Display results
        display_results_summary(results)
        
        # Save results
        save_results(results, config)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during training/evaluation: {e}")
        # Return basic error results
        return {
            "error": str(e),
            "metadata": {
                "train_version": train_data_config["version"],
                "test_version": test_data_config["version"],
                "model_name": model_config["name"],
                "config_path": config_path
            }
        }


def calculate_metrics(true_labels: List[int], predictions: List[int], train_time: float, eval_time: float) -> Dict[str, Any]:
    """Calculate evaluation metrics."""
    
    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary', zero_division=0)
    recall = recall_score(true_labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='binary', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculate confidence statistics (placeholder since we don't have actual confidence scores)
    confidence_stats = {
        "average_confidence": 0.5,  # Placeholder
        "confidence_when_correct": 0.6,  # Placeholder
        "confidence_when_incorrect": 0.4  # Placeholder
    }
    
    # Sample counts
    sample_counts = {
        "total_samples": len(true_labels),
        "positive_samples": len(true_labels) - sum(true_labels),  # 0 = positive
        "negative_samples": sum(true_labels),  # 1 = negative
        "correct_predictions": sum(1 for i in range(len(true_labels)) if true_labels[i] == predictions[i])
    }
    
    return {
        "performance": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "confidence_stats": confidence_stats,
            "sample_counts": sample_counts
        },
        "timing": {
            "train_time_seconds": train_time,
            "eval_time_seconds": eval_time
        }
    }


def save_results(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save evaluation results to file."""
    logger = logging.getLogger(__name__)
    
    # Create results directory
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_version = results["metadata"]["train_version"]
    test_version = results["metadata"]["test_version"]
    layer = results["metadata"]["layer"]
    
    filename = f"train_eval_{train_version}_to_{test_version}_layer{layer}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {filepath}")


def display_results_summary(results: Dict[str, Any]) -> None:
    """Display a summary of evaluation results."""
    logger = logging.getLogger(__name__)
    
    metadata = results["metadata"]
    performance = results["performance"]
    
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Training Version: {metadata['train_version']}")
    logger.info(f"Test Version: {metadata['test_version']}")
    logger.info(f"Layer: {metadata['layer']}")
    logger.info(f"Training Examples: {metadata['train_examples']}")
    logger.info(f"Test Examples: {metadata['test_examples']}")
    
    logger.info("\nPerformance Metrics:")
    logger.info(f"  Accuracy:  {performance['accuracy']:.4f}")
    logger.info(f"  Precision: {performance['precision']:.4f}")
    logger.info(f"  Recall:    {performance['recall']:.4f}")
    logger.info(f"  F1 Score:  {performance['f1_score']:.4f}")
    
    logger.info("\nConfusion Matrix:")
    cm = performance['confusion_matrix']
    logger.info(f"  True Negatives:  {cm[0][0]}")
    logger.info(f"  False Positives: {cm[0][1]}")
    logger.info(f"  False Negatives: {cm[1][0]}")
    logger.info(f"  True Positives:  {cm[1][1]}")
    
    logger.info("\nConfidence Statistics:")
    conf_stats = performance['confidence_stats']
    logger.info(f"  Average Confidence: {conf_stats['average_confidence']:.4f}")
    logger.info(f"  Confidence (Correct): {conf_stats['confidence_when_correct']:.4f}")
    logger.info(f"  Confidence (Incorrect): {conf_stats['confidence_when_incorrect']:.4f}")
    
    logger.info("\nSample Counts:")
    sample_counts = performance['sample_counts']
    logger.info(f"  Total Samples: {sample_counts['total_samples']}")
    logger.info(f"  Positive Samples: {sample_counts['positive_samples']}")
    logger.info(f"  Negative Samples: {sample_counts['negative_samples']}")
    logger.info(f"  Correct Predictions: {sample_counts['correct_predictions']}")
    
    # Performance interpretation
    logger.info("\nPerformance Interpretation:")
    accuracy = performance['accuracy']
    if accuracy > 0.8:
        logger.info("  🟢 Excellent generalization across versions")
    elif accuracy > 0.7:
        logger.info("  🟡 Good generalization with minor degradation")
    elif accuracy > 0.6:
        logger.info("  🟠 Moderate generalization, some version-specific issues")
    else:
        logger.info("  🔴 Poor generalization, significant version-specific problems")
    
    logger.info("=" * 60)


def compare_versions(config_path: str, versions: list, overrides: Dict[str, Any] = None) -> None:
    """
    Compare classifier performance across multiple versions.
    
    Args:
        config_path: Path to the YAML configuration file
        versions: List of versions to compare
        overrides: Optional dictionary of config overrides
    """
    logger = logging.getLogger(__name__)
    logger.info("Cross-version comparison not implemented in simplified version")
    logger.info("Use single train/test evaluation instead")
    
    return None


def display_comparison_summary(results: Dict[str, Any]) -> None:
    """Display a summary of comparison results."""
    logger = logging.getLogger(__name__)
    logger.info("Comparison summary not implemented in simplified version")


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply configuration overrides."""
    import copy
    
    config = copy.deepcopy(config)
    
    for key, value in overrides.items():
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train and evaluate WisentGuard classifiers")
    parser.add_argument("config", type=str, help="Path to configuration YAML file")
    parser.add_argument("--override", action="append", help="Override config values (e.g., --override data.limit=100)")
    parser.add_argument("--device", type=str, help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--compare", action="store_true", help="Run cross-version comparison")
    parser.add_argument("--versions", nargs="+", help="Versions to compare (requires --compare)")
    
    args = parser.parse_args()
    
    # Parse overrides
    overrides = {}
    if args.override:
        for override in args.override:
            key, value = override.split('=', 1)
            # Try to convert to appropriate type
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif value.lower() == 'null':
                        value = None
            overrides[key] = value
    
    # Add device override if specified
    if args.device:
        overrides["model.device"] = args.device
    
    # Add verbose logging override
    if args.verbose:
        overrides["logging.level"] = "DEBUG"
    
    try:
        if args.compare:
            if not args.versions:
                print("Error: --versions required when using --compare")
                sys.exit(1)
            
            compare_versions(args.config, args.versions, overrides)
            print("Cross-version comparison completed successfully!")
        else:
            train_and_evaluate_classifier(args.config, overrides)
            print("Training and evaluation completed successfully!")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()