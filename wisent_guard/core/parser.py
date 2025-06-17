"""
Command-line argument parser for wisent-guard.
"""

import argparse
from typing import List, Optional


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation benchmarks through wisent-guard pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m wisent_guard tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B --token-aggregation final
  python -m wisent_guard tasks hellaswag,mmlu --layer 10 --model meta-llama/Llama-3.1-8B --shots 5 --token-aggregation max
  python -m wisent_guard tasks truthfulqa --optimize --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks truthfulqa --layer -1 --auto-optimize --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks truthfulqa --optimize --optimize-layers "10-20" --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks data.csv --from-csv --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks data.json --from-json --model meta-llama/Llama-3.1-8B
  
  # Detection handling examples:
  python -m wisent_guard tasks data.csv --from-csv --detection-action replace_with_placeholder
  python -m wisent_guard tasks data.csv --from-csv --detection-action regenerate_until_safe --max-regeneration-attempts 5
  python -m wisent_guard tasks data.csv --from-csv --detection-action replace_with_placeholder --placeholder-message "Content flagged for review"
  
  # Training/Inference mode examples:
  python -m wisent_guard tasks truthfulqa --train-only --save-classifier my_classifier --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks truthfulqa --layer 14-16 --train-only --classifier-dir ./models --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks data.csv --from-csv --inference-only --load-classifier ./models/my_classifier --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks data.csv --from-csv --layer 14-16 --inference-only --load-classifier ./models/truthfulqa_classifier --model meta-llama/Llama-3.1-8B
        """
    )
    
    parser.add_argument("command", choices=["tasks"], help="Command to run")
    parser.add_argument("task_names", help="Comma-separated list of task names, or path to CSV/JSON file with --from-csv/--from-json")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--layer", type=str, default="15", help="Layer(s) to extract activations from. Can be a single layer (15), range (14-16), or comma-separated list (14,15,16)")
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents per task")
    parser.add_argument("--output", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--classifier-type", type=str, choices=["logistic", "mlp"], default="logistic", help="Type of classifier")
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Maximum new tokens for generation")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--token-aggregation", type=str, choices=["average", "final", "first", "max", "min"], 
                       default="average", help="How to aggregate token scores for classification")
    parser.add_argument("--ground-truth-method", type=str, 
                       choices=["none", "exact_match", "substring_match", "user_specified", "interactive", "manual_review", "good"],
                       default="none", 
                       help="Method for ground truth evaluation. 'none' skips evaluation, 'exact_match' and 'substring_match' are problematic for free-form generation, 'user_specified' allows manual labeling, 'interactive' prompts for y/n labeling, 'manual_review' marks for review, 'good' marks everything as truthful (for debugging)")
    parser.add_argument("--user-labels", type=str, nargs="*", default=None,
                       help="User-specified ground truth labels for responses ('truthful' or 'hallucination'). Used with --ground-truth-method user_specified")
    
    # File input arguments
    parser.add_argument("--from-csv", action="store_true", 
                       help="Load task data from CSV file. Requires columns: question, correct_answer, incorrect_answer")
    parser.add_argument("--from-json", action="store_true",
                       help="Load task data from JSON file. Expected format: list of objects with question, correct_answer, incorrect_answer")
    parser.add_argument("--question-col", type=str, default="question",
                       help="Column name for questions in CSV file (default: question)")
    parser.add_argument("--correct-col", type=str, default="correct_answer",
                       help="Column name for correct answers in CSV file (default: correct_answer)")
    parser.add_argument("--incorrect-col", type=str, default="incorrect_answer", 
                       help="Column name for incorrect answers in CSV file (default: incorrect_answer)")
    
    # Optimization arguments
    parser.add_argument("--optimize", action="store_true",
                       help="Enable hyperparameter optimization. When enabled, will find optimal layer, threshold, and aggregation method")
    parser.add_argument("--optimize-layers", type=str, default="all",
                       help="Layer range for optimization (e.g., '8-24' or '10,15,20' or 'all'). Default: all (uses all model layers)")
    parser.add_argument("--optimize-metric", type=str, choices=["accuracy", "f1", "precision", "recall", "auc"],
                       default="f1", help="Metric to optimize for. Default: f1")
    parser.add_argument("--optimize-max-combinations", type=int, default=100,
                       help="Maximum number of hyperparameter combinations to test. Default: 100")
    parser.add_argument("--auto-optimize", action="store_true",
                       help="Automatically enable optimization when layer is not specified or is -1")
    
    # Dataset validation arguments
    parser.add_argument("--allow-small-dataset", action="store_true",
                       help="Allow training with datasets smaller than 4 samples (may cause training issues)")
    
    # Detection handling arguments
    parser.add_argument("--detection-action", type=str, 
                       choices=["pass_through", "replace_with_placeholder", "regenerate_until_safe"],
                       default="pass_through",
                       help="Action to take when problematic content is detected (default: pass_through)")
    parser.add_argument("--placeholder-message", type=str, default=None,
                       help="Custom placeholder message for detected content (if not specified, uses default)")
    parser.add_argument("--max-regeneration-attempts", type=int, default=3,
                       help="Maximum attempts to regenerate safe content (default: 3)")
    parser.add_argument("--detection-threshold", type=float, default=0.6,
                       help="Threshold for classification (higher = more strict detection) (default: 0.6)")
    parser.add_argument("--log-detections", action="store_true",
                       help="Enable logging of detection events")
    
    # Steering mode arguments
    parser.add_argument("--steering-mode", action="store_true",
                       help="Enable steering mode (uses CAA vectors instead of classification)")
    parser.add_argument("--steering-strength", type=float, default=1.0,
                       help="Steering vector strength multiplier (default: 1.0)")
    parser.add_argument("--save-steering-vector", type=str, default=None,
                       help="Path to save the computed steering vector")
    parser.add_argument("--load-steering-vector", type=str, default=None,
                       help="Path to load a pre-computed steering vector")
    
    # Training/Inference mode arguments
    parser.add_argument("--train-only", action="store_true",
                       help="Training-only mode: train classifiers/vectors and save them, skip inference")
    parser.add_argument("--inference-only", action="store_true", 
                       help="Inference-only mode: load pre-trained classifiers/vectors and use for monitoring/steering")
    parser.add_argument("--save-classifier", type=str, default=None,
                       help="Path to save trained classifier(s). In multi-layer mode, saves one file per layer with layer suffix")
    parser.add_argument("--load-classifier", type=str, default=None,
                       help="Path to load pre-trained classifier(s). In multi-layer mode, expects files with layer suffix")
    parser.add_argument("--classifier-dir", type=str, default="./models",
                       help="Directory for saving/loading classifiers and vectors (default: ./models)")
    
    return parser


def parse_layers_from_arg(layer_arg: str, model=None) -> List[int]:
    """
    Parse layer argument into list of integers.
    
    Args:
        layer_arg: String like "15", "14-16", "14,15,16", or "-1" (for auto-optimization)
        model: Model object (needed for determining available layers)
        
    Returns:
        List of layer indices
    """
    # Handle special cases
    if layer_arg == "-1":
        # Signal for auto-optimization - return single layer list
        return [-1]
    
    # Use existing parse_layer_range logic
    layers = parse_layer_range(layer_arg, model)
    if layers is None:
        # "all" case - would need model to determine actual layers
        # For now, return a reasonable default range
        return list(range(8, 25))  # Common transformer range
    
    return layers


def parse_layer_range(layer_range_str: str, model=None) -> Optional[List[int]]:
    """
    Parse layer range string into list of integers.
    
    Args:
        layer_range_str: String like "8-24", "10,15,20", or "all"
        model: Model object (needed for "all" option)
        
    Returns:
        List of layer indices, or None if "all" (will be auto-detected later)
    """
    if layer_range_str.lower() == "all":
        # Return None to signal auto-detection
        return None
    elif '-' in layer_range_str:
        # Range format: "8-24"
        start, end = map(int, layer_range_str.split('-'))
        return list(range(start, end + 1))
    elif ',' in layer_range_str:
        # Comma-separated format: "10,15,20"
        return [int(x.strip()) for x in layer_range_str.split(',')]
    else:
        # Single layer
        return [int(layer_range_str)]


def aggregate_token_scores(token_scores: List[float], method: str) -> float:
    """
    Aggregate token scores using the specified method.
    
    Args:
        token_scores: List of token scores (probabilities)
        method: Aggregation method ("average", "final", "first", "max", "min")
        
    Returns:
        Aggregated score
    """
    if not token_scores:
        return 0.5
    
    if method == "average":
        return sum(token_scores) / len(token_scores)
    elif method == "final":
        return token_scores[-1]
    elif method == "first":
        return token_scores[0]
    elif method == "max":
        return max(token_scores)
    elif method == "min":
        return min(token_scores)
    else:
        # Default to average if unknown method
        return sum(token_scores) / len(token_scores)
