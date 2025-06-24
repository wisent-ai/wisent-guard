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
  # Traditional task evaluation
  python -m wisent_guard tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks hellaswag,mmlu --layer 10 --model meta-llama/Llama-3.1-8B --shots 5
  
  # Generate synthetic contrastive pairs
  python -m wisent_guard generate-pairs --trait "The model should refuse harmful requests politely" --output ./my_pairs.json
  python -m wisent_guard generate-pairs --trait "The model should be helpful and honest" --num-pairs 50 --output ./helpful_pairs.json
  
  # Use synthetic pairs for training and testing
  python -m wisent_guard synthetic --trait "The model should refuse harmful requests" --steering-method KSteering
  python -m wisent_guard synthetic --pairs-file ./my_pairs.json --steering-method CAA --steering-strength 1.5
  
  # File-based evaluation
  python -m wisent_guard tasks data.csv --from-csv --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks data.json --from-json --model meta-llama/Llama-3.1-8B
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Tasks subcommand (existing functionality)
    tasks_parser = subparsers.add_parser("tasks", help="Run evaluation tasks")
    setup_tasks_parser(tasks_parser)
    
    # Generate-pairs subcommand
    generate_parser = subparsers.add_parser("generate-pairs", help="Generate synthetic contrastive pairs")
    setup_generate_pairs_parser(generate_parser)
    
    # Synthetic subcommand (generate + train + test in one go)
    synthetic_parser = subparsers.add_parser("synthetic", help="Generate synthetic contrastive pairs and run full pipeline")
    setup_synthetic_parser(synthetic_parser)
    
    # Add test-nonsense subcommand
    nonsense_parser = subparsers.add_parser('test-nonsense', help='Test nonsense detection on sample text')
    nonsense_parser.add_argument("text", type=str, nargs='?',
                                 help="Text to analyze (if not provided, will use interactive mode)")
    nonsense_parser.add_argument("--max-word-length", type=int, default=20,
                                 help="Maximum reasonable word length (default: 20)")
    nonsense_parser.add_argument("--repetition-threshold", type=float, default=0.7,
                                 help="Threshold for repetitive content detection (0-1, default: 0.7)")
    nonsense_parser.add_argument("--gibberish-threshold", type=float, default=0.3,
                                 help="Threshold for gibberish word detection (0-1, default: 0.3)")
    nonsense_parser.add_argument("--disable-dictionary-check", action="store_true",
                                 help="Disable dictionary-based word validation")
    nonsense_parser.add_argument("--verbose", action="store_true",
                                 help="Show detailed analysis")
    nonsense_parser.add_argument("--examples", action="store_true",
                                 help="Test with built-in example texts")
    
    return parser


def setup_tasks_parser(parser):
    """Set up the tasks subcommand parser."""
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
                       help="Strength of steering vector application (default: 1.0)")
    
    # Steering method selection
    parser.add_argument("--steering-method", type=str, default="CAA",
                        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
                        help="Steering method to use")
    
    # HPR-specific parameters
    parser.add_argument("--hpr-beta", type=float, default=1.0,
                        help="Beta parameter for HPR method")
    
    # DAC-specific parameters
    parser.add_argument("--dac-dynamic-control", action="store_true",
                        help="Enable dynamic control for DAC method")
    parser.add_argument("--dac-entropy-threshold", type=float, default=1.0,
                        help="Entropy threshold for DAC dynamic control")
    
    # BiPO-specific parameters
    parser.add_argument("--bipo-beta", type=float, default=0.1,
                        help="Beta parameter for BiPO method")
    parser.add_argument("--bipo-learning-rate", type=float, default=5e-4,
                        help="Learning rate for BiPO method")
    parser.add_argument("--bipo-epochs", type=int, default=100,
                        help="Number of epochs for BiPO training")
    
    # K-Steering-specific parameters
    parser.add_argument("--ksteering-num-labels", type=int, default=6,
                        help="Number of labels for K-steering classifier")
    parser.add_argument("--ksteering-hidden-dim", type=int, default=512,
                        help="Hidden dimension for K-steering classifier")
    parser.add_argument("--ksteering-learning-rate", type=float, default=1e-3,
                        help="Learning rate for K-steering classifier training")
    parser.add_argument("--ksteering-classifier-epochs", type=int, default=100,
                        help="Number of epochs for K-steering classifier training")
    parser.add_argument("--ksteering-target-labels", type=str, default="0",
                        help="Comma-separated target label indices for K-steering (e.g., '0,1,2')")
    parser.add_argument("--ksteering-avoid-labels", type=str, default="",
                        help="Comma-separated avoid label indices for K-steering (e.g., '3,4,5')")
    parser.add_argument("--ksteering-alpha", type=float, default=50.0,
                        help="Alpha parameter (step size) for K-steering")
    
    # Token steering arguments
    parser.add_argument("--enable-token-steering", action="store_true",
                        help="Enable token-level steering control")
    parser.add_argument("--token-steering-strategy", type=str, default="last_only",
                        choices=["last_only", "first_only", "all_equal", "exponential_decay", 
                                "exponential_growth", "linear_decay", "linear_growth", "custom"],
                        help="Token steering strategy (default: last_only)")
    parser.add_argument("--token-decay-rate", type=float, default=0.5,
                        help="Decay rate for exponential token steering strategies (0-1, default: 0.5)")
    parser.add_argument("--token-min-strength", type=float, default=0.1,
                        help="Minimum steering strength for token strategies (default: 0.1)")
    parser.add_argument("--token-max-strength", type=float, default=1.0,
                        help="Maximum steering strength for token strategies (default: 1.0)")
    parser.add_argument("--token-apply-to-prompt", action="store_true",
                        help="Apply steering to prompt tokens as well as generated tokens")
    parser.add_argument("--token-prompt-strength-multiplier", type=float, default=0.1,
                        help="Strength multiplier for prompt tokens (default: 0.1)")
    
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
    
    # Prompt construction and token targeting strategy arguments
    parser.add_argument("--prompt-construction-strategy", type=str,
                       choices=["multiple_choice", "role_playing", "direct_completion", "instruction_following"],
                       default="multiple_choice",
                       help="Strategy for constructing prompts from question-answer pairs (default: multiple_choice)")
    parser.add_argument("--token-targeting-strategy", type=str, 
                       choices=["choice_token", "continuation_token", "last_token", "first_token", "mean_pooling", "max_pooling"],
                       default="choice_token",
                       help="Strategy for targeting tokens during activation extraction (default: choice_token)")
    
    # Normalization options
    parser.add_argument("--normalize-mode", action="store_true",
                        help="Enable normalization mode (legacy flag)")
    parser.add_argument("--normalization-method", type=str, default="none",
                        choices=["none", "l2_unit", "cross_behavior", "layer_wise_mean"],
                        help="Vector normalization method to apply")
    parser.add_argument("--target-norm", type=float, default=None,
                        help="Target norm for certain normalization methods")
    
    # Nonsense detection options
    parser.add_argument("--enable-nonsense-detection", action="store_true",
                        help="Enable nonsense detection to stop lobotomized responses")
    parser.add_argument("--max-word-length", type=int, default=20,
                        help="Maximum reasonable word length for nonsense detection (default: 20)")
    parser.add_argument("--repetition-threshold", type=float, default=0.7,
                        help="Threshold for repetitive content detection (0-1, default: 0.7)")
    parser.add_argument("--gibberish-threshold", type=float, default=0.3,
                        help="Threshold for gibberish word detection (0-1, default: 0.3)")
    parser.add_argument("--disable-dictionary-check", action="store_true",
                        help="Disable dictionary-based word validation (faster but less accurate)")
    parser.add_argument("--nonsense-action", type=str, default="regenerate",
                        choices=["regenerate", "stop", "flag"],
                        help="Action when nonsense is detected: regenerate, stop generation, or flag for review")
    
    parser.add_argument("--save-steering-vector", type=str, default=None,
                        help="Path to save the computed steering vector")
    parser.add_argument("--load-steering-vector", type=str, default=None,
                        help="Path to load a pre-computed steering vector")


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


def setup_generate_pairs_parser(parser):
    """Set up the generate-pairs subcommand parser."""
    parser.add_argument("--trait", type=str, required=True,
                        help="Natural language description of the desired trait or behavior")
    parser.add_argument("--num-pairs", type=int, default=30,
                        help="Number of contrastive pairs to generate (default: 30)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path for the generated pairs (JSON format)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path to use for generation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                        help="Similarity threshold for deduplication (0-1, higher = more strict)")


def setup_synthetic_parser(parser):
    """Set up the synthetic subcommand parser."""
    # Either generate new pairs or load existing ones
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trait", type=str,
                       help="Natural language description of the desired trait or behavior (generates new pairs)")
    group.add_argument("--pairs-file", type=str,
                       help="Path to existing JSON file with contrastive pairs")
    
    # Generation parameters (only used if --trait is specified)
    parser.add_argument("--num-pairs", type=int, default=30,
                        help="Number of contrastive pairs to generate (default: 30, only used with --trait)")
    parser.add_argument("--save-pairs", type=str, default=None,
                        help="Save generated pairs to this file (optional, only used with --trait)")
    
    # Model and device
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on")
    
    # Training/evaluation parameters
    parser.add_argument("--layer", type=str, default="15",
                        help="Layer(s) to extract activations from")
    parser.add_argument("--steering-method", type=str, default="CAA",
                        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
                        help="Steering method to use")
    parser.add_argument("--steering-strength", type=float, default=1.0,
                        help="Strength of steering vector application")
    parser.add_argument("--test-questions", type=int, default=5,
                        help="Number of test questions to generate for evaluation")
    
    # Output
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    # K-Steering specific parameters
    parser.add_argument("--ksteering-target-labels", type=str, default="0",
                        help="Comma-separated target label indices for K-steering")
    parser.add_argument("--ksteering-avoid-labels", type=str, default="",
                        help="Comma-separated avoid label indices for K-steering")
    parser.add_argument("--ksteering-alpha", type=float, default=50.0,
                        help="Alpha parameter for K-steering")
    
    # Nonsense detection options
    parser.add_argument("--enable-nonsense-detection", action="store_true",
                        help="Enable nonsense detection to stop lobotomized responses")
    parser.add_argument("--max-word-length", type=int, default=20,
                        help="Maximum reasonable word length for nonsense detection (default: 20)")
    parser.add_argument("--repetition-threshold", type=float, default=0.7,
                        help="Threshold for repetitive content detection (0-1, default: 0.7)")
    parser.add_argument("--gibberish-threshold", type=float, default=0.3,
                        help="Threshold for gibberish word detection (0-1, default: 0.3)")
    parser.add_argument("--disable-dictionary-check", action="store_true",
                        help="Disable dictionary-based word validation (faster but less accurate)")
    parser.add_argument("--nonsense-action", type=str, default="regenerate",
                        choices=["regenerate", "stop", "flag"],
                        help="Action when nonsense is detected: regenerate, stop generation, or flag for review")
