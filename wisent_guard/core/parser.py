"""
Command-line argument parser for wisent-guard.
"""

import argparse
from typing import List, Optional


def setup_parser() -> argparse.ArgumentParser:
    """Set up the main CLI parser with subcommands."""
    parser = argparse.ArgumentParser(description="Wisent-Guard: Advanced AI Safety and Alignment Toolkit")

    # Global arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Tasks command (main evaluation pipeline)
    tasks_parser = subparsers.add_parser("tasks", help="Run evaluation tasks")
    setup_tasks_parser(tasks_parser)

    # Generate pairs command
    generate_parser = subparsers.add_parser("generate-pairs", help="Generate synthetic contrastive pairs")
    setup_generate_pairs_parser(generate_parser)

    # Synthetic command (generate + train + test)
    synthetic_parser = subparsers.add_parser("synthetic", help="Run synthetic contrastive pair pipeline")
    setup_synthetic_parser(synthetic_parser)

    # Test nonsense detection command
    test_nonsense_parser = subparsers.add_parser("test-nonsense", help="Test nonsense detection system")
    setup_test_nonsense_parser(test_nonsense_parser)

    # Monitor command for performance monitoring
    monitor_parser = subparsers.add_parser("monitor", help="Performance monitoring and system information")
    setup_monitor_parser(monitor_parser)

    # Agent command for autonomous agent interaction
    agent_parser = subparsers.add_parser("agent", help="Interact with autonomous agent")
    setup_agent_parser(agent_parser)

    # Model configuration command for managing optimal parameters
    model_config_parser = subparsers.add_parser("model-config", help="Manage model-specific optimal parameters")
    setup_model_config_parser(model_config_parser)

    # Configure model command for setting up new/unsupported models
    configure_model_parser = subparsers.add_parser(
        "configure-model", help="Configure tokens and layer access for unsupported models"
    )
    setup_configure_model_parser(configure_model_parser)

    # Classification optimization command for finding optimal classification parameters
    classification_optimizer_parser = subparsers.add_parser(
        "optimize-classification", help="Optimize classification parameters across all tasks"
    )
    setup_classification_optimizer_parser(classification_optimizer_parser)

    # Steering optimization command for finding optimal steering parameters
    steering_optimizer_parser = subparsers.add_parser(
        "optimize-steering", help="Optimize steering parameters for different methods"
    )
    setup_steering_optimizer_parser(steering_optimizer_parser)

    # Sample size optimization command for finding optimal training sample sizes
    sample_size_optimizer_parser = subparsers.add_parser(
        "optimize-sample-size", help="Find optimal training sample size for classifiers"
    )
    setup_sample_size_optimizer_parser(sample_size_optimizer_parser)

    # Full optimization command that runs both classification and sample size optimization
    full_optimizer_parser = subparsers.add_parser(
        "full-optimize", help="Run full optimization: classification parameters then sample size"
    )
    setup_full_optimizer_parser(full_optimizer_parser)

    # Generate vector command for creating steering vectors without tasks
    generate_vector_parser = subparsers.add_parser(
        "generate-vector", help="Generate steering vectors from contrastive pairs (file or description)"
    )
    setup_generate_vector_parser(generate_vector_parser)

    # Multi-vector steering command for combining multiple vectors at inference time
    multi_steer_parser = subparsers.add_parser(
        "multi-steer", help="Combine multiple steering vectors dynamically at inference time"
    )
    setup_multi_steer_parser(multi_steer_parser)

    # Single-prompt evaluation command for real-time steering assessment
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate single prompt with steering vector and return quality scores"
    )
    setup_evaluate_parser(evaluate_parser)

    return parser


def setup_tasks_parser(parser):
    """Set up the tasks subcommand parser."""

    # Task listing options (mutually exclusive with task execution)
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all 37 available benchmark tasks organized by priority (excludes 28 known problematic benchmarks)",
    )
    list_group.add_argument(
        "--task-info", type=str, metavar="TASK_NAME", help="Show detailed information about a specific task"
    )
    list_group.add_argument("--all", action="store_true", help="Run all 37 available benchmarks automatically")

    # Task execution argument (optional when using listing commands or --all)
    parser.add_argument(
        "task_names",
        nargs="?",
        help="Comma-separated list of available task names (37 working benchmarks), or path to CSV/JSON file with --from-csv/--from-json (not needed with --all)",
    )

    # Skills/risks based task selection
    parser.add_argument(
        "--skills", type=str, nargs="+", help="Select tasks by skill categories (e.g., coding, mathematics, reasoning)"
    )
    parser.add_argument(
        "--risks",
        type=str,
        nargs="+",
        help="Select tasks by risk categories (e.g., harmfulness, toxicity, hallucination)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to randomly select from matched tasks (default: all)",
    )
    parser.add_argument(
        "--min-quality-score",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Minimum quality score for tasks when using --skills/--risks (default: 2)",
    )
    parser.add_argument(
        "--task-seed", type=int, default=None, help="Random seed for task selection (for reproducibility)"
    )

    # Mixed sampling from multiple benchmarks
    parser.add_argument(
        "--tag",
        type=str,
        nargs="+",
        help="Sample randomly from all benchmarks with these tags (e.g., --tag coding). Creates a mixed dataset from multiple benchmarks.",
    )
    parser.add_argument(
        "--mixed-samples",
        type=int,
        default=1000,
        help="Total number of samples to collect when using --tag (default: 1000)",
    )
    parser.add_argument(
        "--tag-mode",
        type=str,
        choices=["any", "all"],
        default="any",
        help="Whether benchmarks must have ANY or ALL specified tags (default: any)",
    )

    # Cross-benchmark evaluation
    parser.add_argument(
        "--train-task", type=str, help="Task/benchmark to train on (can be a task name or --tag for mixed)"
    )
    parser.add_argument(
        "--eval-task", type=str, help="Task/benchmark to evaluate on (can be a task name or --tag for mixed)"
    )
    parser.add_argument(
        "--train-tag", type=str, nargs="+", help="Tags for training data when using cross-benchmark evaluation"
    )
    parser.add_argument(
        "--eval-tag", type=str, nargs="+", help="Tags for evaluation data when using cross-benchmark evaluation"
    )
    parser.add_argument(
        "--cross-benchmark",
        action="store_true",
        help="Enable cross-benchmark evaluation mode (train on one, eval on another)",
    )

    # Synthetic pair generation
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic contrastive pairs from a trait description"
    )
    parser.add_argument(
        "--trait",
        type=str,
        help="Natural language description of desired model behavior (e.g., 'hallucinates less', 'more factual', 'less verbose')",
    )
    parser.add_argument(
        "--num-synthetic-pairs", type=int, default=30, help="Number of synthetic pairs to generate (default: 30)"
    )
    parser.add_argument("--save-synthetic", type=str, help="Path to save generated synthetic pairs as JSON")
    parser.add_argument(
        "--load-synthetic", type=str, help="Path to load previously generated synthetic pairs from JSON"
    )

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument(
        "--layer",
        type=str,
        default="15",
        help="Layer(s) to extract activations from. Can be a single layer (15), range (14-16), or comma-separated list (14,15,16)",
    )
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents per task")
    parser.add_argument(
        "--training-limit",
        type=int,
        default=None,
        help="Limit number of training documents (overrides limit for training)",
    )
    parser.add_argument(
        "--testing-limit",
        type=int,
        default=None,
        help="Limit number of testing documents (overrides limit for testing)",
    )
    parser.add_argument("--output", type=str, default="./results", help="Output directory for results")
    parser.add_argument(
        "--classifier-type", type=str, choices=["logistic", "mlp"], default="logistic", help="Type of classifier"
    )
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Maximum new tokens for generation")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--token-aggregation",
        type=str,
        choices=["average", "final", "first", "max", "min"],
        default="average",
        help="How to aggregate token scores for classification",
    )
    parser.add_argument(
        "--ground-truth-method",
        type=str,
        choices=[
            "none",
            "exact_match",
            "substring_match",
            "user_specified",
            "interactive",
            "manual_review",
            "good",
            "lm-eval-harness",
        ],
        default="lm-eval-harness",
        help="Method for ground truth evaluation. 'lm-eval-harness' uses lm-eval-harness tasks for evaluation (default for most tasks), 'none' skips evaluation, 'exact_match' and 'substring_match' are problematic for free-form generation, 'user_specified' allows manual labeling, 'interactive' prompts for y/n labeling, 'manual_review' marks for review, 'good' marks everything as truthful (for debugging)",
    )
    parser.add_argument(
        "--user-labels",
        type=str,
        nargs="*",
        default=None,
        help="User-specified ground truth labels for responses ('truthful' or 'hallucination'). Used with --ground-truth-method user_specified",
    )

    # File input arguments
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Load task data from CSV file. Requires columns: question, correct_answer, incorrect_answer",
    )
    parser.add_argument(
        "--from-json",
        action="store_true",
        help="Load task data from JSON file. Expected format: list of objects with question, correct_answer, incorrect_answer",
    )
    parser.add_argument(
        "--question-col", type=str, default="question", help="Column name for questions in CSV file (default: question)"
    )
    parser.add_argument(
        "--correct-col",
        type=str,
        default="correct_answer",
        help="Column name for correct answers in CSV file (default: correct_answer)",
    )
    parser.add_argument(
        "--incorrect-col",
        type=str,
        default="incorrect_answer",
        help="Column name for incorrect answers in CSV file (default: incorrect_answer)",
    )

    # Optimization arguments
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable hyperparameter optimization. When enabled, will find optimal layer, threshold, and aggregation method",
    )
    parser.add_argument(
        "--optimize-layers",
        type=str,
        default="all",
        help="Layer range for optimization (e.g., '8-24' or '10,15,20' or 'all'). Default: all (uses all model layers)",
    )
    parser.add_argument(
        "--optimize-metric",
        type=str,
        choices=["accuracy", "f1", "precision", "recall", "auc"],
        default="f1",
        help="Metric to optimize for. Default: f1",
    )
    parser.add_argument(
        "--optimize-max-combinations",
        type=int,
        default=100,
        help="Maximum number of hyperparameter combinations to test. Default: 100",
    )
    parser.add_argument(
        "--auto-optimize",
        action="store_true",
        help="Automatically enable optimization when layer is not specified or is -1",
    )

    # Dataset validation arguments
    parser.add_argument(
        "--allow-small-dataset",
        action="store_true",
        help="Allow training with datasets smaller than 4 samples (may cause training issues)",
    )

    # Detection handling arguments
    parser.add_argument(
        "--detection-action",
        type=str,
        choices=["pass_through", "replace_with_placeholder", "regenerate_until_safe"],
        default="pass_through",
        help="Action to take when problematic content is detected (default: pass_through)",
    )
    parser.add_argument(
        "--placeholder-message",
        type=str,
        default=None,
        help="Custom placeholder message for detected content (if not specified, uses default)",
    )
    parser.add_argument(
        "--max-regeneration-attempts",
        type=int,
        default=3,
        help="Maximum attempts to regenerate safe content (default: 3)",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.6,
        help="Threshold for classification (higher = more strict detection) (default: 0.6)",
    )
    parser.add_argument("--log-detections", action="store_true", help="Enable logging of detection events")

    # Code execution security arguments
    parser.add_argument(
        "--trust-code-execution",
        action="store_true",
        help="⚠️  UNSAFE: Allow code execution without Docker in trusted sandbox environments (e.g., RunPod containers). Use only in secure, isolated environments!",
    )

    # Steering mode arguments
    parser.add_argument(
        "--steering-mode", action="store_true", help="Enable steering mode (uses CAA vectors instead of classification)"
    )
    parser.add_argument(
        "--steering-strength", type=float, default=1.0, help="Strength of steering vector application (default: 1.0)"
    )

    # Steering method selection
    parser.add_argument(
        "--steering-method",
        type=str,
        default="CAA",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use",
    )

    # Steering output mode selection
    parser.add_argument(
        "--output-mode",
        type=str,
        default="both",
        choices=["likelihoods", "responses", "both"],
        help="Type of comparison to show: 'likelihoods' for log-likelihood comparison only, 'responses' for response generation only, 'both' for both (default: both)",
    )

    # HPR-specific parameters
    parser.add_argument("--hpr-beta", type=float, default=1.0, help="Beta parameter for HPR method")

    # DAC-specific parameters
    parser.add_argument("--dac-dynamic-control", action="store_true", help="Enable dynamic control for DAC method")
    parser.add_argument(
        "--dac-entropy-threshold", type=float, default=1.0, help="Entropy threshold for DAC dynamic control"
    )

    # BiPO-specific parameters
    parser.add_argument("--bipo-beta", type=float, default=0.1, help="Beta parameter for BiPO method")
    parser.add_argument("--bipo-learning-rate", type=float, default=5e-4, help="Learning rate for BiPO method")
    parser.add_argument("--bipo-epochs", type=int, default=100, help="Number of epochs for BiPO training")

    # K-Steering-specific parameters
    parser.add_argument(
        "--ksteering-num-labels", type=int, default=6, help="Number of labels for K-steering classifier"
    )
    parser.add_argument(
        "--ksteering-hidden-dim", type=int, default=512, help="Hidden dimension for K-steering classifier"
    )
    parser.add_argument(
        "--ksteering-learning-rate", type=float, default=1e-3, help="Learning rate for K-steering classifier training"
    )
    parser.add_argument(
        "--ksteering-classifier-epochs",
        type=int,
        default=100,
        help="Number of epochs for K-steering classifier training",
    )
    parser.add_argument(
        "--ksteering-target-labels",
        type=str,
        default="0",
        help="Comma-separated target label indices for K-steering (e.g., '0,1,2')",
    )
    parser.add_argument(
        "--ksteering-avoid-labels",
        type=str,
        default="",
        help="Comma-separated avoid label indices for K-steering (e.g., '3,4,5')",
    )
    parser.add_argument(
        "--ksteering-alpha", type=float, default=50.0, help="Alpha parameter (step size) for K-steering"
    )

    # Token steering arguments
    parser.add_argument("--enable-token-steering", action="store_true", help="Enable token-level steering control")
    parser.add_argument(
        "--token-steering-strategy",
        type=str,
        default="last_only",
        choices=[
            "last_only",
            "first_only",
            "all_equal",
            "exponential_decay",
            "exponential_growth",
            "linear_decay",
            "linear_growth",
            "custom",
        ],
        help="Token steering strategy (default: last_only)",
    )
    parser.add_argument(
        "--token-decay-rate",
        type=float,
        default=0.5,
        help="Decay rate for exponential token steering strategies (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--token-min-strength",
        type=float,
        default=0.1,
        help="Minimum steering strength for token strategies (default: 0.1)",
    )
    parser.add_argument(
        "--token-max-strength",
        type=float,
        default=1.0,
        help="Maximum steering strength for token strategies (default: 1.0)",
    )
    parser.add_argument(
        "--token-apply-to-prompt",
        action="store_true",
        help="Apply steering to prompt tokens as well as generated tokens",
    )
    parser.add_argument(
        "--token-prompt-strength-multiplier",
        type=float,
        default=0.1,
        help="Strength multiplier for prompt tokens (default: 0.1)",
    )

    # Training/Inference mode arguments
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Training-only mode: train classifiers/vectors and save them, skip inference",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Inference-only mode: load pre-trained classifiers/vectors and use for monitoring/steering",
    )
    parser.add_argument(
        "--save-classifier",
        type=str,
        default=None,
        help="Path to save trained classifier(s). In multi-layer mode, saves one file per layer with layer suffix",
    )
    parser.add_argument(
        "--load-classifier",
        type=str,
        default=None,
        help="Path to load pre-trained classifier(s). In multi-layer mode, expects files with layer suffix",
    )
    parser.add_argument(
        "--classifier-dir",
        type=str,
        default="./models",
        help="Directory for saving/loading classifiers and vectors (default: ./models)",
    )

    # Prompt construction and token targeting strategy arguments
    parser.add_argument(
        "--prompt-construction-strategy",
        type=str,
        choices=["multiple_choice", "role_playing", "direct_completion", "instruction_following"],
        default="multiple_choice",
        help="Strategy for constructing prompts from question-answer pairs (default: multiple_choice)",
    )
    parser.add_argument(
        "--token-targeting-strategy",
        type=str,
        choices=["choice_token", "continuation_token", "last_token", "first_token", "mean_pooling", "max_pooling"],
        default="choice_token",
        help="Strategy for targeting tokens during activation extraction (default: choice_token)",
    )

    # Normalization options
    parser.add_argument("--normalize-mode", action="store_true", help="Enable normalization mode (legacy flag)")
    parser.add_argument(
        "--normalization-method",
        type=str,
        default="none",
        choices=["none", "l2_unit", "cross_behavior", "layer_wise_mean"],
        help="Vector normalization method to apply",
    )
    parser.add_argument("--target-norm", type=float, default=None, help="Target norm for certain normalization methods")

    # Nonsense detection options
    parser.add_argument(
        "--enable-nonsense-detection",
        action="store_true",
        help="Enable nonsense detection to stop lobotomized responses",
    )
    parser.add_argument(
        "--max-word-length",
        type=int,
        default=20,
        help="Maximum reasonable word length for nonsense detection (default: 20)",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=0.7,
        help="Threshold for repetitive content detection (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        default=0.3,
        help="Threshold for gibberish word detection (0-1, default: 0.3)",
    )
    parser.add_argument(
        "--disable-dictionary-check",
        action="store_true",
        help="Disable dictionary-based word validation (faster but less accurate)",
    )
    parser.add_argument(
        "--nonsense-action",
        type=str,
        default="regenerate",
        choices=["regenerate", "stop", "flag"],
        help="Action when nonsense is detected: regenerate, stop generation, or flag for review",
    )

    # Performance monitoring options
    parser.add_argument(
        "--enable-memory-tracking", action="store_true", help="Enable memory usage tracking and reporting"
    )
    parser.add_argument(
        "--enable-latency-tracking", action="store_true", help="Enable latency/timing tracking and reporting"
    )
    parser.add_argument(
        "--memory-sampling-interval", type=float, default=0.1, help="Memory sampling interval in seconds (default: 0.1)"
    )
    parser.add_argument("--track-gpu-memory", action="store_true", help="Track GPU memory usage (requires CUDA)")
    parser.add_argument(
        "--detailed-performance-report",
        action="store_true",
        help="Generate detailed performance report with all metrics",
    )
    parser.add_argument("--export-performance-csv", type=str, default=None, help="Export performance data to CSV file")
    parser.add_argument(
        "--show-memory-usage", action="store_true", help="Show current memory usage without full tracking"
    )
    parser.add_argument("--show-timing-summary", action="store_true", help="Show timing summary after evaluation")

    # Test-time activation saving/loading options
    parser.add_argument(
        "--save-test-activations", type=str, default=None, help="Save test activations to file for future use"
    )
    parser.add_argument(
        "--load-test-activations", type=str, default=None, help="Load test activations from file instead of computing"
    )

    # Priority-aware benchmark selection options
    parser.add_argument(
        "--priority",
        type=str,
        default="all",
        choices=["all", "high", "medium", "low"],
        help="Priority level for benchmark selection (default: all)",
    )
    parser.add_argument(
        "--fast-only", action="store_true", help="Only use fast benchmarks (high priority, < 13.5s loading time)"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Time budget in minutes for benchmark selection (auto-selects fast benchmarks)",
    )
    parser.add_argument(
        "--max-benchmarks",
        type=int,
        default=None,
        help="Maximum number of benchmarks to select (combines with priority filtering)",
    )
    parser.add_argument(
        "--smart-selection", action="store_true", help="Use smart benchmark selection based on relevance and priority"
    )
    parser.add_argument(
        "--prefer-fast",
        action="store_true",
        help="Prefer fast benchmarks in selection when multiple options are available",
    )

    parser.add_argument(
        "--save-steering-vector", type=str, default=None, help="Path to save the computed steering vector"
    )
    parser.add_argument(
        "--load-steering-vector", type=str, default=None, help="Path to load a pre-computed steering vector"
    )

    # Additional output options
    parser.add_argument("--csv-output", type=str, default=None, help="Path to save results in CSV format")
    parser.add_argument("--evaluation-report", type=str, default=None, help="Path to save evaluation report")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing other tasks if one fails")

    # Benchmark caching arguments
    parser.add_argument(
        "--cache-benchmark",
        action="store_true",
        default=True,
        help="Cache the benchmark data locally for faster future access (default: True)",
    )
    parser.add_argument("--no-cache", dest="cache_benchmark", action="store_false", help="Disable benchmark caching")
    parser.add_argument(
        "--use-cached", action="store_true", default=True, help="Use cached benchmark data if available (default: True)"
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Force fresh download even if cached version exists"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./benchmark_cache",
        help="Directory to store cached benchmark data (default: ./benchmark_cache)",
    )
    parser.add_argument("--cache-status", action="store_true", help="Show cache status and exit")
    parser.add_argument("--cleanup-cache", type=int, metavar="DAYS", help="Clean up cache entries older than DAYS days")


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
        # "all" case - auto-detect model layers
        if model is not None:
            from .hyperparameter_optimizer import detect_model_layers

            total_layers = detect_model_layers(model)
            return list(range(total_layers))
        # If no model provided, we cannot determine layers - this should not happen
        raise ValueError("Cannot determine layer range without model instance")

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
    if "-" in layer_range_str:
        # Range format: "8-24"
        start, end = map(int, layer_range_str.split("-"))
        return list(range(start, end + 1))
    if "," in layer_range_str:
        # Comma-separated format: "10,15,20"
        return [int(x.strip()) for x in layer_range_str.split(",")]
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

    # Convert any tensor values to floats and filter out None values
    clean_scores = []
    for i, score in enumerate(token_scores):
        if score is None:
            raise ValueError(
                f"Token score at index {i} is None! This indicates a bug in the classifier output handling."
            )
        if hasattr(score, "item"):  # Handle tensors
            raise ValueError(
                f"Token score at index {i} is a tensor ({type(score)})! Expected float but got tensor: {score}"
            )
        if not isinstance(score, (int, float)):
            raise ValueError(
                f"Token score at index {i} has invalid type: {type(score)}. Expected float but got {type(score).__name__}: {score}"
            )
        clean_scores.append(float(score))

    if not clean_scores:
        return 0.5

    if method == "average":
        return sum(clean_scores) / len(clean_scores)
    if method == "final":
        return clean_scores[-1]
    if method == "first":
        return clean_scores[0]
    if method == "max":
        return max(clean_scores)
    if method == "min":
        return min(clean_scores)
    # Default to average if unknown method
    return sum(clean_scores) / len(clean_scores)


def setup_generate_pairs_parser(parser):
    """Set up the generate-pairs subcommand parser."""
    parser.add_argument(
        "--trait", type=str, required=True, help="Natural language description of the desired trait or behavior"
    )
    parser.add_argument(
        "--num-pairs", type=int, default=30, help="Number of contrastive pairs to generate (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output file path for the generated pairs (JSON format)"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path to use for generation"
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for deduplication (0-1, higher = more strict)",
    )
    parser.add_argument("--timing", action="store_true", help="Show detailed timing for each generation step")
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Number of parallel workers for generation (default: 4)"
    )


def setup_synthetic_parser(parser):
    """Set up the synthetic subcommand parser."""
    # Either generate new pairs or load existing ones
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--trait", type=str, help="Natural language description of the desired trait or behavior (generates new pairs)"
    )
    group.add_argument("--pairs-file", type=str, help="Path to existing JSON file with contrastive pairs")

    # Generation parameters (only used if --trait is specified)
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=30,
        help="Number of contrastive pairs to generate (default: 30, only used with --trait)",
    )
    parser.add_argument(
        "--save-pairs",
        type=str,
        default=None,
        help="Save generated pairs to this file (optional, only used with --trait)",
    )

    # Model and device
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")

    # Training/evaluation parameters
    parser.add_argument("--layer", type=str, default="15", help="Layer(s) to extract activations from")
    parser.add_argument(
        "--steering-method",
        type=str,
        default="CAA",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use",
    )
    parser.add_argument("--steering-strength", type=float, default=1.0, help="Strength of steering vector application")
    parser.add_argument(
        "--test-questions", type=int, default=5, help="Number of test questions to generate for evaluation"
    )

    # Output
    parser.add_argument("--output", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # K-Steering specific parameters
    parser.add_argument(
        "--ksteering-target-labels", type=str, default="0", help="Comma-separated target label indices for K-steering"
    )
    parser.add_argument(
        "--ksteering-avoid-labels", type=str, default="", help="Comma-separated avoid label indices for K-steering"
    )
    parser.add_argument("--ksteering-alpha", type=float, default=50.0, help="Alpha parameter for K-steering")

    # Nonsense detection options
    parser.add_argument(
        "--enable-nonsense-detection",
        action="store_true",
        help="Enable nonsense detection to stop lobotomized responses",
    )
    parser.add_argument(
        "--max-word-length",
        type=int,
        default=20,
        help="Maximum reasonable word length for nonsense detection (default: 20)",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=0.7,
        help="Threshold for repetitive content detection (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        default=0.3,
        help="Threshold for gibberish word detection (0-1, default: 0.3)",
    )
    parser.add_argument(
        "--disable-dictionary-check",
        action="store_true",
        help="Disable dictionary-based word validation (faster but less accurate)",
    )
    parser.add_argument(
        "--nonsense-action",
        type=str,
        default="regenerate",
        choices=["regenerate", "stop", "flag"],
        help="Action when nonsense is detected: regenerate, stop generation, or flag for review",
    )


def setup_test_nonsense_parser(parser):
    """Set up the test-nonsense subcommand parser."""
    parser.add_argument(
        "text", type=str, nargs="?", help="Text to analyze (if not provided, will use interactive mode)"
    )
    parser.add_argument("--max-word-length", type=int, default=20, help="Maximum reasonable word length (default: 20)")
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=0.7,
        help="Threshold for repetitive content detection (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        default=0.3,
        help="Threshold for gibberish word detection (0-1, default: 0.3)",
    )
    parser.add_argument(
        "--disable-dictionary-check", action="store_true", help="Disable dictionary-based word validation"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed analysis")
    parser.add_argument("--examples", action="store_true", help="Test with built-in example texts")


def setup_monitor_parser(parser):
    """Set up the monitor subcommand parser."""
    parser.add_argument("--memory-info", action="store_true", help="Show current memory usage information")
    parser.add_argument("--system-info", action="store_true", help="Show system information and capabilities")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--test-gpu", action="store_true", help="Test GPU availability and memory")
    parser.add_argument("--continuous", action="store_true", help="Continuous monitoring mode (Ctrl+C to stop)")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds (default: 1.0)")
    parser.add_argument("--export-csv", type=str, default=None, help="Export monitoring data to CSV file")
    parser.add_argument(
        "--duration", type=int, default=60, help="Duration for continuous monitoring in seconds (default: 60)"
    )
    parser.add_argument("--track-gpu", action="store_true", help="Include GPU monitoring (requires CUDA)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed monitoring information")


def setup_agent_parser(parser):
    """Set up the agent subcommand parser."""
    parser.add_argument("prompt", type=str, help="Prompt to send to the autonomous agent")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use")
    parser.add_argument("--layer", type=int, help="Layer to use (overrides parameter file)")
    parser.add_argument(
        "--quality-threshold", type=float, default=0.3, help="Quality threshold for classifiers (default: 0.3)"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=10.0,
        help="Time budget in minutes for creating classifiers (default: 10.0)",
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum improvement attempts (default: 3)")
    parser.add_argument(
        "--max-classifiers", type=int, default=None, help="Maximum classifiers to use (default: no limit)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Steering method arguments
    parser.add_argument(
        "--steering-method",
        type=str,
        default="CAA",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use (default: CAA)",
    )
    parser.add_argument(
        "--steering-strength", type=float, default=1.0, help="Strength of steering vector application (default: 1.0)"
    )
    parser.add_argument("--steering-mode", action="store_true", help="Enable steering mode")

    # Normalization parameters
    parser.add_argument("--normalize-mode", action="store_true", help="Enable normalization of steering vectors")
    parser.add_argument(
        "--normalization-method",
        type=str,
        default="none",
        choices=["none", "l2_unit", "l2_norm", "max_norm"],
        help="Normalization method for steering vectors (default: none)",
    )
    parser.add_argument("--target-norm", type=float, default=None, help="Target norm for steering vectors")

    # HPR (Householder Pseudo-Rotation) parameters
    parser.add_argument("--hpr-beta", type=float, default=1.0, help="Beta parameter for HPR steering (default: 1.0)")

    # DAC (Dynamic Activation Composition) parameters
    parser.add_argument("--dac-dynamic-control", action="store_true", help="Enable dynamic control for DAC steering")
    parser.add_argument(
        "--dac-entropy-threshold", type=float, default=1.0, help="Entropy threshold for DAC steering (default: 1.0)"
    )

    # BiPO (Bi-directional Preference Optimization) parameters
    parser.add_argument("--bipo-beta", type=float, default=0.1, help="Beta parameter for BiPO steering (default: 0.1)")
    parser.add_argument(
        "--bipo-learning-rate", type=float, default=5e-4, help="Learning rate for BiPO steering (default: 5e-4)"
    )
    parser.add_argument(
        "--bipo-epochs", type=int, default=100, help="Number of epochs for BiPO steering (default: 100)"
    )

    # KSteering parameters
    parser.add_argument(
        "--ksteering-num-labels", type=int, default=6, help="Number of labels for K-steering (default: 6)"
    )
    parser.add_argument(
        "--ksteering-hidden-dim", type=int, default=512, help="Hidden dimension for K-steering (default: 512)"
    )
    parser.add_argument(
        "--ksteering-learning-rate", type=float, default=1e-3, help="Learning rate for K-steering (default: 1e-3)"
    )
    parser.add_argument(
        "--ksteering-classifier-epochs", type=int, default=100, help="Classifier epochs for K-steering (default: 100)"
    )
    parser.add_argument(
        "--ksteering-target-labels",
        type=str,
        default="0",
        help="Target labels for K-steering (comma-separated, default: '0')",
    )
    parser.add_argument(
        "--ksteering-avoid-labels",
        type=str,
        default="",
        help="Avoid labels for K-steering (comma-separated, default: '')",
    )
    parser.add_argument(
        "--ksteering-alpha", type=float, default=50.0, help="Alpha parameter for K-steering (default: 50.0)"
    )

    # Quality Control System parameters
    parser.add_argument(
        "--enable-quality-control",
        action="store_true",
        default=True,
        help="Enable new quality control system (default: True)",
    )
    parser.add_argument(
        "--max-quality-attempts",
        type=int,
        default=5,
        help="Maximum attempts to achieve acceptable quality (default: 5)",
    )
    parser.add_argument(
        "--show-parameter-reasoning", action="store_true", help="Display model's reasoning for parameter choices"
    )


def setup_classification_optimizer_parser(parser):
    """Set up the classification-optimizer subcommand parser."""
    parser.add_argument("model", type=str, help="Model name or path to optimize")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum samples per task (default: 1000)")
    parser.add_argument(
        "--optimization-metric",
        type=str,
        default="f1",
        choices=["f1", "accuracy", "precision", "recall"],
        help="Metric to optimize (default: f1)",
    )
    parser.add_argument(
        "--max-time-per-task", type=float, default=15.0, help="Maximum time per task in minutes (default: 15.0)"
    )
    parser.add_argument(
        "--layer-range", type=str, default=None, help="Layer range to test (e.g., '10-20', if None uses all layers)"
    )
    parser.add_argument(
        "--aggregation-methods",
        type=str,
        nargs="+",
        default=["average", "final", "first", "max", "min"],
        help="Token aggregation methods to test",
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Detection thresholds to test",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--results-file", type=str, default=None, help="Custom file path for saving results")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to model config")
    parser.add_argument("--save-logs-json", type=str, default=None, help="Save detailed optimization logs to JSON file")
    parser.add_argument(
        "--save-classifiers",
        action="store_true",
        default=True,
        help="Save best classifiers for each task (default: True)",
    )
    parser.add_argument(
        "--no-save-classifiers",
        dest="save_classifiers",
        action="store_false",
        help="Don't save classifiers (overrides --save-classifiers)",
    )
    parser.add_argument(
        "--classifiers-dir",
        type=str,
        default=None,
        help="Directory to save classifiers (default: ./optimized_classifiers/model_name/)",
    )

    # Timing calibration options
    parser.add_argument(
        "--skip-timing-estimation", action="store_true", help="Skip timing estimation and proceed without time warnings"
    )
    parser.add_argument("--calibration-file", type=str, default=None, help="File to save/load calibration data")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Only run calibration and exit (saves to --calibration-file if provided)",
    )


def setup_configure_model_parser(parser):
    """Set up the configure-model subcommand parser."""
    parser.add_argument("model", type=str, help="Model name to configure")
    parser.add_argument("--force", action="store_true", help="Force reconfiguration even if model already has a config")


def setup_steering_optimizer_parser(parser):
    """Set up the steering-optimizer subcommand parser."""
    # Create subparsers for different steering optimization types
    steering_subparsers = parser.add_subparsers(dest="steering_action", help="Steering optimization actions")

    # Auto optimization subcommand (NEW - runs after classification optimization)
    auto_parser = steering_subparsers.add_parser(
        "auto", help="Automatically optimize steering based on classification config"
    )
    auto_parser.add_argument("model", type=str, help="Model name or path")
    auto_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to optimize (defaults to all classification-optimized tasks)",
    )
    auto_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        default=["CAA", "HPR"],
        help="Steering methods to test (default: CAA, HPR)",
    )
    auto_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")
    auto_parser.add_argument("--max-time", type=float, default=60.0, help="Maximum time in minutes (default: 60)")
    auto_parser.add_argument(
        "--strength-range",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Steering strengths to test (default: 0.5 1.0 1.5 2.0)",
    )
    auto_parser.add_argument(
        "--layer-range",
        type=str,
        default=None,
        help="Explicit layer range to search (e.g., '0-5' or '0,2,4'). If not specified, uses classification layer or defaults to 0-5",
    )

    # Method comparison subcommand
    method_parser = steering_subparsers.add_parser(
        "compare-methods", help="Compare different steering methods for a task"
    )
    method_parser.add_argument("model", type=str, help="Model name or path")
    method_parser.add_argument(
        "--task", type=str, default="truthfulqa_mc1", help="Task to optimize steering for (default: truthfulqa_mc1)"
    )
    method_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        default=["CAA", "HPR"],
        help="Steering methods to compare",
    )
    method_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")
    method_parser.add_argument(
        "--max-time", type=float, default=30.0, help="Maximum optimization time in minutes (default: 30.0)"
    )

    # Layer optimization subcommand
    layer_parser = steering_subparsers.add_parser("optimize-layer", help="Find optimal steering layer for a method")
    layer_parser.add_argument("model", type=str, help="Model name or path")
    layer_parser.add_argument(
        "--task", type=str, default="truthfulqa_mc1", help="Task to optimize for (default: truthfulqa_mc1)"
    )
    layer_parser.add_argument(
        "--method",
        type=str,
        default="CAA",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use (default: CAA)",
    )
    layer_parser.add_argument("--layer-range", type=str, default=None, help="Layer range to search (e.g., '10-20')")
    layer_parser.add_argument(
        "--strength", type=float, default=1.0, help="Fixed steering strength during layer search (default: 1.0)"
    )
    layer_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")

    # Strength optimization subcommand
    strength_parser = steering_subparsers.add_parser("optimize-strength", help="Find optimal steering strength")
    strength_parser.add_argument("model", type=str, help="Model name or path")
    strength_parser.add_argument(
        "--task", type=str, default="truthfulqa_mc1", help="Task to optimize for (default: truthfulqa_mc1)"
    )
    strength_parser.add_argument(
        "--method",
        type=str,
        default="CAA",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use (default: CAA)",
    )
    strength_parser.add_argument(
        "--layer", type=int, default=None, help="Steering layer to use (defaults to classification layer)"
    )
    strength_parser.add_argument(
        "--strength-range",
        type=float,
        nargs=2,
        default=[0.1, 2.0],
        help="Min and max strength to test (default: 0.1 2.0)",
    )
    strength_parser.add_argument(
        "--strength-steps", type=int, default=10, help="Number of strength values to test (default: 10)"
    )
    strength_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")

    # Comprehensive optimization subcommand
    comprehensive_parser = steering_subparsers.add_parser(
        "comprehensive", help="Run comprehensive steering optimization"
    )
    comprehensive_parser.add_argument("model", type=str, help="Model name or path")
    comprehensive_parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to optimize (defaults to classification-optimized tasks)",
    )
    comprehensive_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        default=["CAA", "HPR"],
        help="Steering methods to test",
    )
    comprehensive_parser.add_argument("--limit", type=int, default=100, help="Sample limit per task (default: 100)")
    comprehensive_parser.add_argument(
        "--max-time-per-task", type=float, default=20.0, help="Time limit per task in minutes (default: 20.0)"
    )
    comprehensive_parser.add_argument("--no-save", action="store_true", help="Don't save results to model config")

    # Common arguments for all steering optimization subcommands
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")


def setup_model_config_parser(parser):
    """Set up the model-config subcommand parser."""
    # Create subparsers for different model config actions
    config_subparsers = parser.add_subparsers(dest="config_action", help="Model configuration actions")

    # Save configuration subcommand
    save_parser = config_subparsers.add_parser("save", help="Save optimal parameters for a model")
    save_parser.add_argument("model", type=str, help="Model name or path")
    save_parser.add_argument("--classification-layer", type=int, required=True, help="Optimal layer for classification")
    save_parser.add_argument(
        "--steering-layer", type=int, default=None, help="Optimal layer for steering (defaults to classification layer)"
    )
    save_parser.add_argument(
        "--token-aggregation",
        type=str,
        default="average",
        choices=["average", "final", "first", "max", "min"],
        help="Token aggregation method",
    )
    save_parser.add_argument("--detection-threshold", type=float, default=0.6, help="Detection threshold")
    save_parser.add_argument(
        "--optimization-method", type=str, default="manual", help="How these parameters were determined"
    )
    save_parser.add_argument("--metrics", type=str, default=None, help="JSON string with optimization metrics")

    # List configurations subcommand
    list_parser = config_subparsers.add_parser("list", help="List all saved model configurations")
    list_parser.add_argument("--detailed", action="store_true", help="Show detailed configuration information")

    # Show configuration subcommand
    show_parser = config_subparsers.add_parser("show", help="Show configuration for a specific model")
    show_parser.add_argument("model", type=str, help="Model name or path")
    show_parser.add_argument("--task", type=str, default=None, help="Show task-specific overrides if available")

    # Remove configuration subcommand
    remove_parser = config_subparsers.add_parser("remove", help="Remove configuration for a model")
    remove_parser.add_argument("model", type=str, help="Model name or path")
    remove_parser.add_argument("--confirm", action="store_true", help="Confirm removal without prompting")

    # Test configuration subcommand
    test_parser = config_subparsers.add_parser("test", help="Test if saved configuration works")
    test_parser.add_argument("model", type=str, help="Model name or path")
    test_parser.add_argument(
        "--task", type=str, default="truthfulqa_mc1", help="Task to test with (default: truthfulqa_mc1)"
    )
    test_parser.add_argument("--limit", type=int, default=5, help="Number of samples to test with (default: 5)")
    test_parser.add_argument("--device", type=str, default=None, help="Device to run on")

    # Common arguments for all subcommands
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Custom directory for configuration files (default: ~/.wisent-guard/model_configs/)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")


def setup_sample_size_optimizer_parser(parser):
    """Set up the sample-size-optimizer subcommand parser."""
    parser.add_argument("model", type=str, help="Model name or path to optimize")
    parser.add_argument("--task", type=str, required=True, help="Task to optimize for (REQUIRED)")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to use (REQUIRED)")
    parser.add_argument(
        "--token-aggregation",
        type=str,
        required=True,
        choices=["average", "final", "first", "max", "min"],
        help="Token aggregation method (REQUIRED)",
    )

    # Classification-specific arguments
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Detection threshold for classification (default: 0.5)"
    )

    # Steering mode
    parser.add_argument("--steering-mode", action="store_true", help="Optimize for steering instead of classification")
    parser.add_argument(
        "--steering-method",
        type=str,
        default="CAA",
        choices=["CAA", "CAA_L2", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use (default: CAA)",
    )
    parser.add_argument("--steering-strength", type=float, default=1.0, help="Steering strength to use (default: 1.0)")
    parser.add_argument(
        "--token-targeting-strategy",
        type=str,
        default="LAST_TOKEN",
        choices=["CHOICE_TOKEN", "LAST_TOKEN", "FIRST_TOKEN", "ALL_TOKENS"],
        help="Token targeting strategy for steering (default: LAST_TOKEN)",
    )

    # Common optimization parameters
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100, 200, 500],
        help="Sample sizes to test (default: 5 10 20 50 100 200 500)",
    )
    parser.add_argument("--test-size", type=int, default=200, help="Fixed test set size (default: 200)")
    parser.add_argument("--test-split", type=float, default=0.2, help="DEPRECATED: Use --test-size instead")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to load from dataset")
    parser.add_argument("--save-plot", action="store_true", help="Save performance plot")
    parser.add_argument("--no-save-config", action="store_true", help="Don't save optimal sample size to model config")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--force", action="store_true", help="Force optimization even without matching classifier parameters"
    )


def setup_full_optimizer_parser(parser):
    """Set up the full-optimize subcommand parser."""
    parser.add_argument("model", type=str, help="Model name or path to optimize")

    # Task selection - mutually exclusive options
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--tasks", type=str, nargs="+", help="Specific tasks to optimize")
    task_group.add_argument(
        "--skills", type=str, nargs="+", help="Select tasks by skill categories (e.g., coding, mathematics, reasoning)"
    )
    task_group.add_argument(
        "--risks",
        type=str,
        nargs="+",
        help="Select tasks by risk categories (e.g., harmfulness, toxicity, hallucination)",
    )

    # General limit that applies to all optimizations unless overridden
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Sample limit for all optimizations (default: 100). Can be overridden by specific limits below",
    )

    # Specific limits (override general limit if provided)
    parser.add_argument(
        "--classification-limit",
        type=int,
        default=None,
        help="Sample limit for classification optimization (overrides --limit)",
    )
    parser.add_argument(
        "--sample-size-limit",
        type=int,
        default=None,
        help="Sample limit for sample size optimization (overrides --limit)",
    )
    parser.add_argument(
        "--steering-limit", type=int, default=None, help="Sample limit for steering optimization (overrides --limit)"
    )

    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100, 200, 500],
        help="Sample sizes to test (default: 5 10 20 50 100 200 500)",
    )
    parser.add_argument(
        "--skip-classification", action="store_true", help="Skip classification optimization and use existing config"
    )
    parser.add_argument("--skip-sample-size", action="store_true", help="Skip sample size optimization")
    parser.add_argument("--skip-classifier-training", action="store_true", help="Skip final classifier training step")
    parser.add_argument("--skip-control-vectors", action="store_true", help="Skip control vector training step")

    # Steering optimization options
    parser.add_argument("--skip-steering", action="store_true", help="Skip steering optimization")
    parser.add_argument(
        "--steering-methods",
        type=str,
        nargs="+",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        default=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering methods to test (default: all methods with parameter variations)",
    )
    parser.add_argument(
        "--steering-layer-range", type=str, default=None, help="Layer range for steering optimization (e.g., '0-5')"
    )
    parser.add_argument(
        "--steering-strength-range",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Steering strengths to test (default: 0.5 1.0 1.5 2.0)",
    )
    # Task selection options
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to randomly select from matched tasks (default: all)",
    )
    parser.add_argument(
        "--min-quality-score",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Minimum quality score for tasks (default: 2)",
    )
    parser.add_argument(
        "--task-seed", type=int, default=None, help="Random seed for task selection (for reproducibility)"
    )

    parser.add_argument(
        "--max-time-per-task", type=float, default=20.0, help="Maximum time per task in minutes (default: 20.0)"
    )

    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--save-plots", action="store_true", help="Save plots for both optimizations")

    # Timing calibration options
    parser.add_argument(
        "--skip-timing-estimation", action="store_true", help="Skip timing estimation and proceed without time warnings"
    )
    parser.add_argument("--calibration-file", type=str, default=None, help="File to save/load calibration data")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Only run calibration and exit (saves to --calibration-file if provided)",
    )


def setup_configure_model_parser(parser):
    """Set up the configure-model subcommand parser."""
    parser.add_argument("model", type=str, help="Model name to configure")
    parser.add_argument("--force", action="store_true", help="Force reconfiguration even if model already has a config")


def setup_generate_vector_parser(parser):
    """Set up the generate-vector subcommand parser."""
    # Source of contrastive pairs - mutually exclusive for single property
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--from-pairs",
        type=str,
        metavar="FILE",
        help="Path to JSON file containing contrastive pairs (single property)",
    )
    source_group.add_argument(
        "--from-description",
        type=str,
        metavar="TRAIT",
        help="Natural language description of the trait (single property)",
    )

    # Multi-property support
    parser.add_argument("--multi-property", action="store_true", help="Enable multi-property steering (DAC only)")
    parser.add_argument(
        "--property-files",
        type=str,
        nargs="+",
        metavar="NAME:FILE:LAYER",
        help="Property definitions from files (format: property_name:pairs_file:layer)",
    )
    parser.add_argument(
        "--property-descriptions",
        type=str,
        nargs="+",
        metavar="NAME:DESC:LAYER",
        help="Property definitions from descriptions (format: property_name:description:layer)",
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name or path (default: distilgpt2)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (default: auto-detect)")

    # Steering method configuration
    parser.add_argument(
        "--method",
        type=str,
        default="DAC",
        choices=["DAC", "CAA", "HPR", "BiPO", "ControlVectorSteering"],
        help="Steering method to use (default: DAC)",
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer index to apply steering (default: 0)")

    # Output configuration
    parser.add_argument("--output", type=str, required=True, help="Output path for the generated steering vector")

    # Pair generation options (only used with --from-description)
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=30,
        help="Number of pairs to generate when using --from-description (default: 30)",
    )
    parser.add_argument(
        "--save-pairs", type=str, default=None, help="Save generated pairs to this file when using --from-description"
    )

    # Method-specific parameters
    parser.add_argument("--dynamic-control", action="store_true", help="Enable dynamic control for DAC method")
    parser.add_argument(
        "--entropy-threshold", type=float, default=1.0, help="Entropy threshold for DAC method (default: 1.0)"
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for HPR method (default: 1.0)")

    # Activation extraction configuration
    parser.add_argument(
        "--prompt-construction",
        type=str,
        default="multiple_choice",
        choices=["multiple_choice", "role_playing", "direct_completion", "instruction_following"],
        help="Strategy for constructing prompts from question-answer pairs (default: multiple_choice)",
    )
    parser.add_argument(
        "--token-targeting",
        type=str,
        default="choice_token",
        choices=["choice_token", "continuation_token", "last_token", "first_token", "mean_pooling", "max_pooling"],
        help="Strategy for targeting tokens in activation extraction (default: choice_token)",
    )

    # General options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")


def setup_multi_steer_parser(parser):
    """Set up the multi-steer subcommand parser for dynamic vector combination."""
    # Vector inputs - can specify multiple vector-weight pairs
    parser.add_argument(
        "--vector",
        type=str,
        action="append",
        required=True,
        metavar="PATH:WEIGHT",
        help="Path to steering vector and its weight (format: path/to/vector.pt:0.5). Can be specified multiple times.",
    )

    # Model configuration
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to apply combined steering")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (default: auto-detect)")
    
    # Steering method configuration
    parser.add_argument(
        "--method",
        type=str,
        default="CAA",
        choices=["CAA", "DAC"],
        help="Steering method to use for combination (default: CAA)",
    )

    # Generation configuration
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate with combined steering")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum new tokens to generate (default: 100)")

    # Weight normalization
    parser.add_argument("--normalize-weights", action="store_true", help="Normalize weights to sum to 1.0")
    parser.add_argument(
        "--allow-unnormalized", action="store_true", help="Allow weights that don't sum to 1.0 (for stronger effects)"
    )
    parser.add_argument(
        "--target-norm", type=float, default=None, help="Scale the combined vector to have this norm (e.g., 10.0)"
    )

    # Output options
    parser.add_argument(
        "--save-combined", type=str, default=None, help="Save the combined steering vector to this path"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output showing weight calculations")


def setup_evaluate_parser(parser):
    """Set up the evaluate subcommand parser for single-prompt evaluation."""

    # Required arguments
    parser.add_argument("--vector", type=str, required=True, help="Path to steering vector file (.pt)")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to evaluate")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name or path (used for both generation and evaluation)"
    )
    parser.add_argument("--trait", type=str, required=True, help="Trait name (e.g., 'catholic', 'cynical')")

    # Optional model configuration
    parser.add_argument("--device", type=str, default=None, help="Device to run on (default: auto-detect)")

    # Optional steering parameters
    parser.add_argument(
        "--steering-strength", type=float, default=2.0, help="Steering strength to apply (default: 2.0)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum new tokens to generate (default: 100)")
    parser.add_argument(
        "--trait-description",
        type=str,
        default=None,
        help="Optional description of the trait (default: use trait name)",
    )

    # Optional threshold parameters
    parser.add_argument(
        "--trait-threshold", type=float, default=None, help="Minimum trait quality threshold (-1 to 1 scale)"
    )
    parser.add_argument(
        "--answer-threshold", type=float, default=None, help="Minimum answer quality threshold (0 to 1 scale)"
    )

    # Output options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
