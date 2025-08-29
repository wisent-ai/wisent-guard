#!/usr/bin/env python3
"""
WandB LM-Eval Math Evaluation Script

Evaluates Qwen2.5-Math models on various math reasoning tasks using lm-evaluation-harness.
Supports both baseline (unsteered) and CAA-steered models with WandB logging.

Key features:
- Single GPU evaluation with explicit device selection
- Native WandB integration via lm-evaluation-harness
- Support for multiple math tasks (GSM8K, MATH, MathQA, ASDIV, Minerva Math)
- CAA parameter loading from model config
- Memory-efficient evaluation options
- Result caching and reproducible seeds

Usage:
    python evaluate_lm_eval_wandb.py --method unsteered --device 0 --tasks gsm8k
    python evaluate_lm_eval_wandb.py --method caa --device 1 --load_in_8bit --tasks hendrycks_math
    python evaluate_lm_eval_wandb.py --method caa --tasks mathqa --num_fewshot 0 --seed 42
    python evaluate_lm_eval_wandb.py --method unsteered --tasks asdiv --use_cache /tmp/lm_cache
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Model configuration
UNSTEERED_MODEL_PATH = "Qwen/Qwen2.5-Math-7B-Instruct"
CAA_MODEL_PATH = "/workspace/wisent-guard/huggingface_qwen25-7b-math-caa"
MODEL_BASE_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"

# Default parameters
DEFAULT_BATCH_SIZE = "auto"
DEFAULT_NUM_FEWSHOT = 5
DEFAULT_PROJECT = "qwen2.5_math_eval"
DEFAULT_TASKS = "gsm8k"
DEFAULT_DEVICE = 0
SUPPORTED_DEVICES = [0, 1, 2, 3]

# Supported math tasks
MATH_TASKS = [
    "gsm8k",  # Grade School Math 8K
    "gsm8k_cot",  # GSM8K with Chain-of-Thought
    "gsm8k_cot_zeroshot",  # GSM8K CoT zero-shot
    "hendrycks_math",  # MATH dataset (Competition Mathematics)
    "mathqa",  # MathQA dataset
    "asdiv",  # ASDIV dataset (arithmetic word problems)
    "minerva_math",  # Minerva Math problems
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Math models on various math tasks using lm-evaluation-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Unsteered model on GSM8K
    python evaluate_lm_eval_wandb.py --method unsteered --device 0 --tasks gsm8k
    
    # CAA model on MATH dataset with caching
    python evaluate_lm_eval_wandb.py --method caa --tasks hendrycks_math --use_cache /tmp/lm_cache
    
    # Zero-shot evaluation on MathQA
    python evaluate_lm_eval_wandb.py --method caa --tasks mathqa --num_fewshot 0
    
    # Comprehensive evaluation with specific seed
    python evaluate_lm_eval_wandb.py --method unsteered --tasks asdiv --seed 42 --max_batch_size 16
        """,
    )

    # Method selection
    parser.add_argument(
        "--method",
        choices=["unsteered", "caa"],
        required=True,
        help="Evaluation method: unsteered baseline or CAA-steered model",
    )

    # Task configuration
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS,
        help=f"Math task to evaluate (default: {DEFAULT_TASKS}). Options: {', '.join(MATH_TASKS)}",
    )

    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=DEFAULT_NUM_FEWSHOT,
        help=f"Number of few-shot examples (default: {DEFAULT_NUM_FEWSHOT})",
    )

    # Model and evaluation parameters
    parser.add_argument(
        "--device",
        type=int,
        choices=SUPPORTED_DEVICES,
        default=DEFAULT_DEVICE,
        help=f"GPU device ID to use (default: {DEFAULT_DEVICE})",
    )

    parser.add_argument(
        "--batch_size", default=DEFAULT_BATCH_SIZE, help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE})"
    )

    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to evaluate (for testing)")

    # Memory management
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Load model in 8-bit quantization for memory efficiency"
    )

    # WandB configuration
    parser.add_argument(
        "--wandb_project", default=DEFAULT_PROJECT, help=f"WandB project name (default: {DEFAULT_PROJECT})"
    )

    parser.add_argument("--wandb_entity", default=None, help="WandB entity (team/username)")

    # Output configuration
    parser.add_argument(
        "--output_dir",
        default="evaluation_results",
        help="Directory to save evaluation results (default: evaluation_results)",
    )

    parser.add_argument(
        "--log_samples", action="store_true", help="Log individual samples to WandB for detailed analysis"
    )

    # Additional lm-eval parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    parser.add_argument(
        "--max_batch_size", type=int, default=None, help="Maximum batch size when using --batch_size auto"
    )

    parser.add_argument(
        "--use_cache",
        type=str,
        default=None,
        help="Path for sqlite cache database (e.g., /path/to/cache/lm_eval_cache)",
    )

    return parser.parse_args()


def load_caa_params_from_config(model_path):
    """Load CAA parameters from model's config.json file."""
    try:
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        return {
            "layer_id": config.get("caa_layer_id"),
            "alpha": config.get("caa_alpha"),
            "steering_method": config.get("steering_method"),
            "steering_vector_path": config.get("steering_vector_path"),
            "caa_enabled": config.get("caa_enabled"),
            "optimization_info": config.get("wisent_optimization", {}),
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ Warning: Could not load CAA config from {config_path}: {e}")
        return None


# WandB initialization handled by lm-eval harness


def build_lm_eval_command(args, output_path, caa_params=None):
    """Build the lm_eval command with all necessary arguments."""

    # Base command
    cmd = ["python", "-m", "lm_eval"]

    # Model configuration
    cmd.extend(["--model", "hf"])

    # Model arguments
    model_path = CAA_MODEL_PATH if args.method == "caa" else UNSTEERED_MODEL_PATH
    model_args = [f"pretrained={model_path}"]

    # Add trust_remote_code for custom WisentQwen2 architecture
    if args.method == "caa":
        model_args.append("trust_remote_code=True")

    if args.load_in_8bit:
        model_args.append("load_in_8bit=True")

    cmd.extend(["--model_args", ",".join(model_args)])

    # Task and evaluation settings
    cmd.extend(["--tasks", args.tasks])
    cmd.extend(["--num_fewshot", str(args.num_fewshot)])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--device", f"cuda:{args.device}"])

    # Output settings
    cmd.extend(["--output_path", str(output_path)])

    # Optional limit
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    # Seed for reproducibility
    cmd.extend(["--seed", str(args.seed)])

    # Max batch size for auto batch sizing
    if args.max_batch_size:
        cmd.extend(["--max_batch_size", str(args.max_batch_size)])

    # Cache configuration
    if args.use_cache:
        cmd.extend(["--use_cache", str(args.use_cache)])

    # Enable trust_remote_code for datasets by default (required for mathqa and other tasks)
    cmd.extend(["--trust_remote_code"])

    # Native lm-eval WandB integration with enhanced CAA metadata
    wandb_args = [f"project={args.wandb_project}"]
    if args.wandb_entity:
        wandb_args.append(f"entity={args.wandb_entity}")

    cmd.extend(["--wandb_args", ",".join(wandb_args)])

    # Pass CAA metadata through lm-eval's config system
    wandb_config = {
        "method": args.method,
        "model_base": MODEL_BASE_NAME,
        "wisent_model_type": "qwen2.5_math_caa" if args.method == "caa" else "qwen2.5_math_baseline",
    }

    # Add CAA-specific parameters if available
    if args.method == "caa" and caa_params:
        wandb_config.update(
            {
                "caa_alpha": caa_params.get("alpha"),
                "caa_layer_id": caa_params.get("layer_id"),
                "steering_method": caa_params.get("steering_method"),
                "caa_enabled": caa_params.get("caa_enabled"),
                "optimization_dataset": caa_params.get("optimization_info", {}).get("dataset"),
                "optimization_performance": caa_params.get("optimization_info", {}).get("best_value"),
            }
        )

    # Convert config to lm-eval format (only non-None values)
    config_items = [f"{k}={v}" for k, v in wandb_config.items() if v is not None]
    if config_items:
        cmd.extend(["--wandb_config_args", ",".join(config_items)])

    # Always enable fine-grained sample logging for rich WandB integration
    cmd.append("--log_samples")

    return cmd


def run_evaluation(cmd, output_path):
    """Execute the lm_eval command and return results."""
    print("🧮 Running math task evaluation on Qwen2.5-Math...")
    print(f"📁 Output directory: {output_path}")
    print(f"🔧 Command: {' '.join(cmd)}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Record start time
    start_time = time.time()

    # Run evaluation
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd="/workspace/wisent-guard/lm-evaluation-harness"
        )

        duration = time.time() - start_time

        print("✅ Evaluation completed successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds ({duration / 60:.1f} minutes)")

        return {"success": True, "duration": duration, "stdout": result.stdout, "stderr": result.stderr}

    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return {"success": False, "error": e.stderr, "stdout": e.stdout}


def parse_lm_eval_results(output_path, args):
    """Parse and extract results from lm_eval output files."""
    import glob

    results = {}

    # LM-eval saves results in nested directories with model-specific naming
    # Search comprehensively for results files in all possible locations
    possible_paths = [
        # Direct path (if lm-eval changes behavior)
        output_path / "results.json",
        # Standard nested paths in lm-evaluation-harness directory
        Path("/workspace/wisent-guard/lm-evaluation-harness") / output_path.name / "**" / "results_*.json",
        Path("/workspace/wisent-guard/lm-evaluation-harness/evaluation_results")
        / output_path.name
        / "**"
        / "results_*.json",
        # Working directory relative paths
        Path("lm-evaluation-harness") / output_path.name / "**" / "results_*.json",
        Path("lm-evaluation-harness/evaluation_results") / output_path.name / "**" / "results_*.json",
        # Test evaluation results paths
        Path("/workspace/wisent-guard/lm-evaluation-harness/test_evaluation_results")
        / output_path.name
        / "**"
        / "results_*.json",
        # Direct model subdirectory patterns (for different model naming)
        output_path / "*" / "results_*.json",
        output_path / "**" / "results_*.json",
    ]

    results_file = None

    # Find the results file
    for path_pattern in possible_paths:
        if "*" in str(path_pattern):
            # Use glob for wildcard patterns
            matches = glob.glob(str(path_pattern), recursive=True)
            if matches:
                results_file = Path(matches[0])  # Take the first match
                break
        else:
            # Direct path check
            if path_pattern.exists():
                results_file = path_pattern
                break

    if results_file and results_file.exists():
        try:
            with open(results_file) as f:
                lm_eval_results = json.load(f)

            print(f"📁 Found results file: {results_file}")

            # Extract task results
            task_name = args.tasks
            if task_name in lm_eval_results.get("results", {}):
                task_results = lm_eval_results["results"][task_name]

                # Handle different metric naming conventions
                exact_match_key = None
                for key in task_results.keys():
                    if "exact_match" in key and "stderr" not in key:
                        exact_match_key = key
                        break

                exact_match_stderr_key = None
                for key in task_results.keys():
                    if "exact_match" in key and "stderr" in key:
                        exact_match_stderr_key = key
                        break

                results.update(
                    {
                        "exact_match": task_results.get(exact_match_key, 0.0) if exact_match_key else 0.0,
                        "exact_match_stderr": task_results.get(exact_match_stderr_key, 0.0)
                        if exact_match_stderr_key
                        else 0.0,
                        "alias": task_results.get("alias", task_name),
                    }
                )

            # Extract configuration
            if "config" in lm_eval_results:
                results["lm_eval_config"] = lm_eval_results["config"]

            # Extract sample count from n-samples
            if "n-samples" in lm_eval_results and task_name in lm_eval_results["n-samples"]:
                results["total_samples"] = lm_eval_results["n-samples"][task_name].get("effective", 0)

            print(f"📊 Results parsed: exact_match = {results.get('exact_match', 'N/A'):.3f}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Warning: Could not parse results file: {e}")
    else:
        print("⚠️ Warning: Results file not found in any expected location")
        print("   Searched paths:")
        for path in possible_paths:
            print(f"   - {path}")

    return results


# All WandB logging handled by lm-eval harness


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("🧮 WandB LM-Eval Math Evaluation - Qwen2.5-Math")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Base Model: {MODEL_BASE_NAME}")
    print(f"Tasks: {args.tasks}")
    print(f"Device: cuda:{args.device}")
    print(f"Batch Size: {args.batch_size}")
    if args.max_batch_size:
        print(f"Max Batch Size: {args.max_batch_size}")
    print(f"Few-shot: {args.num_fewshot}")
    print(f"Seed: {args.seed}")
    if args.use_cache:
        print(f"Cache: {args.use_cache}")
    if args.limit:
        print(f"Sample Limit: {args.limit}")
    print(f"WandB Project: {args.wandb_project}")
    print("=" * 60)

    # Load CAA parameters if using CAA method
    caa_params = None
    if args.method == "caa":
        caa_params = load_caa_params_from_config(CAA_MODEL_PATH)
        if caa_params:
            print(f"📋 Loaded Math CAA config: α={caa_params.get('alpha')}, layer={caa_params.get('layer_id')}")
            opt_info = caa_params.get("optimization_info", {})
            if opt_info.get("dataset"):
                print(f"📊 Optimized for: {opt_info['dataset']} (accuracy: {opt_info.get('best_value', 'N/A')})")
        else:
            print("⚠️ Could not load CAA configuration - model may not be available")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.method}_{args.tasks}_{timestamp}"

    # Build and execute lm_eval command (includes WandB integration)
    cmd = build_lm_eval_command(args, output_path, caa_params)
    evaluation_result = run_evaluation(cmd, output_path)

    if evaluation_result["success"]:
        # Parse results for our own summary
        results = parse_lm_eval_results(output_path, args)

        print("\n🎉 Evaluation completed successfully!")
        print(f"📁 Local results: {output_path}")
        print("📊 All results logged to WandB by lm-eval harness")

    else:
        print("\n❌ Evaluation failed!")
        print("Check the error output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
