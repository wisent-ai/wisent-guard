#!/usr/bin/env python3
"""
WandB BigCode Evaluation Script

Evaluates Qwen2.5-Coder models on coding benchmarks using bigcode-evaluation-harness.
Supports both baseline (unsteered) and CAA-steered models with WandB logging.

IMPORTANT: Before running this script, configure accelerate for proper GPU/memory management:
    $ accelerate config

Recommended accelerate settings:
    - Compute environment: This machine
    - Machine type: No distributed training
    - GPU selection: Choose your GPU(s) (e.g., 0 for single GPU)
    - Mixed precision: fp16 (for memory efficiency)

The BigCode harness uses accelerate to manage device placement and memory allocation.
Manual memory/precision settings can conflict with accelerate's configuration.

Usage:
    python evaluate_bigcode_wandb.py --method unsteered
    python evaluate_bigcode_wandb.py --method caa
    python evaluate_bigcode_wandb.py --method caa --load-in-8bit  # For memory-constrained systems
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration Dataclasses
@dataclass
class ModelConfig:
    """Model configuration settings."""
    unsteered_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    caa_path: str = "/workspace/wisent-guard/huggingface_qwen25-7b-coder-caa"
    base_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

@dataclass
class EvaluationDefaults:
    """Default evaluation parameters."""
    temperature: float = 0.2
    batch_size: int = 16
    n_samples: int = 1
    top_p: float = 0.95
    precision: str = "fp16"
    seed: int = 0
    project: str = "qwen2.5_coder_bigcode"

# Initialize configurations
MODEL_CONFIG = ModelConfig()
EVAL_DEFAULTS = EvaluationDefaults()

# Supported choices
SUPPORTED_METHODS = ["unsteered", "caa"]
SUPPORTED_DEVICES = [0, 1, 2, 3]
SUPPORTED_PRECISIONS = ["fp32", "fp16", "bf16"]
SUPPORTED_DO_SAMPLE = ["True", "False"]

# Dataset configurations
INSTRUCTION_DATASETS = ["instruct-humaneval", "instruct-humaneval-nocontext"]

RECODE_DATASETS = {
    "perturbed-humaneval-format-num_seeds_5",
    "perturbed-humaneval-func_name-num_seeds_5",
    "perturbed-humaneval-natgen-num_seeds_5",
    "perturbed-humaneval-nlaugmenter-num_seeds_5"
}

SUPPORTED_DATASETS = [
    "mbpp", "mbppplus",
    "humaneval", "humanevalplus",
    "multiple-js",
    *INSTRUCTION_DATASETS,
    *RECODE_DATASETS
]

# Special tokens for instruction-following datasets
QWEN_INSTRUCTION_TOKENS = "<|im_start|>user\n,<|im_end|>\n,<|im_start|>assistant\n"

# Directory paths
EVALUATION_RESULTS_DIR = "evaluation_results"
BIGCODE_HARNESS_DIR = "/workspace/wisent-guard/bigcode-evaluation-harness"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WandB BigCode Evaluation for Qwen2.5-Coder (requires accelerate configuration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Note: Run 'accelerate config' before using this script to configure GPU/memory settings.",
    )

    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        required=True,
        help="Evaluation method: 'unsteered' (baseline) or 'caa' (CAA-steered)",
    )

    parser.add_argument(
        "--dataset",
        default=SUPPORTED_DATASETS[0],
        choices=SUPPORTED_DATASETS,
        help="Dataset to evaluate on (default: mbpp)",
    )

    parser.add_argument(
        "--limit", type=int, default=None, help="Number of samples to evaluate (default: all samples in dataset)"
    )

    parser.add_argument("--temperature", type=float, default=EVAL_DEFAULTS.temperature, help=f"Generation temperature (default: {EVAL_DEFAULTS.temperature})")

    parser.add_argument("--batch_size", type=int, default=EVAL_DEFAULTS.batch_size, help=f"Batch size for evaluation (default: {EVAL_DEFAULTS.batch_size})")

    parser.add_argument("--n_samples", type=int, default=EVAL_DEFAULTS.n_samples, help=f"Number of completions to generate per problem (default: {EVAL_DEFAULTS.n_samples})")

    parser.add_argument(
        "--project", default=EVAL_DEFAULTS.project, help=f"WandB project name (default: {EVAL_DEFAULTS.project})"
    )

    parser.add_argument(
        "--do_sample",
        type=str,
        choices=SUPPORTED_DO_SAMPLE,
        default=SUPPORTED_DO_SAMPLE[0],
        help="Use sampling for generation (default: True)",
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=SUPPORTED_PRECISIONS,
        default=EVAL_DEFAULTS.precision,
        help=f"Model precision (default: {EVAL_DEFAULTS.precision})",
    )

    parser.add_argument("--seed", type=int, default=EVAL_DEFAULTS.seed, help=f"Random seed for reproducible generation (default: {EVAL_DEFAULTS.seed})")

    # Memory management parameters
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8bit quantization for memory efficiency (requires bitsandbytes)",
    )

    # Device selection parameter
    parser.add_argument(
        "--device", type=int, choices=SUPPORTED_DEVICES, default=SUPPORTED_DEVICES[0], help=f"GPU device ID to use for evaluation (default: {SUPPORTED_DEVICES[0]})"
    )

    return parser.parse_args()


def load_caa_params_from_config(model_path: str) -> Optional[Dict[str, Any]]:
    """Load CAA parameters from model's config.json file."""
    try:
        config_path = Path(model_path) / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {
            "layer_id": config.get("caa_layer_id"),
            "alpha": config.get("caa_alpha"), 
            "steering_method": config.get("steering_method"),
            "steering_vector_path": config.get("steering_vector_path"),
            "caa_enabled": config.get("caa_enabled"),
            "optimization_info": config.get("wisent_optimization", {})
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ Warning: Could not load CAA config from {config_path}: {e}")
        return None


def initialize_wandb(args: argparse.Namespace, timestamp: str):
    """Initialize WandB with appropriate configuration."""

    # Method-specific configuration
    if args.method == "caa":
        method_params = load_caa_params_from_config(MODEL_CONFIG.caa_path)
    else:
        method_params = None

    # WandB configuration
    config = {
        "type": "test",
        "dataset": args.dataset,
        "dataset_size": args.limit,
        "method": args.method,
        "method_parameters": method_params,
        "model_base": MODEL_CONFIG.base_name,
        "evaluation_framework": "bigcode-evaluation-harness",
        "generation_parameters": {
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "do_sample": args.do_sample,
            "precision": args.precision,
            "seed": args.seed,
            "top_p": EVAL_DEFAULTS.top_p,
            "load_in_8bit": args.load_in_8bit,
            "device": args.device,
        },
    }

    # Initialize WandB run
    run = wandb.init(
        project=args.project,
        name=f"{args.method}_{args.dataset}_{timestamp}",
        config=config,
        tags=[args.method, args.dataset, "bigcode", "minimal_eval"],
    )

    return run


def load_model(method: str) -> str:
    """Load the appropriate model based on the method."""

    if method == "unsteered":
        print("Loading baseline Qwen2.5-Coder-7B-Instruct...")
        model_path = MODEL_CONFIG.unsteered_path

    elif method == "caa":
        print("Loading CAA-steered Wisent model...")
        model_path = MODEL_CONFIG.caa_path

        # Verify the CAA model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"CAA model not found at {model_path}. Make sure the huggingface_qwen_generated directory exists."
            )

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Model path: {model_path}")

    # Test model loading to ensure it works
    try:
        print("Testing model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"   Device: {next(model.parameters()).device}")

        # Free memory - we just needed to test loading
        del model, tokenizer
        torch.cuda.empty_cache()

        return model_path

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def run_bigcode_evaluation(model_path: str, args: argparse.Namespace, timestamp: str) -> Tuple[Dict[str, Any], Path, float, Dict[str, Any]]:
    """Run bigcode-evaluation-harness evaluation."""

    print("\\n🚀 Running BigCode evaluation...")
    print(f"   Model: {model_path}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Samples: {args.limit if args.limit is not None else 'all'}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Sampling: {args.do_sample}")

    # Create output directory and convert to absolute path for BigCode harness
    output_dir = Path(EVALUATION_RESULTS_DIR) / f"{args.method}_{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_abs = Path.cwd() / output_dir

    # Convert relative model path to absolute path for BigCode harness
    if model_path.startswith("./"):
        abs_model_path = str(Path.cwd() / model_path.lstrip("./"))
    else:
        abs_model_path = model_path

    # Prepare BigCode command (using accelerate launch for proper device management)
    cmd = [
        "accelerate",
        "launch",
        "--gpu_ids",
        str(args.device),
        "main.py",
        "--model",
        abs_model_path,
        "--tasks",
        args.dataset,
        "--batch_size",
        str(args.batch_size),
        "--temperature",
        str(args.temperature),
        "--do_sample",
        args.do_sample,
        "--top_p",
        str(EVAL_DEFAULTS.top_p),
        "--precision",
        args.precision,
        "--seed",
        str(args.seed),
        "--allow_code_execution",
        "--trust_remote_code",
        "--save_generations",
        "--save_generations_path",
        str(output_dir_abs / "generations.json"),
        "--save_references",
        "--save_references_path",
        str(output_dir_abs / "references.json"),
        "--metric_output_path",
        str(output_dir_abs / "metrics.json"),
    ]

    # Add limit only if specified (None means evaluate full dataset)
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    # Add n_samples parameter  
    cmd.extend(["--n_samples", str(args.n_samples)])

    # Add memory management parameters
    if args.load_in_8bit:
        cmd.append("--load_in_8bit")
    
    # Add instruction tokens for instruct-humaneval datasets
    if args.dataset in INSTRUCTION_DATASETS:
        cmd.extend(["--instruction_tokens", QWEN_INSTRUCTION_TOKENS])
        print(f"📝 Using instruction tokens for {args.dataset}")
    
    # Log if this is a Recode dataset (parameters should be set by shell script)
    if args.dataset in RECODE_DATASETS:
        print(f"📊 Evaluating Recode dataset: {args.dataset}")

    print("\\n🔧 BigCode command:")
    print(" ".join(cmd))
    print(f"🎯 Using GPU device: {args.device}")

    # Run evaluation
    start_time = time.time()

    # Set CUDA_VISIBLE_DEVICES to control GPU selection
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=BIGCODE_HARNESS_DIR, env=env)

        duration = time.time() - start_time

        print("\\n✅ Evaluation completed successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds ({duration / 60:.1f} minutes)")

        # Load aggregate results
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            print("\\n📊 Results:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for subkey, subvalue in value.items():
                        print(f"     {subkey}: {subvalue}")
                else:
                    print(f"   {key}: {value}")

            # Extract detailed per-sample results
            detailed_results = extract_detailed_results(output_dir, args.dataset, timestamp, args)
            
            return metrics, output_dir, duration, detailed_results

        print(f"⚠️ Metrics file not found at {metrics_path}")
        return {}, output_dir, duration, {}

    except subprocess.CalledProcessError as e:
        print("\\n❌ BigCode evaluation failed!")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

    except Exception as e:
        print(f"\\n❌ Error running evaluation: {e}")
        raise


def extract_detailed_results(output_dir: Path, dataset: str, timestamp: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Load BigCode evaluation outputs for WandB logging."""
    print("\\n🔍 Loading BigCode evaluation results...")
    
    try:
        # Load metrics file
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"⚠️ Metrics file not found at {metrics_path}")
            return {}
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load generations file (BigCode saves it as generations_{task_name}.json)
        generations_path = output_dir / f"generations_{dataset}.json"
        if not generations_path.exists():
            generations_path = output_dir / "generations.json"
        
        generations = []
        if generations_path.exists():
            with open(generations_path, 'r') as f:
                generations = json.load(f)
            print(f"✅ Found generations file with {len(generations)} samples")
        
        # Load references file
        references_path = output_dir / f"references_{dataset}.json"
        if not references_path.exists():
            references_path = output_dir / "references.json"
        if not references_path.exists():
            # Fallback: BigCode harness saves references in its own directory
            bigcode_references_path = Path("/workspace/wisent-guard/bigcode-evaluation-harness") / f"references_{dataset}.json"
            if bigcode_references_path.exists():
                references_path = bigcode_references_path
        
        references = []
        if references_path.exists():
            with open(references_path, 'r') as f:
                references = json.load(f)
            print(f"✅ Found references file with {len(references)} entries")
        
        # Load detailed results if available (from modified BigCode)
        detailed_results_path = output_dir / f"detailed_results_{dataset}.json"
        detailed_results = []
        if detailed_results_path.exists():
            with open(detailed_results_path, 'r') as f:
                detailed_data = json.load(f)
            print(f"✅ Found detailed results from BigCode")
            
            # Process BigCode's detailed results format
            for task_id, task_results in detailed_data.items():
                if task_results:
                    # Process ALL completions for each task (for n_samples > 1)
                    # task_results is like [[0, {"task_id": 0, "passed": false, ...}], [1, {...}], ...]
                    for completion_data in task_results:
                        completion_id = completion_data[0]  # Completion index
                        result_dict = completion_data[1]    # The result dictionary
                        
                        # Get the specific generation for this completion
                        task_idx = int(task_id)
                        if task_idx < len(generations) and completion_id < len(generations[task_idx]):
                            generation = generations[task_idx][completion_id]
                        else:
                            generation = ""
                        
                        # Get reference for this task (same for all completions)
                        reference = references[task_idx] if task_idx < len(references) else ""
                        
                        detailed_results.append({
                            "task_id": task_idx,
                            "completion_id": completion_id,
                            "generation": generation,
                            "reference": reference,
                            "passed": result_dict.get("passed", False),
                            "result": result_dict.get("result", "unknown")
                        })
        
        # Prepare final results structure
        final_results = {
            "metadata": {
                "dataset": dataset,
                "timestamp": timestamp,
                "method": args.method,
                "model_path": getattr(args, 'model_path', 'unknown'),
                "total_samples": len(generations),
                "evaluation_parameters": {
                    "temperature": args.temperature,
                    "batch_size": args.batch_size,
                    "do_sample": args.do_sample,
                    "precision": args.precision,
                    "seed": args.seed,
                    "device": args.device,
                }
            },
            "metrics": metrics,
            "detailed_results": detailed_results if detailed_results else None,
            "generations_count": len(generations),
            "references_count": len(references)
        }
        
        # Save consolidated results
        consolidated_path = output_dir / f"consolidated_results_{timestamp}.json"
        with open(consolidated_path, 'w') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved consolidated results to: {consolidated_path}")
        
        # Log detailed results statistics
        if detailed_results:
            total_completions = len(detailed_results)
            unique_tasks = len(set(r['task_id'] for r in detailed_results))
            avg_completions_per_task = total_completions / unique_tasks if unique_tasks > 0 else 0
            print(f"📊 Detailed results: {total_completions} completions across {unique_tasks} tasks (avg: {avg_completions_per_task:.1f} per task)")
        
        return final_results
        
    except Exception as e:
        print(f"⚠️ Error loading results: {e}")
        return {}


def log_results_to_wandb(metrics: Dict[str, Any], args: argparse.Namespace, duration: float, output_dir: Path, detailed_results: Optional[Dict[str, Any]] = None) -> None:
    """Log final results to WandB."""

    print("\\n📈 Logging results to WandB...")

    # Extract key metrics from nested task structure
    # BigCode saves results as: {"task_name": {"pass@1": value}, "config": {...}}
    task_results = None
    for key, value in metrics.items():
        if key != "config" and isinstance(value, dict):
            task_results = value
            break

    pass_at_1 = task_results.get("pass@1", 0.0) if task_results else 0.0

    # Calculate problems solved - need to get actual number from results since limit might be None
    if task_results and "total" in task_results:
        total_problems = task_results["total"]
        problems_solved = int(pass_at_1 * total_problems) if pass_at_1 > 0 else 0
    elif args.limit is not None:
        total_problems = args.limit
        problems_solved = int(pass_at_1 * args.limit) if pass_at_1 > 0 else 0
    else:
        # Try to get from detailed results
        if detailed_results and 'results' in detailed_results:
            total_problems = len(detailed_results['results'])
            problems_solved = int(pass_at_1 * total_problems) if pass_at_1 > 0 else 0
        else:
            total_problems = None
            problems_solved = None

    # Log final results
    log_dict = {
        "pass_at_1": float(pass_at_1),
        "evaluation_duration_min": duration / 60,
        "evaluation_duration_sec": duration,
    }

    # Add problem counts if available
    if problems_solved is not None:
        log_dict["problems_solved"] = problems_solved
    if total_problems is not None:
        log_dict["total_problems"] = total_problems
        log_dict["avg_time_per_problem"] = duration / total_problems if total_problems > 0 else 0

    # Add detailed results summary
    if detailed_results and 'results' in detailed_results:
        log_dict["detailed_samples_extracted"] = len(detailed_results['results'])
        
        # Add execution summary statistics if available
        if 'summary_statistics' in detailed_results:
            stats = detailed_results['summary_statistics']
            log_dict.update({
                "detailed_correct_samples": stats.get("correct_samples", 0),
                "detailed_failed_samples": stats.get("failed_samples", 0), 
                "detailed_timeout_samples": stats.get("timeout_samples", 0),
                "detailed_error_samples": stats.get("error_samples", 0),
                "detailed_no_reference_samples": stats.get("no_reference_samples", 0),
                "detailed_accuracy": stats.get("accuracy", 0.0)
            })

    wandb.log(log_dict)

    # Log any additional metrics from BigCode task results
    if task_results:
        for key, value in task_results.items():
            if key != "pass@1" and isinstance(value, (int, float)):
                wandb.log({f"bigcode_metrics/{key}": value})

    # Upload result artifacts
    try:
        if (output_dir / "metrics.json").exists():
            wandb.save(str(output_dir / "metrics.json"), base_path=str(output_dir.parent))

        # Upload generations file (with correct BigCode naming)
        generations_files = [
            output_dir / f"generations_{args.dataset}.json",
            output_dir / "generations.json"
        ]
        for gen_file in generations_files:
            if gen_file.exists():
                wandb.save(str(gen_file), base_path=str(output_dir.parent))
                break

        # Upload references file (with correct BigCode naming)
        references_files = [
            output_dir / f"references_{args.dataset}.json",
            output_dir / "references.json"
        ]
        for ref_file in references_files:
            if ref_file.exists():
                wandb.save(str(ref_file), base_path=str(output_dir.parent))
                break

        # Upload consolidated results (comprehensive processed results)
        consolidated_files = list(output_dir.glob("consolidated_results_*.json"))
        for consolidated_file in consolidated_files:
            wandb.save(str(consolidated_file), base_path=str(output_dir.parent))
            print(f"📊 Uploaded consolidated results: {consolidated_file.name}")

        print("✅ Artifacts uploaded to WandB")

    except Exception as e:
        print(f"⚠️ Error uploading artifacts: {e}")

    print("✅ Results logged to WandB successfully!")


def main() -> bool:
    """Main evaluation function."""

    # Parse arguments
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("🧪 WandB BigCode Evaluation - Minimal")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.limit if args.limit is not None else 'all'}")
    print(f"WandB Project: {args.project}")
    print("=" * 60)

    try:
        # Initialize WandB
        wandb_run = initialize_wandb(args, timestamp)
        print(f"✅ WandB initialized: {wandb_run.url}")

        # Set model path (BigCode harness will handle loading to save memory)
        if args.method == "unsteered":
            model_path = MODEL_CONFIG.unsteered_path
        elif args.method == "caa":
            model_path = MODEL_CONFIG.caa_path

        # Run BigCode evaluation
        metrics, output_dir, duration, detailed_results = run_bigcode_evaluation(model_path, args, timestamp)

        # Log results to WandB
        log_results_to_wandb(metrics, args, duration, output_dir, detailed_results)

        print("\\n🎉 Evaluation completed successfully!")
        print(f"📁 Local results: {output_dir}")
        print(f"📊 WandB dashboard: {wandb_run.url}")

        return True

    except Exception as e:
        print(f"\\n❌ Evaluation failed: {e}")
        wandb.log({"error": str(e), "status": "failed"})
        return False

    finally:
        # Always finish WandB run
        wandb.finish()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
