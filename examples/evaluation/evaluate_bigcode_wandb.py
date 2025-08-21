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
import time
from datetime import datetime
from pathlib import Path
import sys

import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WandB BigCode Evaluation for Qwen2.5-Coder (requires accelerate configuration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Note: Run 'accelerate config' before using this script to configure GPU/memory settings."
    )
    
    parser.add_argument(
        "--method",
        choices=["unsteered", "caa"],
        required=True,
        help="Evaluation method: 'unsteered' (baseline) or 'caa' (CAA-steered)"
    )
    
    parser.add_argument(
        "--dataset",
        default="mbpp",
        choices=["mbpp", "mbppplus", "humaneval"],
        help="Dataset to evaluate on (default: mbpp)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all samples in dataset)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature (default: 0.2)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 1)"
    )
    
    parser.add_argument(
        "--project",
        default="qwen2.5_coder_bigcode",
        help="WandB project name (default: qwen2.5_coder_bigcode)"
    )
    
    parser.add_argument(
        "--do_sample", 
        type=str,
        choices=["True", "False"],
        default="True",
        help="Use sampling for generation (default: True)"
    )
    
    parser.add_argument(
        "--precision", 
        type=str, 
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Model precision (default: fp16)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible generation (default: 0)"
    )
    
    # Memory management parameters
    parser.add_argument(
        "--load-in-8bit", 
        action="store_true",
        help="Load model in 8bit quantization for memory efficiency (requires bitsandbytes)"
    )
    
    # Device selection parameter
    parser.add_argument(
        "--device",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="GPU device ID to use for evaluation (default: 0)"
    )
    
    return parser.parse_args()


def initialize_wandb(args, timestamp):
    """Initialize WandB with appropriate configuration."""
    
    # Method-specific configuration
    if args.method == "caa":
        method_params = {
            "layer_id": 24,
            "alpha": 0.9,
            "normalization": "l2_unit",
            "target_norm": 0.7,
            "steering_method": "CAA"
        }
    else:
        method_params = None
    
    # WandB configuration
    config = {
        "type": "test",
        "dataset": args.dataset,
        "dataset_size": args.limit,
        "method": args.method,
        "method_parameters": method_params,
        "model_base": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "evaluation_framework": "bigcode-evaluation-harness",
        "generation_parameters": {
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "do_sample": args.do_sample,
            "precision": args.precision,
            "seed": args.seed,
            "top_p": 0.95,
            "load_in_8bit": args.load_in_8bit,
            "device": args.device
        }
    }
    
    # Initialize WandB run
    run = wandb.init(
        project=args.project,
        name=f"{args.method}_{args.dataset}_{timestamp}",
        config=config,
        tags=[args.method, args.dataset, "bigcode", "minimal_eval"]
    )
    
    return run


def load_model(method):
    """Load the appropriate model based on the method."""
    
    if method == "unsteered":
        print("Loading baseline Qwen2.5-Coder-7B-Instruct...")
        model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
        
    elif method == "caa":
        print("Loading CAA-steered Wisent model...")
        model_path = "./huggingface_qwen25-7b-coder-caa"
        
        # Verify the CAA model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"CAA model not found at {model_path}. "
                "Make sure the huggingface_qwen_generated directory exists."
            )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Model path: {model_path}")
    
    # Test model loading to ensure it works
    try:
        print("Testing model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"✅ Model loaded successfully!")
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


def run_bigcode_evaluation(model_path, args, timestamp):
    """Run bigcode-evaluation-harness evaluation."""
    
    print(f"\\n🚀 Running BigCode evaluation...")
    print(f"   Model: {model_path}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Samples: {args.limit if args.limit is not None else 'all'}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Sampling: {args.do_sample}")
    
    # Create output directory and convert to absolute path for BigCode harness
    output_dir = Path("evaluation_results") / f"{args.method}_{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_abs = Path.cwd() / output_dir
    
    # Convert relative model path to absolute path for BigCode harness
    if model_path.startswith("./huggingface_qwen25-7b-coder-caa"):
        abs_model_path = str(Path.cwd() / model_path.lstrip("./"))
    else:
        abs_model_path = model_path
    
    # Prepare BigCode command (using accelerate launch for proper device management)
    cmd = [
        "accelerate", "launch", "--gpu_ids", str(args.device), "main.py",
        "--model", abs_model_path,
        "--tasks", args.dataset,
        "--batch_size", str(args.batch_size),
        "--temperature", str(args.temperature),
        "--do_sample", args.do_sample,
        "--top_p", "0.95",
        "--precision", args.precision,
        "--seed", str(args.seed),
        "--allow_code_execution",
        "--trust_remote_code",
        "--save_generations",
        "--save_generations_path", str(output_dir_abs / "generations.json"),
        "--save_references",  
        "--save_references_path", str(output_dir_abs / "references.json"),
        "--metric_output_path", str(output_dir_abs / "metrics.json"),
    ]
    
    # Add limit only if specified (None means evaluate full dataset)
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    
    # Add memory management parameters
    if args.load_in_8bit:
        cmd.append("--load_in_8bit")
    
    print(f"\\n🔧 BigCode command:")
    print(" ".join(cmd))
    print(f"🎯 Using GPU device: {args.device}")
    
    # Run evaluation
    start_time = time.time()
    
    # Set CUDA_VISIBLE_DEVICES to control GPU selection
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd="./bigcode-evaluation-harness",
            env=env
        )
        
        duration = time.time() - start_time
        
        print(f"\\n✅ Evaluation completed successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        
        # Load results
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            print(f"\\n📊 Results:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for subkey, subvalue in value.items():
                        print(f"     {subkey}: {subvalue}")
                else:
                    print(f"   {key}: {value}")
            
            return metrics, output_dir, duration
            
        else:
            print(f"⚠️ Metrics file not found at {metrics_path}")
            return {}, output_dir, duration
    
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ BigCode evaluation failed!")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    
    except Exception as e:
        print(f"\\n❌ Error running evaluation: {e}")
        raise


def log_results_to_wandb(metrics, args, duration, output_dir):
    """Log final results to WandB."""
    
    print(f"\\n📈 Logging results to WandB...")
    
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
        # Fallback - can't determine exact numbers
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
            
        if (output_dir / "generations.json").exists():
            wandb.save(str(output_dir / "generations.json"), base_path=str(output_dir.parent))
            
        if (output_dir / "references.json").exists():
            wandb.save(str(output_dir / "references.json"), base_path=str(output_dir.parent))
            
        print("✅ Artifacts uploaded to WandB")
        
    except Exception as e:
        print(f"⚠️ Error uploading artifacts: {e}")
    
    print(f"✅ Results logged to WandB successfully!")


def main():
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
            model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
        elif args.method == "caa":
            model_path = "./huggingface_qwen25-7b-coder-caa"
        
        # Run BigCode evaluation
        metrics, output_dir, duration = run_bigcode_evaluation(model_path, args, timestamp)
        
        # Log results to WandB
        log_results_to_wandb(metrics, args, duration, output_dir)
        
        print(f"\\n🎉 Evaluation completed successfully!")
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