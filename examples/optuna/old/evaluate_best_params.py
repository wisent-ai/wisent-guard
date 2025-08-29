#!/usr/bin/env python3
"""
Evaluate Best Parameters from Optuna Study

This script demonstrates how to evaluate the best parameters found by Optuna
on new datasets or with different configurations.

USAGE EXAMPLES:

1. Evaluate best params from existing study:
   python evaluate_best_params.py --study-db outputs/optimization_pipeline/optuna_study_20250729_161937.db

2. Evaluate on different dataset:
   python evaluate_best_params.py --study-db path/to/study.db --test-dataset aime --test-limit 50

3. Use different model:
   python evaluate_best_params.py --study-db path/to/study.db --model-name Qwen/Qwen2.5-7B

4. Load saved config and override specific settings:
   python evaluate_best_params.py --study-db path/to/study.db --config path/to/config.json --test-limit 500

5. Manual parameter specification:
   python evaluate_best_params.py --manual-eval --layer-id 10 --steering-method caa --steering-alpha 0.18
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add wisent-guard to path
sys.path.append(str(Path(__file__).parent.parent.parent))


from wisent_guard.core.optuna.steering.optuna_pipeline import OptimizationConfig, OptimizationPipeline


def evaluate_from_study(args) -> dict[str, Any]:
    """Evaluate using parameters from existing Optuna study."""
    print(f"📊 Loading study from: {args.study_db}")

    # Load pipeline and study
    pipeline, study = OptimizationPipeline.from_saved_study(
        study_path=args.study_db,
        config_path=args.config,
        override_config={
            "model_name": args.model_name,
            "test_dataset": args.test_dataset,
            "test_limit": args.test_limit,
            "device": args.device,
            "batch_size": args.batch_size,
        }
        if any([args.model_name, args.test_dataset, args.test_limit, args.device, args.batch_size])
        else None,
    )

    print(f"🏆 Best trial: {study.best_trial.number}")
    print(f"📈 Best validation score: {study.best_value:.4f}")
    print(f"⚙️  Best parameters: {study.best_params}")

    # Run evaluation
    if args.cross_dataset:
        # Evaluate on multiple datasets
        datasets = ["gsm8k", "hendrycks_math", "aime"]
        results = {}

        for dataset in datasets:
            print(f"\n🔬 Evaluating on {dataset}...")
            try:
                result = pipeline.evaluate_on_dataset(study.best_params, dataset, args.test_limit)
                results[dataset] = result

                # Print summary
                baseline_acc = result["baseline_benchmark_metrics"]["accuracy"]
                steered_acc = result["steered_benchmark_metrics"]["accuracy"]
                improvement = result["accuracy_improvement"]

                print(f"   Baseline:   {baseline_acc:.4f}")
                print(f"   Steered:    {steered_acc:.4f}")
                print(f"   Improvement: {improvement:+.4f}")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[dataset] = {"error": str(e)}

        return results
    # Single dataset evaluation
    return pipeline.evaluate_only(study.best_params)


def evaluate_manual_params(args) -> dict[str, Any]:
    """Evaluate using manually specified parameters."""
    print("🔧 Manual evaluation mode")

    # Create manual parameters
    best_params = {
        "layer_id": args.layer_id,
        "steering_method": args.steering_method,
        "steering_alpha": args.steering_alpha,
        "probe_type": "logistic_regression",
        "probe_c": 1.0,
    }

    # Add method-specific parameters
    if args.steering_method == "dac":
        best_params.update(
            {"entropy_threshold": args.entropy_threshold, "ptop": args.ptop, "max_alpha": args.max_alpha}
        )

    print(f"⚙️  Parameters: {best_params}")

    # Create pipeline with configuration matching minimal example
    config_kwargs = {
        "model_name": args.model_name,
        "test_dataset": args.test_dataset,
        "test_limit": args.test_limit,
        "layer_search_range": (args.layer_id, args.layer_id),  # Single layer range
        "train_limit": 20,
        "val_limit": 20,
        "temperature": 0.0,
        "do_sample": False,
        "output_dir": "outputs/evaluate_manual",
        "cache_dir": "cache/evaluate_manual",
    }

    # Only add non-None values to avoid overriding defaults
    if args.device:
        config_kwargs["device"] = args.device
    if args.batch_size:
        config_kwargs["batch_size"] = args.batch_size

    config = OptimizationConfig(**config_kwargs)

    pipeline = OptimizationPipeline(config)
    return pipeline.evaluate_only(best_params)


def print_results(results: dict[str, Any], cross_dataset: bool = False):
    """Print evaluation results in a nice format."""
    if cross_dataset:
        print("\n" + "=" * 80)
        print("🏆 CROSS-DATASET EVALUATION RESULTS")
        print("=" * 80)

        for dataset, result in results.items():
            if "error" in result:
                print(f"{dataset}: ❌ {result['error']}")
            else:
                baseline = result["baseline_benchmark_metrics"]["accuracy"]
                steered = result["steered_benchmark_metrics"]["accuracy"]
                improvement = result["accuracy_improvement"]
                print(f"{dataset:15} | Baseline: {baseline:.4f} | Steered: {steered:.4f} | Δ: {improvement:+.4f}")
    else:
        print("\n" + "=" * 60)
        print("🏆 EVALUATION RESULTS")
        print("=" * 60)

        baseline_metrics = results["baseline_benchmark_metrics"]
        steered_metrics = results["steered_benchmark_metrics"]

        print(f"📊 Dataset: {results.get('config', {}).get('test_dataset', 'Unknown')}")
        print(f"📈 Test samples: {results.get('num_test_samples', 'Unknown')}")
        print(f"🎯 Best params: {results['best_trial_params']}")
        print()
        print(f"Baseline accuracy:  {baseline_metrics['accuracy']:.4f}")
        print(f"Steered accuracy:   {steered_metrics['accuracy']:.4f}")
        print(f"Improvement:        {results['accuracy_improvement']:+.4f}")
        print(f"Probe AUC:          {results['test_probe_metrics']['auc']:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate best parameters from Optuna study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--manual-eval", action="store_true", help="Use manually specified parameters instead of loading from study"
    )

    # Study loading options
    parser.add_argument("--study-db", type=str, help="Path to Optuna study database (.db file)")
    parser.add_argument("--config", type=str, help="Path to saved configuration JSON")

    # Evaluation options
    parser.add_argument("--test-dataset", type=str, default="gsm8k", help="Dataset to evaluate on (default: gsm8k)")
    parser.add_argument("--test-limit", type=int, help="Limit number of test samples")
    parser.add_argument(
        "--cross-dataset", action="store_true", help="Evaluate on multiple datasets (gsm8k, hendrycks_math, aime)"
    )

    # Model configuration
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--device", type=str, help="Device to use (cuda, cpu, mps)")
    parser.add_argument("--batch-size", type=int, help="Override batch size")

    # Manual parameter specification
    parser.add_argument("--layer-id", type=int, default=15, help="Layer ID for manual evaluation")
    parser.add_argument(
        "--steering-method",
        type=str,
        default="caa",
        choices=["caa", "dac"],
        help="Steering method for manual evaluation",
    )
    parser.add_argument("--steering-alpha", type=float, default=0.5, help="Steering alpha for manual evaluation")
    parser.add_argument("--entropy-threshold", type=float, default=1.5, help="DAC entropy threshold (manual mode)")
    parser.add_argument("--ptop", type=float, default=0.5, help="DAC ptop parameter (manual mode)")
    parser.add_argument("--max-alpha", type=float, default=2.0, help="DAC max alpha parameter (manual mode)")

    # Utility options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Validate arguments
    if not args.manual_eval and not args.study_db:
        parser.error("Must provide --study-db or use --manual-eval")

    # Set default model for manual evaluation if not provided
    if args.manual_eval and not args.model_name:
        args.model_name = "realtreetune/rho-1b-sft-GSM8K"  # Same as minimal example

    try:
        # Run evaluation
        results = evaluate_manual_params(args) if args.manual_eval else evaluate_from_study(args)

        # Print results
        print_results(results, args.cross_dataset)

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {args.output}")

        return results

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
