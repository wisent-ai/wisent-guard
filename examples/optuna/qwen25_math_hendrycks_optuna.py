"""
Qwen2.5-Math-7B Hendrycks Math Optimization with Optuna

This script demonstrates a mathematics-focused optimization pipeline for Qwen2.5-Math-7B
model using the MATH (hendrycks_math) dataset with Optuna hyperparameter optimization,
SQLite persistence, and WandB experiment tracking.

RECOMMENDED BATCH SIZES:
- Qwen2.5-Math-7B: batch_size=1-2 (math problems can be very long)
- Training samples: 50-100 (MATH has 12500+ problems total)
- Validation samples: 30-50 (enough for reliable optimization signal)
- Test samples: 50-100 (comprehensive final evaluation)

STEERING METHOD FOCUS:
- CAA (Contrastive Activation Addition) - Optimized for mathematical reasoning

DATASET:
- Training: hendrycks_math (MATH dataset - competition mathematics)
- Validation: hendrycks_math (same dataset for consistency)
- Test: hendrycks_math (same dataset for testing)

MODEL SPECIALIZATION:
- Qwen2.5-Math-7B uses Chain-of-Thought (COT) reasoning
- Requires higher max_tokens (1024-2048) for detailed mathematical solutions
- Benefits from deterministic generation (low temperature)

CONTRASTIVE PAIRS GENERATION:
Uses specialized MATH extractors that create mathematical reasoning pairs:
- Correct: Proper mathematical solution with reasoning steps
- Incorrect: Flawed mathematical reasoning or wrong approach

USAGE:
    # Basic usage with default settings
    python qwen25_math_hendrycks_optuna.py

    # Custom model path and batch size
    python qwen25_math_hendrycks_optuna.py --model-path Qwen/Qwen2.5-Math-7B-Instruct --batch-size 1

    # Enable WandB logging
    python qwen25_math_hendrycks_optuna.py --use-wandb --wandb-project qwen-math-optimization

    # Quick test run
    python qwen25_math_hendrycks_optuna.py --n-trials 10 --train-limit 20 --val-limit 15

EXPECTED RESULTS:
- Baseline Qwen2.5-Math-7B typically achieves ~25-40% on MATH (very challenging dataset)
- Optimal steering can improve mathematical reasoning performance
- Over-steering (high α values) typically degrades logical consistency
- Layer selection is crucial - deeper layers (20-28) often work best for reasoning

OUTPUTS:
- SQLite database: optuna_studies.db (persistent across runs)
- Results: outputs/qwen25_math_hendrycks_optimization/
- WandB logs: https://wandb.ai/your-project/qwen-math-optimization

MEMORY OPTIMIZATION:
- Uses efficient attention mechanisms
- Caches activations efficiently
- Consider CUDA_VISIBLE_DEVICES to limit GPU usage
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# Add wisent-guard to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from wisent_guard.core.optuna.optuna_pipeline import OptimizationConfig, OptimizationPipeline


def get_recommended_config_for_qwen25_math() -> Dict[str, Any]:
    """Get recommended configuration values for Qwen2.5-Math-7B MATH optimization."""
    return {
        "model_name": "Qwen/Qwen2.5-Math-7B-Instruct",  # Qwen2.5-Math specialized for mathematics
        "batch_size": 8,
        "max_new_tokens": 1024,  # Higher for COT mathematical reasoning
        "layer_search_range": (18, 27),  # Qwen2.5-Math has 28 layers (0-27), deeper layers for complex reasoning
        "train_limit": 75,  # Reasonable for MATH dataset
        "contrastive_pairs_limit": 50,  # Bounded by train_limit
        "val_limit": 40,
        "test_limit": 75,
        "n_trials": 25,  # Focused optimization for CAA only
        "n_startup_trials": 8,  # Some random exploration
    }


def create_qwen25_math_config(args) -> OptimizationConfig:
    """Create optimized configuration for Qwen2.5-Math-7B MATH optimization."""

    # Get base recommendations
    defaults = get_recommended_config_for_qwen25_math()

    return OptimizationConfig(
        # Model configuration - Qwen2.5-Math-7B specialized for mathematics
        model_name=args.model_path or defaults["model_name"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Dataset configuration - Mathematics focus with MATH dataset
        train_dataset="hendrycks_math",  # MATH dataset - competition mathematics
        val_dataset="hendrycks_math",  # Same dataset for consistency
        test_dataset="hendrycks_math",  # Same dataset for testing
        # Training configuration
        train_limit=args.train_limit or defaults["train_limit"],
        contrastive_pairs_limit=args.contrastive_pairs_limit or defaults["contrastive_pairs_limit"],
        # Evaluation configuration
        val_limit=args.val_limit or defaults["val_limit"],
        test_limit=args.test_limit or defaults["test_limit"],
        # Layer search configuration - Qwen2.5-Math has 28 layers (0-27)
        # Deeper layers (18-27) typically capture complex mathematical reasoning
        layer_search_range=args.layer_range or defaults["layer_search_range"],
        # Probe type - Fixed to logistic regression
        probe_type="logistic_regression",
        # Steering methods - Focus on CAA only for mathematical reasoning
        steering_methods=["caa"],
        # Optuna study configuration
        study_name=args.study_name or "qwen25_math_hendrycks_optimization",
        db_url=f"sqlite:///{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/optuna_studies.db",
        n_trials=args.n_trials or defaults["n_trials"],
        n_startup_trials=args.n_startup_trials or defaults["n_startup_trials"],
        sampler="TPE",  # Tree-structured Parzen Estimator
        pruner="MedianPruner",  # Aggressive pruning for efficiency
        # WandB configuration
        wandb_project=args.wandb_project or "qwen25-math-hendrycks-optimization",
        use_wandb=args.use_wandb,
        # Generation configuration - Optimized for mathematical reasoning
        batch_size=args.batch_size or defaults["batch_size"],
        max_length=2048,  # Longer for complex mathematical solutions
        max_new_tokens=defaults["max_new_tokens"],
        temperature=0.0,  # Deterministic for mathematical reasoning
        do_sample=False,  # Deterministic generation for math
        # Performance optimization
        seed=42,
        # Output configuration
        output_dir="outputs/qwen25_math_hendrycks_optimization",
        cache_dir="cache/qwen25_math_hendrycks_optimization",
        # Search space constraints
        max_layers_to_search=10,  # Search range of 10 layers (18-27)
        early_stopping_patience=12,  # More patience for math reasoning
    )


class Qwen25MathPipeline(OptimizationPipeline):
    """Specialized pipeline for Qwen2.5-Math-7B optimization with MATH dataset."""

    def _objective_function(self, trial) -> float:
        """Enhanced objective function for mathematical reasoning tasks with CAA focus."""
        try:
            self.logger.info(f"🔬 Trial {trial.number}: Starting MATH dataset optimization")

            # Sample layer and probe hyperparameters
            layer_id = trial.suggest_int(
                "layer_id", self.config.layer_search_range[0], self.config.layer_search_range[1]
            )

            # Fixed probe type and regularization
            probe_type = self.config.probe_type  # Always logistic_regression
            probe_c = 1.0  # Default regularization strength

            # CAA hyperparameters - optimized for mathematical reasoning
            steering_method = "caa"  # Fixed to CAA only

            steering_alpha = trial.suggest_float(
                "steering_alpha",
                0.05,
                1.5,
                step=0.05,  # Fine-grained control for math reasoning
            )

            steering_params = {
                "steering_alpha": steering_alpha,
            }

            self.logger.info(f"🎯 Trial {trial.number}: CAA with α={steering_alpha:.3f} (Layer {layer_id})")

            # Step 1: Train and evaluate probe
            probe_score = self._train_and_evaluate_probe(trial, layer_id, probe_type, probe_c)
            self.logger.info(f"📊 Trial {trial.number}: Probe {probe_type} AUC = {probe_score:.4f}")

            # Step 2: Train CAA steering method
            steering_instance = self._train_steering_method(trial, steering_method, layer_id, steering_params)

            # Step 3: Evaluate steering on validation set
            validation_accuracy = self._evaluate_steering_on_validation(
                steering_instance, steering_method, layer_id, steering_params, trial.number, trial
            )

            self.logger.info(f"🎯 Trial {trial.number}: Validation MATH accuracy = {validation_accuracy:.4f}")
            trial.report(validation_accuracy, step=1)

            # Enhanced WandB logging with math-specific metrics
            metrics = {
                "validation_accuracy": validation_accuracy,
                "probe_score": probe_score,
                "method": steering_method,
                "layer": layer_id,
                "task_type": "mathematics",
                "dataset": "hendrycks_math",
                "model": "qwen2.5-math-7b",
                "steering_alpha": steering_alpha,
            }
            self._log_trial_to_wandb(trial, metrics)

            return validation_accuracy

        except Exception as e:
            self.logger.error(f"❌ Trial {trial.number} failed: {e}")
            import traceback

            traceback.print_exc()
            return 0.0

    def _log_enhanced_results(self, study, final_results):
        """Log enhanced results with MATH-specific analysis."""
        self.logger.info("=" * 80)
        self.logger.info("🧮 QWEN2.5-MATH-7B HENDRYCKS MATH OPTIMIZATION RESULTS")
        self.logger.info("=" * 80)

        best_trial = study.best_trial
        best_method = "caa"  # Always CAA
        best_layer = best_trial.params.get("layer_id", -1)
        best_alpha = best_trial.params.get("steering_alpha", 0.0)
        best_norm_method = best_trial.params.get("normalization_method", "none")

        self.logger.info(f"🥇 Best Method: {best_method.upper()}")
        self.logger.info(f"📊 Best Layer: {best_layer}")
        self.logger.info(f"⚡ Best Alpha: {best_alpha:.4f}")
        self.logger.info(f"🔧 Best Normalization: {best_norm_method}")
        self.logger.info(f"🎯 Best Validation Accuracy: {study.best_value:.4f}")

        baseline_acc = final_results["baseline_benchmark_metrics"]["accuracy"]
        steered_acc = final_results["steered_benchmark_metrics"]["accuracy"]
        improvement = final_results["accuracy_improvement"]

        self.logger.info("📈 Test Results on MATH Dataset:")
        self.logger.info(f"   Baseline:  {baseline_acc:.4f}")
        self.logger.info(f"   Steered:   {steered_acc:.4f}")
        self.logger.info(f"   Improvement: {improvement:+.4f}")

        # Mathematics-specific insights
        self.logger.info("🧮 MATH Dataset Task Insights:")
        self.logger.info(f"   - Model: Qwen2.5-Math-7B (28 layers, specialized for mathematics)")
        self.logger.info(f"   - Dataset: MATH (Competition mathematics with step-by-step solutions)")
        self.logger.info(f"   - Reasoning: Chain-of-Thought with {self.config.max_new_tokens} max tokens")
        self.logger.info(f"   - Generation: Deterministic (temperature=0.0) for consistent math")
        self.logger.info(
            f"   - Best layer {best_layer} suggests {'early' if best_layer < 11 else 'middle' if best_layer < 22 else 'deep'} reasoning processing"
        )

        # Performance context for MATH
        if baseline_acc > 0.35:
            self.logger.info("🎉 Excellent baseline performance on MATH!")
        elif baseline_acc > 0.25:
            self.logger.info("👍 Good baseline performance for MATH")
        elif baseline_acc > 0.15:
            self.logger.info("📚 Reasonable baseline - MATH is very challenging")
        else:
            self.logger.info("🤔 Lower baseline - check model loading and mathematical prompt format")

        # CAA-specific analysis
        caa_trials = [t for t in study.trials if t.value is not None]
        if len(caa_trials) > 1:
            caa_values = [t.value for t in caa_trials]
            self.logger.info(f"📊 CAA Method Statistics:")
            self.logger.info(f"   Trials: {len(caa_values)}")
            self.logger.info(f"   Mean: {sum(caa_values) / len(caa_values):.4f}")
            self.logger.info(f"   Best: {max(caa_values):.4f}")
            self.logger.info(
                f"   Std: {(sum((x - sum(caa_values) / len(caa_values)) ** 2 for x in caa_values) / len(caa_values)) ** 0.5:.4f}"
            )

            # Analyze normalization impact
            none_trials = [t.value for t in caa_trials if t.params.get("normalization_method") == "none"]
            l2_trials = [t.value for t in caa_trials if t.params.get("normalization_method") == "l2_unit"]

            if none_trials and l2_trials:
                self.logger.info(f"   No normalization avg: {sum(none_trials) / len(none_trials):.4f}")
                self.logger.info(f"   L2 normalization avg: {sum(l2_trials) / len(l2_trials):.4f}")

        self.logger.info("=" * 80)

        # Call parent logging
        self._log_final_results_to_wandb(study, final_results)


def main():
    """Main entry point for Qwen2.5-Math-7B MATH optimization."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Math-7B MATH Dataset Task Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model (default: Qwen/Qwen2.5-Math-7B-Instruct)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: 1 for Qwen2.5-Math-7B)",
    )

    # Dataset configuration
    parser.add_argument(
        "--train-limit", type=int, default=None, help="Number of training samples to load (default: 75)"
    )
    parser.add_argument(
        "--contrastive-pairs-limit",
        type=int,
        default=None,
        help="Number of contrastive pairs for steering training (default: 50, bounded by train-limit)",
    )
    parser.add_argument(
        "--val-limit", type=int, default=None, help="Number of validation samples to load (default: 40)"
    )
    parser.add_argument("--test-limit", type=int, default=None, help="Number of test samples to load (default: 75)")

    # Optimization configuration
    parser.add_argument(
        "--study-name", type=str, default=None, help="Optuna study name (default: qwen25_math_hendrycks_optimization)"
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Number of optimization trials (default: 25)")
    parser.add_argument(
        "--n-startup-trials", type=int, default=None, help="Random exploration trials before TPE kicks in (default: 8)"
    )
    parser.add_argument(
        "--layer-range", type=int, nargs=2, default=None, help="Layer search range as two integers (default: 18 27)"
    )

    # WandB configuration
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable WandB experiment tracking (requires 'wandb login' first)"
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")

    # Utility options
    parser.add_argument(
        "--quick-test", action="store_true", help="Quick test run (10 trials, 20 train, 15 val samples)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default=None, help="Log output to file (in addition to console)")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Configure logging handlers
    handlers = [logging.StreamHandler()]  # Console output
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))  # File output

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=handlers
    )

    logger = logging.getLogger(__name__)

    # Quick test mode overrides
    if args.quick_test:
        args.n_trials = 10
        args.train_limit = 20
        args.val_limit = 15
        args.test_limit = 20
        logger.info("🚀 Quick test mode enabled")

    # Display configuration
    logger.info("🧮 QWEN2.5-MATH-7B HENDRYCKS MATH OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("🔧 CONFIGURATION:")
    logger.info(f"   Model: {args.model_path or get_recommended_config_for_qwen25_math()['model_name']}")
    logger.info(f"   Batch Size: {args.batch_size or get_recommended_config_for_qwen25_math()['batch_size']}")
    logger.info(f"   Trials: {args.n_trials or get_recommended_config_for_qwen25_math()['n_trials']}")
    logger.info(f"   Train/Val/Test: {args.train_limit or 75}/{args.val_limit or 40}/{args.test_limit or 75}")
    logger.info(f"   Dataset: MATH (Hendrycks) - Competition mathematics with step-by-step reasoning")
    logger.info(f"   Method: CAA only (optimized for mathematical reasoning)")
    logger.info(f"   WandB: {'Enabled' if args.use_wandb else 'Disabled'}")

    if torch.cuda.is_available():
        logger.info(f"🔥 GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} devices)")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        # Memory warning for Qwen2.5-Math-7B
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 18:
            logger.warning(f"⚠️  GPU has {vram_gb:.1f}GB VRAM. Qwen2.5-Math-7B requires ~16GB+. Consider:")
            logger.warning("   - Using smaller batch size (--batch-size 1)")
            logger.warning("   - Setting CUDA_VISIBLE_DEVICES to limit GPU usage")
    else:
        logger.error("❌ No CUDA detected - Qwen2.5-Math-7B requires GPU!")
        return None

    logger.info("=" * 80)

    # Create configuration and pipeline
    try:
        config = create_qwen25_math_config(args)
        pipeline = Qwen25MathPipeline(config)

        # Run optimization
        logger.info("🚀 Starting MATH optimization for Qwen2.5-Math-7B...")
        results = pipeline.run_optimization()

        # Enhanced result display
        pipeline._log_enhanced_results(pipeline._create_optuna_study(), results)

        logger.info("✅ Qwen2.5-Math-7B MATH optimization completed successfully!")
        logger.info(f"📂 Results saved to: {config.output_dir}")
        logger.info(f"🗄️  Study database: {config.db_url}")

        if config.use_wandb:
            logger.info("📊 WandB: Check your WandB dashboard for run details")

        return results

    except KeyboardInterrupt:
        logger.info("🛑 Optimization interrupted by user")
        return None

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        # Cleanup
        if "pipeline" in locals():
            pipeline.cleanup_memory()


if __name__ == "__main__":
    main()
