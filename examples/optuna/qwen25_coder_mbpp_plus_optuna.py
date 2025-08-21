"""
Qwen2.5-Coder-7B MBPP Plus Optimization with Optuna

This script demonstrates a coding-focused optimization pipeline for Qwen2.5-Coder-7B
model using MBPP Plus dataset with Optuna hyperparameter optimization,
SQLite persistence, and WandB experiment tracking.

RECOMMENDED BATCH SIZES:
- Qwen2.5-Coder-7B: batch_size=2-4 (depending on GPU memory)
- Training samples: 100-200 (MBPP Plus has 399 problems total)
- Validation samples: 50-100 (enough for reliable optimization signal)
- Test samples: 100-200 (comprehensive final evaluation)

STEERING METHODS INVESTIGATED:
1. CAA (Contrastive Activation Addition) - Classic vector steering
2. DAC (Dynamic Activation Control) - Adaptive steering with entropy thresholds
3. DAC_TENSOR (Tensor-based DAC) - Multi-dimensional steering with attention heads

DATASETS (CODING FOCUS):
- Training: mbpp_plus (Extended Python programming problems)
- Validation: mbpp_plus (same dataset for consistency)
- Test: mbpp_plus (same dataset for testing)

CONTRASTIVE PAIRS GENERATION:
Uses specialized MBPP Plus extractors that create "obscured correct answer" pairs:
- Correct: Original working code solution
- Incorrect: Syntactically corrupted code (missing tokens, wrong syntax)
  that "obscures" the correct answer

USAGE:
    # Basic usage with default settings
    HF_ALLOW_CODE_EVAL="1" python qwen25_coder_mbpp_plus_optuna.py

    # Custom model path and batch size
    HF_ALLOW_CODE_EVAL="1" python qwen25_coder_mbpp_plus_optuna.py --model-path Qwen/Qwen2.5-Coder-7B-Instruct --batch-size 2

    # Enable WandB logging
    HF_ALLOW_CODE_EVAL="1" python qwen25_coder_mbpp_plus_optuna.py --use-wandb --wandb-project qwen-mbpp-plus-optimization

    # Quick test run
    HF_ALLOW_CODE_EVAL="1" python qwen25_coder_mbpp_plus_optuna.py --n-trials 10 --train-limit 50 --val-limit 30

EXPECTED RESULTS:
- Baseline Qwen2.5-Coder-7B typically achieves ~50-70% on MBPP Plus (challenging extended dataset)
- Optimal steering can improve performance on complex coding tasks
- Over-steering (high α values) typically degrades performance
- Layer selection is crucial - middle layers (16-24) often work best for Qwen

OUTPUTS:
- SQLite database: optuna_studies.db (persistent across runs)
- Results: outputs/qwen25_coder_mbpp_plus_optimization/
- WandB logs: https://wandb.ai/your-project/qwen-mbpp-plus-optimization

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
from wisent_guard.core.steering_methods_tensor.dac_attention import DAC as TensorDAC


def get_recommended_config_for_qwen25_coder() -> Dict[str, Any]:
    """Get recommended configuration values for Qwen2.5-Coder-7B MBPP Plus optimization."""
    return {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",  # Qwen2.5-Coder specialized for coding
        "batch_size": 32,  # RTX 4090 24 GB
        "max_new_tokens": 512,  # Longer for coding tasks - Qwen can handle complex code
        "layer_search_range": (16, 28),  # Qwen has 32 layers (0-31), middle-to-late layers work well for code
        "train_limit": 378,  # Good balance for MBPP Plus
        "contrastive_pairs_limit": 378,  # Bounded by train_limit
        "val_limit": 378,
        "test_limit": 150,
        "n_trials": 50,  # More trials for better optimization
        "n_startup_trials": 10,  # More random exploration
    }


def create_qwen25_coder_config(args) -> OptimizationConfig:
    """Create optimized configuration for Qwen2.5-Coder-7B MBPP Plus optimization."""

    # Get base recommendations
    defaults = get_recommended_config_for_qwen25_coder()

    return OptimizationConfig(
        # Model configuration - Qwen2.5-Coder-7B specialized for coding
        model_name=args.model_path or defaults["model_name"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Dataset configuration - Coding focus with MBPP Plus
        train_dataset="mbpp_plus",  # Extended Python programming problems
        val_dataset="mbpp_plus",  # Same dataset for consistency
        test_dataset="mbpp",  # Same dataset for testing
        # Training configuration
        train_limit=args.train_limit or defaults["train_limit"],
        contrastive_pairs_limit=args.contrastive_pairs_limit or defaults["contrastive_pairs_limit"],
        # Evaluation configuration
        val_limit=args.val_limit or defaults["val_limit"],
        test_limit=args.test_limit or defaults["test_limit"],
        # Layer search configuration - Qwen has 32 layers (0-31)
        # Middle-to-late layers (16-24) typically capture code semantics well
        layer_search_range=args.layer_range or defaults["layer_search_range"],
        # Probe type - Fixed to logistic regression
        probe_type="logistic_regression",
        # Steering methods - Currently implemented methods
        # steering_methods=["caa", "dac", "dac_tensor"],
        # steering_methods=["caa", "dac_tensor"],
        steering_methods=["caa"],
        # Optuna study configuration
        study_name=args.study_name or "qwen25_coder_mbpp_plus_optimization",
        db_url=f"sqlite:///{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/optuna_studies.db",
        n_trials=args.n_trials or defaults["n_trials"],
        n_startup_trials=args.n_startup_trials or defaults["n_startup_trials"],
        sampler="TPE",  # Tree-structured Parzen Estimator
        pruner="MedianPruner",  # Aggressive pruning for efficiency
        # WandB configuration
        wandb_project=args.wandb_project or "qwen25-coder-mbpp-plus-optimization",
        use_wandb=args.use_wandb,
        # Generation configuration - Optimized for coding tasks
        batch_size=args.batch_size or defaults["batch_size"],
        max_length=1024,  # Longer for complex coding problems
        max_new_tokens=defaults["max_new_tokens"],
        temperature=0.1,  # Lower temperature for more deterministic code generation
        do_sample=True,
        # Performance optimization
        seed=42,
        # Output configuration
        output_dir="outputs/qwen25_coder_mbpp_plus_optimization",
        cache_dir="cache/qwen25_coder_mbpp_plus_optimization",
        # Search space constraints
        max_layers_to_search=9,  # Search more layers for better coverage
        early_stopping_patience=15,  # More patience for specialized model
    )


class Qwen25CoderMBPPPlusPipeline(OptimizationPipeline):
    """Specialized pipeline for Qwen2.5-Coder-7B optimization with MBPP Plus dataset."""

    def _objective_function(self, trial) -> float:
        """Enhanced objective function for coding tasks with method-specific hyperparameter spaces."""
        try:
            self.logger.info(f"🔬 Trial {trial.number}: Starting MBPP Plus optimization")

            # Sample layer and probe hyperparameters
            layer_id = trial.suggest_int(
                "layer_id", self.config.layer_search_range[0], self.config.layer_search_range[1]
            )

            # Fixed probe type and regularization
            probe_type = self.config.probe_type  # Always logistic_regression
            probe_c = 1.0  # Default regularization strength

            # Sample steering method and method-specific hyperparameters
            steering_method = trial.suggest_categorical("steering_method", self.config.steering_methods)

            if steering_method == "caa":
                # CAA hyperparameters - adjusted for coding-specialized model
                steering_alpha = trial.suggest_float(  # maps to `strength`
                    "steering_alpha",
                    0.05,
                    1.5,
                    step=0.05,  # Moderate range for specialized model
                )

                steering_params = {
                    "steering_alpha": steering_alpha,
                }

            elif steering_method == "dac":
                # DAC: Dynamic control with entropy-based adaptation
                steering_params = {
                    "base_strength": trial.suggest_float(
                        "base_strength", 0.3, 1.2, step=0.05
                    ),  # Moderate for coding model
                    "ptop": trial.suggest_float("ptop", 0.25, 0.55, step=0.05),
                    "max_alpha": trial.suggest_float("max_alpha", 0.8, 2.5, step=0.1),  # Reasonable max for coding
                    "entropy_threshold": trial.suggest_float("entropy_threshold", 1.8, 3.5, step=0.1),
                }

            elif steering_method == "dac_tensor":
                # DAC_TENSOR: Tensor-based steering with attention heads
                steering_params = {
                    "steering_alpha": trial.suggest_float("steering_alpha", 0.0, 1.5, step=0.05),  # Steering strength
                    "icl_examples": trial.suggest_int("icl_examples", 0, 4),  # In-context learning examples
                }

            else:
                raise ValueError(f"steering_method: {steering_method} not implemented")

            alpha_str = (
                f"{steering_params.get('steering_alpha', 'N/A'):.3f}"
                if steering_params.get("steering_alpha") is not None
                else "N/A"
            )
            self.logger.info(
                f"🎯 Trial {trial.number}: {steering_method.upper()} with α={alpha_str} (Layer {layer_id})"
            )

            # Step 1: Train and evaluate probe
            probe_score = self._train_and_evaluate_probe(trial, layer_id, probe_type, probe_c)
            self.logger.info(f"📊 Trial {trial.number}: Probe {probe_type} AUC = {probe_score:.4f}")

            # Step 2: Train steering method
            if steering_method == "dac_tensor":
                steering_instance = self._train_tensor_dac(trial, layer_id, steering_params)
            else:
                steering_instance = self._train_steering_method(trial, steering_method, layer_id, steering_params)

            # Step 3: Evaluate steering on validation set
            if steering_method == "dac_tensor":
                validation_accuracy = self._evaluate_tensor_dac_on_validation(
                    steering_instance, steering_params, layer_id, trial
                )
            else:
                validation_accuracy = self._evaluate_steering_on_validation(
                    steering_instance, steering_method, layer_id, steering_params, trial.number, trial
                )

            self.logger.info(f"🎯 Trial {trial.number}: Validation MBPP Plus accuracy = {validation_accuracy:.4f}")
            trial.report(validation_accuracy, step=1)

            # Enhanced WandB logging with coding-specific metrics
            metrics = {
                "validation_accuracy": validation_accuracy,
                "probe_score": probe_score,
                "method": steering_method,
                "layer": layer_id,
                "task_type": "coding",
                "dataset": "mbpp_plus",
                "model": "qwen2.5-coder-7b",
            }
            self._log_trial_to_wandb(trial, metrics)

            return validation_accuracy

        except Exception as e:
            self.logger.error(f"❌ Trial {trial.number} failed: {e}")
            import traceback

            traceback.print_exc()
            return 0.0

    def _train_tensor_dac(self, trial, layer_id: int, hyperparams: dict) -> TensorDAC:
        """Train tensor-based DAC using standard pattern like CAA."""
        self.logger.info(f"🧪 Training Tensor DAC for layer {layer_id}")

        # Initialize tensor DAC (no internal model loading!)
        tensor_dac = TensorDAC(
            device=self.device,
            max_new_tokens=hyperparams.get("max_new_tokens", 384),
            icl_examples=hyperparams.get("icl_examples", 2),
            legacy_behavior=hyperparams.get("legacy_behavior", False),
        )

        # Inject model reference to avoid duplicate loading
        tensor_dac.load_model_with_reference(self.model, self.tokenizer)

        # Use standard contrastive pair creation (reuses existing pipeline method)
        contrastive_limit = min(self.config.contrastive_pairs_limit, len(self.train_samples))
        contrastive_pairs = self._create_contrastive_pairs(
            self.train_samples, layer_id, self.config.train_dataset, limit=contrastive_limit
        )

        # Train using property-based interface (restored DAC method)
        import time

        training_start = time.time()
        property_name = f"coding_capability_layer_{layer_id}"
        training_stats = tensor_dac.train_property(property_name, contrastive_pairs)
        training_time = time.time() - training_start

        self.logger.info(f"   ✅ Tensor DAC training completed in {training_time:.1f}s")
        steering_tensor = tensor_dac.get_steering_tensor()
        self.logger.info(f"   Tensor shape: {steering_tensor.shape}")

        return tensor_dac

    def _evaluate_tensor_dac_on_validation(
        self, tensor_dac: TensorDAC, hyperparams: dict, layer_id: int, trial=None
    ) -> float:
        """Evaluate tensor DAC on validation set."""
        from wisent_guard.core.optuna import metrics
        from wisent_guard.core.task_interface import get_task

        if tensor_dac is None:
            return 0.0

        # Check if trained (but don't fail if attribute doesn't exist)
        if hasattr(tensor_dac, "is_trained") and not tensor_dac.is_trained:
            return 0.0

        self.logger.info("🎯 Evaluating Tensor DAC on validation set")

        # Get validation task
        task = get_task(self.config.val_dataset)
        extractor = task.get_extractor()

        # Collect validation questions and task docs
        questions = []
        ground_truths = []
        valid_samples = []  # Keep track of samples for task_docs (needed for BigCode evaluation)

        for sample in self.val_samples:
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                continue

            questions.append(qa_pair["formatted_question"])
            ground_truths.append(qa_pair["correct_answer"])
            valid_samples.append(sample)  # Store original sample for BigCode evaluation

        if not questions:
            return 0.0

        # Generate predictions with tensor DAC
        predictions = []
        steering_strength = hyperparams.get("steering_alpha", 1.0)

        # Construct property weights for dynamic steering
        property_name = f"coding_capability_layer_{layer_id}"
        property_weights = {property_name: 1.0}  # Use full weight for the trained property

        self.logger.debug(f"   Generating {len(questions)} predictions with α={steering_strength:.3f}")
        self.logger.debug(f"   Using dynamic strategy with property: {property_name}")

        for i, question in enumerate(questions):
            try:
                # Generate with tensor DAC steering
                generated_text = tensor_dac.generate_with_steering(
                    prompt=question,
                    property_weights=property_weights,
                    max_new_tokens=hyperparams.get("max_new_tokens", 384),
                    steering_strength=steering_strength,
                    timing_strategy="dynamic",
                )

                # For coding tasks, don't try to extract - let BigCode handle it
                # Just clean up the response
                prediction = generated_text.strip()
                predictions.append(prediction)

            except Exception as e:
                self.logger.warning(f"Generation failed for question {i}: {e}")
                predictions.append("")  # Empty prediction for failed generation

        # Log sample predictions for debugging (like parent class)
        for i, (pred, gt) in enumerate(zip(predictions[:3], ground_truths[:3])):
            self.logger.debug(f"DAC_TENSOR Sample {i} - Model: ...{pred[-50:] if pred else 'None'}")
            self.logger.debug(f"DAC_TENSOR Sample {i} - Ground truth: {gt}")

        # Save detailed validation results (like parent class)
        trial_number = trial.number if trial and hasattr(trial, "number") else 0
        self._save_detailed_validation_results(
            questions,
            ground_truths,
            predictions,
            trial_number,
            trial=trial,
            steering_method="dac_tensor",
            layer_id=layer_id,
            hyperparams=hyperparams,
        )

        # Prepare task docs for BigCode evaluation (critical for MBPP Plus)
        task_docs = valid_samples[: len(predictions)] if valid_samples else []

        # Use proper benchmark evaluation (same as parent class)
        benchmark_metrics = metrics.evaluate_benchmark_performance(
            predictions, ground_truths, self.config.val_dataset, task_docs=task_docs
        )
        accuracy = benchmark_metrics.get("accuracy", 0.0)

        self.logger.info(f"   Validation accuracy: {accuracy:.4f}")

        # Report to Optuna for pruning if trial is available
        if trial and hasattr(trial, "report"):
            trial.report(accuracy, step=1)

        return accuracy

    def _log_enhanced_results(self, study, final_results):
        """Log enhanced results with MBPP Plus-specific analysis."""
        self.logger.info("=" * 80)
        self.logger.info("🤖 QWEN2.5-CODER-7B MBPP PLUS OPTIMIZATION RESULTS")
        self.logger.info("=" * 80)

        best_trial = study.best_trial
        best_method = best_trial.params.get("steering_method", "unknown")
        best_layer = best_trial.params.get("layer_id", -1)

        # Get the appropriate alpha/strength parameter based on method
        if best_method == "dac_tensor" or best_method == "caa":
            best_alpha = best_trial.params.get("steering_alpha", 0.0)
        elif best_method == "dac":
            best_alpha = best_trial.params.get("base_strength", 0.0)
        else:
            best_alpha = 0.0

        self.logger.info(f"🥇 Best Method: {best_method.upper()}")
        self.logger.info(f"📊 Best Layer: {best_layer}")
        self.logger.info(f"⚡ Best Alpha/Strength: {best_alpha:.4f}")
        self.logger.info(f"🎯 Best Validation Accuracy: {study.best_value:.4f}")

        baseline_acc = final_results["baseline_benchmark_metrics"]["accuracy"]
        steered_acc = final_results["steered_benchmark_metrics"]["accuracy"]
        improvement = final_results["accuracy_improvement"]

        self.logger.info("📈 Test Results on MBPP Plus:")
        self.logger.info(f"   Baseline:  {baseline_acc:.4f}")
        self.logger.info(f"   Steered:   {steered_acc:.4f}")
        self.logger.info(f"   Improvement: {improvement:+.4f}")

        # Coding-specific insights
        self.logger.info("🔧 MBPP Plus Task Insights:")
        self.logger.info("   - Model: Qwen2.5-Coder-7B (32 layers, specialized for coding)")
        self.logger.info("   - Training: MBPP Plus (Extended Python problems)")
        self.logger.info("   - Testing: MBPP Plus (same dataset)")
        self.logger.info("   - More challenging than standard MBPP")
        self.logger.info(
            f"   - Best layer {best_layer} suggests {'early' if best_layer < 11 else 'middle' if best_layer < 22 else 'late'} processing"
        )

        # Performance context
        if baseline_acc > 0.6:
            self.logger.info("🎉 Excellent baseline performance on MBPP Plus!")
        elif baseline_acc > 0.4:
            self.logger.info("👍 Good baseline performance for MBPP Plus")
        else:
            self.logger.info("🤔 Lower than expected baseline - check model loading and prompt format")

        # Method-specific insights
        method_trials = [t for t in study.trials if t.params.get("steering_method") == best_method]
        if len(method_trials) > 1:
            method_values = [t.value for t in method_trials if t.value is not None]
            if method_values:
                self.logger.info(f"📊 {best_method.upper()} Method Statistics:")
                self.logger.info(f"   Trials: {len(method_values)}")
                self.logger.info(f"   Mean: {sum(method_values) / len(method_values):.4f}")
                self.logger.info(f"   Best: {max(method_values):.4f}")
                self.logger.info(
                    f"   Std: {(sum((x - sum(method_values) / len(method_values)) ** 2 for x in method_values) / len(method_values)) ** 0.5:.4f}"
                )

                # Additional info for tensor DAC
                if best_method == "dac_tensor":
                    self.logger.info(f"   ICL Examples: {best_trial.params.get('icl_examples', 'N/A')}")
                    self.logger.info(f"   Max Examples: {best_trial.params.get('max_examples', 'N/A')}")
                    self.logger.info(f"   Legacy Format: {best_trial.params.get('legacy_behavior', 'N/A')}")

        self.logger.info("=" * 80)

        # Call parent logging
        self._log_final_results_to_wandb(study, final_results)


def main():
    """Main entry point for Qwen2.5-Coder-7B MBPP Plus optimization."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Coder-7B MBPP Plus Task Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model (default: Qwen/Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: 2 for Qwen2.5-Coder-7B)",
    )

    # Dataset configuration
    parser.add_argument("--train-limit", type=int, default=None, help="Number of training samples to load")
    parser.add_argument(
        "--contrastive-pairs-limit",
        type=int,
        default=None,
        help="Number of contrastive pairs for steering training (default: 75, bounded by train-limit)",
    )
    parser.add_argument("--val-limit", type=int, default=None, help="Number of validation samples to load")
    parser.add_argument("--test-limit", type=int, default=None, help="Number of test samples to load")

    # Optimization configuration
    parser.add_argument(
        "--study-name", type=str, default=None, help="Optuna study name (default: qwen25_coder_mbpp_plus_optimization)"
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Number of optimization trials")
    parser.add_argument(
        "--n-startup-trials", type=int, default=None, help="Random exploration trials before TPE kicks in"
    )
    parser.add_argument("--layer-range", type=int, nargs=2, default=None, help="Layer search range as two integers")

    # WandB configuration
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable WandB experiment tracking (requires 'wandb login' first)"
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")

    # Utility options
    parser.add_argument("--quick-test", action="store_true", help="Quick test run")
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
        args.train_limit = 30
        args.val_limit = 20
        args.test_limit = 30
        logger.info("🚀 Quick test mode enabled")

    run_config = get_recommended_config_for_qwen25_coder()
    # Display configuration
    logger.info("🤖 QWEN2.5-CODER-7B MBPP PLUS OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("🔧 CONFIGURATION:")
    logger.info(f"   Model: {args.model_path or run_config['model_name']}")
    logger.info(f"   Batch Size: {args.batch_size or run_config['batch_size']}")
    logger.info(f"   Trials: {args.n_trials or run_config['n_trials']}")
    logger.info(f"   Datasets: MBPP Plus (train/val/test) - Extended Python programming problems")
    logger.info(f"   WandB: {'Enabled' if args.use_wandb else 'Disabled'}")

    if torch.cuda.is_available():
        logger.info(f"🔥 GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} devices)")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        # Memory warning for Qwen2.5-Coder-7B
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 20:
            logger.warning(f"⚠️  GPU has {vram_gb:.1f}GB VRAM. Qwen2.5-Coder-7B requires ~14GB+. Consider:")
            logger.warning("   - Using smaller batch size (--batch-size 1)")
            logger.warning("   - Setting CUDA_VISIBLE_DEVICES to limit GPU usage")
    else:
        logger.error("❌ No CUDA detected - Qwen2.5-Coder-7B requires GPU!")
        return None

    logger.info("=" * 80)

    # Environment variable check
    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        logger.warning("⚠️  HF_ALLOW_CODE_EVAL not set. MBPP Plus requires code execution.")
        logger.warning("   Set: export HF_ALLOW_CODE_EVAL='1'")
        logger.info("=" * 80)

    # Create configuration and pipeline
    try:
        config = create_qwen25_coder_config(args)
        pipeline = Qwen25CoderMBPPPlusPipeline(config)

        # Run optimization
        logger.info("🚀 Starting MBPP Plus optimization for Qwen2.5-Coder-7B...")
        results = pipeline.run_optimization()

        # Enhanced result display
        pipeline._log_enhanced_results(pipeline._create_optuna_study(), results)

        logger.info("✅ Qwen2.5-Coder-7B MBPP Plus optimization completed successfully!")
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
