"""
TruthfulQA-MC1 Optuna Minimal Example - Random Model Testing

This example demonstrates steering behavior with tiny-random-gpt2 on TruthfulQA-MC1.

EXPECTED BEHAVIOR:
- This model has random weights (no pretraining), so baseline should be ~0%
- Random models produce gibberish, not valid letter choices (A/B/C/D)
- True random guessing would be 25%, but this model doesn't attempt letter answers
- With optimal steering, we should see improvement toward meaningful letter choices
- This tests if steering can help untrained models learn from activation patterns
- Optuna should find optimal α > 0.0 that provides meaningful learning signal

TESTING HYPOTHESIS:
- Random/untrained models benefit significantly from steering
- Multiple choice tasks show clear steering improvement signals
- Optimal α should be substantial (0.3-0.8) for untrained models
- This complements the fine-tuned model example (where steering hurts)
"""

import logging
import sys
from pathlib import Path

import optuna
import torch

# Add wisent-guard to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from wisent_guard.core.optuna.optuna_pipeline import OptimizationConfig, OptimizationPipeline

# No environment variables needed - all config in script


def create_mmlu_config() -> OptimizationConfig:
    """Create minimal config for MMLU multiple choice testing."""
    return OptimizationConfig(
        # Model: Tiny random GPT2 (untrained, pure random baseline)
        model_name="hf-internal-testing/tiny-random-gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Datasets: Use truthfulqa_mc1 which is a proven working multiple choice task
        train_dataset="truthfulqa_mc1",
        val_dataset="truthfulqa_mc1",
        test_dataset="truthfulqa_mc1",
        # Training configuration
        train_limit=30,  # Training samples to load (MMLU has many subjects)
        contrastive_pairs_limit=20,  # Contrastive pairs for steering training
        # Evaluation configuration
        val_limit=40,  # Validation samples to load
        test_limit=60,  # Test samples to load (more samples for reliable MC evaluation)
        # Optuna configuration: Test both steering methods (NEW STUDY)
        study_name="truthfulqa_random_gpt2_mc_clean",
        db_url="sqlite:///optuna_truthfulqa_random_gpt2_mc_clean.db",
        n_trials=1,  # Single trial for quick testing
        n_startup_trials=1,  # Single random trial
        sampler="TPE",
        pruner="MedianPruner",
        # Search space: Test layers appropriate for tiny-random-gpt2 (5 layers: 0-4)
        layer_search_range=(2, 4),  # Middle to late layers for tiny model
        probe_type="logistic_regression",  # Fixed probe type
        steering_methods=["caa", "dac"],  # Test both methods
        # Generation: Multiple choice typically uses deterministic generation
        temperature=0.0,
        do_sample=False,
        batch_size=4,
        max_length=512,
        max_new_tokens=32,  # Shorter for MC answers (A, B, C, D)
        # Output
        output_dir="outputs/truthfulqa_random_gpt2_minimal",
        cache_dir="cache/truthfulqa_random_gpt2_minimal",
        # Disable WandB for testing
        use_wandb=False,
        wandb_project="mmlu_minimal_test",
        # Efficiency settings
        seed=42,
        max_layers_to_search=3,
        early_stopping_patience=5,
    )


class MMLUSteeringPipeline(OptimizationPipeline):
    """Custom pipeline with appropriate steering range for MMLU multiple choice."""

    def _objective_function(self, trial):
        """Optuna objective optimized for multiple choice tasks."""
        try:
            # Search space for multiple choice optimization
            layer_id = trial.suggest_int("layer_id", 2, 4)
            probe_type = trial.suggest_categorical("probe_type", ["logistic_regression"])
            probe_c = trial.suggest_float("probe_c", 0.1, 10.0, log=True)
            steering_method = trial.suggest_categorical("steering_method", ["caa", "dac"])

            # Set up method-specific hyperparameters
            hyperparams = {}
            if steering_method == "dac":
                # DAC-specific parameters for MC tasks
                hyperparams["steering_alpha"] = trial.suggest_float("steering_alpha", 0.1, 1.0)
                hyperparams["entropy_threshold"] = trial.suggest_float("entropy_threshold", 0.3, 1.5)
                hyperparams["ptop"] = trial.suggest_float("ptop", 0.2, 0.7)
                hyperparams["max_alpha"] = trial.suggest_float("max_alpha", 1.0, 4.0)
            elif steering_method == "caa":
                # CAA-specific parameters for MC tasks
                hyperparams["steering_alpha"] = trial.suggest_float("steering_alpha", 0.1, 1.0)

            self.logger.info(
                f"Trial {trial.number}: Testing {steering_method} on TruthfulQA-MC1 with α={hyperparams['steering_alpha']:.3f}"
            )

            # Step 1: Train probe
            probe_score = self._train_and_evaluate_probe(trial, layer_id, probe_type, probe_c)
            trial.report(probe_score, step=0)

            if trial.should_prune():
                raise optuna.TrialPruned()

            # Step 2: Train steering method
            steering_instance = self._train_steering_method(trial, steering_method, layer_id, hyperparams)

            # Step 3: Evaluate on validation (MC evaluation)
            validation_accuracy = self._evaluate_steering_on_validation(
                steering_instance, steering_method, layer_id, hyperparams
            )

            self.logger.info(
                f"Trial {trial.number}: TruthfulQA-MC1 {steering_method} α={hyperparams['steering_alpha']:.3f} → accuracy={validation_accuracy:.3f}"
            )

            trial.report(validation_accuracy, step=1)
            return validation_accuracy

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0


def main():
    """Run MMLU minimal steering example."""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(__name__)
    logger.info("🧪 TRUTHFULQA-MC1 OPTUNA MINIMAL EXAMPLE - Random Model Testing")
    logger.info("=" * 80)
    logger.info("Model: hf-internal-testing/tiny-random-gpt2 (random weights, untrained)")
    logger.info("Task: TruthfulQA-MC1 multiple choice truthfulness evaluation")
    logger.info("Expected: ~0% baseline (gibberish), steering should help significantly")
    logger.info("Hypothesis: Optimal α > 0.3 for untrained models, improvement > 10%")
    logger.info("=" * 80)

    # Create config and pipeline
    config = create_mmlu_config()
    pipeline = MMLUSteeringPipeline(config)

    try:
        # Run optimization
        results = pipeline.run_optimization()

        # Extract key results
        best_alpha = results["best_trial_params"]["steering_alpha"]
        best_val_score = results["best_validation_score"]
        baseline_acc = results["baseline_benchmark_metrics"]["accuracy"]
        steered_acc = results["steered_benchmark_metrics"]["accuracy"]
        improvement = results["accuracy_improvement"]

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("📊 TRUTHFULQA-MC1 MINIMAL EXAMPLE RESULTS")
        logger.info("=" * 80)
        logger.info(f"Best steering α: {best_alpha:.4f}")
        logger.info(f"Best validation score: {best_val_score:.4f}")
        logger.info("Final test results:")
        logger.info(f"  Baseline accuracy: {baseline_acc:.4f}")
        logger.info(f"  Steered accuracy:  {steered_acc:.4f}")
        logger.info(f"  Improvement:       {improvement:+.4f}")
        logger.info("=" * 80)

        # Analysis for TruthfulQA-MC1 with random model
        if baseline_acc < 0.05:
            logger.info("✅ EXPECTED: Low baseline (gibberish responses, no valid letters)")
        elif baseline_acc < 0.15:
            logger.info("🤔 MODERATE: Low baseline (mostly gibberish with some valid letters)")
        elif abs(baseline_acc - 0.25) < 0.05:
            logger.info("🤔 INTERESTING: Baseline ≈ 25% (random letter selection)")
        elif baseline_acc > 0.30:
            logger.info("⚠️  UNEXPECTED: High baseline (random model shouldn't know answers)")
        else:
            logger.info("🤔 MODERATE: Baseline above expected gibberish level")

        if improvement > 0.15:
            logger.info("✅ EXCELLENT: Steering provided major improvement (>15%)")
        elif improvement > 0.10:
            logger.info("✅ GOOD: Steering significantly helped (>10%)")
        elif improvement > 0.05:
            logger.info("🤔 MODERATE: Some improvement from steering (>5%)")
        else:
            logger.info("⚠️  CONCERNING: Little/no steering benefit (<5%)")

        if best_alpha > 0.4:
            logger.info("✅ EXPECTED: High α for untrained model (needs strong signal)")
        elif best_alpha > 0.2:
            logger.info("🤔 MODERATE: Medium α (reasonable for untrained model)")
        else:
            logger.info("⚠️  UNEXPECTED: Low α (untrained model should need more steering)")

        logger.info("=" * 80)

        return results

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        # Cleanup
        pipeline.cleanup_memory()


if __name__ == "__main__":
    main()
