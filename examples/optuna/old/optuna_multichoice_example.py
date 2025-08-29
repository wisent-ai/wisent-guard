"""
TruthfulQA-MC1 Optuna Example - Qwen3 Model Testing

This example demonstrates steering behavior with Qwen/Qwen3-0.6B on TruthfulQA-MC1.

EXPECTED BEHAVIOR:
- Qwen3-0.6B is a modern pretrained model with good language understanding
- Model should produce coherent responses and attempt meaningful choices
- Baseline accuracy should be well above random (>25%) due to language capability
- With optimal steering, we may see improvement or preservation of performance
- This tests steering effects on capable language models for MC tasks
- Optuna should find optimal α that enhances reasoning without disruption

TESTING HYPOTHESIS:
- Modern language models should perform well on MC tasks (>30% baseline)
- Multiple choice tasks benefit from language understanding capability
- Optimal α should be low-moderate (0.1-0.4) to enhance without disrupting
- This demonstrates proper MC evaluation on a capable model
"""

import logging
import sys
from pathlib import Path

import optuna
import torch

# Add wisent-guard to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from wisent_guard.core.optuna.steering.optuna_pipeline import OptimizationConfig, OptimizationPipeline


def create_multichoice_config() -> OptimizationConfig:
    """Create config for fine-tuned model multiple choice testing."""
    return OptimizationConfig(
        # Model: Qwen3-0.6B modern pretrained model (should have good baseline)
        model_name="Qwen/Qwen3-0.6B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Datasets: Use truthfulqa_mc1 which is a proven working multiple choice task
        train_dataset="truthfulqa_mc1",
        val_dataset="truthfulqa_mc1",
        test_dataset="truthfulqa_mc1",
        # Training configuration
        train_limit=30,  # Training samples to load
        contrastive_pairs_limit=20,  # Contrastive pairs for steering training
        # Evaluation configuration
        val_limit=40,  # Validation samples to load
        test_limit=60,  # Test samples to load (more samples for reliable MC evaluation)
        # Optuna configuration: Test both steering methods
        study_name="truthfulqa_qwen3_mc_clean",
        db_url="sqlite:///optuna_truthfulqa_qwen3_mc_clean.db",
        n_trials=5,  # More trials for fine-tuned model
        n_startup_trials=2,  # Two random trials
        sampler="TPE",
        pruner="MedianPruner",
        # Search space: Test layers appropriate for Qwen3-0.6B (24 layers: 0-23)
        layer_search_range=(18, 23),  # Later layers for language model
        probe_type="logistic_regression",  # Fixed probe type
        steering_methods=["caa", "dac"],  # Test both methods
        # Generation: Multiple choice typically uses deterministic generation
        temperature=0.0,
        do_sample=False,
        batch_size=4,
        max_length=512,
        max_new_tokens=64,  # Moderate length for MC reasoning
        # Output
        output_dir="outputs/truthfulqa_qwen3_multichoice",
        cache_dir="cache/truthfulqa_qwen3_multichoice",
        # Disable WandB for testing
        use_wandb=False,
        wandb_project="qwen3_multichoice_test",
        # Efficiency settings
        seed=42,
        max_layers_to_search=5,
        early_stopping_patience=5,
    )


class MultiChoiceSteeringPipeline(OptimizationPipeline):
    """Custom pipeline with appropriate steering range for fine-tuned model multiple choice."""

    def _objective_function(self, trial):
        """Optuna objective optimized for Qwen3 model multiple choice tasks."""
        try:
            # Search space for Qwen3 model optimization
            layer_id = trial.suggest_int("layer_id", 18, 23)
            probe_type = trial.suggest_categorical("probe_type", ["logistic_regression"])
            probe_c = trial.suggest_float("probe_c", 0.1, 10.0, log=True)
            steering_method = trial.suggest_categorical("steering_method", ["caa", "dac"])

            # Set up method-specific hyperparameters - moderate range for Qwen3 model
            hyperparams = {}
            if steering_method == "dac":
                # DAC-specific parameters for Qwen3 model
                hyperparams["steering_alpha"] = trial.suggest_float("steering_alpha", 0.1, 0.6)
                hyperparams["entropy_threshold"] = trial.suggest_float("entropy_threshold", 0.3, 1.5)
                hyperparams["ptop"] = trial.suggest_float("ptop", 0.2, 0.7)
                hyperparams["max_alpha"] = trial.suggest_float("max_alpha", 0.8, 3.0)
            elif steering_method == "caa":
                # CAA-specific parameters for Qwen3 model
                hyperparams["steering_alpha"] = trial.suggest_float("steering_alpha", 0.1, 0.6)

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
    """Run Qwen3 multiple choice steering example."""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(__name__)
    logger.info("🧪 TRUTHFULQA-MC1 OPTUNA EXAMPLE - Qwen3-0.6B Model Testing")
    logger.info("=" * 80)
    logger.info("Model: Qwen/Qwen3-0.6B (modern pretrained model)")
    logger.info("Task: TruthfulQA-MC1 multiple choice truthfulness evaluation")
    logger.info("Expected: >25% baseline (language model), steering should enhance or preserve")
    logger.info("Hypothesis: Moderate optimal α for language models, positive steering effects")
    logger.info("=" * 80)

    # Create config and pipeline
    config = create_multichoice_config()
    pipeline = MultiChoiceSteeringPipeline(config)

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
        logger.info("📊 TRUTHFULQA-MC1 QWEN3-0.6B MULTICHOICE EXAMPLE RESULTS")
        logger.info("=" * 80)
        logger.info(f"Best steering α: {best_alpha:.4f}")
        logger.info(f"Best validation score: {best_val_score:.4f}")
        logger.info("Final test results:")
        logger.info(f"  Baseline accuracy: {baseline_acc:.4f}")
        logger.info(f"  Steered accuracy:  {steered_acc:.4f}")
        logger.info(f"  Improvement:       {improvement:+.4f}")
        logger.info("=" * 80)

        # Analysis for TruthfulQA-MC1 with Qwen3 language model
        if baseline_acc > 0.50:
            logger.info("🎉 EXCELLENT: High baseline (language model excelling at MC)")
        elif baseline_acc > 0.35:
            logger.info("✅ VERY GOOD: Strong baseline (language model working well)")
        elif baseline_acc > 0.25:
            logger.info("✅ GOOD: Above random baseline (language capability evident)")
        elif baseline_acc > 0.15:
            logger.info("🤔 MODERATE: Some capability (model has language skills but struggles with MC format)")
        else:
            logger.info("⚠️  CONCERNING: Low baseline (model not adapting to MC format)")

        if improvement > 0.15:
            logger.info("🎉 OUTSTANDING: Steering provided major enhancement (>15%)")
        elif improvement > 0.10:
            logger.info("✅ EXCELLENT: Steering significantly helped (>10%)")
        elif improvement > 0.05:
            logger.info("✅ GOOD: Steering meaningfully improved performance (>5%)")
        elif abs(improvement) < 0.03:
            logger.info("🤔 NEUTRAL: Steering had minimal effect (±3%)")
        elif improvement < -0.05:
            logger.info("⚠️  CONCERNING: Steering hurt performance significantly (<-5%)")
        else:
            logger.info("🤔 SLIGHT NEGATIVE: Steering had modest negative effect")

        if best_alpha < 0.2:
            logger.info("✅ CONSERVATIVE: Low α for language model (gentle enhancement)")
        elif best_alpha < 0.4:
            logger.info("✅ BALANCED: Moderate α (good balance of enhancement and preservation)")
        elif best_alpha < 0.6:
            logger.info("🤔 AGGRESSIVE: High α (strong steering, monitor for disruption)")
        else:
            logger.info("⚠️  VERY HIGH: Very high α (may be over-steering)")

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
