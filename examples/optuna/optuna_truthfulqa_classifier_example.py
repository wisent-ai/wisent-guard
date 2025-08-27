#!/usr/bin/env python3
"""
TruthfulQA Classifier Optuna Example - Qwen3 Model Testing

This example demonstrates classifier training and optimization with Qwen/Qwen3-0.6B on TruthfulQA.
The example shows how to use the efficient Optuna-based classifier optimization system to find
the best layer, aggregation method, and model hyperparameters for truthfulness detection.

EXPECTED BEHAVIOR:
- Qwen3-0.6B is a modern pretrained model with good language understanding
- Pre-generated activations should provide rich features for classification
- Baseline classification accuracy should be well above random (>60%) for truthfulness detection
- With optimal hyperparameters, we should achieve strong performance (>75%)
- MLP models may outperform logistic regression on complex activation patterns
- Optimal layers should be in the middle-to-late range (layers 16-22) for semantic understanding

TESTING HYPOTHESIS:
- Modern language models contain rich truthfulness representations in their activations
- Later layers capture semantic truthfulness better than early layers
- MLP classifiers can capture non-linear patterns better than logistic regression
- Activation aggregation method significantly impacts classification performance
- Optimal α should enhance truthfulness detection without disrupting language capability

ARCHITECTURE:
- Uses pre-generated activations for efficiency (no re-extraction per trial)
- Tests both logistic regression and MLP classifiers
- Intelligent caching to avoid retraining identical configurations
- Cross-validation for robust evaluation
"""

import logging
import sys
from pathlib import Path

import torch

# Add wisent-guard to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair

# Import Model wrapper and ContrastivePair for proper data conversion
from wisent_guard.core.model import Model
from wisent_guard.core.optuna import (
    CacheConfig,
    ClassifierOptimizationConfig as OptimizationConfig,
    GenerationConfig,
    OptunaClassifierOptimizer,
)

# Import data loading utilities from steering system
from wisent_guard.core.optuna.steering.data_utils import get_task_contrastive_pairs, load_dataset_samples


def convert_steering_pairs_to_contrastive_pairs(steering_pairs: list) -> list[ContrastivePair]:
    """Convert steering system dictionary format to ContrastivePair objects."""
    contrastive_pairs = []

    for pair in steering_pairs:
        if isinstance(pair, dict):
            # Extract the necessary fields from the dictionary
            # The exact structure may vary, but common fields are:
            prompt = pair.get("prompt", pair.get("question", ""))
            positive = pair.get("positive_response", pair.get("correct_answer", ""))
            negative = pair.get("negative_response", pair.get("incorrect_answer", ""))

            # Handle TruthfulQA MC format
            if not positive and "choices" in pair:
                choices = pair["choices"]
                if len(choices) >= 2:
                    positive = choices[0]  # First choice as positive
                    negative = choices[1]  # Second choice as negative

            if prompt and positive and negative:
                contrastive_pair = ContrastivePair(
                    prompt=prompt, positive_response=positive, negative_response=negative
                )
                contrastive_pairs.append(contrastive_pair)

    return contrastive_pairs


def create_truthfulqa_config() -> tuple[OptimizationConfig, GenerationConfig, CacheConfig]:
    """Create configuration for TruthfulQA classifier optimization."""

    # Optimization configuration - tuned for TruthfulQA classifier task
    optimization_config = OptimizationConfig(
        # Model configuration
        model_name="Qwen/Qwen3-0.6B",
        device="auto",  # Auto-detect best available device
        # Optuna settings for thorough search (reduced for testing)
        n_trials=5,  # Further reduced for debugging
        timeout=1800,  # 30 minutes max
        n_jobs=1,
        sampler_seed=42,
        # Model types: test both approaches
        model_types=["logistic", "mlp"],
        # Hyperparameter search ranges - tuned for truthfulness detection
        hidden_dim_range=(64, 256),  # MLP hidden dimensions
        threshold_range=(0.4, 0.8),  # Classification thresholds
        # Training hyperparameters - balanced for stability and performance
        num_epochs_range=(30, 80),  # Sufficient training for complex patterns
        learning_rate_range=(1e-4, 5e-3),  # Conservative but effective range
        batch_size_options=[16, 32, 64],  # Various batch sizes
        # Evaluation settings
        cv_folds=3,  # 3-fold cross-validation for robust evaluation
        test_size=0.2,  # 80/20 train/test split
        random_state=42,
        # Optimization objective - F1 for balanced precision/recall
        primary_metric="f1",  # Good for potentially imbalanced truthfulness data
        # Pruning for efficiency (disabled for debugging)
        enable_pruning=False,  # Temporarily disabled
        pruning_patience=5,  # Early stopping if no improvement
    )

    # Generation configuration - layers and aggregation methods to test
    generation_config = GenerationConfig(
        # Layer range: middle to late layers for semantic understanding
        layer_search_range=(16, 22),  # Qwen3-0.6B has 24 layers (0-23)
        # Aggregation methods: comprehensive set for activation processing
        aggregation_methods=["average", "final", "first", "max", "min"],
        # Infrastructure
        cache_dir="./cache/truthfulqa_activations",
        device=None,  # Will be set from optimization_config
        batch_size=32,
    )

    # Cache configuration for trained models
    cache_config = CacheConfig(
        cache_dir="./cache/truthfulqa_classifiers",
        max_cache_size_gb=1.5,  # Generous cache for different configurations
        max_age_days=7.0,  # Keep models for a week
        memory_cache_size=10,  # Keep recent models in memory
    )

    # Ensure device consistency
    generation_config.device = optimization_config.device

    return optimization_config, generation_config, cache_config


class TruthfulQAClassifierPipeline:
    """Custom pipeline for TruthfulQA classifier optimization with detailed analysis."""

    def __init__(
        self, optimization_config: OptimizationConfig, generation_config: GenerationConfig, cache_config: CacheConfig
    ):
        self.optimizer = OptunaClassifierOptimizer(
            optimization_config=optimization_config, generation_config=generation_config, cache_config=cache_config
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_optimization(self, model, contrastive_pairs: list, task_name: str, model_name: str, limit: int):
        """Run the TruthfulQA classifier optimization with detailed tracking."""

        self.logger.info("Starting TruthfulQA classifier optimization")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Task: {task_name}")
        self.logger.info(f"Contrastive pairs: {len(contrastive_pairs)}")
        self.logger.info(f"Data limit: {limit}")
        self.logger.info("=" * 60)

        # Run optimization
        try:
            result = self.optimizer.optimize(
                model=model,
                contrastive_pairs=contrastive_pairs,
                task_name=task_name,
                model_name=model_name,
                limit=limit,
            )

            # Get detailed analysis
            best_config = result.get_best_config()
            summary = self.optimizer.get_optimization_summary(result)

            return {
                "optimization_result": result,
                "best_configuration": best_config,
                "summary": summary,
                "best_f1_score": result.best_value,
                "optimization_time": result.optimization_time,
                "cache_efficiency": result.cache_hits / (result.cache_hits + result.cache_misses)
                if (result.cache_hits + result.cache_misses) > 0
                else 0,
            }

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def cleanup_memory(self):
        """Clean up memory after optimization."""
        # Clear activation data
        if hasattr(self.optimizer, "activation_data"):
            self.optimizer.activation_data.clear()

        # Force garbage collection
        import gc

        gc.collect()

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Run TruthfulQA classifier optimization example."""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(__name__)
    logger.info("🔍 TRUTHFULQA CLASSIFIER OPTUNA EXAMPLE - Qwen3-0.6B Model")
    logger.info("=" * 80)
    logger.info("Model: Qwen/Qwen3-0.6B (modern pretrained model)")
    logger.info("Task: TruthfulQA truthfulness classification with activations")
    logger.info("Expected: >60% baseline accuracy, >75% optimized accuracy")
    logger.info("Hypothesis: Later layers + MLP models should perform best")
    logger.info("=" * 80)

    # Create configurations
    opt_config, gen_config, cache_config = create_truthfulqa_config()

    # Create pipeline
    pipeline = TruthfulQAClassifierPipeline(opt_config, gen_config, cache_config)

    # Load real model and data
    logger.info("🤖 Loading model and TruthfulQA data...")

    # Configuration
    task_name = "truthfulqa_mc1"  # Use the MC1 variant that works
    model_name = opt_config.model_name
    limit = 30  # Small limit for initial testing

    try:
        # Load model using proper Model wrapper
        logger.info(f"Loading model {model_name} on {opt_config.device}...")
        model = Model(name=model_name, device=opt_config.device)
        num_layers = model.hf_model.config.to_dict().get("num_hidden_layers", 24)
        logger.info(f"✅ Model loaded successfully with {num_layers} layers")

        # Load dataset samples
        logger.info(f"Loading {limit} samples from {task_name}...")
        samples = load_dataset_samples(task_name, limit)

        # Extract contrastive pairs
        logger.info("Extracting contrastive pairs...")
        steering_pairs = get_task_contrastive_pairs(samples, task_name)

        # Debug: Print what we got from steering system
        if steering_pairs:
            logger.info(f"Sample steering pair structure: {type(steering_pairs[0])}")
            if hasattr(steering_pairs[0], "__dict__"):
                logger.info(f"Sample steering pair attributes: {list(vars(steering_pairs[0]).keys())}")
            elif isinstance(steering_pairs[0], dict):
                logger.info(f"Sample steering pair keys: {list(steering_pairs[0].keys())}")

        # Convert to proper ContrastivePair objects
        logger.info("Converting to ContrastivePair objects...")
        contrastive_pairs = convert_steering_pairs_to_contrastive_pairs(steering_pairs)

        logger.info(f"✅ Loaded model and {len(contrastive_pairs)} contrastive pairs")

    except Exception as e:
        logger.error(f"Failed to load model or data: {e}")
        logger.info("💡 Troubleshooting suggestions:")
        logger.info("   - Make sure you have internet connection for model download")
        logger.info("   - Check if you have enough disk space for model files")
        logger.info("   - Verify the model name 'Qwen/Qwen3-0.6B' is accessible")
        logger.info("   - Try running with a smaller model if memory is limited")
        logger.info("   - Make sure the 'truthfulqa_mc1' dataset can be loaded")
        import traceback

        traceback.print_exc()
        return None

    # Run optimization if model and data loaded successfully
    if model is not None and len(contrastive_pairs) > 0:
        try:
            # Run optimization
            results = pipeline.run_optimization(
                model=model,
                contrastive_pairs=contrastive_pairs,
                task_name=task_name,
                model_name=model_name,
                limit=limit,
            )

            if results and results["optimization_result"].best_value > 0:
                # Extract key results
                best_config = results["best_configuration"]
                best_f1 = results["best_f1_score"]
                opt_time = results["optimization_time"]
                cache_eff = results["cache_efficiency"]
                summary = results["summary"]

                # Display results
                logger.info("\n" + "=" * 80)
                logger.info("🏆 TRUTHFULQA CLASSIFIER OPTIMIZATION RESULTS")
                logger.info("=" * 80)
                logger.info(f"Best Model Type: {best_config['model_type']}")
                logger.info(f"Best Layer: {best_config['layer']}")
                logger.info(f"Best Aggregation: {best_config['aggregation']}")
                logger.info(f"Best Threshold: {best_config['threshold']:.3f}")
                if best_config["model_type"] == "mlp":
                    logger.info(f"Hidden Dimensions: {best_config['hyperparameters'].get('hidden_dim', 'N/A')}")
                logger.info(f"Best F1 Score: {best_f1:.4f}")
                logger.info(f"Optimization Time: {opt_time:.1f}s")
                logger.info(f"Cache Efficiency: {cache_eff:.2%}")
                logger.info(f"Total Trials: {summary['study_info']['n_trials']}")
                logger.info(f"Pruned Trials: {summary['study_info']['pruned_trials']}")
                logger.info("=" * 80)

                # Analysis for TruthfulQA classification
                if best_f1 > 0.85:
                    logger.info("🎉 EXCELLENT: Outstanding classification performance (>85% F1)")
                elif best_f1 > 0.75:
                    logger.info("✅ VERY GOOD: Strong truthfulness classification (>75% F1)")
                elif best_f1 > 0.65:
                    logger.info("✅ GOOD: Solid performance above baseline (>65% F1)")
                elif best_f1 > 0.55:
                    logger.info("🤔 MODERATE: Some truthfulness signal detected (>55% F1)")
                else:
                    logger.info("⚠️  CONCERNING: Performance near random (≤55% F1)")

                if best_config["model_type"] == "mlp":
                    logger.info("✅ COMPLEX MODEL: MLP captured non-linear patterns")
                else:
                    logger.info("📊 LINEAR MODEL: Logistic regression sufficient")

                layer = best_config["layer"]
                if layer >= 20:
                    logger.info("🧠 SEMANTIC LAYER: Very late layer optimal (semantic understanding)")
                elif layer >= 16:
                    logger.info("🧠 MID-LATE LAYER: Good semantic processing layer")
                elif layer >= 12:
                    logger.info("🧠 MIDDLE LAYER: Intermediate representation layer")
                else:
                    logger.info("🧠 EARLY LAYER: Surprisingly early layer optimal")

                agg = best_config["aggregation"]
                if agg == "average":
                    logger.info("📊 AGGREGATION: Average pooling captured global patterns")
                elif agg == "final":
                    logger.info("📊 AGGREGATION: Final token position most informative")
                elif agg in ["max", "min"]:
                    logger.info("📊 AGGREGATION: Extreme values most discriminative")
                else:
                    logger.info(f"📊 AGGREGATION: {agg} aggregation optimal")

                logger.info("=" * 80)

                # Activation data analysis
                logger.info("\n📊 ACTIVATION DATA SUMMARY:")
                for key, info in summary["activation_data_info"].items():
                    logger.info(f"  {key}: {info['samples']} samples, {info['features']} features")

                return results

            # Handle case where no trials completed successfully
            logger.info("\n" + "=" * 80)
            logger.info("❌ OPTIMIZATION INCOMPLETE")
            logger.info("=" * 80)
            if results:
                opt_result = results["optimization_result"]
                logger.info(f"Optimization Time: {results['optimization_time']:.1f}s")
                logger.info(f"Total Trials Attempted: {len(opt_result.study.trials)}")

                # Show what went wrong
                trial_states = {}
                for trial in opt_result.study.trials:
                    state = trial.state.name
                    trial_states[state] = trial_states.get(state, 0) + 1
                logger.info(f"Trial Results: {trial_states}")

                logger.info("\n💡 Debugging Suggestions:")
                logger.info("   - All trials were pruned - likely classifier training issues")
                logger.info("   - Check if the activation data is properly formatted")
                logger.info("   - Try reducing threshold ranges or increasing data size")
                logger.info("   - Consider using different aggregation methods")

            return None

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            return None

        finally:
            # Cleanup
            pipeline.cleanup_memory()

    logger.info("🎉 TruthfulQA Classifier Example Completed!")
    return None  # Return None if no results due to errors


if __name__ == "__main__":
    main()
