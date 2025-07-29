"""
Optuna Minimal Example - Small Steering Range Testing

This example demonstrates steering behavior with realtreetune/rho-1b-sft-GSM8K model.

EXPECTED BEHAVIOR:
- This model was fine-tuned on GSM8K, so it has high baseline accuracy (~27%)
- With zero steering (α=0), accuracy should remain high (equivalent to baseline)
- With small positive steering (α=0.05-0.3), accuracy should gradually DECREASE
- This is because the model is already optimized, and steering disrupts its learned patterns
- Optuna should find optimal α close to 0.0 to preserve the fine-tuned performance

TESTING HYPOTHESIS:
- Fine-tuned models perform best with minimal or zero steering
- Steering typically hurts performance on models already optimized for the task
- Small steering range (0.0-0.3) should show clear degradation pattern
"""

import logging
import sys
from pathlib import Path
import torch
import optuna

# Add wisent-guard to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from comprehensive_evaluation.optimization_pipeline import OptimizationPipeline, OptimizationConfig

# No environment variables needed - all config in script


def create_minimal_config() -> OptimizationConfig:
    """Create minimal config for fine-tuned model testing."""
    return OptimizationConfig(
        # Model: Fine-tuned on GSM8K with ~27% accuracy
        model_name="realtreetune/rho-1b-sft-GSM8K",
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # Datasets: Use hendrycks_math for training, GSM8K for val/test
        train_dataset="hendrycks_math",
        val_dataset="gsm8k",
        test_dataset="gsm8k",
        
        # Training configuration
        train_limit=20,              # Training samples to load
        contrastive_pairs_limit=15,  # Contrastive pairs for steering training (bounded by train_limit)
        
        # Evaluation configuration  
        val_limit=20,                # Validation samples to load (all used for evaluation)
        test_limit=50,               # Test samples to load
        
        # Optuna configuration: Few trials to see clear pattern
        study_name="minimal_example_test",
        db_url="sqlite:///optuna_minimal_test.db",
        n_trials=3,  # 3 trials to test the integration
        n_startup_trials=1,  # 1 random trial then TPE
        sampler="TPE",
        pruner="MedianPruner",
        
        # Search space: Single layer, single method for clarity
        layer_search_range=(10, 10),  # Single layer for speed
        probe_type="logistic_regression",  # Fixed probe type
        steering_methods=["caa"],  # CAA is simpler than DAC
        
        # Generation: Deterministic for reproducible results
        temperature=0.0,
        do_sample=False,
        batch_size=4,
        max_length=512,
        max_new_tokens=128,
        
        # Output
        output_dir="outputs/optuna_minimal_example",
        cache_dir="cache/optuna_minimal_example",
        
        # Disable WandB for testing
        use_wandb=False,
        wandb_project="minimal_test",
        
        # Efficiency settings
        seed=42,
        max_layers_to_search=1,
        early_stopping_patience=5
    )


class MinimalSteeringPipeline(OptimizationPipeline):
    """Custom pipeline with small steering range for fine-tuned models."""
    
    def _objective_function(self, trial):
        """Optuna objective with SMALL steering range (0.0-0.3)."""
        try:
            # Fixed hyperparameters for clarity
            layer_id = trial.suggest_int("layer_id", 10, 10)  # Fixed but still in trial params
            probe_type = trial.suggest_categorical("probe_type", ["logistic_regression"])
            probe_c = trial.suggest_float("probe_c", 0.1, 10.0, log=True)
            steering_method = trial.suggest_categorical("steering_method", ["caa"])
            
            # SMALL steering range: expect optimal near 0.0
            steering_alpha = trial.suggest_float("steering_alpha", 0.0, 0.3)
            
            self.logger.info(f"Trial {trial.number}: Testing α={steering_alpha:.3f}")
            
            # Step 1: Train probe
            probe_score = self._train_and_evaluate_probe(trial, layer_id, probe_type, probe_c)
            trial.report(probe_score, step=0)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Step 2: Train steering method
            steering_instance = self._train_steering_method(
                trial, steering_method, layer_id, {"steering_alpha": steering_alpha}
            )
            
            # Step 3: Evaluate on validation
            validation_accuracy = self._evaluate_steering_on_validation(
                steering_instance, steering_method, layer_id, {"steering_alpha": steering_alpha}
            )
            
            self.logger.info(f"Trial {trial.number}: α={steering_alpha:.3f} → accuracy={validation_accuracy:.3f}")
            
            trial.report(validation_accuracy, step=1)
            return validation_accuracy
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0


def main():
    """Run minimal steering example."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🧪 OPTUNA MINIMAL EXAMPLE - Small Steering Range Testing")
    logger.info("="*80)
    logger.info("Model: realtreetune/rho-1b-sft-GSM8K (fine-tuned)")
    logger.info("Expected: Optimal steering α ≈ 0.0 (fine-tuned models resist steering)")
    logger.info("Hypothesis: Accuracy decreases as α increases from 0.0 to 0.3")
    logger.info("="*80)
    
    # Create config and pipeline
    config = create_minimal_config()
    pipeline = MinimalSteeringPipeline(config)
    
    try:
        # Run optimization
        results = pipeline.run_optimization()
        
        # Extract key results
        best_alpha = results['best_trial_params']['steering_alpha']
        best_val_score = results['best_validation_score']
        baseline_acc = results['baseline_benchmark_metrics']['accuracy']
        steered_acc = results['steered_benchmark_metrics']['accuracy']
        improvement = results['accuracy_improvement']
        
        # Display results
        logger.info("\n" + "="*80)
        logger.info("📊 MINIMAL EXAMPLE RESULTS")
        logger.info("="*80)
        logger.info(f"Best steering α: {best_alpha:.4f}")
        logger.info(f"Best validation score: {best_val_score:.4f}")
        logger.info(f"Final test results:")
        logger.info(f"  Baseline accuracy: {baseline_acc:.4f}")
        logger.info(f"  Steered accuracy:  {steered_acc:.4f}")
        logger.info(f"  Improvement:       {improvement:+.4f}")
        logger.info("="*80)
        
        # Analysis
        if best_alpha < 0.05:
            logger.info("✅ EXPECTED: Optimal α near zero (fine-tuned model)")
        elif best_alpha > 0.15:
            logger.info("⚠️  UNEXPECTED: High optimal α (investigate further)")
        else:
            logger.info("🤔 MODERATE: Small optimal α (reasonable for fine-tuned model)")
        
        if improvement < 0:
            logger.info("✅ EXPECTED: Steering hurt performance (fine-tuned model)")
        else:
            logger.info("⚠️  UNEXPECTED: Steering helped (unusual for fine-tuned model)")
        
        logger.info("="*80)
        
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