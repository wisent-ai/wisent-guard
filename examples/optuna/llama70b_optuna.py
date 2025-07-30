"""
Llama 3.1 70B Comprehensive Steering Optimization with Optuna

This script demonstrates a production-ready optimization pipeline for Llama 3.1 70B Instruct
model using multiple steering vector methods with Optuna hyperparameter optimization,
SQLite persistence, and WandB experiment tracking.

RECOMMENDED BATCH SIZES:
- Llama 70B: batch_size=1 (recommended for memory constraints)
- Training samples: 50-100 (balance between training quality and memory)
- Validation samples: 30-50 (enough for reliable optimization signal)
- Test samples: 100-200 (comprehensive final evaluation)

STEERING METHODS INVESTIGATED:
1. CAA (Contrastive Activation Addition) - Classic vector steering
2. DAC (Dynamic Activation Control) - Adaptive steering with entropy thresholds

DATASETS:
- Training: hendrycks_math (diverse mathematical reasoning)
- Validation: gsm8k (grade school math - optimization target)
- Test: gsm8k (final evaluation)

USAGE:
    # Basic usage with default settings
    python llama70b_optuna.py
    
    # Custom model path and batch size
    python llama70b_optuna.py --model-path /path/to/llama70b --batch-size 1
    
    # Enable WandB logging
    python llama70b_optuna.py --use-wandb --wandb-project my-project
    
    # Resume existing study
    python llama70b_optuna.py --study-name existing_study_name
    
    # Quick test run
    python llama70b_optuna.py --n-trials 10 --train-limit 25 --val-limit 15

EXPECTED RESULTS:
- Baseline Llama 3.1 70B Instruct typically achieves ~80-85% on GSM8K
- Optimal steering should preserve or slightly improve this performance
- Over-steering (high α values) typically degrades performance
- CAA often works well with α in range [0.1, 1.0]
- DAC can handle higher α values due to adaptive control

OUTPUTS:
- SQLite database: optuna_studies.db (persistent across runs)
- Results: outputs/llama70b_optimization/
- WandB logs: https://wandb.ai/your-project/wisent-guard-optimization
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


def get_recommended_config_for_llama70b() -> Dict[str, Any]:
    """Get recommended configuration values for Llama 3.1 70B optimization."""
    return {
        "model_name": "/workspace/models/llama31-70b-instruct-hf",
        "batch_size": 1,  # Recommended: conservative for 70B model memory requirements
        "max_new_tokens": 256,
        "layer_search_range": (30, 40),  # Layers 30-40 typically good for 70B models
        "train_limit": 50,
        "contrastive_pairs_limit": 25,  # Bounded by train_limit
        "val_limit": 30,
        "test_limit": 100,
        "n_trials": 30,
        "n_startup_trials": 8,  # Random warmup before TPE
    }


def create_llama70b_config(args) -> OptimizationConfig:
    """Create optimized configuration for Llama 3.1 70B steering optimization."""
    
    # Get base recommendations
    defaults = get_recommended_config_for_llama70b()
    
    return OptimizationConfig(
        # Model configuration - Llama 3.1 70B Instruct
        model_name=args.model_path or defaults["model_name"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # Dataset configuration - Mathematical reasoning focus
        train_dataset="hendrycks_math",  # Diverse math training data
        val_dataset="gsm8k",            # GSM8K for optimization target
        test_dataset="gsm8k",           # GSM8K for final evaluation
        
        # Training configuration
        train_limit=args.train_limit or defaults["train_limit"],
        contrastive_pairs_limit=args.contrastive_pairs_limit or defaults["contrastive_pairs_limit"],
        
        # Evaluation configuration  
        val_limit=args.val_limit or defaults["val_limit"],
        test_limit=args.test_limit or defaults["test_limit"],
        
        # Layer search configuration - Llama 70B has 80 layers
        # Layers 30-40 typically good for steering in 70B models
        layer_search_range=args.layer_range or defaults["layer_search_range"],
        
        # Probe type - Fixed to logistic regression
        probe_type="logistic_regression",
        
        # Steering methods - Currently implemented methods
        steering_methods=["caa", "dac"],

        # Optuna study configuration
        study_name=args.study_name or "llama_70b_steering_optimizationv2",
        db_url=f"sqlite:///{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/optuna_studies.db",
        n_trials=args.n_trials or defaults["n_trials"],
        n_startup_trials=args.n_startup_trials or defaults["n_startup_trials"],
        sampler="TPE",  # Tree-structured Parzen Estimator
        pruner="MedianPruner",  # Aggressive pruning for efficiency
        
        # WandB configuration
        wandb_project=args.wandb_project or "llama70b-steering-optimization",
        use_wandb=args.use_wandb,
        
        # Generation configuration - Sampling enabled
        batch_size=args.batch_size or defaults["batch_size"],
        max_length=512,  # Adequate for math problems
        max_new_tokens=defaults["max_new_tokens"],
        temperature=0.7,  # Enable sampling
        do_sample=True,
        
        # Performance optimization
        seed=42,
        
        # Output configuration
        output_dir="outputs/llama70b_optimization",
        cache_dir="cache/llama70b_optimization",
        
        # Search space constraints
        max_layers_to_search=6,  # Limit layer search for efficiency
        early_stopping_patience=10,  # Stop unpromising trials early
    )


class Llama70BSteeringPipeline(OptimizationPipeline):
    """Specialized pipeline for Llama 70B steering optimization with multiple methods."""
    
    def _objective_function(self, trial) -> float:
        """Enhanced objective function with method-specific hyperparameter spaces."""
        try:
            self.logger.info(f"🔬 Trial {trial.number}: Starting hyperparameter optimization")
            
            # Sample layer and probe hyperparameters
            layer_id = trial.suggest_int(
                "layer_id", 
                self.config.layer_search_range[0], 
                self.config.layer_search_range[1]
            )
            
            # Fixed probe type and regularization
            probe_type = self.config.probe_type  # Always logistic_regression
            probe_c = 1.0  # Default regularization strength
            
            # Sample steering method and method-specific hyperparameters
            steering_method = trial.suggest_categorical("steering_method", self.config.steering_methods)
            
            if steering_method == "caa":
                steering_alpha = trial.suggest_float(      # maps to `strength`
                    "steering_alpha", 0.10, 1.80, step=0.05
                )
                
                normalization_method = trial.suggest_categorical(
                    "normalization_method", ["none", "l2_unit"]
                )

                target_norm = None                         # ask only if relevant
                if normalization_method != "none":
                    target_norm = trial.suggest_float(
                        "target_norm", 0.8, 1.5, step=0.1
                    )

                steering_params = {
                    "steering_alpha": steering_alpha,
                    "normalization_method": normalization_method,
                    "target_norm":    target_norm,
                }
                
            elif steering_method == "dac":
                # DAC: Dynamic control with entropy-based adaptation
                
                steering_params = {
                    "base_strength":      trial.suggest_float("base_strength", 0.7, 1.3, step=0.05),
                    "ptop":               trial.suggest_float("ptop", 0.30, 0.60, step=0.05),
                    "max_alpha":          trial.suggest_float("max_alpha", 1.0, 3.0, step=0.2),
                    "entropy_threshold":  trial.suggest_float("entropy_threshold", 2.0, 3.5, step=0.1),
                }
                
            else:
                raise ValueError(f"steering_method: {steering_method} not implemented")
            
            alpha_str = f"{steering_params.get('steering_alpha', 'N/A'):.3f}" if steering_params.get('steering_alpha') is not None else 'N/A'
            self.logger.info(f"🎯 Trial {trial.number}: {steering_method.upper()} with α={alpha_str}")
            
            # Step 1: Train and evaluate probe
            probe_score = self._train_and_evaluate_probe(trial, layer_id, probe_type, probe_c)
            self.logger.info(f"📊 Trial {trial.number}: Probe {probe_type} AUC = {probe_score:.4f}")
            
            # Step 2: Train steering method
            steering_instance = self._train_steering_method(
                trial, steering_method, layer_id, steering_params
            )
            
            # Step 3: Evaluate steering on validation set
            validation_accuracy = self._evaluate_steering_on_validation(
                steering_instance, steering_method, layer_id, steering_params
            )
            
            self.logger.info(f"🎯 Trial {trial.number}: Validation accuracy = {validation_accuracy:.4f}")
            trial.report(validation_accuracy, step=1)
            
            # Enhanced WandB logging with method-specific metrics
            metrics = {
                "validation_accuracy": validation_accuracy,
                "probe_score": probe_score,
                "method": steering_method,
                "layer": layer_id
            }
            self._log_trial_to_wandb(trial, metrics)
            
            return validation_accuracy
            
        except Exception as e:
            self.logger.error(f"❌ Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _log_enhanced_results(self, study, final_results):
        """Log enhanced results with method-specific analysis."""
        self.logger.info("="*80)
        self.logger.info("🏆 LLAMA 70B STEERING OPTIMIZATION RESULTS")
        self.logger.info("="*80)
        
        best_trial = study.best_trial
        best_method = best_trial.params.get("steering_method", "unknown")
        best_alpha = best_trial.params.get("steering_alpha", 0.0)
        best_layer = best_trial.params.get("layer_id", -1)
        
        self.logger.info(f"🥇 Best Method: {best_method.upper()}")
        self.logger.info(f"📊 Best Layer: {best_layer}")
        self.logger.info(f"⚡ Best Alpha: {best_alpha:.4f}")
        self.logger.info(f"🎯 Best Validation Accuracy: {study.best_value:.4f}")
        
        baseline_acc = final_results["baseline_benchmark_metrics"]["accuracy"]
        steered_acc = final_results["steered_benchmark_metrics"]["accuracy"]
        improvement = final_results["accuracy_improvement"]
        
        self.logger.info("📈 Test Results:")
        self.logger.info(f"   Baseline:  {baseline_acc:.4f}")
        self.logger.info(f"   Steered:   {steered_acc:.4f}")
        self.logger.info(f"   Improvement: {improvement:+.4f}")
        
        # Method-specific insights
        method_trials = [t for t in study.trials if t.params.get("steering_method") == best_method]
        if len(method_trials) > 1:
            method_values = [t.value for t in method_trials if t.value is not None]
            if method_values:
                self.logger.info(f"📊 {best_method.upper()} Method Statistics:")
                self.logger.info(f"   Trials: {len(method_values)}")
                self.logger.info(f"   Mean: {sum(method_values)/len(method_values):.4f}")
                self.logger.info(f"   Best: {max(method_values):.4f}")
        
        self.logger.info("="*80)
        
        # Call parent logging
        self._log_final_results_to_wandb(study, final_results)


def main():
    """Main entry point for Llama 70B optimization."""
    parser = argparse.ArgumentParser(
        description="Llama 3.1 70B Steering Optimization with Multiple Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model configuration
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to Llama 70B model (default: /workspace/models/llama31-70b-instruct-hf)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for inference. Recommendations: "
                            "1 (conservative for 70B model memory requirements)")
    
    # Dataset configuration
    parser.add_argument("--train-limit", type=int, default=None,
                       help="Number of training samples to load (default: 50)")
    parser.add_argument("--contrastive-pairs-limit", type=int, default=None,
                       help="Number of contrastive pairs for steering training (default: 25, bounded by train-limit)")
    parser.add_argument("--val-limit", type=int, default=None,
                       help="Number of validation samples to load (default: 30)")
    parser.add_argument("--test-limit", type=int, default=None,
                       help="Number of test samples to load (default: 100)")
    
    # Optimization configuration
    parser.add_argument("--study-name", type=str, default=None,
                       help="Optuna study name (default: llama_70b_steering_optimization)")
    parser.add_argument("--n-trials", type=int, default=None,
                       help="Number of optimization trials (default: 30)")
    parser.add_argument("--n-startup-trials", type=int, default=None,
                       help="Random exploration trials before TPE kicks in (default: 8)")
    parser.add_argument("--layer-range", type=int, nargs=2, default=None,
                       help="Layer search range as two integers (default: 30 40)")
    
    # WandB configuration
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable WandB experiment tracking (requires 'wandb login' first)")
    parser.add_argument("--wandb-project", type=str, default=None,
                       help="WandB project name")
    
    # Utility options
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test run (8 trials, 15 train, 8 val samples)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Log output to file (in addition to console)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Configure logging handlers
    handlers = [logging.StreamHandler()]  # Console output
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))  # File output
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    
    # Quick test mode overrides
    if args.quick_test:
        args.n_trials = 8
        args.train_limit = 15
        args.val_limit = 8
        args.test_limit = 15
        logger.info("🚀 Quick test mode enabled")
    
    # Display configuration
    logger.info("🦙 LLAMA 3.1 70B STEERING OPTIMIZATION")
    logger.info("="*80)
    logger.info("🔧 CONFIGURATION:")
    logger.info(f"   Model: {args.model_path or get_recommended_config_for_llama70b()['model_name']}")
    logger.info(f"   Batch Size: {args.batch_size or get_recommended_config_for_llama70b()['batch_size']}")
    logger.info(f"   Trials: {args.n_trials or get_recommended_config_for_llama70b()['n_trials']}")
    logger.info(f"   Train/Val/Test: {args.train_limit or 50}/{args.val_limit or 30}/{args.test_limit or 100}")
    logger.info(f"   WandB: {'Enabled' if args.use_wandb else 'Disabled'}")
    
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} devices)")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        logger.warning("⚠️  No CUDA detected - running on CPU (very slow!)")
    
    logger.info("="*80)
    
    # Create configuration and pipeline
    try:
        config = create_llama70b_config(args)
        pipeline = Llama70BSteeringPipeline(config)
        
        # Run optimization
        logger.info("🚀 Starting optimization...")
        results = pipeline.run_optimization()
        
        # Enhanced result display
        pipeline._log_enhanced_results(pipeline._create_optuna_study(), results)
        
        logger.info("✅ Optimization completed successfully!")
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