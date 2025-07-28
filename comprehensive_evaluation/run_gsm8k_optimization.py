"""
Simple runner script for GSM8K optimization pipeline.

This script provides an easy way to run the GSM8K optimization with different configurations.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add wisent-guard to path
sys.path.append(str(Path(__file__).parent.parent))

from gsm8k_optimization_pipeline import GSM8KOptimizationPipeline, OptimizationConfig


def create_debug_config() -> OptimizationConfig:
    """Create a small configuration for quick debugging."""
    return OptimizationConfig(
        model_name="realtreetune/rho-1b-sft-GSM8K",  # Try a different model for testing
        train_limit=10,  # Reduced for faster testing
        val_limit=10,   # Reduced for faster testing
        test_limit=10,  # Reduced for faster testing
        n_trials=3,    # Just 2 trials for testing
        layer_search_range=(8, 12),  # Adjust for DialoGPT layers
        steering_methods=["dac"],  # Start with just DAC
        output_dir="outputs/gsm8k_dialogpt",
        cache_dir="cache/gsm8k_dialogpt"
    )


def create_full_config() -> OptimizationConfig:
    """Create the full configuration as specified in the requirements."""
    return OptimizationConfig(
        model_name="PradhyumnaPoralla/gpt2_gsm8k_CLM",  # 5% accuracy on GSM8K
        train_limit=500,
        val_limit=200,
        test_limit=300,
        n_trials=50,
        layer_search_range=(6, 12),  # Last 6 blocks
        steering_methods=["dac", "caa"],
        output_dir="outputs/gsm8k_full",
        cache_dir="cache/gsm8k_full"
    )


def main():
    parser = argparse.ArgumentParser(description="Run GSM8K optimization pipeline")
    parser.add_argument("--mode", choices=["debug", "full"], default="debug",
                       help="Run in debug mode (small) or full mode")
    parser.add_argument("--model", type=str, 
                       default=None,
                       help="Model name to use")
    parser.add_argument("--n-trials", type=int, default=None,
                       help="Number of Optuna trials to run")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Generation temperature (0.0=deterministic, >0.0=random)")
    parser.add_argument("--do-sample", action="store_true",
                       help="Enable sampling (vs greedy decoding)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Create configuration
    if args.mode == "debug":
        config = create_debug_config()
        logger.info("Running in DEBUG mode with small dataset")
    else:
        config = create_full_config()
        logger.info("Running in FULL mode with complete dataset")
    
    # Override with command line arguments
    if args.model:
        config.model_name = args.model
    if args.n_trials:
        config.n_trials = args.n_trials
    if args.output_dir:
        config.output_dir = args.output_dir
        config.cache_dir = f"{args.output_dir}_cache"
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.do_sample:
        config.do_sample = True
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Train/Val/Test: {config.train_limit}/{config.val_limit}/{config.test_limit}")
    logger.info(f"  Trials: {config.n_trials}")
    logger.info(f"  Layer range: {config.layer_search_range}")
    logger.info(f"  Generation: temp={config.temperature}, sample={config.do_sample}")
    logger.info(f"  Output: {config.output_dir}")
    
    # Run optimization
    pipeline = None
    try:
        pipeline = GSM8KOptimizationPipeline(config)
        results = pipeline.run_optimization()
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Best validation score: {results['best_validation_score']:.4f}")
        print(f"Baseline accuracy: {results['baseline_benchmark_metrics']['accuracy']:.4f}")
        print(f"Steered accuracy: {results['steered_benchmark_metrics']['accuracy']:.4f}")
        print(f"Accuracy improvement: {results['accuracy_improvement']:+.4f}")
        print(f"Probe AUC: {results['test_probe_metrics']['auc']:.4f}")
        print(f"Best configuration: {results['best_trial_params']}")
        print(f"Results saved to: {config.output_dir}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up memory
        if pipeline:
            pipeline.cleanup_memory()


if __name__ == "__main__":
    main()