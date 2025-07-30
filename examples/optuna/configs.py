"""
Example Configurations for Optuna Optimization and Evaluation

This module provides pre-configured settings for common use cases:
- Minimal config: Fast testing with small model
- Production config: Full optimization with larger models  
- Debug config: Quick debugging with minimal data
"""


import torch

from wisent_guard.core.optuna.optuna_pipeline import OptimizationConfig


def create_minimal_config() -> OptimizationConfig:
    """
    Minimal config for fine-tuned model testing.
    
    This configuration is designed for:
    - Quick testing and debugging
    - Fine-tuned models (realtreetune/rho-1b-sft-GSM8K)
    - Small steering ranges (0.0-0.3)
    - Fast execution (~5-10 minutes)
    
    Expected behavior:
    - Baseline accuracy: ~14-27% on GSM8K
    - Optimal steering α should be close to 0.0
    - Steering typically hurts performance on fine-tuned models
    """
    return OptimizationConfig(
        # Model: Fine-tuned on GSM8K, small and fast
        model_name="realtreetune/rho-1b-sft-GSM8K",
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # Datasets: Mixed training, GSM8K validation/test
        train_dataset="hendrycks_math",  # Diverse math for steering training
        val_dataset="gsm8k",            # GSM8K for optimization target  
        test_dataset="gsm8k",           # GSM8K for final evaluation
        
        # Data limits: Small for speed
        train_limit=20,              # Training samples to load
        contrastive_pairs_limit=15,  # Contrastive pairs for steering training
        val_limit=20,                # Validation samples (all used for evaluation)
        test_limit=50,               # Test samples for final evaluation
        
        # Optuna configuration: Few trials for quick testing
        study_name="minimal_example_test",
        db_url="sqlite:///optuna_minimal_test.db",
        n_trials=6,                  # 6 trials to test both CAA and DAC
        n_startup_trials=2,          # 2 random trials, then TPE
        sampler="TPE",
        pruner="MedianPruner",
        
        # Search space: Single layer for speed
        layer_search_range=(10, 10),  # Single layer (layer 10)
        probe_type="logistic_regression",
        steering_methods=["caa", "dac"],  # Test both methods
        
        # Generation: Deterministic for reproducible results
        temperature=0.0,
        do_sample=False,
        batch_size=4,
        max_length=512,
        max_new_tokens=128,
        
        # Output directories
        output_dir="outputs/optuna_minimal_example",
        cache_dir="cache/optuna_minimal_example",
        
        # Disable WandB for simple testing
        use_wandb=False,
        wandb_project="minimal_test",
        
        # Efficiency settings
        seed=42,
        max_layers_to_search=1,
        early_stopping_patience=5
    )


def create_production_config() -> OptimizationConfig:
    """
    Production config for serious hyperparameter optimization.
    
    This configuration is designed for:
    - Full-scale optimization with larger models
    - Comprehensive search spaces
    - Production-ready results
    - Longer execution time (~2-4 hours)
    """
    return OptimizationConfig(
        # Model: Larger, more capable model
        model_name="Qwen/Qwen2.5-7B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # Datasets: Same as minimal but larger
        train_dataset="hendrycks_math",
        val_dataset="gsm8k", 
        test_dataset="gsm8k",
        
        # Data limits: Larger for better optimization
        train_limit=200,
        contrastive_pairs_limit=100,
        val_limit=100,
        test_limit=500,
        
        # Optuna configuration: More thorough search
        study_name="production_optimization",
        db_url="sqlite:///optuna_production_study.db",
        n_trials=100,                # More trials for better optimization
        n_startup_trials=20,         # More random exploration
        sampler="TPE",
        pruner="MedianPruner",
        
        # Search space: Multiple layers
        layer_search_range=(14, 20),  # Search across multiple layers
        probe_type="logistic_regression",
        steering_methods=["caa", "dac"],
        
        # Generation: Sampling for diversity
        temperature=0.7,
        do_sample=True,
        batch_size=8,               # Larger batch size
        max_length=512,
        max_new_tokens=256,         # More tokens for complex problems
        
        # Output directories
        output_dir="outputs/production_optimization",
        cache_dir="cache/production_optimization",
        
        # Enable WandB for tracking
        use_wandb=True,
        wandb_project="wisent-guard-production",
        
        # Efficiency settings
        seed=42,
        max_layers_to_search=7,     # Search more layers
        early_stopping_patience=15
    )


def create_debug_config() -> OptimizationConfig:
    """
    Debug config for rapid iteration and testing.
    
    This configuration is designed for:
    - Very fast execution (~1-2 minutes)
    - Testing code changes
    - Debugging pipeline issues
    - Minimal resource usage
    """
    return OptimizationConfig(
        # Model: Same small model as minimal
        model_name="realtreetune/rho-1b-sft-GSM8K",
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # Datasets: Same as minimal
        train_dataset="hendrycks_math",
        val_dataset="gsm8k",
        test_dataset="gsm8k",
        
        # Data limits: Very small for speed
        train_limit=10,              # Minimal training data
        contrastive_pairs_limit=5,   # Minimal pairs
        val_limit=10,                # Minimal validation
        test_limit=20,               # Minimal test
        
        # Optuna configuration: Minimal trials
        study_name="debug_test",
        db_url="sqlite:///optuna_debug_test.db",
        n_trials=3,                  # Just 3 trials
        n_startup_trials=1,          # Minimal startup
        sampler="Random",            # Random for speed
        pruner="MedianPruner",
        
        # Search space: Fixed single layer
        layer_search_range=(10, 10),
        probe_type="logistic_regression",
        steering_methods=["caa"],    # Test only CAA for speed
        
        # Generation: Deterministic and fast
        temperature=0.0,
        do_sample=False,
        batch_size=2,               # Small batch
        max_length=256,             # Shorter sequences
        max_new_tokens=64,          # Fewer tokens
        
        # Output directories
        output_dir="outputs/debug_test",
        cache_dir="cache/debug_test",
        
        # Disable WandB for debugging
        use_wandb=False,
        wandb_project="debug",
        
        # Efficiency settings
        seed=42,
        max_layers_to_search=1,
        early_stopping_patience=2   # Stop early
    )


def create_evaluation_config(
    model_name: str = "realtreetune/rho-1b-sft-GSM8K",
    test_dataset: str = "gsm8k",
    test_limit: int = 50,
    layer_id: int = 10
) -> OptimizationConfig:
    """
    Config specifically for evaluation (not optimization).
    
    Args:
        model_name: Model to evaluate with
        test_dataset: Dataset to evaluate on  
        test_limit: Number of test samples
        layer_id: Single layer to use for evaluation
        
    This configuration is designed for:
    - Evaluating specific parameters
    - Cross-dataset evaluation
    - Quick evaluation runs
    """
    return OptimizationConfig(
        # Model configuration
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # Datasets: Minimal train/val, focus on test
        train_dataset="hendrycks_math",
        val_dataset="gsm8k",
        test_dataset=test_dataset,
        
        # Data limits: Minimal train/val for required setup
        train_limit=20,              # Minimal for steering training
        contrastive_pairs_limit=15,
        val_limit=20,
        test_limit=test_limit,       # Focus on test evaluation
        
        # Optuna: Not used for evaluation, but required
        study_name="evaluation_only",
        db_url="sqlite:///evaluation_temp.db",
        n_trials=1,                  # Not used
        n_startup_trials=1,
        sampler="Random",
        pruner="MedianPruner", 
        
        # Search space: Single fixed layer
        layer_search_range=(layer_id, layer_id),
        probe_type="logistic_regression",
        steering_methods=["caa", "dac"],
        
        # Generation: Deterministic for consistent evaluation
        temperature=0.0,
        do_sample=False,
        batch_size=4,
        max_length=512,
        max_new_tokens=128,
        
        # Output directories
        output_dir="outputs/evaluation_temp",
        cache_dir="cache/evaluation_temp",
        
        # Disable WandB for evaluation
        use_wandb=False,
        wandb_project="evaluation",
        
        # Efficiency settings
        seed=42,
        max_layers_to_search=1,
        early_stopping_patience=1
    )


# Pre-configured parameter sets for common evaluation scenarios
EXAMPLE_BEST_PARAMS = {
    "minimal_caa": {
        "layer_id": 10,
        "probe_type": "logistic_regression", 
        "probe_c": 1.0,
        "steering_method": "caa",
        "steering_alpha": 0.18,  # Optimal from minimal example
    },
    
    "minimal_dac": {
        "layer_id": 10,
        "probe_type": "logistic_regression",
        "probe_c": 1.0, 
        "steering_method": "dac",
        "steering_alpha": 0.15,
        "entropy_threshold": 1.5,
        "ptop": 0.5,
        "max_alpha": 2.0,
    },
    
    "production_caa": {
        "layer_id": 16,
        "probe_type": "logistic_regression",
        "probe_c": 1.0,
        "steering_method": "caa", 
        "steering_alpha": 0.8,  # Higher for non-fine-tuned models
    },
    
    "zero_steering": {
        "layer_id": 10,
        "probe_type": "logistic_regression",
        "probe_c": 1.0,
        "steering_method": "caa",
        "steering_alpha": 0.0,  # No steering baseline
    }
}