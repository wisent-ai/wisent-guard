"""
Optuna-based Optimization Framework for Wisent Guard

This module provides Optuna-based hyperparameter optimization for both steering and classifier systems:

STEERING OPTIMIZATION:
1. Hyperparameter Optimization: Optuna-driven search for best steering parameters
2. Evaluation Pipeline: Comprehensive evaluation on multiple datasets
3. Reproducibility: Complete experiment tracking and reproduction

CLASSIFIER OPTIMIZATION:
1. Activation Pre-generation: Efficient caching of model activations
2. Model Training: Optimized logistic regression and MLP classifiers
3. Intelligent Caching: Avoid retraining identical configurations
4. Cross-validation: Robust performance evaluation

Key components:
- Steering: OptimizationPipeline, OptimizationConfig, metrics
- Classifier: OptunaClassifierOptimizer, GenerationConfig, CacheConfig
"""

# Steering optimization components
# Classifier optimization components
from wisent_guard.core.optuna.classifier import (
    ActivationGenerator,
    CacheConfig,
    ClassifierCache,
    ClassifierOptimizationConfig as ClassifierOptimizationConfig,
    GenerationConfig,
    OptimizationResult,
    OptunaClassifierOptimizer,
)
from wisent_guard.core.optuna.steering.metrics import (
    calculate_comprehensive_metrics,
    evaluate_benchmark_performance,
    evaluate_probe_performance,
    generate_performance_summary,
)
from wisent_guard.core.optuna.steering.optuna_pipeline import OptimizationConfig, OptimizationPipeline

__all__ = [
    # Steering optimization
    "OptimizationConfig",
    "OptimizationPipeline",
    "calculate_comprehensive_metrics",
    "evaluate_benchmark_performance",
    "evaluate_probe_performance",
    "generate_performance_summary",
    # Classifier optimization
    "OptunaClassifierOptimizer",
    "ClassifierOptimizationConfig",
    "GenerationConfig",
    "CacheConfig",
    "ActivationGenerator",
    "ClassifierCache",
    "OptimizationResult",
]
