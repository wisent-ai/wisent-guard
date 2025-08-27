"""
Optuna-based classifier optimization module.

This module provides modern, efficient classifier optimization using Optuna with
intelligent caching and pre-generation of activations for maximum performance.
"""

from .activation_generator import ActivationGenerator, GenerationConfig, ActivationData

from .classifier_cache import ClassifierCache, CacheConfig, CacheMetadata

from .optuna_classifier_optimizer import OptunaClassifierOptimizer, OptimizationConfig, OptimizationResult

__all__ = [
    # Activation generation
    "ActivationGenerator",
    "GenerationConfig",
    "ActivationData",
    # Classifier caching
    "ClassifierCache",
    "CacheConfig",
    "CacheMetadata",
    # Optuna optimization
    "OptunaClassifierOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
]
