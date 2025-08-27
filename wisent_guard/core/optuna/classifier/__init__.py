"""
Optuna-based classifier optimization module.

This module provides modern, efficient classifier optimization using Optuna with
intelligent caching and pre-generation of activations for maximum performance.
"""

from .activation_generator import ActivationData, ActivationGenerator, GenerationConfig
from .classifier_cache import CacheConfig, CacheMetadata, ClassifierCache
from .optuna_classifier_optimizer import ClassifierOptimizationConfig, OptimizationResult, OptunaClassifierOptimizer

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
    "ClassifierOptimizationConfig",
    "OptimizationResult",
]
