"""
Optuna-based Optimization Framework for Wisent Guard

This module provides Optuna-based hyperparameter optimization for steering methods:
1. Hyperparameter Optimization: Optuna-driven search for best parameters
2. Evaluation Pipeline: Comprehensive evaluation on multiple datasets
3. Reproducibility: Complete experiment tracking and reproduction

Key components:
- OptimizationPipeline: Main Optuna optimization pipeline
- OptimizationConfig: Configuration management
- Evaluation utilities for parameter assessment
- Data loading and metrics calculation
"""

from .metrics import (
    calculate_comprehensive_metrics,
    evaluate_benchmark_performance,
    evaluate_probe_performance,
    generate_performance_summary,
)
from .optuna_pipeline import OptimizationConfig, OptimizationPipeline

__all__ = [
    "OptimizationConfig",
    "OptimizationPipeline",
    "calculate_comprehensive_metrics",
    "evaluate_benchmark_performance",
    "evaluate_probe_performance",
    "generate_performance_summary",
]
