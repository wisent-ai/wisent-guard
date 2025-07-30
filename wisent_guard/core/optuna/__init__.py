"""
Comprehensive Evaluation Framework for Wisent Guard

This module provides a comprehensive evaluation framework that properly separates:
1. Benchmark Performance: Model's ability to solve problems  
2. Probe Performance: Probe's ability to detect correctness from activations
3. Steering Optimization: Grid search based on benchmark results

Key components:
- ComprehensiveEvaluationPipeline: Main evaluation pipeline
- ComprehensiveEvaluationConfig: Configuration management
- Visualization utilities for human-readable results
- Modular design for easy extension and testing
"""

from .optuna_pipeline import OptimizationConfig, OptimizationPipeline
from .metrics import (
    evaluate_benchmark_performance,
    evaluate_probe_performance,  
    calculate_comprehensive_metrics,
    generate_performance_summary
)

__all__ = [
    "OptimizationConfig",
    "OptimizationPipeline", 
    "evaluate_benchmark_performance",
    "evaluate_probe_performance",
    "calculate_comprehensive_metrics",
    "generate_performance_summary",
]