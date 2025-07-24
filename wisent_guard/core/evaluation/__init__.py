"""
Evaluation orchestration for multiple code generation benchmarks and comprehensive evaluation.
"""

from .benchmarks import BenchmarkConfig, BenchmarkResult
from .exporters import ModelExporter

# Import comprehensive evaluation components if available
try:
    from .comprehensive import (
        ComprehensiveEvaluationConfig,
        ComprehensiveEvaluationPipeline,
        plot_evaluation_results,
        create_results_dashboard,
        generate_summary_report
    )
    _comprehensive_available = True
except ImportError:
    _comprehensive_available = False

if _comprehensive_available:
    __all__ = [
        "BenchmarkConfig", "BenchmarkResult", "ModelExporter",
        "ComprehensiveEvaluationConfig", "ComprehensiveEvaluationPipeline",
        "plot_evaluation_results", "create_results_dashboard", "generate_summary_report"
    ]
else:
    __all__ = ["BenchmarkConfig", "BenchmarkResult", "ModelExporter"]