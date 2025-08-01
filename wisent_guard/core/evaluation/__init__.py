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
        create_results_dashboard,
        generate_summary_report,
        plot_evaluation_results,
    )

    _comprehensive_available = True
except ImportError:
    _comprehensive_available = False

if _comprehensive_available:
    __all__ = [
        "BenchmarkConfig",
        "BenchmarkResult",
        "ComprehensiveEvaluationConfig",
        "ComprehensiveEvaluationPipeline",
        "ModelExporter",
        "create_results_dashboard",
        "generate_summary_report",
        "plot_evaluation_results",
    ]
else:
    __all__ = ["BenchmarkConfig", "BenchmarkResult", "ModelExporter"]
