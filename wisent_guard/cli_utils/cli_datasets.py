"""
Benchmarks dataset configurations for CLI tools. 
"""

from __future__ import annotations
from typing import Any, Sequence
import logging


logger = logging.getLogger(__name__) 


try:
    from wisent_guard.core.download_full_benchmarks import FullBenchmarkDownloader  
except Exception: 
    FullBenchmarkDownloader = None  

# --- Allowed tasks (must exist; fail fast with a clear error) --------------------
try:
    from wisent_guard.parameters.task_config import ALLOWED_TASKS 
except Exception as exc:  
    raise ImportError(
        "Failed to import ALLOWED_TASKS from wisent_guard.parameters.task_config. "
        "Ensure your project is installed and PYTHONPATH is set correctly."
    ) from exc


# --- Resolvers for CORE_BENCHMARKS and UNAVAILABLE_BENCHMARKS --------------------
def _resolve_benchmark_catalogs() -> tuple[dict[str, dict[str, Any]], Sequence[str]]:
    """Resolve CORE_BENCHMARKS and UNAVAILABLE_BENCHMARKS with fallbacks.

    Returns:
        tuple[dict[str, dict[str, Any]], Sequence[str]]: (CORE_BENCHMARKS, UNAVAILABLE_BENCHMARKS)
    """
    # 1) Primary import path
    try:
        from wisent_guard.core.lm_harness_integration.only_benchmarks import CORE_BENCHMARKS 
        try:
            # Prefer canonical source for unavailable set if available
            if FullBenchmarkDownloader is not None:
                UNAVAILABLE_BENCHMARKS = getattr(FullBenchmarkDownloader, "UNAVAILABLE_BENCHMARKS", set())
            else:
                UNAVAILABLE_BENCHMARKS = set()
        except Exception:  
            UNAVAILABLE_BENCHMARKS = set()
        return CORE_BENCHMARKS, UNAVAILABLE_BENCHMARKS
    except Exception:  
        pass

    # 2) Minimal fallback — keep the pipeline usable without full deps
    logger.warning(
        "Could not import CORE_BENCHMARKS; using minimal fallback list (reduced coverage)."
    )
    CORE_BENCHMARKS = {
        "truthfulqa_mc1": {"task": "truthfulqa_mc1", "tags": ["hallucination"], "priority": "high"},
        "hellaswag": {"task": "hellaswag", "tags": ["reasoning"], "priority": "high"},
        "mmlu": {"task": "mmlu", "tags": ["knowledge"], "priority": "high"},
    }
    UNAVAILABLE_BENCHMARKS = set()
    return CORE_BENCHMARKS, UNAVAILABLE_BENCHMARKS


CORE_BENCHMARKS, UNAVAILABLE_BENCHMARKS = _resolve_benchmark_catalogs()

AVAILABLE_BENCHMARKS: dict[str, dict[str, Any]] = {
    name: cfg
    for name, cfg in CORE_BENCHMARKS.items()
    if name not in UNAVAILABLE_BENCHMARKS and name in ALLOWED_TASKS
}


__all__ = [
    "CORE_BENCHMARKS",
    "UNAVAILABLE_BENCHMARKS",
    "AVAILABLE_BENCHMARKS",
]