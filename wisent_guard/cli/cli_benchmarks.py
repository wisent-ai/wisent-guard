"""
Benchmark availability definitions.
"""

from __future__ import annotations

import warnings

try:
    from wisent_guard.core.lm_harness_integration.only_benchmarks import CORE_BENCHMARKS  # type: ignore
except ImportError:  # pragma: no cover - executed only in constrained envs
    CORE_BENCHMARKS = {
        "truthfulqa_mc1": {
            "task": "truthfulqa_mc1",
            "tags": ["hallucination"],
            "priority": "high",
        },
        "hellaswag": {
            "task": "hellaswag",
            "tags": ["reasoning"],
            "priority": "high",
        },
        "mmlu": {"task": "mmlu", "tags": ["knowledge"], "priority": "high"},
    }
    warnings.warn(
        "Could not import full CORE_BENCHMARKS; using minimal fallback list.",
        RuntimeWarning,
        stacklevel=1,
    )

# 2. Attempt to load unavailable benchmark names from downloader helper
try:
    from wisent_guard.core.download_full_benchmarks import FullBenchmarkDownloader  # type: ignore
    UNAVAILABLE_BENCHMARKS = getattr(FullBenchmarkDownloader, "UNAVAILABLE_BENCHMARKS", set())
except ImportError:  # pragma: no cover - optional dependency
    UNAVAILABLE_BENCHMARKS = set()

# Import allowed tasks from centralized configuration
from wisent_guard.parameters.task_config import ALLOWED_TASKS

# Filter to only available (working) benchmarks - this gives us the validated benchmarks
AVAILABLE_BENCHMARKS = {
    name: config
    for name, config in CORE_BENCHMARKS.items()
    if name not in UNAVAILABLE_BENCHMARKS and name in ALLOWED_TASKS
}