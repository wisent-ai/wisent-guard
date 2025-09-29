"""Aggregate interface for contrastive pair diagnostics."""

from __future__ import annotations

from typing import Iterable

from .base import DiagnosticsConfig, DiagnosticsReport
from .divergence import compute_divergence_metrics
from .duplicates import compute_duplicate_metrics
from .coverage import compute_coverage_metrics
from .activations import compute_activation_metrics
from .control_vectors import ControlVectorDiagnosticsConfig, run_control_vector_diagnostics

__all__ = [
    "DiagnosticsConfig",
    "DiagnosticsReport",
    "run_all_diagnostics",
    "ControlVectorDiagnosticsConfig",
    "run_control_vector_diagnostics",
]


def run_all_diagnostics(pairs: Iterable, config: DiagnosticsConfig | None = None) -> DiagnosticsReport:
    """Run all registered diagnostics for the provided contrastive pairs.

    Args:
        pairs: Iterable of contrastive pair objects implementing the required interface.
        config: Optional diagnostics configuration overrides.

    Returns:
        Aggregated diagnostics report capturing metric summaries and issues.
    """

    cfg = config or DiagnosticsConfig()

    metric_reports = [
        compute_divergence_metrics(pairs, cfg),
        compute_duplicate_metrics(pairs, cfg),
        compute_coverage_metrics(pairs, cfg),
        compute_activation_metrics(pairs, cfg),
    ]

    combined = DiagnosticsReport.from_metrics(metric_reports)
    return combined
