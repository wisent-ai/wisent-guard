"""Shared dataclasses and helpers for contrastive pair diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass(slots=True)
class DiagnosticsConfig:
    """Threshold configuration for diagnostics.

    Attributes:
        min_divergence: Minimum acceptable divergence between positive and negative responses.
        max_low_divergence_fraction: Maximum allowed share of pairs falling below the divergence threshold.
        near_duplicate_prompt_threshold: Similarity threshold (0-1) at which prompts are treated as near duplicates.
        max_exact_duplicate_fraction: Maximum allowed share of exact duplicate prompts or responses.
        min_unique_prompt_ratio: Minimum ratio of unique prompts to total pairs for coverage diagnostics.
        min_average_length: Minimum average response length (characters) indicating sufficient content.
        warn_on_missing_activations: Whether missing activations should be reported as issues.
    """

    min_divergence: float = 0.3
    max_low_divergence_fraction: float = 0.1
    near_duplicate_prompt_threshold: float = 0.9
    max_exact_duplicate_fraction: float = 0.05
    min_unique_prompt_ratio: float = 0.75
    min_average_length: int = 15
    warn_on_missing_activations: bool = True


@dataclass(slots=True)
class DiagnosticsIssue:
    """Represents a single diagnostics issue detected in a pair set."""

    metric: str
    severity: str
    message: str
    pair_index: int | None = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricReport:
    """Stores summary statistics for a single diagnostics metric."""

    name: str
    summary: Dict[str, Any]
    issues: List[DiagnosticsIssue] = field(default_factory=list)


@dataclass(slots=True)
class DiagnosticsReport:
    """Aggregated diagnostics results across metrics."""

    metrics: Dict[str, MetricReport]
    issues: List[DiagnosticsIssue]
    summary: Dict[str, Any]
    has_critical_issues: bool

    @classmethod
    def from_metrics(cls, reports: Iterable[MetricReport]) -> "DiagnosticsReport":
        metrics_map: Dict[str, MetricReport] = {}
        all_issues: List[DiagnosticsIssue] = []

        for report in reports:
            metrics_map[report.name] = report
            all_issues.extend(report.issues)

        summary = {name: report.summary for name, report in metrics_map.items()}
        has_critical = any(issue.severity == "critical" for issue in all_issues)

        return cls(metrics=metrics_map, issues=all_issues, summary=summary, has_critical_issues=has_critical)
