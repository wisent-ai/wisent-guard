"""Divergence diagnostics for contrastive pairs."""

from __future__ import annotations

from difflib import SequenceMatcher
from statistics import mean
from typing import Iterable, List

from .base import DiagnosticsConfig, DiagnosticsIssue, MetricReport


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def compute_divergence_metrics(pairs: Iterable, config: DiagnosticsConfig) -> MetricReport:
    """Evaluate textual divergence between positive and negative responses."""

    pairs_list = list(pairs)

    divergences: List[float] = []
    issues: List[DiagnosticsIssue] = []

    if not pairs_list:
        return MetricReport(
            name="divergence",
            summary={
                "mean_divergence": 0.0,
                "min_divergence": 0.0,
                "max_divergence": 0.0,
                "low_divergence_fraction": 0.0,
            },
            issues=[],
        )

    for idx, pair in enumerate(pairs_list):
        positive = getattr(pair.positive_response, "model_response", "")
        negative = getattr(pair.negative_response, "model_response", "")

        norm_pos = _normalize_text(positive)
        norm_neg = _normalize_text(negative)

        if not norm_pos or not norm_neg:
            issues.append(
                DiagnosticsIssue(
                    metric="divergence",
                    severity="critical",
                    message="Missing positive or negative response text.",
                    pair_index=idx,
                    details={"positive": bool(norm_pos), "negative": bool(norm_neg)},
                )
            )
            divergences.append(0.0)
            continue

        similarity = SequenceMatcher(None, norm_pos, norm_neg).ratio()
        divergence = 1.0 - similarity
        divergences.append(divergence)

        if divergence < config.min_divergence:
            issues.append(
                DiagnosticsIssue(
                    metric="divergence",
                    severity="warning",
                    message="Positive and negative responses are highly similar.",
                    pair_index=idx,
                    details={"divergence": divergence, "similarity": similarity},
                )
            )

    low_divergence_fraction = 0.0
    low_divergence_count = sum(1 for value in divergences if value < config.min_divergence)
    low_divergence_fraction = low_divergence_count / len(divergences)

    if low_divergence_fraction > config.max_low_divergence_fraction:
        issues.append(
            DiagnosticsIssue(
                metric="divergence",
                severity="critical",
                message="Too many pairs fall below divergence threshold.",
                pair_index=None,
                details={
                    "fraction": low_divergence_fraction,
                    "threshold": config.max_low_divergence_fraction,
                    "count": low_divergence_count,
                    "total": len(divergences),
                },
            )
        )

    summary = {
        "mean_divergence": mean(divergences) if divergences else 0.0,
        "min_divergence": min(divergences) if divergences else 0.0,
        "max_divergence": max(divergences) if divergences else 0.0,
        "low_divergence_fraction": low_divergence_fraction,
    }

    return MetricReport(name="divergence", summary=summary, issues=issues)
