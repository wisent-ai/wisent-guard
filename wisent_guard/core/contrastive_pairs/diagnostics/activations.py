"""Activation completeness diagnostics for contrastive pairs."""

from __future__ import annotations

from typing import Iterable, List

from .base import DiagnosticsConfig, DiagnosticsIssue, MetricReport


def compute_activation_metrics(pairs: Iterable, config: DiagnosticsConfig) -> MetricReport:
    """Check for presence of activations across the contrastive pair set."""

    pairs_list = list(pairs)

    if not pairs_list:
        return MetricReport(name="activations", summary={"total_pairs": 0}, issues=[])

    has_positive = 0
    has_negative = 0
    mismatch_indices: List[int] = []

    for idx, pair in enumerate(pairs_list):
        pos_has = getattr(pair.positive_response, "layers_activations", None) is not None
        neg_has = getattr(pair.negative_response, "layers_activations", None) is not None

        has_positive += int(pos_has)
        has_negative += int(neg_has)

        if pos_has != neg_has:
            mismatch_indices.append(idx)

    total_pairs = len(pairs_list)
    issues: List[DiagnosticsIssue] = []

    if mismatch_indices and config.warn_on_missing_activations:
        issues.append(
            DiagnosticsIssue(
                metric="activations",
                severity="warning",
                message="Positive/negative activation availability mismatch detected.",
                pair_index=None,
                details={"indices": mismatch_indices},
            )
        )

    summary = {
        "total_pairs": total_pairs,
        "pairs_with_positive_activations": has_positive,
        "pairs_with_negative_activations": has_negative,
        "mismatch_pairs": len(mismatch_indices),
    }

    return MetricReport(name="activations", summary=summary, issues=issues)
