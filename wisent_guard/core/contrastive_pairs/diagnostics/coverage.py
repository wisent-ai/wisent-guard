"""Coverage and diversity diagnostics for contrastive pairs."""

from __future__ import annotations

from statistics import mean
from typing import Iterable, List

from .base import DiagnosticsConfig, DiagnosticsIssue, MetricReport


def compute_coverage_metrics(pairs: Iterable, config: DiagnosticsConfig) -> MetricReport:
    """Assess dataset coverage such as prompt diversity and response length."""

    pairs_list = list(pairs)

    if not pairs_list:
        return MetricReport(name="coverage", summary={"total_pairs": 0}, issues=[])

    unique_prompts = {getattr(pair, "prompt", "").strip().lower() for pair in pairs_list}
    prompt_ratio = len(unique_prompts) / len(pairs_list)

    pos_lengths: List[int] = []
    neg_lengths: List[int] = []
    labels = set()

    for pair in pairs_list:
        pos_text = getattr(pair.positive_response, "model_response", "")
        neg_text = getattr(pair.negative_response, "model_response", "")
        pos_lengths.append(len(pos_text))
        neg_lengths.append(len(neg_text))

        if pair.label:
            labels.add(pair.label)

    avg_positive_length = mean(pos_lengths) if pos_lengths else 0.0
    avg_negative_length = mean(neg_lengths) if neg_lengths else 0.0

    issues: List[DiagnosticsIssue] = []

    if prompt_ratio < config.min_unique_prompt_ratio:
        issues.append(
            DiagnosticsIssue(
                metric="coverage",
                severity="warning",
                message="Prompt diversity below configured ratio.",
                pair_index=None,
                details={
                    "ratio": prompt_ratio,
                    "threshold": config.min_unique_prompt_ratio,
                    "unique_prompts": len(unique_prompts),
                    "total_pairs": len(pairs_list),
                },
            )
        )

    if avg_positive_length < config.min_average_length or avg_negative_length < config.min_average_length:
        issues.append(
            DiagnosticsIssue(
                metric="coverage",
                severity="warning",
                message="Average response length below minimum threshold.",
                pair_index=None,
                details={
                    "avg_positive_length": avg_positive_length,
                    "avg_negative_length": avg_negative_length,
                    "threshold": config.min_average_length,
                },
            )
        )

    summary = {
        "total_pairs": len(pairs_list),
        "unique_prompt_ratio": prompt_ratio,
        "avg_positive_length": avg_positive_length,
        "avg_negative_length": avg_negative_length,
        "label_coverage": len(labels),
    }

    return MetricReport(name="coverage", summary=summary, issues=issues)
