"""Duplicate detection diagnostics for contrastive pairs."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, Iterable, List

from .base import DiagnosticsConfig, DiagnosticsIssue, MetricReport


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def compute_duplicate_metrics(pairs: Iterable, config: DiagnosticsConfig) -> MetricReport:
    """Detect exact and near duplicates across prompts and responses."""

    pairs_list = list(pairs)

    prompt_counter: Counter[str] = Counter()
    positive_counter: Counter[str] = Counter()
    negative_counter: Counter[str] = Counter()
    indexed_prompts: Dict[str, List[int]] = defaultdict(list)

    for idx, pair in enumerate(pairs_list):
        prompt = _norm(getattr(pair, "prompt", ""))
        pos = _norm(getattr(pair.positive_response, "model_response", ""))
        neg = _norm(getattr(pair.negative_response, "model_response", ""))

        if prompt:
            prompt_counter[prompt] += 1
            indexed_prompts[prompt].append(idx)
        if pos:
            positive_counter[pos] += 1
        if neg:
            negative_counter[neg] += 1

    total_pairs = len(pairs_list)
    issues: List[DiagnosticsIssue] = []

    if total_pairs == 0:
        return MetricReport(name="duplicates", summary={"total_pairs": 0}, issues=[])

    def _collect_exact(counter: Counter[str], label: str) -> List[DiagnosticsIssue]:
        duplicates: List[DiagnosticsIssue] = []
        for value, count in counter.items():
            if count > 1:
                duplicates.append(
                    DiagnosticsIssue(
                        metric="duplicates",
                        severity="warning",
                        message=f"Exact duplicate detected in {label}.",
                        pair_index=None,
                        details={"value": value, "count": count, "field": label},
                    )
                )
        return duplicates

    issues.extend(_collect_exact(prompt_counter, "prompt"))
    issues.extend(_collect_exact(positive_counter, "positive_response"))
    issues.extend(_collect_exact(negative_counter, "negative_response"))

    exact_duplicate_fraction = sum(max(0, count - 1) for count in prompt_counter.values()) / total_pairs
    if exact_duplicate_fraction > config.max_exact_duplicate_fraction:
        issues.append(
            DiagnosticsIssue(
                metric="duplicates",
                severity="critical",
                message="Too many exact duplicate prompts detected.",
                pair_index=None,
                details={
                    "fraction": exact_duplicate_fraction,
                    "threshold": config.max_exact_duplicate_fraction,
                    "duplicates": [
                        {"prompt": prompt, "count": count}
                        for prompt, count in prompt_counter.items()
                        if count > 1
                    ],
                },
            )
        )

    near_duplicate_pairs: List[tuple[int, int, float]] = []
    prompt_items = list(prompt_counter.keys())
    for i, prompt_a in enumerate(prompt_items):
        for prompt_b in prompt_items[i + 1 :]:
            similarity = SequenceMatcher(None, prompt_a, prompt_b).ratio()
            if similarity >= config.near_duplicate_prompt_threshold:
                indices_a = indexed_prompts[prompt_a]
                indices_b = indexed_prompts[prompt_b]
                near_duplicate_pairs.append((indices_a[0], indices_b[0], similarity))
                issues.append(
                    DiagnosticsIssue(
                        metric="duplicates",
                        severity="warning",
                        message="Near-duplicate prompts detected.",
                        pair_index=None,
                        details={
                            "prompt_a": prompt_a,
                            "prompt_b": prompt_b,
                            "similarity": similarity,
                            "a_indices": indices_a,
                            "b_indices": indices_b,
                        },
                    )
                )

    summary = {
        "total_pairs": total_pairs,
        "exact_duplicate_fraction": exact_duplicate_fraction,
        "unique_prompts": len(prompt_counter),
        "near_duplicate_count": len(near_duplicate_pairs),
    }

    return MetricReport(name="duplicates", summary=summary, issues=issues)
