"""Diagnostics for steering/control vectors."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping

import torch

from wisent_guard.core.activations.core.atoms import LayerActivations, RawActivationMap

from .base import DiagnosticsIssue, DiagnosticsReport, MetricReport

__all__ = [
    "ControlVectorDiagnosticsConfig",
    "run_control_vector_diagnostics",
]


@dataclass(slots=True)
class ControlVectorDiagnosticsConfig:
    """Thresholds and options for control vector diagnostics."""

    min_norm: float = 1e-4
    max_norm: float | None = None
    zero_value_threshold: float = 1e-8
    max_zero_fraction: float = 0.999
    warn_on_missing: bool = True


def _to_layer_activations(vectors: LayerActivations | RawActivationMap | Mapping[str, object] | None) -> LayerActivations:
    if isinstance(vectors, LayerActivations):
        return vectors
    data: RawActivationMap = vectors or {}
    return LayerActivations(data)


def run_control_vector_diagnostics(
    vectors: LayerActivations | RawActivationMap | Mapping[str, object] | None,
    config: ControlVectorDiagnosticsConfig | None = None,
) -> DiagnosticsReport:
    """Evaluate steering/control vectors for basic health metrics."""

    cfg = config or ControlVectorDiagnosticsConfig()
    activations = _to_layer_activations(vectors)

    issues: list[DiagnosticsIssue] = []
    norms: list[float] = []
    zero_fractions: list[float] = []
    per_layer: dict[str, dict[str, float]] = {}

    for layer, tensor in activations.to_dict().items():
        if tensor is None:
            if cfg.warn_on_missing:
                issues.append(
                    DiagnosticsIssue(
                        metric="control_vectors",
                        severity="warning",
                        message=f"Layer {layer} has no control vector",
                        details={"layer": layer},
                    )
                )
            continue

        detached = tensor.detach()
        if detached.numel() == 0:
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity="critical",
                    message=f"Layer {layer} control vector is empty",
                    details={"layer": layer},
                )
            )
            continue

        flat = detached.to(dtype=torch.float32, device="cpu").reshape(-1)

        if not torch.isfinite(flat).all():
            non_finite = (~torch.isfinite(flat)).sum().item()
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity="critical",
                    message=f"Layer {layer} contains non-finite values",
                    details={"layer": layer, "non_finite_entries": int(non_finite)},
                )
            )
            continue

        norm_value = float(torch.linalg.vector_norm(flat).item())
        norms.append(norm_value)

        zero_fraction = float((flat.abs() <= cfg.zero_value_threshold).sum().item()) / float(flat.numel())
        zero_fractions.append(zero_fraction)

        per_layer[layer] = {
            "norm": norm_value,
            "zero_fraction": zero_fraction,
        }

        if norm_value < cfg.min_norm:
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity="critical",
                    message=f"Layer {layer} control vector norm {norm_value:.3e} below minimum {cfg.min_norm}",
                    details={"layer": layer, "norm": norm_value},
                )
            )

        if cfg.max_norm is not None and norm_value > cfg.max_norm:
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity="warning",
                    message=f"Layer {layer} control vector norm {norm_value:.3e} exceeds maximum {cfg.max_norm}",
                    details={"layer": layer, "norm": norm_value},
                )
            )

        if zero_fraction >= cfg.max_zero_fraction:
            severity = "critical" if zero_fraction >= 1.0 - 1e-9 else "warning"
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity=severity,
                    message=(
                        f"Layer {layer} control vector is {zero_fraction:.3%} zero-valued, exceeding allowed {cfg.max_zero_fraction:.3%}"
                    ),
                    details={"layer": layer, "zero_fraction": zero_fraction},
                )
            )

    summary: dict[str, object] = {
        "evaluated_layers": len(norms),
        "norm_min": min(norms) if norms else None,
        "norm_max": max(norms) if norms else None,
        "norm_mean": statistics.mean(norms) if norms else None,
        "norm_median": statistics.median(norms) if norms else None,
        "zero_fraction_max": max(zero_fractions) if zero_fractions else None,
        "per_layer": per_layer,
    }

    if not norms and not issues:
        issues.append(
            DiagnosticsIssue(
                metric="control_vectors",
                severity="critical",
                message="No control vectors were provided for diagnostics",
                details={},
            )
        )

    report = MetricReport(name="control_vectors", summary=summary, issues=issues)
    return DiagnosticsReport.from_metrics([report])
