"""Public interface for contrastive pair utilities."""

from .core.pair import ContrastivePair
from .core.set import ContrastivePairSet
from .core.buliders import from_phrase_pairs
from .diagnostics import DiagnosticsConfig, DiagnosticsReport, run_all_diagnostics

__all__ = [
    "ContrastivePair",
    "ContrastivePairSet",
    "from_phrase_pairs",
    "DiagnosticsConfig",
    "DiagnosticsReport",
    "run_all_diagnostics",
]