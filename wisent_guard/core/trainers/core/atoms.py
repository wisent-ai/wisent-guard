from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from wisent_guard.core.activations.core.atoms import LayerActivations
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet

__all__ = [
    "TrainingResult",
]

@dataclass(slots=True)
class TrainingResult:
    """
    Container returned by a trainer after running the full pipeline.

    attributes:
        steered_vectors:
            Per-layer steering vectors in a LayerActivations mapping. Each value
            is typically a 1D tensor of shape [H].
        pair_set_with_activations:
            The original ContrastivePairSet, but with per-pair, per-layer activations
            collected and stored in the Positive/NegativeResponse objects.
        metadata:
            A JSON-serializable dictionary with run metadata
            (date, model_name, layers, method, hyperparams, aggregation, etc.).
    """
    steered_vectors: LayerActivations
    pair_set_with_activations: ContrastivePairSet
    metadata: Dict[str, Any]

class BaseSteeringTrainer(ABC):
    """
    Abstract interface for a trainer that orchestrates:
        1) Collecting activations for a set of contrastive pairs
        2) Training a steering vector(s) using a chosen method
        3) Returning a TrainingResult and (optionally) saving artifacts
    """

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> TrainingResult:
        """
        Execute the full pipeline and return a TrainingResult.
        """
        ...