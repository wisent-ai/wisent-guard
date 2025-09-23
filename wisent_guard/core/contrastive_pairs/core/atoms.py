from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class AtomResponse(ABC):
    """Abstract base for a single model response."""
    model_response: str
    activations: torch.Tensor | None
    label: str | None


class AtomContrastivePair(ABC):
    """Abstract base for a (prompt, positive, negative) trio."""
    prompt: str
    positive_response: AtomResponse
    negative_response: AtomResponse
    label: str | None
    trait_description: str | None


class AtomContrastivePairSet(ABC):
    """Abstract base for a named collection of pairs."""
    name: str
    pairs: list[AtomContrastivePair]
    task_type: str | None

    @abstractmethod
    def add(self, pair: AtomContrastivePair) -> None: ...
    @abstractmethod
    def extend(self, pairs: Iterable[AtomContrastivePair]) -> None: ...

    def __len__(self) -> int:
        return len(self.pairs)