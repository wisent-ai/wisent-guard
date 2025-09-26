"""Minimal container class for contrastive pairs with light orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from typing import TYPE_CHECKING

from wisent_guard.core.contrastive_pairs.core.atoms import AtomContrastivePairSet

if TYPE_CHECKING:
    from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair 


@dataclass
class ContrastivePairSet(AtomContrastivePairSet):
    """
    A named set of contrastive pairs, with optional task type.

    Attributes:
        name: The name of the contrastive pair set.
        pairs: The list of contrastive pairs in the set.
        task_type: The optional task type associated with the pair set.
    """
    name: str
    pairs: list[ContrastivePair] = field(default_factory=list)
    task_type: Optional[str] = None

    def add(self, pair: ContrastivePair) -> None:
        """Append a pair with an assert for correctness.

        Arguments:
            pair: The ContrastivePair to add.
        
        Raises:
            AssertionError: If the provided pair is not an instance of ContrastivePair.  
        """
        assert isinstance(pair, ContrastivePair), "pair must be a ContrastivePair"
        self.pairs.append(pair)

    def extend(self, pairs: list[ContrastivePair]) -> None:
        """Extend with multiple pairs.
        
        Arguments:
            pairs: A list of ContrastivePair instances to add.
        """
        for p in pairs:
            self.add(p)

    def __len__(self) -> int:
        return len(self.pairs)

    def __repr__(self) -> str:
        return f"ContrastivePairSet(name={self.name!r}, pairs={len(self.pairs)}, task_type={self.task_type!r})"

    def statistics(self) -> dict[str, str | int | None]:
        """Return simple statistics about this set.

        Returns:
            A dictionary with statistics about the pair set.
        """
        pos = sum(1 for p in self.pairs if getattr(p.positive_response, "layers_activations", None) is not None)
        neg = sum(1 for p in self.pairs if getattr(p.negative_response, "layers_activations", None) is not None)
        both = sum(
            1
            for p in self.pairs
            if getattr(p.positive_response, "layers_activations", None) is not None
            and getattr(p.negative_response, "activations", None) is not None
        )

        assert pos == neg, "Number of positive and negative layers_activations should be equal."

        return {
            "name": self.name,
            "total_pairs": len(self.pairs),
            "pairs_with_positive_activations": pos,
            "pairs_with_negative_activations": neg,
            "pairs_with_both_activations": both,
            "task_type": self.task_type,
            "example_pair": repr(self.pairs[0]) if self.pairs else None,
        }