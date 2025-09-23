from __future__ import annotations

from dataclasses import dataclass, replace 

import numpy as np
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wisent_guard.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse, Response
    from wisent_guard.core.contrastive_pairs.core.atoms import AtomContrastivePair


@dataclass(frozen=True, slots=True)
class ContrastivePair(AtomContrastivePair):
    """A single contrastive pair: (prompt, positive_response, negative_response).
    
    Attributes:
        prompt: The input prompt string. For example, a question or instruction.
        positive_response: The response considered "harmless" or "correct".
        negative_response: The response considered "harmful" or "incorrect".
        label: Optional label for the pair, e.g., "toxic", "biased", etc.
        trait_description: Optional description of the trait being tested. For example, "hallucinatory", "toxic", "biased", etc.
    """

    prompt: str
    positive_response: PositiveResponse
    negative_response: NegativeResponse
    label: str | None = None
    trait_description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("'prompt' must be a non-empty string.")
        if not isinstance(self.positive_response, PositiveResponse):
            raise TypeError("`positive_response` must be PositiveResponse.")
        if not isinstance(self.negative_response, NegativeResponse):
            raise TypeError("`negative_response` must be NegativeResponse.")

    def with_activations(
        self,
        positive: torch.Tensor | np.ndarray | None,
        negative: torch.Tensor | np.ndarray | None,
    ) -> ContrastivePair:
        """Return a copy of the ContrastivePair with updated activations.

        Arguments:
            positive: New activations for the positive response.
            negative: New activations for the negative response.
        
        Returns:
            A new ContrastivePair with updated activations.

        For example:
            new_pair = pair.with_activations(positive=new_pos_acts, negative=new_neg_acts)
        """
        new_pos = self.positive_response if positive is None else self.positive_response.with_activations(positive)
        new_neg = self.negative_response if negative is None else self.negative_response.with_activations(negative)
        return replace(self, positive_response=new_pos, negative_response=new_neg)

    def to_dict(self, include_metadata: bool = True) -> dict[str, Response]:
        data: dict[str, Response] = {
            "prompt": self.prompt,
            "positive_response": self.positive_response.to_dict(include_label=True),
            "negative_response": self.negative_response.to_dict(include_label=True),
        }
        if include_metadata:
            data.update({"label": self.label, "trait_description": self.trait_description})
        return data

    @classmethod
    def from_dict(cls, data: dict[str, str | torch.Tensor | None]) -> ContrastivePair:

        from wisent_guard.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse

        return cls(
            prompt=str(data["prompt"]),
            positive_response=PositiveResponse.from_dict(text=data.get("positive_response"), activations=data.get("positive_activation"), label=data.get("label")),
            negative_response=NegativeResponse.from_dict(text=data.get("negative_response"), activations=data.get("negative_activation"), label=data.get("label")),
            label=data.get("label"),
            trait_description=data.get("trait_description"),
        )