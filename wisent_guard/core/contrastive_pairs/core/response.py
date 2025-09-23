from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wisent_guard.core.contrastive_pairs.core.atoms import AtomResponse

@dataclass(frozen=True, slots=True)
class Response(AtomResponse):
    """A single model response with optional activations and label.

    Atributes:
        model_response: The text response from the model.
        activations: Optional tensor of model activations for this response.
        label: Optional label for the response, e.g., "harmless", "toxic", etc.
    """

    model_response: str
    activations: torch.Tensor | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_response, str) or not self.model_response.strip():
            raise ValueError("'model_response' must be a non-empty string.")
        object.__setattr__(self, "activations", self._to_tensor(self.activations))

    def with_activations(self, activations: torch.Tensor | np.ndarray | None) -> Response:
        """Return a copy of the Response with updated activations.

        Arguments:
            activations: New activations tensor or array.

        Returns:
            A new Response with updated activations.
        """
        return replace(self, activations=self._to_tensor(activations))

    def with_label(self, label: str | None) -> Response:
        """Return a copy of the Response with updated label.

        Arguments:
            label: New label for the response.

        Returns:
            A new Response with updated label.
        """
        return replace(self, label=label)

    def to_dict(self, include_label: bool = True) -> dict[str, torch.Tensor | str | None]:
        """Convert the Response to a dictionary.
        
        Arguments:
            include_label: Whether to include the 'label' field in the output (default: True).
            
        Returns:
            A dictionary with keys 'model_response', 'activations', and optionally 'label'.
        """
        data: dict[str, torch.Tensor | str | None] = {"model_response": self.model_response, "activations": self.activations}
        if include_label:
            data["label"] = self.label
        return data

    def _to_tensor(
        self,
        x: torch.Tensor | np.ndarray | None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        """Convert input to torch.Tensor with appropriate dtype.

        Arguments:
            x: Input data, either torch.Tensor, np.ndarray, or None.
            dtype: Desired torch dtype. If None, infer based on input type.

        Returns:
            A torch.Tensor with the specified or inferred dtype, or None if input is None.
        """
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x if dtype is None else x.to(dtype)
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
            if dtype is not None:
                return t.to(dtype)
            if t.is_floating_point():
                return t.to(torch.get_default_dtype())
            if t.is_complex():
                default_c = torch.complex64 if torch.get_default_dtype() == torch.float32 else torch.complex128
                return t.to(default_c)
            if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                return t.to(torch.get_default_dtype())
            return t
        raise TypeError(f"Activations must be torch.Tensor or np.ndarray, got {type(x)!r}.")

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor | str | None]) -> Response:
        """Create a Response from a dictionary.

        Arguments:
            data: Dictionary with keys 'model_response', 'activations', and optionally 'label'.

        Returns:
            A Response instance.
        """
        return cls(model_response=str(data["model_response"]), activations=data.get("activations"), label=data.get("label"))


class PositiveResponse(Response):
    """Marker subtype for harmless/correct responses."""
    pass


class NegativeResponse(Response):
    """Marker subtype for harmful/incorrect responses."""
    pass