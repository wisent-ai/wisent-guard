from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from wisent_guard.core.activations.core.atoms import RawActivationMap 


__all__ = [
    "SteeringVector",
    "SteeringPlan",
    "HookHandleGroup",
    "TopLogits",
    "GenerationStats",
]


@dataclass(slots=True)
class SteeringVector:
    """
    Single steering vector added to a layer's residual stream (output).

    attributes:
        vector:
            tensor whose last dim == hidden_size. Shape may be [H], [1, H], [1, 1, H] or [B, T, H].
        scale:
            scalar coefficient (alpha) multiplied before adding.
        normalize:
            L2-normalize the vector (safe + epsilon).
    
    example:
        >>> sv = SteeringVector(torch.randn(4096), scale=0.8, normalize=True)
    """
    vector: torch.Tensor
    scale: float = 1.0
    normalize: bool = False

    def materialize(self, like: torch.Tensor) -> torch.Tensor:
        """
        #1 Broadcast and cast the steering vector so it's addable to `like` ([B, T, H]).
        #2 Returns a tensor on like.device and like.dtype.
        """
        v = self.vector
        if self.normalize and torch.is_floating_point(v):
            denom = torch.linalg.vector_norm(v.float(), dim=-1, keepdim=True).clamp_min(1e-12)
            v = v / denom

        if v.dim() == 1:
            v = v.view(1, 1, -1)
        elif v.dim() == 2:
            v = v.view(1, *v.shape)
        elif v.dim() == 3:
            pass
        else:
            raise ValueError(f"Unsupported steering vector shape {tuple(v.shape)}; expected [H], [1,H], [1,1,H], or [B,T,H].")

        v = v.to(dtype=like.dtype, device=like.device)
        return v * float(self.scale)


@dataclass(slots=True)
class SteeringPlan:
    """
    Mapping: layer_name -> list of SteeringVector(s). Multiple vectors per layer are summed.
    Build with 'from_raw' to convert {layer: tensor} into a plan quickly.

    attributes:
        layers:
            dict mapping layer names (str) to list of SteeringVector(s).

    example:
        >>> raw = {"3": torch.randn(4096), "7": torch.randn(4096)}
        >>> plan = SteeringPlan.from_raw(raw, scale=1.2, normalize=True)
        >>> plan.layers["3"][0].scale
        1.2
    """
    layers: dict[str, list[SteeringVector]] = field(default_factory=dict)

    @staticmethod
    def from_raw(raw: RawActivationMap, *, scale: float = 1.0, normalize: bool = False) -> SteeringPlan:
        """
        Convert a raw dict of {layer_name: tensor} into a SteeringPlan.

        notes:
        In practice you often want to combine directions (e.g., “politeness” + “conciseness”) or run A/B vectors for the same layer.
        Activation-steering work commonly adds/combines multiple vectors linearly during the forward pass; keeping a list makes that
        trivial and avoids stacking multiple hooks on the same module.

        example of combining vectors:
            >>> plan = SteeringPlan.from_raw({"6": v6, "12": v12}, scale=0.8)
            # Later, decide layer 12 should combine two vectors (with different coeffs):
            >>> plan.layers["12"].append(SteeringVector(v12_extra, scale=0.4, normalize=True))
            # Or construct directly with multiple vectors per layer:
            >>> plan = SteeringPlan({
            ...     "6": [SteeringVector(v6_a, scale=0.7), SteeringVector(v6_b, scale=0.3)],
            ...     "12": [SteeringVector(v12, scale=1.0)],
            ... })

        arguments:
            raw:
                dict mapping layer names (str) to tensors (or None to skip).
            scale:
                scalar coefficient (alpha) for all vectors.
            normalize:
                L2-normalize all vectors (safe + epsilon).
        
        returns:
            SteeringPlan instance.
        """
        out: dict[str, list[SteeringVector]] = {}
        for k, v in (raw or {}).items():
            if v is None:
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            out[str(k)] = [SteeringVector(v, scale=scale, normalize=normalize)]
        return SteeringPlan(out)

    def validate_hidden_size(self, hidden_size: int) -> None:
        """
        Check that all vectors have last dim == hidden_size.
        Accepts [H], [1,H], [1,1,H] or [B,T,H]; we only check the last dimension.

        arguments:
            hidden_size:
                expected hidden size (last dim of steering vectors).
        
        raises:
            ValueError if any vector has a mismatched last dimension.
        """
        for k, vecs in self.layers.items():
            for sv in vecs:
                if sv.vector.shape[-1] != hidden_size:
                    raise ValueError(f"Layer {k} steering last dim {sv.vector.shape[-1]} != hidden_size {hidden_size}")

    def is_empty(self) -> bool:
        "True if no non-empty layer entry exists."
        return not any(self.layers.values())


class HookHandleGroup:
    """
    Manage a set of torch hooks to ensure clean detach.
    """
    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def add(self, handle: torch.utils.hooks.RemovableHandle) -> None:
        self._handles.append(handle)

    def remove_all(self) -> None:
        while self._handles:
            h = self._handles.pop()
            try:
                h.remove()
            except Exception:
                pass


@dataclass(slots=True)
class TopLogits:
    """
    Info for a generated step.

    attributes:
        token_id: 
            chosen token id at this step.
        logit: 
            raw logit for that token.
        prob: 
            softmax probability for that token.
        topk_ids/topk_probs:
            optional top-k for analysis/visualization.
    """
    token_id: int
    logit: float
    prob: float
    topk_ids: list[int] | None = None
    topk_probs: list[float] | None = None


@dataclass(slots=True)
class GenerationStats:
    """
    Per-sequence stats for a generation call.

    attributes:
        tokens:
            the generated token ids (excluding the prompt).
        per_step: 
            optional list of TopLogits, one per generated step.
    """
    tokens: list[int]
    per_step: list[TopLogits] | None = None
