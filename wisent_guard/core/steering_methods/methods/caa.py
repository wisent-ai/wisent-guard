from __future__ import annotations

from typing import List
import torch

from wisent_guard.core.steering_methods.core.atoms import PerLayerBaseSteeringMethod

__all__ = [
    "CAAMethod",
]

class CAAMethod(PerLayerBaseSteeringMethod):
    """
    Contrastive Activation Additions (CAA).
    For each layer: v = mean(positives) - mean(negatives),
    optionally L2-normalized (kwargs: normalize=True, dtype=..., activation_aggregation_strategy=...).
    """
    name = "caa"
    description = "Per-layer mean(pos)-mean(neg) over ContrastivePairSet."

    def train_for_layer(self, pos_list: List[torch.Tensor], neg_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Train CAA vector for a single layer.

        arguments:
            pos_list: List of positive activations (torch.Tensor) for this layer.
            neg_list: List of negative activations (torch.Tensor) for this layer.
            
        returns:
            torch.Tensor steering vector for the layer.
        """
        if not pos_list or not neg_list:
            raise ValueError("Both positive and negative lists must be non-empty.")
        pos = torch.stack([t.detach().to("cpu").float().reshape(-1) for t in pos_list], dim=0)  # [N_pos, H]
        neg = torch.stack([t.detach().to("cpu").float().reshape(-1) for t in neg_list], dim=0)  # [N_neg, H]
        v = pos.mean(dim=0) - neg.mean(dim=0)
        if bool(self.kwargs.get("normalize", True)):
            v = self._safe_l2_normalize(v)
        return v

    def _safe_l2_normalize(self, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if v.ndim != 1:
            v = v.reshape(-1)
        return v / (torch.linalg.norm(v) + eps)