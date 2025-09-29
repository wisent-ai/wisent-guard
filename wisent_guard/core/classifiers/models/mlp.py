from __future__ import annotations

import torch
from torch import nn

from wisent_guard.core.classifiers.core.atoms import BaseClassifier

__all__ = ["MLPClassifier"]

class MLPModel(nn.Module):
    """Multi-layer perceptron for activation classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out


class MLPClassifier(BaseClassifier):
    name = "mlp"
    description = "Two-layer MLP with dropout and ReLU"

    def __init__(self, *, hidden_dim: int = 128, **base_kwargs):
        super().__init__(**base_kwargs)
        self._hidden_dim = int(hidden_dim)

    def build_model(self, input_dim: int, **model_params: object) -> nn.Module:
        hd = int(model_params.get("hidden_dim", self._hidden_dim))  # type: ignore[arg-type]
        self._hidden_dim = hd
        return MLPModel(input_dim, hidden_dim=hd)

    def model_hyperparams(self) -> dict[str, int]:
        return {"hidden_dim": self._hidden_dim}
