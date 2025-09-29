from __future__ import annotations

import torch
from torch import nn

from wisent_guard.core.classifiers.core.atoms import BaseClassifier

__all__ = ["LogisticClassifier"]

class LogisticModel(nn.Module):
    """Simple logistic regression model for activation classification."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        return self.sigmoid(logits)


class LogisticClassifier(BaseClassifier):
    name = "logistic"
    description = "One-layer logistic regression over dense features"

    def build_model(self, input_dim: int, **_: object) -> nn.Module:
        return LogisticModel(input_dim)
