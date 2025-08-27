import torch
from torch import nn as nn


class MLPModel(nn.Module):
    """Multi-layer perceptron model for activation classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
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
        logits = self.network(x)
        # Keep output shape consistent regardless of batch size
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(1)
        return logits
