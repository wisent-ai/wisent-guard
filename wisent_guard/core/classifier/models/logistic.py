import torch
from torch import nn as nn


class LogisticModel(nn.Module):
    """Simple logistic regression model for activation classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure output has proper dimensions by keeping the batch dimension
        logits = self.linear(x)
        # Keep output shape consistent regardless of batch size
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(1)
        return self.sigmoid(logits)
