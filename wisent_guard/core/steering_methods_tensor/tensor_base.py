"""
Base class for tensor-based steering methods.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class SteeringMethodTensor(ABC):
    """
    Abstract base class for tensor-based steering methods (like DAC).

    Unlike SteeringMethod which works with single vectors per layer,
    SteeringMethodTensor works with multi-dimensional tensors spanning
    multiple layers/heads/timesteps.

    This is designed for methods like DAC that operate on tensors of shape
    [steps, n_layers, n_heads, d_head] rather than single vectors.
    """

    def __init__(self, name: str, device: Optional[str] = None):
        """
        Initialize the tensor-based steering method.

        Args:
            name: Name of the steering method
            device: Device to use for computations (None for auto-detection)
        """
        self.name = name
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.is_trained = False

    @abstractmethod
    def get_steering_tensor(self) -> torch.Tensor:
        """
        Get the steering tensor (multi-dimensional).

        For DAC, this typically returns a tensor of shape:
        [steps, n_layers, n_heads, d_head]

        Returns:
            Steering tensor
        """

    @abstractmethod
    def apply_steering_tensor(self, activations: torch.Tensor, strength: float = 1.0, **kwargs) -> torch.Tensor:
        """
        Apply tensor-based steering to the given activations.

        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier
            **kwargs: Method-specific parameters (e.g., layer_index, step_index)

        Returns:
            Steered activations
        """

    @abstractmethod
    def save_steering_tensor(self, path: str) -> bool:
        """
        Save steering tensor data to file.

        Args:
            path: Path to save the steering data

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def load_steering_tensor(self, path: str) -> bool:
        """
        Load steering tensor data from file.

        Args:
            path: Path to load the steering data from

        Returns:
            True if successful, False otherwise
        """

    def __repr__(self) -> str:
        """String representation of the tensor-based steering method."""
        return f"{self.name}TensorSteering(device='{self.device}', is_trained={self.is_trained})"
