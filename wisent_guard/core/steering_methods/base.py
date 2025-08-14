"""
Base class for steering methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from ..contrastive_pairs import ContrastivePairSet


class SteeringMethod(ABC):
    """
    Abstract base class for steering methods.

    All steering methods must implement the train, apply_steering, get_steering_vector,
    save_steering_vector, and load_steering_vector methods.
    """

    def __init__(self, name: str, device: Optional[str] = None):
        """
        Initialize the steering method.

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
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: Optional[int]) -> Dict[str, Any]:
        """
        Train the steering method on contrastive pairs.

        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied

        Returns:
            Dictionary with training statistics
        """

    @abstractmethod
    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Apply steering to the given activations.

        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier

        Returns:
            Steered activations
        """

    @abstractmethod
    def get_steering_vector(self) -> torch.Tensor:
        """
        Get the steering vector/parameters.

        Returns:
            Steering vector or parameters
        """

    @abstractmethod
    def save_steering_vector(self, path: str) -> bool:
        """
        Save steering data to file.

        Args:
            path: Path to save the steering data

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def load_steering_vector(self, path: str) -> bool:
        """
        Load steering data from file.

        Args:
            path: Path to load the steering data from

        Returns:
            True if successful, False otherwise
        """

    def __str__(self) -> str:
        """String representation of the steering method."""
        return f"{self.name}SteeringMethod(device={self.device}, trained={self.is_trained})"

    def __repr__(self) -> str:
        """Detailed representation of the steering method."""
        return self.__str__()
