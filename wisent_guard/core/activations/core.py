from enum import Enum
from typing import Any, Dict

import torch
import torch.nn.functional as F

from wisent_guard.core.layer import Layer


class ActivationAggregationStrategy(Enum):
    """Different strategies for targeting tokens in activation extraction."""

    CHOICE_TOKEN = "choice_token"  # Target A/B choice tokens (for multiple choice)
    CONTINUATION_TOKEN = "continuation_token"  # Target first token of continuation ("I", etc.)
    LAST_TOKEN = "last_token"  # Always use last token
    FIRST_TOKEN = "first_token"  # Always use first token
    MEAN_POOLING = "mean_pooling"  # Use mean of all tokens
    MAX_POOLING = "max_pooling"  # Use max pooling across tokens


class Activations:
    def __init__(
        self,
        tensor,
        layer,
        aggregation_strategy: ActivationAggregationStrategy = ActivationAggregationStrategy.LAST_TOKEN,
    ):
        self.tensor = tensor
        self.layer = layer
        self.aggregation_method = aggregation_strategy

    def get_aggregated(self):
        """
        Get aggregated activations based on the aggregation method.

        Returns:
            torch.Tensor: Aggregated activation tensor
        """
        if self.aggregation_method == ActivationAggregationStrategy.LAST_TOKEN:
            # Return the last token's activations
            if len(self.tensor.shape) >= 3:  # [batch, seq_len, hidden_dim]
                return self.tensor[0, -1, :]  # Take last token of first batch
            if len(self.tensor.shape) == 2:  # [seq_len, hidden_dim]
                return self.tensor[-1, :]  # Take last token
            return self.tensor  # Already aggregated

        if self.aggregation_method == ActivationAggregationStrategy.MEAN_POOLING:
            # Return mean across sequence dimension
            if len(self.tensor.shape) >= 3:  # [batch, seq_len, hidden_dim]
                return self.tensor[0].mean(dim=0)  # Mean across sequence length
            if len(self.tensor.shape) == 2:  # [seq_len, hidden_dim]
                return self.tensor.mean(dim=0)  # Mean across sequence length
            return self.tensor

        if self.aggregation_method == ActivationAggregationStrategy.MAX_POOLING:
            # Return max across sequence dimension
            if len(self.tensor.shape) >= 3:  # [batch, seq_len, hidden_dim]
                return self.tensor[0].max(dim=0)[0]  # Max across sequence length
            if len(self.tensor.shape) == 2:  # [seq_len, hidden_dim]
                return self.tensor.max(dim=0)[0]  # Max across sequence length
            return self.tensor
        # Default to last token
        if len(self.tensor.shape) >= 3:  # [batch, seq_len, hidden_dim]
            return self.tensor[0, -1, :]  # Take last token of first batch
        if len(self.tensor.shape) == 2:  # [seq_len, hidden_dim]
            return self.tensor[-1, :]  # Take last token
        return self.tensor  # Already aggregated

    def calculate_similarity(self, other_tensor: torch.Tensor, method: str = "cosine") -> float:
        """
        Calculate similarity between this activation and another tensor.

        Args:
            other_tensor: Tensor to compare against (e.g., contrastive vector)
            method: Similarity method ("cosine", "dot", "euclidean")

        Returns:
            Similarity score
        """
        # Get aggregated activation
        activation = self.get_aggregated()

        # Ensure tensors are on the same device and have compatible shapes
        if activation.device != other_tensor.device:
            other_tensor = other_tensor.to(activation.device)

        # Flatten tensors if needed
        if len(activation.shape) > 1:
            activation = activation.flatten()
        if len(other_tensor.shape) > 1:
            other_tensor = other_tensor.flatten()

        # Handle dimension mismatch
        if activation.shape[0] != other_tensor.shape[0]:
            min_dim = min(activation.shape[0], other_tensor.shape[0])
            activation = activation[:min_dim]
            other_tensor = other_tensor[:min_dim]

        try:
            if method == "cosine":
                # Cosine similarity
                cos_sim = F.cosine_similarity(activation.unsqueeze(0), other_tensor.unsqueeze(0), dim=1)
                # Convert to similarity score (0 to 1)
                similarity = (cos_sim.item() + 1.0) / 2.0
                return max(0.0, min(1.0, similarity))

            if method == "dot":
                # Dot product similarity
                dot_product = torch.dot(activation, other_tensor)
                return dot_product.item()

            if method == "euclidean":
                # Negative euclidean distance (higher = more similar)
                distance = torch.norm(activation - other_tensor)
                return -distance.item()

            raise ValueError(f"Unknown similarity method: {method}")

        except Exception:
            # Return 0 similarity on error
            return 0.0

    def compare_with_vectors(self, vector_dict: Dict[str, torch.Tensor], threshold: float = 0.7) -> Dict[str, Any]:
        """
        Compare this activation with multiple contrastive vectors.

        Args:
            vector_dict: Dictionary mapping category names to contrastive vectors
            threshold: Threshold for determining harmful content

        Returns:
            Dictionary with comparison results for each category
        """
        results = {}

        for category, vector in vector_dict.items():
            similarity = self.calculate_similarity(vector)
            is_harmful = similarity >= threshold

            results[category] = {"similarity": similarity, "is_harmful": is_harmful, "threshold": threshold}

        return results

    def extract_features_for_classifier(self) -> torch.Tensor:
        """
        Extract features suitable for classifier input.

        Returns:
            Flattened tensor ready for classification
        """
        features = self.get_aggregated()

        # Ensure it's a PyTorch tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        # Flatten if needed
        if len(features.shape) > 1:
            features = features.flatten()

        return features

    def to_device(self, device: str) -> "Activations":
        """
        Move activations to a specific device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu', 'mps')

        Returns:
            New Activations object on the target device
        """
        new_tensor = self.tensor.to(device)
        return Activations(tensor=new_tensor, layer=self.layer, aggregation_strategy=self.aggregation_method)

    def normalize(self) -> "Activations":
        """
        Normalize the activation tensor.

        Returns:
            New Activations object with normalized tensor
        """
        aggregated = self.get_aggregated()

        # L2 normalization
        norm = torch.norm(aggregated, p=2, dim=-1, keepdim=True)
        normalized = aggregated / (norm + 1e-8)  # Add small epsilon to avoid division by zero

        return Activations(tensor=normalized, layer=self.layer, aggregation_strategy=self.aggregation_method)

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical information about the activations.

        Returns:
            Dictionary with statistics
        """
        aggregated = self.get_aggregated()

        return {
            "mean": float(aggregated.mean()),
            "std": float(aggregated.std()),
            "min": float(aggregated.min()),
            "max": float(aggregated.max()),
            "norm": float(torch.norm(aggregated)),
            "shape": list(aggregated.shape),
            "device": str(aggregated.device),
            "dtype": str(aggregated.dtype),
        }

    @classmethod
    def from_model_output(cls, model_outputs, layer: Layer, aggregation_method=None) -> "Activations":
        """
        Create Activations object from model forward pass outputs.

        Args:
            model_outputs: Output from model forward pass
            layer: Layer object specifying which layer to extract
            aggregation_method: How to aggregate the activations

        Returns:
            Activations object
        """
        if hasattr(model_outputs, "hidden_states") and model_outputs.hidden_states is not None:
            hidden_states = model_outputs.hidden_states

            # Get the hidden state for the specified layer (add 1 because hidden_states[0] is embeddings)
            if layer.index + 1 < len(hidden_states):
                layer_hidden_state = hidden_states[layer.index + 1]

                return cls(tensor=layer_hidden_state, layer=layer, aggregation_strategy=aggregation_method)
            raise ValueError(f"Layer {layer.index} not found in model with {len(hidden_states)} layers")
        raise ValueError("Model outputs don't contain hidden_states")

    @classmethod
    def from_tensor_dict(
        cls, tensor_dict: Dict[str, torch.Tensor], layer: Layer, aggregation_method=None
    ) -> "Activations":
        """
        Create Activations object from a dictionary of tensors (legacy compatibility).

        Args:
            tensor_dict: Dictionary containing activation tensors
            layer: Layer object
            aggregation_method: How to aggregate the activations

        Returns:
            Activations object
        """
        # Try to find the activation tensor in the dictionary
        if "activations" in tensor_dict:
            tensor = tensor_dict["activations"]
        elif str(layer.index) in tensor_dict:
            tensor = tensor_dict[str(layer.index)]
        elif layer.index in tensor_dict:
            tensor = tensor_dict[layer.index]
        else:
            raise ValueError(f"No activation tensor found for layer {layer.index} in dictionary")

        return cls(tensor=tensor, layer=layer, aggregation_strategy=aggregation_method)

    def __repr__(self) -> str:
        """String representation of the Activations object."""
        aggregated = self.get_aggregated()
        return f"Activations(layer={self.layer.index}, shape={list(aggregated.shape)}, method={self.aggregation_method.value})"

    # Monitoring functionality
    def cosine_similarity(self, other: "Activations") -> float:
        """Calculate cosine similarity with another activation."""
        return self.calculate_similarity(other.get_aggregated(), method="cosine")

    def dot_product_similarity(self, other: "Activations") -> float:
        """Calculate dot product similarity with another activation."""
        return self.calculate_similarity(other.get_aggregated(), method="dot")

    def euclidean_distance(self, other: "Activations") -> float:
        """Calculate euclidean distance with another activation."""
        return -self.calculate_similarity(other.get_aggregated(), method="euclidean")
