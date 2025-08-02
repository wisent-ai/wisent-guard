"""
Contrastive Activation Addition (CAA) steering method.
"""

import logging
from typing import Any, Dict, Optional

import torch

from ..aggregation import ControlVectorAggregationMethod, create_control_vector_from_contrastive_pairs
from ..contrastive_pairs import ContrastivePairSet
from ..normalization import VectorNormalizationMethod, VectorNormalizer
from .base import SteeringMethod

logger = logging.getLogger(__name__)


class CAA(SteeringMethod):
    """
    Contrastive Activation Addition (CAA) steering method.

    Uses simple activation differences between positive and negative examples
    to create steering vectors. Supports various normalization strategies.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        aggregation_method: ControlVectorAggregationMethod = ControlVectorAggregationMethod.CAA,
        normalization_method: str = "none",
        target_norm: Optional[float] = None,
    ):
        super().__init__("CAA", device)
        self.steering_vector = None
        self.layer_index = None
        self.aggregation_method = aggregation_method
        self.normalization_method = normalization_method
        self.target_norm = target_norm
        self.training_stats = {}
        self.normalizer = VectorNormalizer()  # Always initialize for potential multi-behavior training

    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train CAA by computing activation differences.

        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index where steering will be applied

        Returns:
            Dictionary with training statistics
        """
        self.layer_index = layer_index

        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()

        # Create control vector using aggregation method
        self.steering_vector, training_stats = create_control_vector_from_contrastive_pairs(
            pos_activations, neg_activations, self.aggregation_method, self.device
        )

        # Apply normalization if specified
        if self.normalizer and self.normalization_method != "none":
            if self.normalization_method == "cross_behavior":
                # For CAA, we treat each vector as a separate behavior
                vectors = [self.steering_vector]
                normalized_vectors = self.normalizer.normalize_cross_behavior(vectors, target_norm=self.target_norm)
                self.steering_vector = normalized_vectors[0]
            elif self.normalization_method == "l2_unit":
                self.steering_vector = self.normalizer.normalize_l2_unit(self.steering_vector)
            elif self.normalization_method == "layer_wise_mean":
                # For single layer, this is equivalent to l2_unit
                vectors = {layer_index: [self.steering_vector]}
                normalized_vectors = self.normalizer.normalize_layer_wise_mean(vectors)
                self.steering_vector = normalized_vectors[layer_index][0]

        # Update training statistics
        self.training_stats = training_stats.copy()
        self.training_stats.update(
            {
                "method": "CAA",
                "aggregation_method": self.aggregation_method.value,
                "normalization_method": self.normalization_method,
                "target_norm": self.target_norm,
                "layer_index": layer_index,
                "final_vector_norm": torch.norm(self.steering_vector).item(),
            }
        )

        self.is_trained = True
        return self.training_stats

    def train_multi_behavior(
        self, behavior_pairs: Dict[str, ContrastivePairSet], layer_index: int, normalize_across_behaviors: bool = True
    ) -> Dict[str, Any]:
        """
        Train CAA on multiple behaviors and optionally normalize across them.

        This implements the reference CAA approach where multiple behaviors are
        trained together and normalized to have the same magnitude.

        Args:
            behavior_pairs: Dictionary mapping behavior names to ContrastivePairSets
            layer_index: Layer index where steering will be applied
            normalize_across_behaviors: Whether to apply cross-behavior normalization

        Returns:
            Dictionary with training statistics for all behaviors
        """
        self.layer_index = layer_index
        self.behavior_vectors = {}
        all_stats = {}

        # First, train individual vectors for each behavior
        for behavior_name, pair_set in behavior_pairs.items():
            # Get positive and negative activations
            pos_activations, neg_activations = pair_set.get_activation_pairs()

            # Create control vector using aggregation method
            vector, stats = create_control_vector_from_contrastive_pairs(
                pos_activations, neg_activations, self.aggregation_method, self.device
            )

            self.behavior_vectors[behavior_name] = vector
            all_stats[behavior_name] = stats

        # Apply cross-behavior normalization if requested
        if normalize_across_behaviors and self.normalizer:
            normalized_vectors, norm_stats = self.normalizer.normalize_cross_behavior(
                self.behavior_vectors, method=VectorNormalizationMethod.CROSS_BEHAVIOR
            )
            self.behavior_vectors = normalized_vectors

            # Update stats with normalization info
            for behavior_name in self.behavior_vectors:
                all_stats[behavior_name]["normalization"] = norm_stats
                all_stats[behavior_name]["final_norm"] = torch.norm(self.behavior_vectors[behavior_name]).item()

        # Store the first behavior as the default steering vector for compatibility
        if self.behavior_vectors:
            first_behavior = list(self.behavior_vectors.keys())[0]
            self.steering_vector = self.behavior_vectors[first_behavior]

        self.training_stats = {
            "method": "CAA",
            "multi_behavior": True,
            "behaviors": list(self.behavior_vectors.keys()),
            "layer_index": layer_index,
            "normalized_across_behaviors": normalize_across_behaviors,
            "behavior_stats": all_stats,
        }

        self.is_trained = True
        return self.training_stats

    def apply_steering(
        self,
        activations: torch.Tensor,
        strength: float = 1.0,
        verbose: bool = False,
        behavior_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply CAA steering using additive intervention.

        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier
            verbose: Enable debug logging
            behavior_name: Specific behavior to apply (for multi-behavior steering)

        Returns:
            Steered activations
        """
        if not self.is_trained:
            raise ValueError("CAA method must be trained before applying steering")

        # Select which vector to use
        if behavior_name and hasattr(self, "behavior_vectors") and behavior_name in self.behavior_vectors:
            steering_vector = self.behavior_vectors[behavior_name].to(activations.device)
        else:
            steering_vector = self.steering_vector.to(activations.device)

        if verbose:
            logger.debug("CAA apply_steering called:")
            logger.debug(f"   Input shape: {activations.shape}")
            logger.debug(f"   Strength: {strength}")
            logger.debug(f"   Vector norm: {torch.norm(steering_vector).item():.4f}")
            logger.debug(f"   Input norm (mean): {torch.norm(activations, dim=-1).mean().item():.4f}")

        # Handle different activation shapes
        if len(activations.shape) == 3:  # [batch, seq, hidden]
            # Apply to second-to-last token position (reference behavior)
            steered = activations.clone()
            if activations.shape[1] > 1:
                # Use second-to-last token if sequence has more than 1 token
                before_norm = torch.norm(steered[:, -2:-1, :], dim=-1).mean().item()

                # LOG THE ACTUAL STEERING BEING APPLIED
                logger.debug("   🎯 CAA APPLYING STEERING:")
                logger.debug(f"      Strength parameter: {strength}")
                logger.debug(f"      Vector norm: {torch.norm(steering_vector).item():.4f}")
                logger.debug(f"      Effective addition norm: {torch.norm(strength * steering_vector).item():.4f}")

                steered[:, -2:-1, :] = steered[:, -2:-1, :] + strength * steering_vector.unsqueeze(0).unsqueeze(0)
                after_norm = torch.norm(steered[:, -2:-1, :], dim=-1).mean().item()

                logger.debug(f"      Position -2 norm before: {before_norm:.4f}")
                logger.debug(f"      Position -2 norm after: {after_norm:.4f}")
                logger.debug(f"      Change: {after_norm - before_norm:.4f}")

                # Check if we created inf/nan
                if torch.any(torch.isinf(steered)) or torch.any(torch.isnan(steered)):
                    logger.debug("      ⚠️ CRITICAL: Steering created inf/nan values!")
                    logger.debug(f"      Max in steered: {steered.max().item()}")
                    logger.debug(f"      Min in steered: {steered.min().item()}")
            else:
                # Fallback to last token for single-token sequences
                steered[:, -1:, :] = steered[:, -1:, :] + strength * steering_vector.unsqueeze(0).unsqueeze(0)
        elif len(activations.shape) == 2:  # [batch, hidden]
            steered = activations + strength * steering_vector.unsqueeze(0)
        else:
            steered = activations + strength * steering_vector

        return steered

    def get_steering_vector(self) -> torch.Tensor:
        """Return the CAA steering vector."""
        if not self.is_trained:
            raise ValueError("CAA method must be trained before getting steering vector")
        return self.steering_vector

    def save_steering_vector(self, path: str) -> bool:
        """Save CAA steering data."""
        if not self.is_trained:
            return False
        try:
            save_data = {
                "steering_vector": self.steering_vector,
                "aggregation_method": self.aggregation_method.value,
                "normalization_method": self.normalization_method,
                "target_norm": self.target_norm,
                "layer_index": self.layer_index,
                "training_stats": self.training_stats,
                "method": "CAA",
            }

            # Save multi-behavior data if available
            if hasattr(self, "behavior_vectors"):
                save_data["behavior_vectors"] = self.behavior_vectors
                save_data["multi_behavior"] = True

            torch.save(save_data, path)
            return True
        except Exception:
            return False

    def load_steering_vector(self, path: str) -> bool:
        """Load CAA steering data."""
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
            if data.get("method") != "CAA":
                return False
            self.steering_vector = data["steering_vector"]
            self.aggregation_method = ControlVectorAggregationMethod(data.get("aggregation_method", "CAA"))
            self.normalization_method = data.get("normalization_method", "none")
            self.target_norm = data.get("target_norm")
            self.layer_index = data.get("layer_index")
            self.training_stats = data.get("training_stats", {})

            # Load multi-behavior data if available
            if data.get("multi_behavior", False) and "behavior_vectors" in data:
                self.behavior_vectors = data["behavior_vectors"]

            self.is_trained = True
            return True
        except Exception:
            return False
