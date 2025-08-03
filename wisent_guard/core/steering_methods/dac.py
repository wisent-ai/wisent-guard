"""
Dynamic Activation Composition (DAC) steering method.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..aggregation import ControlVectorAggregationMethod, create_control_vector_from_contrastive_pairs
from ..contrastive_pairs import ContrastivePairSet
from .base import SteeringMethod

logger = logging.getLogger(__name__)


@dataclass
class PropertyVector:
    """Container for a single property's steering vector and metadata."""

    name: str
    vector: torch.Tensor
    layer_index: int
    training_stats: Dict[str, Any]
    aggregation_method: ControlVectorAggregationMethod


class DAC(SteeringMethod):
    """
    Dynamic Activation Composition (DAC) steering method.

    Uses information-theoretic principles to dynamically modulate steering
    intensity throughout generation for multi-property control.

    Supports both single-property (backward compatible) and multi-property steering.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        dynamic_control: bool = True,
        entropy_threshold: float = 1.0,
        aggregation_method: ControlVectorAggregationMethod = ControlVectorAggregationMethod.CAA,
        ptop: float = 0.4,  # For KL-based dynamic control
        max_alpha: float = 2.0,  # Maximum steering intensity
    ):
        super().__init__("DAC", device)
        # Single-property compatibility
        self.steering_vector = None
        self.layer_index = None
        self.dynamic_control = dynamic_control
        self.entropy_threshold = entropy_threshold
        self.aggregation_method = aggregation_method
        self.training_stats = {}

        # Multi-property support
        self.property_vectors: Dict[str, PropertyVector] = {}
        self.ptop = ptop
        self.max_alpha = max_alpha
        self._model_ref = None  # For KL computation

    def train_property(
        self, property_name: str, contrastive_pair_set: ContrastivePairSet, layer_index: int
    ) -> Dict[str, Any]:
        """
        Train a steering vector for a single property.

        Args:
            property_name: Name of the property (e.g., "language_italian", "safe")
            contrastive_pair_set: Set of contrastive pairs for this property
            layer_index: Layer index where steering will be applied

        Returns:
            Dictionary with training statistics
        """
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()
        
        # Check if activations were actually extracted
        if len(pos_activations) == 0 or len(neg_activations) == 0:
            raise ValueError(
                f"No activations found in contrastive pair set '{contrastive_pair_set.name}'. "
                f"Activations must be extracted before training. "
                f"Call pair_set.extract_activations_with_model(model, layer) first."
            )

        # Create control vector
        steering_vector, training_stats = create_control_vector_from_contrastive_pairs(
            pos_activations, neg_activations, self.aggregation_method, self.device
        )

        # Store property vector
        self.property_vectors[property_name] = PropertyVector(
            name=property_name,
            vector=steering_vector,
            layer_index=layer_index,
            training_stats=training_stats,
            aggregation_method=self.aggregation_method,
        )

        # Add property-specific info to stats
        training_stats.update(
            {
                "method": "DAC",
                "property": property_name,
                "layer_index": layer_index,
                "success": True,  # Indicate successful training
            }
        )

        self.is_trained = True
        return training_stats

    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """
        Train DAC for single property (backward compatibility).
        """
        # For backward compatibility, train as "default" property
        stats = self.train_property("default", contrastive_pair_set, layer_index)

        # Also set single-property attributes for compatibility
        self.layer_index = layer_index
        self.steering_vector = self.property_vectors["default"].vector
        self.training_stats = stats

        return stats

    def train_multi_property(
        self, property_pairs: Dict[str, Tuple[ContrastivePairSet, int]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple properties at once.

        Args:
            property_pairs: Dictionary mapping property names to (ContrastivePairSet, layer_index) tuples

        Returns:
            Dictionary mapping property names to training statistics
        """
        all_stats = {}
        for property_name, (pair_set, layer_idx) in property_pairs.items():
            stats = self.train_property(property_name, pair_set, layer_idx)
            all_stats[property_name] = stats
        return all_stats

    def compute_dynamic_alphas(
        self,
        unsteered_logits: torch.Tensor,
        property_steered_logits: Dict[str, torch.Tensor],
        active_properties: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute dynamic alpha values for each property using KL divergence.

        This implements the paper's approach from Section 6.1.

        Args:
            unsteered_logits: Logits from unsteered model
            property_steered_logits: Dict mapping property names to their steered logits
            active_properties: List of properties to compute alphas for (None = all)

        Returns:
            Dictionary mapping property names to alpha values
        """
        if active_properties is None:
            active_properties = list(property_steered_logits.keys())

        alphas = {}

        # Convert logits to probabilities
        p_unsteered = F.softmax(unsteered_logits, dim=-1)

        for property_name in active_properties:
            if property_name not in property_steered_logits:
                continue

            # Get steered probabilities for this property
            p_steered = F.softmax(property_steered_logits[property_name], dim=-1)

            # Apply nucleus sampling to get top-p tokens
            # Sort probabilities in descending order
            p_unsteered_sorted, indices_unsteered = torch.sort(p_unsteered, descending=True)
            p_steered_sorted, indices_steered = torch.sort(p_steered, descending=True)

            # Get cumulative probabilities
            cumsum_unsteered = torch.cumsum(p_unsteered_sorted, dim=-1)
            cumsum_steered = torch.cumsum(p_steered_sorted, dim=-1)

            # Find cutoff indices for top-p
            cutoff_unsteered = (cumsum_unsteered <= self.ptop).sum(dim=-1, keepdim=True)
            cutoff_steered = (cumsum_steered <= self.ptop).sum(dim=-1, keepdim=True)

            # Get union of top tokens from both distributions
            top_indices_unsteered = indices_unsteered[:, : cutoff_unsteered.item()]
            top_indices_steered = indices_steered[:, : cutoff_steered.item()]

            # Union of indices
            all_top_indices = torch.unique(torch.cat([top_indices_unsteered.flatten(), top_indices_steered.flatten()]))

            # Extract probabilities for these indices
            p_unsteered_filtered = p_unsteered[:, all_top_indices]
            p_steered_filtered = p_steered[:, all_top_indices]

            # Renormalize
            p_unsteered_filtered = p_unsteered_filtered / p_unsteered_filtered.sum(dim=-1, keepdim=True)
            p_steered_filtered = p_steered_filtered / p_steered_filtered.sum(dim=-1, keepdim=True)

            # Compute KL divergence: KL(p_steered || p_unsteered)
            # We want to measure how much the steered distribution diverges from unsteered
            kl_div = F.kl_div(p_unsteered_filtered.log(), p_steered_filtered, reduction="batchmean")

            # Bound alpha to [0, max_alpha]
            alpha = min(kl_div.item(), self.max_alpha)
            alphas[property_name] = alpha

        return alphas

    def _compute_dynamic_alphas_online(
        self, model_forward_fn: callable, input_ids: torch.Tensor, active_properties: List[str]
    ) -> Dict[str, float]:
        """
        Compute dynamic alphas during generation using model forward passes.

        This implements the paper's approach:
        1. Get unsteered logits
        2. Get steered logits with α=2 for each property
        3. Compute KL divergence
        4. Use KL as alpha (bounded)

        Args:
            model_forward_fn: Function that takes input_ids and returns logits
            input_ids: Current input token ids
            active_properties: Properties to compute alphas for

        Returns:
            Dictionary mapping property names to computed alphas
        """
        # Debug print
        logger.debug(f"Computing dynamic alphas for properties: {active_properties}")
        logger.debug(f"Input shape: {input_ids.shape}")

        # Get unsteered logits
        with torch.no_grad():
            unsteered_logits = model_forward_fn(input_ids)
            if hasattr(unsteered_logits, "logits"):
                unsteered_logits = unsteered_logits.logits
            unsteered_logits = unsteered_logits[:, -1, :]  # Last token

        logger.debug(f"Unsteered logits shape: {unsteered_logits.shape}")

        alphas = {}

        # For each property, compute KL with α=2 steering
        for prop_name in active_properties:
            if prop_name not in self.property_vectors:
                logger.warning(f"Property {prop_name} not found in property_vectors!")
                continue

            # Get steered logits by applying steering directly with α=2
            # We need to set up a temporary hook for this specific computation
            prop_vec = self.property_vectors[prop_name]
            layer_idx = prop_vec.layer_index

            # Create temporary hook for this property with α=2
            def temp_hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Apply steering with α=2
                steered = self.apply_steering(
                    hidden_states,
                    strength=1.0,
                    active_properties=[prop_name],
                    dynamic_alphas={prop_name: 2.0},  # Force α=2
                    compute_dynamic=False,
                )

                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            # Get the right layer module
            # Use the stored model reference if available
            if hasattr(self, "_model_ref") and self._model_ref is not None:
                model_obj = self._model_ref.hf_model
            elif hasattr(model_forward_fn, "__self__"):  # It's a bound method
                model_obj = model_forward_fn.__self__
            else:
                logger.warning(f"Cannot access model for {prop_name}")
                alpha = 0.0
                alphas[prop_name] = alpha
                continue

            # Find the layer module
            if hasattr(model_obj, "model") and hasattr(model_obj.model, "layers"):
                layer_module = model_obj.model.layers[layer_idx]
            elif hasattr(model_obj, "transformer") and hasattr(model_obj.transformer, "h"):
                layer_module = model_obj.transformer.h[layer_idx]
            else:
                logger.warning(f"Cannot find layer module for {prop_name}")
                alpha = 0.0
                alphas[prop_name] = alpha
                continue

            # Register temporary hook
            handle = layer_module.register_forward_hook(temp_hook_fn)

            # Get steered logits
            with torch.no_grad():
                steered_logits = model_forward_fn(input_ids)
                if hasattr(steered_logits, "logits"):
                    steered_logits = steered_logits.logits
                steered_logits = steered_logits[:, -1, :]  # Last token

            # Remove temporary hook
            handle.remove()

            # Compute KL divergence using nucleus sampling
            p_unsteered = F.softmax(unsteered_logits, dim=-1)
            p_steered = F.softmax(steered_logits, dim=-1)

            # Apply nucleus sampling
            p_unsteered_sorted, indices_unsteered = torch.sort(p_unsteered, descending=True)
            p_steered_sorted, indices_steered = torch.sort(p_steered, descending=True)

            cumsum_unsteered = torch.cumsum(p_unsteered_sorted, dim=-1)
            cumsum_steered = torch.cumsum(p_steered_sorted, dim=-1)

            cutoff_unsteered = (cumsum_unsteered <= self.ptop).sum(dim=-1, keepdim=True).item()
            cutoff_steered = (cumsum_steered <= self.ptop).sum(dim=-1, keepdim=True).item()

            # Get union of top tokens
            top_indices_unsteered = indices_unsteered[:, :cutoff_unsteered]
            top_indices_steered = indices_steered[:, :cutoff_steered]

            all_top_indices = torch.unique(torch.cat([top_indices_unsteered.flatten(), top_indices_steered.flatten()]))

            # Extract and renormalize probabilities
            p_unsteered_filtered = p_unsteered[:, all_top_indices]
            p_steered_filtered = p_steered[:, all_top_indices]

            p_unsteered_filtered = p_unsteered_filtered / p_unsteered_filtered.sum(dim=-1, keepdim=True)
            p_steered_filtered = p_steered_filtered / p_steered_filtered.sum(dim=-1, keepdim=True)

            # Compute KL divergence: KL(p_steered || p_unsteered)
            kl_div = F.kl_div(p_unsteered_filtered.log(), p_steered_filtered, reduction="batchmean").item()

            # Debug prints
            logger.debug(f"{prop_name} - KL divergence: {kl_div:.6f}")
            # Check if distributions are actually different
            max_diff = torch.abs(p_unsteered_filtered - p_steered_filtered).max().item()
            logger.debug(f"{prop_name} - Max prob difference: {max_diff:.6f}")
            logger.debug(f"{prop_name} - Num tokens in nucleus: {len(all_top_indices)}")

            # Bound alpha to [0, max_alpha]
            alpha = min(kl_div, self.max_alpha)
            alphas[prop_name] = alpha
            logger.debug(f"{prop_name} - Final alpha: {alpha:.6f}")

        return alphas

    def apply_steering(
        self,
        activations: torch.Tensor,
        strength: float = 1.0,
        token_probs: Optional[torch.Tensor] = None,
        property_weights: Optional[Dict[str, float]] = None,
        active_properties: Optional[List[str]] = None,
        dynamic_alphas: Optional[Dict[str, float]] = None,
        model_forward_fn: Optional[callable] = None,
        input_ids: Optional[torch.Tensor] = None,
        compute_dynamic: bool = True,
        token_position: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply DAC steering with optional dynamic intensity control.

        Supports both single-property (backward compatible) and multi-property steering.

        Args:
            activations: Input activations to steer
            strength: Base steering strength
            token_probs: Token probabilities for dynamic control (optional)
            property_weights: Manual weights for each property (overrides dynamic alphas)
            active_properties: List of properties to apply (None = all)
            dynamic_alphas: Pre-computed dynamic alphas (if None, computes them if possible)
            model_forward_fn: Function to compute model forward pass (for dynamic alpha computation)
            input_ids: Input token ids (for dynamic alpha computation)
            compute_dynamic: Whether to compute dynamic alphas (requires model_forward_fn and input_ids)
            token_position: Specific token position to steer (for precise position tracking)

        Returns:
            Steered activations
        """
        if not self.is_trained:
            raise ValueError("DAC method must be trained before applying steering")

        # Single-property backward compatibility
        if len(self.property_vectors) == 0 and self.steering_vector is not None:
            # Old single-property mode
            return self._apply_single_property_steering(activations, strength, token_probs)

        # Multi-property mode
        if active_properties is None:
            active_properties = list(self.property_vectors.keys())

        # Compute dynamic alphas if requested and possible
        if compute_dynamic and dynamic_alphas is None and model_forward_fn is not None and input_ids is not None:
            dynamic_alphas = self._compute_dynamic_alphas_online(model_forward_fn, input_ids, active_properties)

        # Start with original activations
        steered = activations.clone()

        # Apply each property's steering vector
        for property_name in active_properties:
            if property_name not in self.property_vectors:
                continue

            prop_vec = self.property_vectors[property_name]
            steering_vector = prop_vec.vector.to(activations.device)

            # Determine alpha for this property
            if dynamic_alphas and property_name in dynamic_alphas:
                alpha = dynamic_alphas[property_name]
            elif property_weights and property_name in property_weights:
                alpha = property_weights[property_name]
            else:
                alpha = 1.0  # Default

            # Apply strength multiplier
            alpha *= strength

            logger.debug(
                f"Applying {property_name} steering: alpha={alpha:.4f}, vector norm={torch.norm(steering_vector).item():.4f}"
            )

            # Handle different activation shapes
            if len(activations.shape) == 3:  # [batch, seq, hidden]
                # Use token_position if specified, otherwise apply to last token
                if token_position is not None:
                    # Apply to specific token position
                    before_norm = torch.norm(steered[:, token_position : token_position + 1, :]).item()
                    steered[:, token_position : token_position + 1, :] = steered[
                        :, token_position : token_position + 1, :
                    ] + alpha * steering_vector.unsqueeze(0).unsqueeze(0)
                    after_norm = torch.norm(steered[:, token_position : token_position + 1, :]).item()
                    logger.debug(
                        f"Applied to token position {token_position}, shape {activations.shape}, norm change: {before_norm:.4f} -> {after_norm:.4f}"
                    )
                else:
                    # Default: apply to last token position (for generation)
                    before_norm = torch.norm(steered[:, -1:, :]).item()
                    steered[:, -1:, :] = steered[:, -1:, :] + alpha * steering_vector.unsqueeze(0).unsqueeze(0)
                    after_norm = torch.norm(steered[:, -1:, :]).item()
                    logger.debug(
                        f"Applied to last token, shape {activations.shape}, norm change: {before_norm:.4f} -> {after_norm:.4f}"
                    )
            elif len(activations.shape) == 2:  # [batch, hidden]
                before_norm = torch.norm(steered).item()
                steered = steered + alpha * steering_vector.unsqueeze(0)
                after_norm = torch.norm(steered).item()
                logger.debug(
                    f"Applied to shape {activations.shape}, norm change: {before_norm:.4f} -> {after_norm:.4f}"
                )
            else:
                before_norm = torch.norm(steered).item()
                steered = steered + alpha * steering_vector
                after_norm = torch.norm(steered).item()
                logger.debug(
                    f"Applied to shape {activations.shape}, norm change: {before_norm:.4f} -> {after_norm:.4f}"
                )

        return steered

    def _apply_single_property_steering(
        self, activations: torch.Tensor, strength: float = 1.0, token_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Legacy single-property steering for backward compatibility."""
        # Calculate dynamic strength if enabled
        if self.dynamic_control:
            if token_probs is not None:
                # Use provided token probabilities
                entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10), dim=-1)
                entropy_factor = torch.clamp(self.entropy_threshold / (entropy + 1e-10), 0.1, 2.0)
                dynamic_strength = strength * entropy_factor.mean().item()
            else:
                # Fallback: estimate entropy from activation variance
                if len(activations.shape) == 3:
                    last_token_acts = activations[:, -1, :]
                else:
                    last_token_acts = activations

                activation_var = torch.var(last_token_acts, dim=-1).mean()
                normalized_var = torch.clamp(activation_var / 10.0, 0.1, 2.0)
                entropy_factor = self.entropy_threshold / normalized_var
                dynamic_strength = strength * entropy_factor.item()
        else:
            dynamic_strength = strength

        # Apply additive steering
        steering_vector = self.steering_vector.to(activations.device)

        # Handle different activation shapes
        if len(activations.shape) == 3:  # [batch, seq, hidden]
            steered = activations.clone()
            if activations.shape[1] > 1:
                steered[:, -2:-1, :] = steered[:, -2:-1, :] + dynamic_strength * steering_vector.unsqueeze(0).unsqueeze(
                    0
                )
            else:
                steered[:, -1:, :] = steered[:, -1:, :] + dynamic_strength * steering_vector.unsqueeze(0).unsqueeze(0)
        elif len(activations.shape) == 2:  # [batch, hidden]
            steered = activations + dynamic_strength * steering_vector.unsqueeze(0)
        else:
            steered = activations + dynamic_strength * steering_vector

        return steered

    def set_model_reference(self, model):
        """
        Set reference to the model for dynamic alpha computation.
        This allows the steering method to compute forward passes for KL divergence.
        """
        self._model_ref = model

    def get_property_vectors(self) -> Dict[str, PropertyVector]:
        """Get all property vectors."""
        return self.property_vectors

    def get_steering_vector(self) -> torch.Tensor:
        """Return the steering vector (single-property compatibility)."""
        if not self.is_trained:
            raise ValueError("DAC method must be trained before getting steering vector")

        # Multi-property mode
        if self.property_vectors:
            if "default" in self.property_vectors:
                return self.property_vectors["default"].vector
            if len(self.property_vectors) == 1:
                return list(self.property_vectors.values())[0].vector
            raise ValueError("Multiple property vectors exist. Use get_property_vectors() instead.")

        # Single-property mode
        return self.steering_vector

    def save_steering_vector(self, path: str) -> bool:
        """Save DAC steering data."""
        if not self.is_trained:
            return False
        try:
            save_data = {
                "method": "DAC",
                "dynamic_control": self.dynamic_control,
                "entropy_threshold": self.entropy_threshold,
                "aggregation_method": self.aggregation_method.value,
                "ptop": self.ptop,
                "max_alpha": self.max_alpha,
            }

            # Save multi-property data if available
            if self.property_vectors:
                save_data["property_vectors"] = {}
                for prop_name, prop_vec in self.property_vectors.items():
                    save_data["property_vectors"][prop_name] = {
                        "vector": prop_vec.vector,
                        "layer_index": prop_vec.layer_index,
                        "training_stats": prop_vec.training_stats,
                        "aggregation_method": prop_vec.aggregation_method.value,
                    }
            else:
                # Single-property compatibility
                save_data["steering_vector"] = self.steering_vector
                save_data["layer_index"] = self.layer_index
                save_data["training_stats"] = self.training_stats

            torch.save(save_data, path)
            return True
        except Exception:
            return False

    def load_steering_vector(self, path: str) -> bool:
        """Load DAC steering data."""
        try:
            data = torch.load(path, map_location=self.device)
            if data.get("method") != "DAC":
                return False

            self.dynamic_control = data.get("dynamic_control", True)
            self.entropy_threshold = data.get("entropy_threshold", 1.0)
            self.ptop = data.get("ptop", 0.4)
            self.max_alpha = data.get("max_alpha", 2.0)

            # Load multi-property data if available
            if "property_vectors" in data:
                for prop_name, prop_data in data["property_vectors"].items():
                    self.property_vectors[prop_name] = PropertyVector(
                        name=prop_name,
                        vector=prop_data["vector"],
                        layer_index=prop_data["layer_index"],
                        training_stats=prop_data.get("training_stats", {}),
                        aggregation_method=ControlVectorAggregationMethod(prop_data.get("aggregation_method", "caa")),
                    )
            else:
                # Load single-property data
                self.steering_vector = data["steering_vector"]
                self.layer_index = data.get("layer_index")
                self.training_stats = data.get("training_stats", {})

            self.is_trained = True
            return True
        except Exception:
            return False

    @staticmethod
    def combine_steering_vectors(
        vectors: List[torch.Tensor], weights: List[float], normalize_weights: bool = True
    ) -> torch.Tensor:
        """
        Combine multiple steering vectors using weighted arithmetic.

        This implements the DAC reference approach: comp_vector = alpha_1 * diff1 + alpha_2 * diff2

        Args:
            vectors: List of steering vectors to combine
            weights: List of weights for each vector
            normalize_weights: Whether to normalize weights to sum to 1.0

        Returns:
            Combined steering vector
        """
        if len(vectors) != len(weights):
            raise ValueError(f"Number of vectors ({len(vectors)}) must match number of weights ({len(weights)})")

        if len(vectors) == 0:
            raise ValueError("Must provide at least one vector")

        # Normalize weights if requested
        if normalize_weights:
            weight_sum = sum(weights)
            if weight_sum == 0:
                raise ValueError("Weights sum to zero")
            weights = [w / weight_sum for w in weights]

        # Ensure all vectors are on the same device
        device = vectors[0].device
        vectors = [v.to(device) for v in vectors]

        # Combine vectors using weighted sum
        combined = torch.zeros_like(vectors[0])
        for vector, weight in zip(vectors, weights):
            combined = combined + weight * vector

        return combined

    def combine_property_vectors(
        self, property_weights: Dict[str, float], normalize_weights: bool = True
    ) -> torch.Tensor:
        """
        Combine loaded property vectors with specified weights.

        Args:
            property_weights: Dictionary mapping property names to weights
            normalize_weights: Whether to normalize weights to sum to 1.0

        Returns:
            Combined steering vector
        """
        if not self.property_vectors:
            raise ValueError("No property vectors loaded")

        vectors = []
        weights = []

        for prop_name, weight in property_weights.items():
            if prop_name not in self.property_vectors:
                raise ValueError(f"Property '{prop_name}' not found in loaded vectors")
            vectors.append(self.property_vectors[prop_name].vector)
            weights.append(weight)

        return self.combine_steering_vectors(vectors, weights, normalize_weights)

    def create_combined_property(
        self, name: str, property_weights: Dict[str, float], normalize_weights: bool = True
    ) -> None:
        """
        Create a new property by combining existing properties.

        Args:
            name: Name for the new combined property
            property_weights: Dictionary mapping property names to weights
            normalize_weights: Whether to normalize weights to sum to 1.0
        """
        combined_vector = self.combine_property_vectors(property_weights, normalize_weights)

        # Use the layer index from the first property (they should all be the same)
        layer_idx = list(self.property_vectors.values())[0].layer_index

        # Create new property vector
        self.property_vectors[name] = PropertyVector(
            name=name,
            vector=combined_vector,
            layer_index=layer_idx,
            training_stats={
                "method": "combined",
                "source_properties": list(property_weights.keys()),
                "weights": property_weights,
                "normalized": normalize_weights,
            },
            aggregation_method=self.aggregation_method,
        )

    def generate_with_dynamic_steering(
        self,
        model,
        prompt: str,
        active_properties: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        verbose: bool = False,
    ) -> Tuple[str, Dict[str, List[float]]]:
        """
        Generate text with true dynamic DAC steering.

        This implements the paper's approach where alphas are recomputed
        at each generation step based on KL divergence.

        Args:
            model: The language model (must have hf_model, tokenizer, device)
            prompt: Input prompt
            active_properties: List of property names to activate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter # TODO The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
            do_sample: Whether to use sampling
            verbose: Whether to print debug info

        Returns:
            Tuple of (generated_text, alpha_history)
        """
        # Format and tokenize prompt
        formatted_prompt = model.format_prompt(prompt)
        inputs = model.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        # Track initial prompt length for precise token position handling
        initial_prompt_len = input_ids.shape[1]

        # Track alpha history
        alpha_history = {prop: [] for prop in active_properties}

        # Track current token position (will be updated in the loop)
        current_generation_step = [0]  # Use list to allow mutation in closure

        # Set up hooks for each property at their respective layers
        hooks = []
        current_alphas = dict.fromkeys(active_properties, 1.0)

        def create_hook(property_name):
            prop_vec = self.property_vectors[property_name]
            layer_idx = prop_vec.layer_index

            def hook_fn(module, input, output):
                logger.debug(f"Hook called for {property_name} at layer {layer_idx}")
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                logger.debug(f"Hidden states shape: {hidden_states.shape}")
                logger.debug(f"Current alpha for {property_name}: {current_alphas[property_name]:.4f}")

                # Apply steering with current alpha and token position info
                # Use the current generation step to determine the exact token position
                token_pos = initial_prompt_len - 1 + current_generation_step[0]
                steered = self.apply_steering(
                    hidden_states,
                    strength=1.0,
                    active_properties=[property_name],
                    dynamic_alphas={property_name: current_alphas[property_name]},
                    compute_dynamic=False,  # Already computed
                    token_position=token_pos,  # Precise position
                )

                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            return layer_idx, hook_fn

        # Register hooks
        for prop_name in active_properties:
            if prop_name in self.property_vectors:
                layer_idx, hook_fn = create_hook(prop_name)
                logger.debug(f"Registering hook for {prop_name} at layer {layer_idx}")

                # Get layer module
                if hasattr(model.hf_model, "model") and hasattr(model.hf_model.model, "layers"):
                    layer_module = model.hf_model.model.layers[layer_idx]
                    logger.debug("Using Llama-style layers")
                elif hasattr(model.hf_model, "transformer") and hasattr(model.hf_model.transformer, "h"):
                    layer_module = model.hf_model.transformer.h[layer_idx]
                    logger.debug("Using GPT-style layers")
                else:
                    raise ValueError("Unsupported model architecture")

                handle = layer_module.register_forward_hook(hook_fn)
                hooks.append(handle)
                logger.debug("Hook registered successfully")

        # Generate tokens one at a time with dynamic alpha computation
        generated_tokens = []

        for step in range(max_new_tokens):
            # Update current generation step for precise token position tracking
            current_generation_step[0] = step

            # Compute dynamic alphas for this step
            # Pass the model's forward method directly
            new_alphas = self._compute_dynamic_alphas_online(model.hf_model, input_ids, active_properties)

            # Update the dictionary in place so hooks see the new values
            for prop in active_properties:
                current_alphas[prop] = new_alphas.get(prop, 0.0)

            # Record alphas
            for prop in active_properties:
                alpha_history[prop].append(current_alphas.get(prop, 0.0))

            if verbose and step % 10 == 0:
                alpha_str = ", ".join([f"{p}={current_alphas.get(p, 0):.2f}" for p in active_properties])
                logger.info(f"Step {step}: α=[{alpha_str}]")

            # Generate next token
            with torch.no_grad():
                outputs = model.hf_model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]

                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature

                # Sample next token
                if do_sample:
                    # Apply top-p filtering
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Set logits to -inf for removed tokens
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float("-inf")

                    # Sample from filtered distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_tokens.append(next_token.item())

            # Check for EOS
            if next_token.item() == model.tokenizer.eos_token_id:
                break

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Decode generated tokens
        generated_ids = torch.tensor(generated_tokens, device=model.device).unsqueeze(0)
        generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text, alpha_history
