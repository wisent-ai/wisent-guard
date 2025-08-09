"""
Dynamic Activation Composition (DAC) steering method - Fixed Implementation.
This implementation matches the original DAC behavior from the provided reference.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..aggregation import ControlVectorAggregationMethod
from ..contrastive_pairs import ContrastivePairSet
from .base import SteeringMethod

logger = logging.getLogger(__name__)


@dataclass
class PropertyVector:
    """Container for a single property's steering vector and metadata."""

    name: str
    vectors: Dict[int, torch.Tensor]  # layer_index -> vector mapping
    training_stats: Dict[str, Any]
    aggregation_method: ControlVectorAggregationMethod


class DAC(SteeringMethod):
    """
    Dynamic Activation Composition (DAC) steering method.

    Uses information-theoretic principles to dynamically modulate steering
    intensity throughout generation for multi-property control.

    Key difference from base implementation: Works on ALL layers simultaneously
    as in the original DAC paper.
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
        # Single-property compatibility - NOTE: Original DAC doesn't use single layer
        self.steering_vectors = None  # Will store Dict[int, torch.Tensor] for layer->vector
        self.layer_index = None  # Always None for DAC compatibility
        self.dynamic_control = dynamic_control
        self.entropy_threshold = entropy_threshold
        self.aggregation_method = aggregation_method
        self.training_stats = {}

        # Multi-property support
        self.property_vectors: Dict[str, PropertyVector] = {}
        self.ptop = ptop
        self.max_alpha = max_alpha
        self._model_ref = None  # For KL computation
        self._model_config = None  # Store model configuration

    def train_property(
        self,
        property_name: str,
        contrastive_pair_set: ContrastivePairSet,
        layer_index: Optional[int] = None,  # Ignored for DAC
    ) -> Dict[str, Any]:
        """
        Train a steering vector for a single property across ALL layers.

        NOTE: layer_index parameter is ignored as DAC operates on all layers.
        This matches the original implementation behavior.
        """
        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()

        if len(pos_activations) == 0 or len(neg_activations) == 0:
            raise ValueError(
                f"No activations found in contrastive pair set '{contrastive_pair_set.name}'. "
                f"Activations must be extracted before training. "
                f"Call pair_set.extract_activations_with_model(model, layer) first."
            )

        # Extract all layer activations from the contrastive pairs
        # NOTE: Original DAC extracts from all attention output projections
        all_layer_vectors = {}
        all_training_stats = {}

        # Assuming activations are structured as Dict[layer_idx, tensors]
        if isinstance(pos_activations, dict):
            # Multi-layer activations
            for layer_idx in pos_activations.keys():
                if layer_idx in neg_activations:
                    # Compute difference vector for this layer
                    pos_mean = torch.stack(pos_activations[layer_idx]).mean(dim=0)
                    neg_mean = torch.stack(neg_activations[layer_idx]).mean(dim=0)
                    diff_vector = pos_mean - neg_mean

                    all_layer_vectors[layer_idx] = diff_vector.to(self.device)
                    all_training_stats[f"layer_{layer_idx}_norm"] = torch.norm(diff_vector).item()
        else:
            # Single layer activations - need to handle this case
            # This is a compatibility issue - original DAC expects multi-layer
            raise ValueError(
                "DAC requires multi-layer activations. "
                "Extract activations from all layers using extract_activations_with_model."
            )

        # Store property vector
        self.property_vectors[property_name] = PropertyVector(
            name=property_name,
            vectors=all_layer_vectors,
            training_stats=all_training_stats,
            aggregation_method=self.aggregation_method,
        )

        # Add property-specific info to stats
        training_stats = {
            "method": "DAC",
            "property": property_name,
            "num_layers": len(all_layer_vectors),
            "layer_norms": all_training_stats,
            "success": True,
        }

        self.is_trained = True
        return training_stats

    def train(
        self,
        contrastive_pair_set: ContrastivePairSet,
        layer_index: Optional[int] = None,  # Ignored for DAC
    ) -> Dict[str, Any]:
        """
        Train DAC for single property (backward compatibility).
        NOTE: layer_index is ignored as DAC operates on all layers.
        """
        # For backward compatibility, train as "default" property
        stats = self.train_property("default", contrastive_pair_set, layer_index)

        # Set single-property attributes for compatibility
        self.layer_index = None  # Always None for DAC
        self.steering_vectors = self.property_vectors["default"].vectors
        self.training_stats = stats

        return stats

    def _extract_all_layer_activations(
        self, model, tokenized_prompt: torch.Tensor, config: Dict[str, Any]
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from all attention layers.
        This mimics the original DAC extraction process.
        """
        all_activations = {}

        # Hook for capturing activations
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                # Store the attention output for this layer
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                all_activations[layer_idx] = hidden_states.detach()

            return hook_fn

        # Register hooks on all attention layers
        hooks = []
        for layer_idx in range(config["n_layers"]):
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                # Llama-style
                layer_module = model.model.layers[layer_idx].self_attn.o_proj
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                # GPT-style
                layer_module = model.transformer.h[layer_idx].attn.out_proj
            else:
                raise ValueError("Unsupported model architecture")

            handle = layer_module.register_forward_hook(create_hook(layer_idx))
            hooks.append(handle)

        # Forward pass to collect activations
        with torch.no_grad():
            _ = model(tokenized_prompt)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return all_activations

    def _compute_dynamic_alphas_online(
        self, model_forward_fn: callable, input_ids: torch.Tensor, active_properties: List[str]
    ) -> Dict[str, float]:
        """
        Compute dynamic alphas during generation using model forward passes.
        This matches the original DAC implementation.
        """
        logger.debug(f"Computing dynamic alphas for properties: {active_properties}")

        # Get unsteered logits
        with torch.no_grad():
            unsteered_logits = model_forward_fn(input_ids)
            if hasattr(unsteered_logits, "logits"):
                unsteered_logits = unsteered_logits.logits
            unsteered_logits = unsteered_logits[:, -1, :]  # Last token

        alphas = {}

        # For each property, compute KL with α=2 steering
        for prop_name in active_properties:
            if prop_name not in self.property_vectors:
                logger.warning(f"Property {prop_name} not found in property_vectors!")
                continue

            # Get steered logits by applying steering with α=2
            prop_vec = self.property_vectors[prop_name]

            # Create temporary hooks for all layers
            temp_hooks = []

            def create_temp_hook(layer_idx, prop_vectors):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # Apply steering with α=2
                    if layer_idx in prop_vectors:
                        steering_vec = prop_vectors[layer_idx].to(hidden_states.device)
                        # Apply to last token position
                        hidden_states[:, -1, :] = hidden_states[:, -1, :] + 2.0 * steering_vec

                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    return hidden_states

                return hook_fn

            # Get the model object
            if hasattr(self, "_model_ref") and self._model_ref is not None:
                model_obj = self._model_ref.hf_model
            elif hasattr(model_forward_fn, "__self__"):
                model_obj = model_forward_fn.__self__
            else:
                logger.warning(f"Cannot access model for {prop_name}")
                alphas[prop_name] = 0.0
                continue

            # Register hooks on all layers
            for layer_idx in prop_vec.vectors.keys():
                if hasattr(model_obj, "model") and hasattr(model_obj.model, "layers"):
                    layer_module = model_obj.model.layers[layer_idx]
                elif hasattr(model_obj, "transformer") and hasattr(model_obj.transformer, "h"):
                    layer_module = model_obj.transformer.h[layer_idx]
                else:
                    continue

                handle = layer_module.register_forward_hook(create_temp_hook(layer_idx, prop_vec.vectors))
                temp_hooks.append(handle)

            # Get steered logits
            with torch.no_grad():
                steered_logits = model_forward_fn(input_ids)
                if hasattr(steered_logits, "logits"):
                    steered_logits = steered_logits.logits
                steered_logits = steered_logits[:, -1, :]  # Last token

            # Remove temporary hooks
            for hook in temp_hooks:
                hook.remove()

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
        layer_index: Optional[int] = None,  # Added for compatibility
    ) -> torch.Tensor:
        """
        Apply DAC steering with optional dynamic intensity control.

        NOTE: This method is called per-layer in the hook during generation.
        The layer_index parameter indicates which layer is being processed.
        """
        if not self.is_trained:
            raise ValueError("DAC method must be trained before applying steering")

        # Single-property backward compatibility
        if len(self.property_vectors) == 0 and self.steering_vectors is not None:
            # Old single-property mode
            if layer_index is not None and layer_index in self.steering_vectors:
                return self._apply_single_layer_steering(
                    activations, strength, self.steering_vectors[layer_index], token_probs
                )
            return activations

        # Multi-property mode
        if active_properties is None:
            active_properties = list(self.property_vectors.keys())

        # Compute dynamic alphas if requested and possible
        if compute_dynamic and dynamic_alphas is None and model_forward_fn is not None and input_ids is not None:
            dynamic_alphas = self._compute_dynamic_alphas_online(model_forward_fn, input_ids, active_properties)

        # Start with original activations
        steered = activations.clone()

        # Apply each property's steering vector for the current layer
        for property_name in active_properties:
            if property_name not in self.property_vectors:
                continue

            prop_vec = self.property_vectors[property_name]

            # Check if this property has a vector for the current layer
            if layer_index is not None and layer_index in prop_vec.vectors:
                steering_vector = prop_vec.vectors[layer_index].to(activations.device)
            else:
                # Skip if no vector for this layer
                continue

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
                f"Applying {property_name} steering at layer {layer_index}: "
                f"alpha={alpha:.4f}, vector norm={torch.norm(steering_vector).item():.4f}"
            )

            # Apply steering to the appropriate token position
            if len(activations.shape) == 3:  # [batch, seq, hidden]
                if token_position is not None:
                    # Apply to specific token position
                    steered[:, token_position, :] = steered[:, token_position, :] + alpha * steering_vector
                else:
                    # Default: apply to last token position
                    steered[:, -1, :] = steered[:, -1, :] + alpha * steering_vector
            elif len(activations.shape) == 2:  # [batch, hidden]
                steered = steered + alpha * steering_vector.unsqueeze(0)
            else:
                steered = steered + alpha * steering_vector

        return steered

    def _apply_single_layer_steering(
        self,
        activations: torch.Tensor,
        strength: float,
        steering_vector: torch.Tensor,
        token_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply steering for a single layer (backward compatibility)."""
        # Calculate dynamic strength if enabled
        if self.dynamic_control and token_probs is not None:
            entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10), dim=-1)
            entropy_factor = torch.clamp(self.entropy_threshold / (entropy + 1e-10), 0.1, 2.0)
            dynamic_strength = strength * entropy_factor.mean().item()
        else:
            dynamic_strength = strength

        # Apply additive steering
        if len(activations.shape) == 3:  # [batch, seq, hidden]
            steered = activations.clone()
            steered[:, -1, :] = steered[:, -1, :] + dynamic_strength * steering_vector
        elif len(activations.shape) == 2:  # [batch, hidden]
            steered = activations + dynamic_strength * steering_vector.unsqueeze(0)
        else:
            steered = activations + dynamic_strength * steering_vector

        return steered

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
        This implements the full multi-layer steering approach.
        """
        # Format and tokenize prompt
        formatted_prompt = model.format_prompt(prompt)
        inputs = model.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        # Track alpha history
        alpha_history = {prop: [] for prop in active_properties}

        # Set up hooks for ALL layers for each property
        hooks = []
        current_alphas = dict.fromkeys(active_properties, 1.0)
        current_step = [0]  # Track generation step

        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Apply steering for all active properties at this layer
                steered = hidden_states.clone()
                for property_name in active_properties:
                    if property_name in self.property_vectors:
                        prop_vec = self.property_vectors[property_name]
                        if layer_idx in prop_vec.vectors:
                            steering_vector = prop_vec.vectors[layer_idx].to(hidden_states.device)
                            alpha = current_alphas.get(property_name, 1.0)
                            # Apply to last token
                            steered[:, -1, :] = steered[:, -1, :] + alpha * steering_vector

                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            return hook_fn

        # Get model configuration if not already stored
        if self._model_config is None:
            if hasattr(model.hf_model, "config"):
                self._model_config = {"n_layers": model.hf_model.config.num_hidden_layers}

        # Register hooks on all layers
        for layer_idx in range(self._model_config["n_layers"]):
            if hasattr(model.hf_model, "model") and hasattr(model.hf_model.model, "layers"):
                layer_module = model.hf_model.model.layers[layer_idx]
            elif hasattr(model.hf_model, "transformer") and hasattr(model.hf_model.transformer, "h"):
                layer_module = model.hf_model.transformer.h[layer_idx]
            else:
                raise ValueError("Unsupported model architecture")

            handle = layer_module.register_forward_hook(create_hook(layer_idx))
            hooks.append(handle)

        # Generate tokens one at a time with dynamic alpha computation
        generated_tokens = []

        for step in range(max_new_tokens):
            current_step[0] = step

            # Compute dynamic alphas for this step
            new_alphas = self._compute_dynamic_alphas_online(model.hf_model, input_ids, active_properties)

            # Update alphas
            for prop in active_properties:
                current_alphas[prop] = new_alphas.get(prop, 0.0)
                alpha_history[prop].append(current_alphas[prop])

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

    def get_steering_vector(self) -> torch.Tensor:
        """
        Return the steering vector (single-property compatibility).
        NOTE: For DAC, this returns vectors for all layers.
        """
        if not self.is_trained:
            raise ValueError("DAC method must be trained before getting steering vector")

        # Multi-property mode
        if self.property_vectors:
            if "default" in self.property_vectors:
                # Return first layer's vector for compatibility
                vectors = self.property_vectors["default"].vectors
                return list(vectors.values())[0] if vectors else None
            if len(self.property_vectors) == 1:
                # Return first layer's vector for compatibility
                vectors = list(self.property_vectors.values())[0].vectors
                return list(vectors.values())[0] if vectors else None
            raise ValueError("Multiple property vectors exist. Use get_property_vectors() instead.")

        # Single-property mode - return first layer vector for compatibility
        if self.steering_vectors:
            return list(self.steering_vectors.values())[0]

        return None

    def get_property_vectors(self) -> Dict[str, PropertyVector]:
        """Get all property vectors."""
        return self.property_vectors

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
                        "vectors": prop_vec.vectors,  # Dict[int, torch.Tensor]
                        "training_stats": prop_vec.training_stats,
                        "aggregation_method": prop_vec.aggregation_method.value,
                    }
            else:
                # Single-property compatibility
                save_data["steering_vectors"] = self.steering_vectors
                save_data["layer_index"] = self.layer_index
                save_data["training_stats"] = self.training_stats

            torch.save(save_data, path)
            return True
        except Exception as e:
            logger.error(f"Error saving DAC vectors: {e}")
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
                        vectors=prop_data["vectors"],  # Dict[int, torch.Tensor]
                        training_stats=prop_data.get("training_stats", {}),
                        aggregation_method=ControlVectorAggregationMethod(prop_data.get("aggregation_method", "caa")),
                    )
            else:
                # Load single-property data
                self.steering_vectors = data["steering_vectors"]
                self.layer_index = data.get("layer_index")  # Should be None
                self.training_stats = data.get("training_stats", {})

            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Error loading DAC vectors: {e}")
            return False

    def set_model_reference(self, model):
        """
        Set reference to the model for dynamic alpha computation.
        Also extract model configuration.
        """
        self._model_ref = model

        # Extract model configuration
        if hasattr(model, "hf_model"):
            hf_model = model.hf_model
            if hasattr(hf_model, "config"):
                self._model_config = {
                    "n_layers": hf_model.config.num_hidden_layers,
                    "n_heads": hf_model.config.num_attention_heads,
                    "d_model": hf_model.config.hidden_size,
                }
