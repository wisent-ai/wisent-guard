"""
HuggingFace-compatible model wrapper that applies steering vectors during inference.
This allows the model to be used with any library (LiveCodeBench, lm-eval, etc.)
while transparently applying steering vectors.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

import torch
from transformers import AutoConfig, AutoModel, GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class SteeringCompatibleModel(GPT2LMHeadModel):
    """
    HuggingFace-compatible model that applies steering vectors during inference.

    This model can be used as a drop-in replacement for any GPT2-based model
    in libraries like LiveCodeBench, lm-eval, etc.
    """

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        # Steering vector storage
        self.steering_vectors = {}  # {layer_idx: steering_vector}
        self.steering_metadata = {}
        self.steering_active = True

        # Register steering vectors as model parameters so they're saved/loaded
        self.register_buffer("_steering_layer_indices", torch.tensor([], dtype=torch.long), persistent=True)

        # Hook to restore steering vectors after loading
        self.register_load_state_dict_post_hook(self._restore_steering_vectors)

    def add_steering_vector(
        self, layer_idx: int, steering_vector: torch.Tensor, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a steering vector for a specific layer.

        Args:
            layer_idx: Layer index to apply steering
            steering_vector: Steering vector tensor
            metadata: Optional metadata about the steering vector
        """
        if layer_idx >= len(self.transformer.h):
            raise ValueError(f"Layer index {layer_idx} out of range for model with {len(self.transformer.h)} layers")

        # Store steering vector as a buffer so it's saved/loaded with the model
        buffer_name = f"_steering_vector_{layer_idx}"
        self.register_buffer(buffer_name, steering_vector)

        # Update tracking
        self.steering_vectors[layer_idx] = steering_vector
        self.steering_metadata[layer_idx] = metadata or {}

        # Update layer indices
        indices = self._steering_layer_indices.tolist()
        if layer_idx not in indices:
            indices.append(layer_idx)
            self._steering_layer_indices = torch.tensor(sorted(indices), dtype=torch.long)

        logger.info(f"Added steering vector for layer {layer_idx}")

    def remove_steering_vector(self, layer_idx: int):
        """Remove steering vector for a specific layer."""
        if layer_idx in self.steering_vectors:
            # Remove buffer
            buffer_name = f"_steering_vector_{layer_idx}"
            if hasattr(self, buffer_name):
                delattr(self, buffer_name)

            # Remove from tracking
            del self.steering_vectors[layer_idx]
            if layer_idx in self.steering_metadata:
                del self.steering_metadata[layer_idx]

            # Update layer indices
            indices = self._steering_layer_indices.tolist()
            if layer_idx in indices:
                indices.remove(layer_idx)
                self._steering_layer_indices = torch.tensor(indices, dtype=torch.long)

            logger.info(f"Removed steering vector for layer {layer_idx}")

    def enable_steering(self):
        """Enable steering vector application."""
        self.steering_active = True
        logger.info("Steering vectors enabled")

    def disable_steering(self):
        """Disable steering vector application."""
        self.steering_active = False
        logger.info("Steering vectors disabled")

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle steering vector buffers."""
        # Handle _steering_layer_indices buffer size mismatch
        indices_key = "_steering_layer_indices"
        if indices_key in state_dict:
            new_size = state_dict[indices_key].shape[0]
            if new_size > 0:
                self.register_buffer(
                    "_steering_layer_indices", torch.zeros(new_size, dtype=torch.long), persistent=True
                )

        # Register steering vector buffers dynamically
        for key, tensor in state_dict.items():
            if key.startswith("_steering_vector_"):
                if not hasattr(self, key):
                    self.register_buffer(key, torch.zeros_like(tensor), persistent=True)

        # Call parent method
        return super().load_state_dict(state_dict, strict)

    def _restore_steering_vectors(self, module, incompatible_keys):
        """Restore steering vectors from saved buffers after loading."""
        self.steering_vectors = {}

        # Find all steering vector buffers
        for name, buffer in self.named_buffers():
            if name.startswith("_steering_vector_"):
                try:
                    layer_idx = int(name.split("_")[-1])
                    self.steering_vectors[layer_idx] = buffer
                    logger.info(f"Restored steering vector for layer {layer_idx}")
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse layer index from buffer name: {name}")

        logger.info(f"Restored {len(self.steering_vectors)} steering vectors from model state")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        """
        Forward pass with steering vector application.
        """
        # If no steering vectors or steering disabled, use original forward
        if not self.steering_active or not self.steering_vectors:
            return super().forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        # Custom forward pass with steering
        return self._forward_with_steering(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def _forward_with_steering(self, **kwargs):
        """Forward pass with steering vector application."""
        # Apply steering using hooks during forward pass
        hooks = []

        def create_steering_hook(layer_idx, steering_vector):
            def hook_fn(module, input, output):
                # Handle different output formats (tuple vs tensor)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Apply steering vector to the output
                # hidden_states shape: (batch_size, seq_len, hidden_size)
                if steering_vector.dim() == 1:
                    # Expand to match batch and sequence dimensions
                    steering_vector_expanded = steering_vector.unsqueeze(0).unsqueeze(0)
                elif steering_vector.dim() == 2:
                    steering_vector_expanded = steering_vector.unsqueeze(0)
                else:
                    steering_vector_expanded = steering_vector

                # Apply steering (additive) to last token position
                hidden_states[:, -1, :] = hidden_states[:, -1, :] + steering_vector_expanded.squeeze(0).squeeze(0)

                # Return in the same format as received
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                return hidden_states

            return hook_fn

        try:
            # Register hooks for layers with steering vectors
            for layer_idx, steering_vector in self.steering_vectors.items():
                if layer_idx < len(self.transformer.h):
                    hook = self.transformer.h[layer_idx].register_forward_hook(
                        create_steering_hook(layer_idx, steering_vector)
                    )
                    hooks.append(hook)

            # Run normal forward pass with steering hooks active
            return super().forward(
                input_ids=kwargs.get("input_ids"),
                past_key_values=kwargs.get("past_key_values"),
                attention_mask=kwargs.get("attention_mask"),
                token_type_ids=kwargs.get("token_type_ids"),
                position_ids=kwargs.get("position_ids"),
                head_mask=kwargs.get("head_mask"),
                inputs_embeds=kwargs.get("inputs_embeds"),
                labels=kwargs.get("labels"),
                use_cache=kwargs.get("use_cache"),
                output_attentions=kwargs.get("output_attentions"),
                output_hidden_states=kwargs.get("output_hidden_states"),
                return_dict=kwargs.get("return_dict"),
            )

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

    def save_steering_vectors(self, save_directory: str):
        """Save steering vectors with metadata."""
        os.makedirs(save_directory, exist_ok=True)

        # Save each steering vector
        for layer_idx, steering_vector in self.steering_vectors.items():
            filename = f"steering_vector_{self.config.model_type}_{layer_idx}_{datetime.now().strftime('%Y%m%d')}.pt"
            filepath = os.path.join(save_directory, filename)
            torch.save(steering_vector, filepath)

            # Save metadata
            metadata_filename = (
                f"steering_vector_{self.config.model_type}_{layer_idx}_{datetime.now().strftime('%Y%m%d')}.json"
            )
            metadata_filepath = os.path.join(save_directory, metadata_filename)

            metadata = self.steering_metadata.get(layer_idx, {})
            metadata.update(
                {
                    "layer_idx": layer_idx,
                    "model_type": self.config.model_type,
                    "vector_shape": list(steering_vector.shape),
                    "save_date": datetime.now().isoformat(),
                    "vector_norm": float(torch.norm(steering_vector).item()),
                }
            )

            with open(metadata_filepath, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Saved {len(self.steering_vectors)} steering vectors to {save_directory}")

    def load_steering_vectors(self, load_directory: str):
        """Load steering vectors from directory."""
        if not os.path.exists(load_directory):
            raise ValueError(f"Directory {load_directory} does not exist")

        loaded_count = 0
        for filename in os.listdir(load_directory):
            if filename.endswith(".pt") and "steering_vector" in filename:
                filepath = os.path.join(load_directory, filename)
                steering_vector = torch.load(filepath, map_location=self.device)

                # Extract layer index from filename
                # Expected format: steering_vector_<model>_<layer>_<date>.pt
                parts = filename.split("_")
                if len(parts) >= 4:
                    try:
                        layer_idx = int(parts[3])  # layer index is at position 3

                        # Load metadata if available
                        metadata_filename = filename.replace(".pt", ".json")
                        metadata_filepath = os.path.join(load_directory, metadata_filename)
                        metadata = {}
                        if os.path.exists(metadata_filepath):
                            with open(metadata_filepath) as f:
                                metadata = json.load(f)

                        self.add_steering_vector(layer_idx, steering_vector, metadata)
                        loaded_count += 1

                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse layer index from filename: {filename}")

        logger.info(f"Loaded {loaded_count} steering vectors from {load_directory}")

    def get_steering_info(self) -> Dict[str, Any]:
        """Get information about current steering vectors."""
        return {
            "active": self.steering_active,
            "num_vectors": len(self.steering_vectors),
            "layers": list(self.steering_vectors.keys()),
            "metadata": self.steering_metadata,
        }

    @classmethod
    def from_pretrained_with_steering(cls, model_name_or_path: str, steering_directory: Optional[str] = None, **kwargs):
        """
        Load a pre-trained model and optionally apply steering vectors.

        Args:
            model_name_or_path: Model name or path
            steering_directory: Directory containing steering vectors
            **kwargs: Additional arguments for model loading
        """
        # Load the base model
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        model = cls(config)

        # Load pre-trained weights
        base_model = GPT2LMHeadModel.from_pretrained(model_name_or_path, **kwargs)
        model.load_state_dict(base_model.state_dict(), strict=False)

        # Load steering vectors if provided
        if steering_directory:
            model.load_steering_vectors(steering_directory)

        return model


# Register the model so it can be used with AutoModel
try:
    AutoModel.register(GPT2Config, SteeringCompatibleModel)
    logger.info("Registered SteeringCompatibleModel with AutoModel")
except Exception as e:
    # This is expected when GPT2Config is already registered - not a real error
    logger.debug(f"Could not register SteeringCompatibleModel (expected): {e}")


def create_steering_compatible_model(
    base_model_name: str = "distilgpt2", steering_directory: Optional[str] = None
) -> SteeringCompatibleModel:
    """
    Create a steering-compatible model.

    Args:
        base_model_name: Base model name (e.g., "distilgpt2")
        steering_directory: Directory containing steering vectors

    Returns:
        SteeringCompatibleModel instance
    """
    return SteeringCompatibleModel.from_pretrained_with_steering(base_model_name, steering_directory=steering_directory)
