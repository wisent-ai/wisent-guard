"""
Wisent-enhanced Qwen2 model with integrated CAA (Contrastive Activation Addition) steering.

This model automatically applies CAA steering during generation without requiring manual hooks.
The steering parameters are optimized using Optuna and stored in the model configuration.
"""

from typing import List, Optional, Tuple, Union

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class WisentQwen2Config(Qwen2Config):
    """Extended Qwen2 configuration with CAA steering parameters."""

    model_type = "wisent_qwen2"

    def __init__(
        self,
        caa_enabled: bool = True,
        caa_layer_id: int = 24,
        caa_alpha: float = 0.9,
        steering_vector_path: str = "./vectors/coding/steering_vector.safetensors",
        steering_method: str = "caa",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.caa_enabled = caa_enabled
        self.caa_layer_id = caa_layer_id
        self.caa_alpha = caa_alpha
        self.steering_vector_path = steering_vector_path
        self.steering_method = steering_method


class WisentQwen2ForCausalLM(Qwen2ForCausalLM):
    """
    Qwen2 model with integrated CAA steering for improved code generation.

    This model automatically applies Contrastive Activation Addition (CAA) steering
    during the forward pass, eliminating the need for manual hook management.
    """

    config_class = WisentQwen2Config

    def __init__(self, config: WisentQwen2Config):
        super().__init__(config)

        # CAA steering parameters
        self.caa_enabled = config.caa_enabled
        self.caa_layer_id = config.caa_layer_id
        self.caa_alpha = config.caa_alpha
        self.steering_method = config.steering_method

        # Steering vector will be loaded in from_pretrained method
        self.steering_vector = None

        # Hook handle for cleanup
        self._steering_hook_handle = None

    def _load_steering_vector(self, model_path: str, vector_path: str):
        """Load the CAA steering vector from safetensors or pytorch file.

        Args:
            model_path: Path to the model directory (for relative path resolution)
            vector_path: Path to the steering vector file (from config)

        Raises:
            FileNotFoundError: If steering vector cannot be found
            ImportError: If safetensors is required but not installed
        """
        import os

        # Try model-relative path first (most reliable)
        model_relative_path = os.path.join(model_path, vector_path)

        # Try absolute path if vector_path is already absolute
        absolute_path = vector_path if os.path.isabs(vector_path) else None

        # Determine which path exists
        if os.path.exists(model_relative_path):
            final_path = model_relative_path
        elif absolute_path and os.path.exists(absolute_path):
            final_path = absolute_path
        else:
            # Fail fast with clear error message
            error_msg = "CAA is enabled but steering vector not found. Tried:\n"
            error_msg += f"  1. Model-relative: {model_relative_path}\n"
            if absolute_path:
                error_msg += f"  2. Absolute: {absolute_path}\n"
            error_msg += "Ensure steering vector exists at one of these locations."
            raise FileNotFoundError(error_msg)

        # Load the steering vector
        if final_path.endswith(".safetensors"):
            # Load from safetensors format (preferred)
            try:
                from safetensors.torch import load_file

                steering_data = load_file(final_path)
                self.steering_vector = steering_data["steering_vector"]
            except ImportError:
                raise ImportError(
                    "safetensors is required for .safetensors files. Install with: pip install safetensors"
                )
        else:
            # Load from pytorch format (fallback)
            steering_data = torch.load(final_path, map_location="cpu")

            # Handle different storage formats
            if isinstance(steering_data, dict):
                if "vector" in steering_data:
                    self.steering_vector = steering_data["vector"]
                elif "steering_vector" in steering_data:
                    self.steering_vector = steering_data["steering_vector"]
                else:
                    # Assume the dict values are the vectors
                    self.steering_vector = next(iter(steering_data.values()))
            else:
                self.steering_vector = steering_data

        # Ensure it's a tensor
        if not isinstance(self.steering_vector, torch.Tensor):
            self.steering_vector = torch.tensor(self.steering_vector)

        print(
            f"✅ Loaded CAA steering vector from {final_path}: shape {self.steering_vector.shape}, norm {torch.norm(self.steering_vector).item():.4f}"
        )

    def _apply_caa_steering(self, module, input, output):
        """
        Hook function that applies CAA steering to the specified layer.

        This follows the implementation from wisent_guard/core/steering_methods/caa.py
        and the patterns from wisent_guard/core/optuna/optuna_pipeline.py
        """
        if not self.caa_enabled or self.steering_vector is None:
            return output

        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Apply steering to the last token position (standard CAA behavior)
        # This matches the implementation in optuna_pipeline.py lines 744-746
        if hidden_states.dim() == 3:  # [batch, seq, hidden]
            # Move steering vector to the same device and dtype
            steering_vector = self.steering_vector.to(hidden_states.device, hidden_states.dtype)

            # Apply steering with configured alpha (strength)
            # Steering is applied to the last token position
            hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + self.caa_alpha * steering_vector.unsqueeze(
                0
            ).unsqueeze(0)

        # Return modified output
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with automatic CAA steering application.

        The steering is applied via a forward hook on the specified layer,
        following the pattern from optuna_pipeline.py.
        """

        # Register CAA steering hook if enabled and not already registered
        if self.caa_enabled and self.steering_vector is not None and self._steering_hook_handle is None:
            target_layer = self.model.layers[self.caa_layer_id]
            self._steering_hook_handle = target_layer.register_forward_hook(self._apply_caa_steering)

        # Call parent forward method
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position if hasattr(self, "cache_position") else None,
        )

        return outputs

    def generate(self, *args, **kwargs):
        """
        Generate method with automatic CAA steering.

        The steering hook is registered before generation and cleaned up after.
        """
        # Register hook if needed
        if self.caa_enabled and self.steering_vector is not None and self._steering_hook_handle is None:
            target_layer = self.model.layers[self.caa_layer_id]
            self._steering_hook_handle = target_layer.register_forward_hook(self._apply_caa_steering)

        try:
            # Call parent generate method
            outputs = super().generate(*args, **kwargs)
        finally:
            # Clean up hook after generation
            if self._steering_hook_handle is not None:
                self._steering_hook_handle.remove()
                self._steering_hook_handle = None

        return outputs

    def set_caa_enabled(self, enabled: bool):
        """Enable or disable CAA steering at runtime."""
        self.caa_enabled = enabled
        if not enabled and self._steering_hook_handle is not None:
            self._steering_hook_handle.remove()
            self._steering_hook_handle = None

    def set_caa_alpha(self, alpha: float):
        """Adjust CAA steering strength at runtime."""
        self.caa_alpha = alpha

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load model with automatic CAA configuration.

        This method ensures the steering vector is loaded from the embedded config.
        If no weights are found locally, it loads from the base Qwen model.
        """
        from pathlib import Path

        # Check if we have local weights
        local_path = Path(pretrained_model_name_or_path)
        has_weights = any(
            (local_path / f).exists()
            for f in [
                "pytorch_model.bin",
                "model.safetensors",
                "pytorch_model.bin.index.json",
                "model.safetensors.index.json",
            ]
        )

        if not has_weights and local_path.exists() and (local_path / "config.json").exists():
            # We have config but no weights - load from base model
            print("Loading weights from base model: Qwen/Qwen2.5-Coder-7B-Instruct")

            # First, load config from local path
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

            # Load model with base weights
            # Remove config from kwargs if it exists to avoid conflict
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop("config", None)

            model = super().from_pretrained(
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                *model_args,
                config=config,  # Use our custom config
                **kwargs_copy,
            )

            # Initialize CAA components
            model.caa_enabled = config.caa_enabled
            model.caa_layer_id = config.caa_layer_id
            model.caa_alpha = config.caa_alpha
            model.steering_method = config.steering_method
            model._steering_hook_handle = None

            # Load steering vector from config (fail fast if CAA enabled but vector missing)
            if model.caa_enabled:
                model._load_steering_vector(pretrained_model_name_or_path, config.steering_vector_path)
        else:
            # Standard loading path
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

            # Load steering vector from config (fail fast if CAA enabled but vector missing)
            if model.caa_enabled and model.steering_vector is None:
                model._load_steering_vector(pretrained_model_name_or_path, model.config.steering_vector_path)

        return model


# Register the model
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("wisent_qwen2", WisentQwen2Config)
AutoModelForCausalLM.register(WisentQwen2Config, WisentQwen2ForCausalLM)
