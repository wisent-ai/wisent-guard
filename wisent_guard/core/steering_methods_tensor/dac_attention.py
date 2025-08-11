"""
Dynamic Activation Composition (DAC) steering method - Tensor-based Implementation.
This implementation uses tensor-based steering compatible with the original DAC format.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..contrastive_pairs import ContrastivePairSet
from .tensor_base import SteeringMethodTensor

logger = logging.getLogger(__name__)

# Model configuration constants (will be updated when model is loaded)
DEFAULT_MODEL_CONFIG = {
    "n_layers": 32,
    "n_heads": 32,
    "d_model": 4096,
    "d_head": 128,
}

# Default constants (can be overridden)
DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MAX_EXAMPLES = 20
DEFAULT_MAX_NEW_TOKENS = 30
DEFAULT_TORCH_DTYPE = torch.float16


class DAC(SteeringMethodTensor):
    """
    Dynamic Activation Composition (DAC) steering method.

    Uses tensor-based steering with shape [steps, n_layers, n_heads, d_head]
    instead of single vectors per layer. This matches the original DAC
    implementation behavior.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_examples: int = DEFAULT_MAX_EXAMPLES,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        torch_dtype=DEFAULT_TORCH_DTYPE,
    ):
        super().__init__("DAC", device)

        # Core DAC tensor storage
        self.steering_tensor = None  # [steps, n_layers, n_heads, d_head]
        self.property_tensors: Dict[str, torch.Tensor] = {}  # For multi-property

        # Model configuration
        self.model_name = model_name
        self.max_examples = max_examples
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self.model_config = DEFAULT_MODEL_CONFIG.copy()

        # Model and tokenizer (loaded lazily)
        self._model = None
        self._tokenizer = None

        # Training statistics
        self.training_stats = {}

    def _load_model(self):
        """Load model and tokenizer for activation extraction."""
        if self._model is not None and self._tokenizer is not None:
            return  # Already loaded

        logger.info(f"Loading model: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype, device_map="auto", trust_remote_code=True
        )

        # Update model configuration
        if hasattr(self._model, "config"):
            config = self._model.config
            self.model_config = {
                "n_layers": getattr(config, "num_hidden_layers", 32),
                "n_heads": getattr(config, "num_attention_heads", 32),
                "d_model": getattr(config, "hidden_size", 4096),
                "d_head": getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32),
            }

        logger.info(f"Model loaded on device: {next(self._model.parameters()).device}")

    def _split_activation(self, activations: List[torch.Tensor]) -> torch.Tensor:
        """
        Split the residual stream (d_model) into n_heads activations for each layer.

        Args:
            activations: List of residual streams for each layer [batch, seq, d_model]

        Returns:
            Reshaped activation in [batch, n_layers, n_heads, seq, d_head]
        """
        # Ensure all activations are on the same device
        target_device = torch.device(self.device)
        activations = [act.to(target_device) for act in activations]

        new_shape = torch.Size(
            [
                activations[0].shape[0],  # batch_size
                activations[0].shape[1],  # seq_len
                self.model_config["n_heads"],  # n_heads
                self.model_config["d_head"],  # d_head
            ]
        )

        attn_activations = torch.stack([act.view(*new_shape) for act in activations])
        # layers batch seq heads dhead -> batch layers heads seq dhead
        attn_activations = torch.einsum("lbshd -> blhsd", attn_activations)
        return attn_activations

    def _extract_step_by_step_activations(self, text: str) -> torch.Tensor:
        """
        Extract activations during step-by-step generation.

        Args:
            text: Input text to process

        Returns:
            Activations tensor [steps, n_layers, n_heads, d_head]
        """
        # Tokenize the full conversation
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(self.device)

        all_activations = []
        current_tokens = input_ids.clone()

        # Hook storage
        layer_activations = {}

        def create_hook(layer_idx):
            def hook_fn(module, input_tensors, output):
                # Store the input to the attention layer (before attention computation)
                del module, output  # Unused parameters
                layer_activations[layer_idx] = input_tensors[0].detach().to(self.device)  # [batch, seq, hidden_dim]

            return hook_fn

        # Register hooks for all layers
        hooks = []
        for layer_idx in range(self.model_config["n_layers"]):
            layer = self._model.model.layers[layer_idx]
            hook = layer.register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)

        try:
            for _ in range(self.max_new_tokens):
                layer_activations.clear()

                # Forward pass
                with torch.no_grad():
                    outputs = self._model(current_tokens)
                    logits = outputs.logits

                # Get activations from all layers
                step_layer_activations = []
                for layer_idx in range(self.model_config["n_layers"]):
                    if layer_idx in layer_activations:
                        step_layer_activations.append(layer_activations[layer_idx])
                    else:
                        # Fallback: create zero tensor if hook didn't fire
                        batch_size, seq_len = current_tokens.shape
                        zero_act = torch.zeros(
                            batch_size, seq_len, self.model_config["d_model"], device=self.device, dtype=logits.dtype
                        )
                        step_layer_activations.append(zero_act)

                # Split heads and extract last token: [batch, n_layers, n_heads, d_head]
                attn_activations = self._split_activation(step_layer_activations)
                last_token_activations = attn_activations[0, :, :, -1, :].detach().cpu()
                all_activations.append(last_token_activations)

                # Get next token
                next_token = torch.argmax(logits[0, -1, :], dim=-1)
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                # Stop if EOS token
                if next_token.item() == self._tokenizer.eos_token_id:
                    break

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Stack activations: [n_steps, n_layers, n_heads, d_head]
        all_activations = torch.stack(all_activations)

        return all_activations

    def _compute_mean_across_examples(self, all_activations: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute mean activations across examples and steps.

        Args:
            all_activations: List of activation tensors [steps, n_layers, n_heads, d_head]

        Returns:
            Mean activations tensor [steps, n_layers, n_heads, d_head]
        """
        # Compute mean across examples and steps
        # Handle different numbers of steps per example
        mean_activations = torch.zeros(
            self.max_new_tokens,
            self.model_config["n_layers"],
            self.model_config["n_heads"],
            self.model_config["d_head"],
        )

        for step in range(self.max_new_tokens):
            step_activations = []
            for example_activations in all_activations:
                if step < example_activations.shape[0]:
                    step_activations.append(example_activations[step, :, :, :])

            if step_activations:
                mean_activations[step, :, :, :] = torch.stack(step_activations).mean(dim=0)

        return mean_activations

    def _extract_mean_activations(self, contrastive_pairs: ContrastivePairSet, response_type: str) -> torch.Tensor:
        """
        Compute mean activations for a set of responses.

        Args:
            contrastive_pairs: ContrastivePairSet containing the pairs
            response_type: "positive" or "negative"

        Returns:
            Mean activations tensor [steps, n_layers, n_heads, d_head]
        """
        all_activations = []
        processed_count = 0

        logger.info(f"Computing mean activations for {response_type} responses...")

        for i, pair in enumerate(contrastive_pairs.pairs[: self.max_examples]):
            logger.info(f"Processing example {i + 1}/{min(len(contrastive_pairs.pairs), self.max_examples)}")

            # Get the appropriate response
            if response_type == "positive":
                response_text = pair.positive_response.text
                full_text = f"{pair.prompt}\n{response_text}"
            else:
                response_text = pair.negative_response.text
                full_text = f"{pair.prompt}\n{response_text}"

            # Extract activations
            activations = self._extract_step_by_step_activations(full_text)  # [steps, layers, heads, d_head]
            all_activations.append(activations)
            processed_count += 1

        logger.info(f"Processed {processed_count} examples for {response_type} responses")

        return self._compute_mean_across_examples(all_activations)

    def train_property(self, property_name: str, contrastive_pair_set: ContrastivePairSet) -> Dict[str, Any]:
        """
        Train a named property tensor.

        Args:
            property_name: Name for this property
            contrastive_pair_set: ContrastivePairSet with positive/negative responses

        Returns:
            Dictionary with training statistics
        """
        # Load model if needed
        self._load_model()

        logger.info(f"Training DAC property '{property_name}' with tensor-based approach...")
        start_time = time.time()

        # Extract activations
        pos_activations = self._extract_mean_activations(contrastive_pair_set, "positive")
        neg_activations = self._extract_mean_activations(contrastive_pair_set, "negative")

        # Compute property tensor
        property_tensor = pos_activations - neg_activations
        self.property_tensors[property_name] = property_tensor

        # Also set as main tensor if this is the first/default property
        if not self.is_trained:
            self.steering_tensor = property_tensor
            self.is_trained = True

        # Training statistics
        elapsed_time = time.time() - start_time
        training_stats = {
            "method": "DAC",
            "property": property_name,
            "tensor_shape": list(property_tensor.shape),
            "tensor_norm": torch.norm(property_tensor).item(),
            "pos_norm": torch.norm(pos_activations).item(),
            "neg_norm": torch.norm(neg_activations).item(),
            "num_pairs": len(contrastive_pair_set.pairs),
            "training_time": elapsed_time,
            "success": True,
        }

        # Store stats
        self.training_stats = training_stats

        logger.info(f"DAC property '{property_name}' training completed in {elapsed_time:.1f} seconds")
        logger.info(f"Property tensor shape: {property_tensor.shape}")
        logger.info(f"Property tensor norm: {training_stats['tensor_norm']:.4f}")

        return training_stats

    def get_steering_tensor(self) -> torch.Tensor:
        """
        Return the full DAC steering tensor [steps, n_layers, n_heads, d_head].

        Returns:
            Steering tensor in DAC format
        """
        if not self.is_trained:
            raise ValueError("DAC must be trained before getting steering tensor")
        return self.steering_tensor

    def get_property_tensor(self, property_name: str) -> torch.Tensor:
        """
        Get a specific property tensor.

        Args:
            property_name: Name of the property

        Returns:
            Property tensor [steps, n_layers, n_heads, d_head]
        """
        if property_name not in self.property_tensors:
            raise ValueError(f"Property '{property_name}' not found. Available: {list(self.property_tensors.keys())}")
        return self.property_tensors[property_name]

    def apply_steering_tensor(
        self,
        activations: torch.Tensor,
        strength: float = 1.0,
        layer_index: Optional[int] = None,
        step_index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply tensor-based steering to activations.

        Args:
            activations: Input activations to steer
            strength: Steering strength multiplier
            layer_index: Specific layer to apply steering to
            step_index: Specific step to use from tensor

        Returns:
            Steered activations
        """
        if not self.is_trained:
            raise ValueError("DAC must be trained before applying steering")

        # Extract appropriate slice from steering tensor based on layer/step
        if layer_index is not None and step_index is not None:
            # Apply specific layer and step
            steering_slice = self.steering_tensor[step_index, layer_index, :, :]  # [n_heads, d_head]
            return activations + strength * steering_slice
        elif layer_index is not None:
            # Apply specific layer (average across steps)
            steering_slice = self.steering_tensor[:, layer_index, :, :].mean(dim=0)  # [n_heads, d_head]
            return activations + strength * steering_slice
        else:
            # Default behavior - need to specify how to apply full tensor
            raise ValueError("Must specify layer_index for tensor application")

    def save_steering_tensor(self, path: str) -> bool:
        """
        Save DAC steering tensor to file.

        Args:
            path: Path to save the tensor data

        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained:
            return False

        try:
            save_data = {
                "method": "DAC",
                "steering_tensor": self.steering_tensor,
                "property_tensors": self.property_tensors,
                "tensor_shape": list(self.steering_tensor.shape),
                "training_stats": self.training_stats,
                "model_config": self.model_config,
                "model_name": self.model_name,
                "max_examples": self.max_examples,
                "max_new_tokens": self.max_new_tokens,
            }
            torch.save(save_data, path)
            logger.info(f"DAC steering tensor saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving DAC tensor: {e}")
            return False

    def load_steering_tensor(self, path: str) -> bool:
        """
        Load DAC steering tensor from file.

        Args:
            path: Path to load the tensor data from

        Returns:
            True if successful, False otherwise
        """
        try:
            data = torch.load(path, map_location=self.device)
            if data.get("method") != "DAC":
                return False

            self.steering_tensor = data["steering_tensor"]
            self.property_tensors = data.get("property_tensors", {})
            self.training_stats = data.get("training_stats", {})
            self.model_config = data.get("model_config", DEFAULT_MODEL_CONFIG.copy())

            # Update configuration if provided
            if "model_name" in data:
                self.model_name = data["model_name"]
            if "max_examples" in data:
                self.max_examples = data["max_examples"]
            if "max_new_tokens" in data:
                self.max_new_tokens = data["max_new_tokens"]

            self.is_trained = True
            logger.info(f"DAC steering tensor loaded from {path}")
            logger.info(f"Loaded tensor shape: {self.steering_tensor.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading DAC tensor: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the DAC method."""
        if not self.is_trained:
            return {"is_trained": False}

        stats = {
            "is_trained": True,
            "method": "DAC",
            "steering_tensor_shape": list(self.steering_tensor.shape),
            "steering_tensor_norm": torch.norm(self.steering_tensor).item(),
            "num_properties": len(self.property_tensors),
            "property_names": list(self.property_tensors.keys()),
            "model_config": self.model_config,
            "training_stats": self.training_stats,
        }

        return stats
