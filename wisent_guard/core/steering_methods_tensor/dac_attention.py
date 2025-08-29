"""
Dynamic Activation Composition (DAC) steering method - Tensor-based Implementation.
This implementation uses tensor-based steering compatible with the original DAC format.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    and dynamic steering by default. Features:

    - Dynamic steering with KL divergence computation and adaptive alpha values
    - Multi-property composition with historical tracking
    - Real-time adaptation based on steering impact
    - Multiple forward passes per token for precise control

    For static tensor steering, explicitly set timing_strategy="normal".
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_examples: int = DEFAULT_MAX_EXAMPLES,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        torch_dtype=DEFAULT_TORCH_DTYPE,
        icl_examples: int = 4,
        legacy_behavior: bool = False,  # Legacy: Q:/A: tokenization format used in original implementation (True) vs modern chat templates (False)
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
        self.icl_examples = icl_examples
        self.legacy_behavior = legacy_behavior
        self.model_config = DEFAULT_MODEL_CONFIG.copy()

        # Model and tokenizer (loaded lazily)
        self._model = None
        self._tokenizer = None

        # Training statistics
        self.training_stats = {}
        
        # For compatibility with multi-steer command
        self.property_vectors = {}

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

    def load_model_with_reference(self, model, tokenizer):
        """
        Inject external model and tokenizer references.

        This allows the Optuna pipeline to provide already-loaded models
        instead of DAC loading them internally, avoiding duplicate model loading.

        Args:
            model: Pre-loaded language model
            tokenizer: Pre-loaded tokenizer
        """
        self._model = model
        self._tokenizer = tokenizer

        # Update model configuration from the injected model
        if hasattr(model, "config"):
            config = model.config
            self.model_config = {
                "n_layers": getattr(config, "num_hidden_layers", 32),
                "n_heads": getattr(config, "num_attention_heads", 32),
                "d_model": getattr(config, "hidden_size", 4096),
                "d_head": getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32),
            }

        logger.info(f"Model reference injected: {type(model).__name__}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model config updated: {self.model_config}")

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

    def _extract_step_by_step_activations(self, prompt_tokens: torch.Tensor, target_answer: str) -> torch.Tensor:
        """
        Extract activations during step-by-step generation using original DAC approach.

        Args:
            prompt_tokens: Tokenized prompt ending at '\nA:' (from original DAC tokenize_from_template)
            target_answer: The expected answer text for generation

        Returns:
            Activations tensor [steps, n_layers, n_heads, d_head]
        """
        # Tokenize the target answer to know when to stop
        target_tokens = self._tokenizer(target_answer, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].squeeze()
        if target_tokens.dim() == 0:
            target_tokens = target_tokens.unsqueeze(0)

        logger.debug(f"Prompt tokens shape: {prompt_tokens.shape}")
        logger.debug(f"Target answer: '{target_answer}' -> {len(target_tokens)} tokens")

        all_activations = []
        current_tokens = prompt_tokens.unsqueeze(0).to(self.device)  # Add batch dimension

        # Hook storage
        layer_activations = {}

        def create_hook(layer_idx):
            def hook_fn(module, input_tensors, output):
                # Store the input to the attention output projection (matching original DAC)
                del module, output  # Unused parameters
                layer_activations[layer_idx] = input_tensors[0].detach().to(self.device)  # [batch, seq, hidden_dim]

            return hook_fn

        # Register hooks for self_attn.o_proj layers (matching original DAC implementation)
        hooks = []
        for layer_idx in range(self.model_config["n_layers"]):
            # Hook the attention output projection, not the entire layer
            attn_o_proj = self._model.model.layers[layer_idx].self_attn.o_proj
            hook = attn_o_proj.register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)

        try:
            tokens_generated = 0
            for step in range(self.max_new_tokens):
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
                tokens_generated += 1

                # Stop if EOS token
                if next_token.item() == self._tokenizer.eos_token_id:
                    logger.debug(f"Generation stopped at EOS token after {tokens_generated} tokens")
                    break

                # NOTE: Original DAC always generates exactly max_new_tokens
                # It does NOT stop early when target answer is reached
                # This ensures consistent tensor shapes across different answer lengths

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Stack activations: [n_steps, n_layers, n_heads, d_head]
        all_activations = torch.stack(all_activations)

        logger.debug(f"Extracted activations shape: {all_activations.shape}")
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

    def _build_prompt_txt(
        self,
        queries: list[str],
        answers: list[str],
        pre_append_instruction: str | None = None,
    ):
        """
        Gratefully taken from original DAC implementation.
        Build the prompt following the default template. Provide a list of queries (length = n ICL examples)
        and a list of answers (length = n ICL examples)

        Args:
            queries (list[str]): queries (ICL examples + final query).
            answers (list[str]): answers (ICL examples), must be one less than queries.
            pre_append_instruction (str | None): Optional instruction at the beginning of each prompt. Defaults to None.

        Returns:
            full_prompt: full prompt following the default template with notation
        """
        assert len(queries) == len(answers) + 1, (
            f"queries (len={len(queries)}) should have one more element than answers (len={len(answers)})"
        )

        full_prompt = []
        if pre_append_instruction:
            full_prompt.append((pre_append_instruction, "sentence"))
            full_prompt.append(("\n", "structural"))

        # prompt default template for structural parts
        begin = [
            ("Q:", "structural"),
        ]
        middle = [
            ("\nA:", "structural"),
        ]
        end = [
            ("\n\n", "structural"),
        ]

        # Use original DAC format (buggy) or corrected format based on parameter
        loop_range = len(answers) - 1 if self.legacy_behavior else len(answers)
        for i in range(loop_range):
            full_prompt.extend(begin)
            full_prompt.append((queries[i], "sentence"))
            full_prompt.extend(middle)
            full_prompt.append((answers[i], "sentence"))
            full_prompt.extend(end)
        # append the final query without the answer
        full_prompt.extend(begin)
        full_prompt.append((queries[-1], "sentence"))
        full_prompt.extend(middle)

        return full_prompt

    @staticmethod
    def _tokenize_from_template(tokenizer, promtp_w_template: tuple[tuple[str, str]]):
        """
        Gratefully taken from original DAC implementation.
        tokenize the prompt following the provided template and return a list of indexes referring to the structural toknes
        and only the last token of sentences (automatically include the bos token at the beginning)

        Args:
            tokenizer (*Tokenizer): huggingface tokenizer
            promtp_w_template (tuple[tuple[str, str]], optional): prompt_template in the form of:
            ```python
                TODO
            ```. Defaults to None.

        Returns:
            torch.LongTensor: tokenized input ids
            list: index of every structural token and last sentence token
        """

        full_tokenized = torch.LongTensor([[tokenizer.bos_token_id]])  # manually adding BOS token
        indexes = [0]

        for prompt_str, prompt_type in promtp_w_template:
            tokenized = tokenizer(
                prompt_str,
                return_tensors="pt",
                # add_special_tokens=False
            ).input_ids.type(torch.int64)
            full_tokenized = torch.cat(
                (full_tokenized, tokenized),
                dim=-1,
            )

            if prompt_type == "structural":
                # all the index of structural tokens must be included
                actual_idxs = list(range(indexes[-1] + 1, tokenized.shape[-1] + indexes[-1] + 1))
                indexes.extend(actual_idxs)
            elif prompt_type == "sentence":
                # include only the last index of the sentence tokens
                indexes.append(indexes[-1] + tokenized.shape[-1])

        full_tokenized = full_tokenized.squeeze()
        return full_tokenized, indexes

    def _build_icl_prompt_chat_template(
        self, contrastive_pairs: ContrastivePairSet, current_pair_idx: int, response_type: str
    ) -> tuple:
        """
        Build an ICL prompt using the tokenizer's chat template for better alignment with generation.

        Args:
            contrastive_pairs: ContrastivePairSet containing all pairs
            current_pair_idx: Index of the current pair to use as the final prompt
            response_type: "positive" or "negative" - determines which responses to use in ICL

        Returns:
            Tuple of (prompt_tokens, target_answer_text) where:
            - prompt_tokens: Tokenized prompt ready for generation
            - target_answer_text: The expected answer text for generation
        """
        logger.debug(
            f"Building chat template: pairs={len(contrastive_pairs.pairs)}, icl={self.icl_examples}, idx={current_pair_idx}"
        )

        # Build conversation messages for ICL examples
        messages = []

        # Add ICL examples as conversation history
        for offset in range(self.icl_examples):
            icl_idx = current_pair_idx + offset
            if icl_idx < len(contrastive_pairs.pairs):
                icl_pair = contrastive_pairs.pairs[icl_idx]

                # Add user message
                messages.append({"role": "user", "content": icl_pair.prompt})

                # Add assistant response
                if response_type == "positive":
                    messages.append({"role": "assistant", "content": icl_pair.positive_response.text})
                else:
                    messages.append({"role": "assistant", "content": icl_pair.negative_response.text})

        # Add the final query (target question)
        target_pair = (
            contrastive_pairs.pairs[current_pair_idx + self.icl_examples]
            if current_pair_idx + self.icl_examples < len(contrastive_pairs.pairs)
            else contrastive_pairs.pairs[current_pair_idx]
        )
        messages.append({"role": "user", "content": target_pair.prompt})

        # Get target answer
        if response_type == "positive":
            target_answer = target_pair.positive_response.text
        else:
            target_answer = target_pair.negative_response.text

        logger.debug(f"Target answer ({response_type}): {target_answer[:50]}...")
        logger.debug(f"Messages before applying template: {messages}")

        # For training, we may want to include the target answer in the prompt
        # This depends on whether we're doing ICL=0 training
        try:
            if self.icl_examples == 0:
                # For ICL=0, include the answer in the messages for training signal
                messages.append({"role": "assistant", "content": target_answer})
                logger.debug("Added target answer to messages (ICL=0)")

                # Apply chat template with generation prompt disabled (since we have the answer)
                prompt_text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                # For ICL>0, stop at the generation prompt
                prompt_text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            logger.debug(f"Chat template result type: {type(prompt_text)}")
            logger.debug(
                f"Chat template result preview: {prompt_text[:100] if isinstance(prompt_text, str) else prompt_text}"
            )

        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            logger.error(f"Messages: {messages}")
            logger.error(f"Tokenizer type: {type(self._tokenizer)}")
            raise

        # Tokenize the prompt
        # Ensure prompt_text is a string
        if not isinstance(prompt_text, str):
            logger.warning(f"prompt_text is not a string: type={type(prompt_text)}, value={prompt_text}")
            # Try to convert to string if possible
            prompt_text = str(prompt_text)

        prompt_tokens = self._tokenizer(prompt_text, return_tensors="pt")["input_ids"].squeeze()
        # Ensure 1D tensor
        if prompt_tokens.dim() == 0:
            prompt_tokens = prompt_tokens.unsqueeze(0)

        logger.debug(f"Built chat template ICL prompt: {len(messages)} messages")
        logger.debug(f"Tokenized prompt length: {prompt_tokens.shape[0]}")
        logger.debug(f"Target answer: {target_answer[:50] if len(target_answer) > 50 else target_answer}...")

        return prompt_tokens, target_answer

    def _build_icl_prompt(
        self, contrastive_pairs: ContrastivePairSet, current_pair_idx: int, response_type: str
    ) -> tuple:
        """
        Build an ICL prompt using either chat templates or original DAC format.

        This method chooses between modern chat templates (for better alignment with generation)
        or the original DAC Q:/A: format (for backward compatibility).

        Args:
            contrastive_pairs: ContrastivePairSet containing all pairs
            current_pair_idx: Index of the current pair to use as the final prompt
            response_type: "positive" or "negative" - determines which responses to use in ICL

        Returns:
            Tuple of (prompt_tokens, target_answer_text) where:
            - prompt_tokens: Tokenized prompt
            - target_answer_text: The expected answer text for generation
        """
        # Use chat templates for better alignment unless original format is requested
        logger.debug(f"Choosing prompt format: legacy_behavior={self.legacy_behavior}")
        logger.debug(f"Tokenizer has chat_template: {hasattr(self._tokenizer, 'chat_template')}")
        logger.debug(f"Chat template exists: {getattr(self._tokenizer, 'chat_template', None) is not None}")

        if not self.legacy_behavior and hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            logger.debug("Using chat template format")
            return self._build_icl_prompt_chat_template(contrastive_pairs, current_pair_idx, response_type)
        logger.debug("Using Q:/A: format")

        # Otherwise use original DAC format for backward compatibility

        # Build queries and answers lists like original DAC
        queries = []
        answers = []

        # Collect ICL examples from the group
        for offset in range(self.icl_examples):
            icl_idx = current_pair_idx + offset
            if icl_idx < len(contrastive_pairs.pairs):
                icl_pair = contrastive_pairs.pairs[icl_idx]
                queries.append(icl_pair.prompt)

                if response_type == "positive":
                    answers.append(icl_pair.positive_response.text)
                else:
                    answers.append(icl_pair.negative_response.text)

        # Add the final query (target question)
        target_pair = (
            contrastive_pairs.pairs[current_pair_idx + self.icl_examples]
            if current_pair_idx + self.icl_examples < len(contrastive_pairs.pairs)
            else contrastive_pairs.pairs[current_pair_idx]
        )
        queries.append(target_pair.prompt)

        # Get target answer
        if response_type == "positive":
            target_answer = target_pair.positive_response.text
        else:
            target_answer = target_pair.negative_response.text

        # Build template using original DAC logic
        template = self._build_prompt_txt(
            queries=queries,
            answers=answers,  # This excludes the target answer - prompt ends at '\nA:'
            pre_append_instruction=None,
        )

        # For ICL=0 with corrected format, append target answer for training signal
        if self.icl_examples == 0 and not self.legacy_behavior:
            template.append((target_answer, "sentence"))
            template.append(("\n\n", "structural"))

        # Tokenize using original DAC method
        prompt_tokens, important_ids = self._tokenize_from_template(self._tokenizer, template)

        logger.debug(f"Built ICL prompt: queries={len(queries)}, answers={len(answers)}")
        logger.debug(f"Tokenized prompt length: {len(prompt_tokens)}")
        logger.debug(f"Target answer: {target_answer[:50]}...")

        # Debug save removed - tokenization issue fixed

        return prompt_tokens, target_answer

    def _extract_mean_activations(self, contrastive_pairs: ContrastivePairSet, response_type: str) -> torch.Tensor:
        """
        Compute mean activations for a set of responses using original DAC example selection strategy.

        This matches the original DAC's grouped example selection:
        - For ICL=0: processes examples 0, 1, 2, 3, ... individually
        - For ICL=4: processes groups [0-4], [5-9], [10-14], ... with each group having unique ICL context

        Args:
            contrastive_pairs: ContrastivePairSet containing the pairs
            response_type: "positive" or "negative"

        Returns:
            Mean activations tensor [steps, n_layers, n_heads, d_head]
        """
        all_activations = []
        processed_count = 0

        logger.info(f"Computing mean activations for {response_type} responses...")
        logger.info(f"Using original DAC example selection strategy with ICL={self.icl_examples}")

        # Use exact same random selection as original DAC (seed=32)
        # Original: random_indexes = [0, 1, 1, 2] from 4 available groups
        step_size = self.icl_examples + 1
        num_groups = len(contrastive_pairs.pairs) // step_size
        original_random_indexes = [0, 1, 1, 2]  # Exact reproduction of original DAC

        logger.info(f"Using original DAC random selection: {original_random_indexes}")
        logger.info(f"Available groups: {num_groups}, selecting: {len(original_random_indexes)} examples")

        for group_idx in original_random_indexes:
            # Stop if we've processed enough examples
            if processed_count >= self.max_examples:
                break

            if group_idx < num_groups:
                i = group_idx * step_size
                logger.info(
                    f"Processing group {group_idx} starting at index {i} (example {processed_count + 1}/{len(original_random_indexes)})"
                )

                # Build ICL prompt for this group using original DAC approach
                prompt_tokens, target_answer = self._build_icl_prompt(contrastive_pairs, i, response_type)

                # Extract activations using the new format
                activations = self._extract_step_by_step_activations(
                    prompt_tokens, target_answer
                )  # [steps, layers, heads, d_head]
                all_activations.append(activations)
                processed_count += 1
            else:
                # Not enough examples for a complete group, skip
                logger.debug(f"Skipping incomplete group {group_idx} (not enough examples)")

        logger.info(f"Processed {processed_count} examples for {response_type} responses using grouped selection")

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
        if layer_index is not None:
            # Apply specific layer (average across steps)
            steering_slice = self.steering_tensor[:, layer_index, :, :].mean(dim=0)  # [n_heads, d_head]
            return activations + strength * steering_slice
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

    def compose_properties(self, property_weights: Dict[str, float]) -> torch.Tensor:
        """
        Compose multiple property tensors using linear combination.

        Args:
            property_weights: Dictionary mapping property names to their weights
                            e.g., {"language_steering": 1.5, "safety_steering": 2.0}

        Returns:
            Composed tensor [steps, n_layers, n_heads, d_head]
        """
        if not self.property_tensors:
            raise ValueError("No property tensors available. Train properties first.")

        # Validate all properties exist
        for prop_name in property_weights:
            if prop_name not in self.property_tensors:
                available = list(self.property_tensors.keys())
                raise ValueError(f"Property '{prop_name}' not found. Available: {available}")

        # Initialize composed tensor with zeros
        if self.property_tensors:
            first_tensor = next(iter(self.property_tensors.values()))
            composed_tensor = torch.zeros_like(first_tensor)
        else:
            raise ValueError("No property tensors to compose")

        # Linear combination: sum(weight_i * property_tensor_i)
        total_weight = 0.0
        for prop_name, weight in property_weights.items():
            composed_tensor += weight * self.property_tensors[prop_name]
            total_weight += abs(weight)

        logger.info(f"Composed {len(property_weights)} properties with total weight: {total_weight:.2f}")
        return composed_tensor

    def get_composed_tensor(
        self, property_weights: Dict[str, float], composition_strategy: str = "linear"
    ) -> torch.Tensor:
        """
        Get a composed steering tensor using specified strategy.

        Args:
            property_weights: Weights for each property
            composition_strategy: Strategy to use ("linear", "normalized", "dynamic")

        Returns:
            Composed steering tensor
        """
        if composition_strategy == "linear":
            return self.compose_properties(property_weights)
        if composition_strategy == "normalized":
            # Normalize weights to sum to 1.0
            total_weight = sum(abs(w) for w in property_weights.values())
            if total_weight == 0:
                raise ValueError("Sum of absolute weights cannot be zero")
            normalized_weights = {k: v / total_weight for k, v in property_weights.items()}
            return self.compose_properties(normalized_weights)
        if composition_strategy == "dynamic":
            # For now, just use linear - dynamic will be implemented later
            logger.warning("Dynamic composition not yet implemented, using linear")
            return self.compose_properties(property_weights)
        raise ValueError(f"Unknown composition strategy: {composition_strategy}")

    def generate_with_steering(
        self,
        prompt: str,
        property_weights: Optional[Dict[str, float]] = None,
        max_new_tokens: Optional[int] = None,
        steering_strength: float = 1.0,
        composition_strategy: str = "linear",
        timing_strategy: str = "dynamic",
        dynamic_config: Optional[Dict] = None,
    ) -> str:
        """
        Generate text with DAC dynamic steering applied.

        Args:
            prompt: Input text prompt
            property_weights: Weights for multi-property composition.
                            If None, uses main steering tensor.
            max_new_tokens: Maximum tokens to generate (defaults to class setting)
            steering_strength: Overall strength multiplier for steering
            composition_strategy: How to combine properties ("linear", "normalized")
            timing_strategy: When to apply steering ("dynamic" default, "normal", "start_only", "diminishing")
            dynamic_config: Configuration for dynamic steering (used by default with "dynamic")

        Returns:
            Generated text with dynamic steering applied

        Note:
            Default behavior uses dynamic steering with KL divergence computation and adaptive alpha values.
            For static steering, explicitly set timing_strategy="normal".
        """
        if not self.is_trained:
            raise ValueError("DAC must be trained before generating with steering")

        # Load model if needed
        self._load_model()

        # Use class defaults if not specified
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Determine which tensor to use (for non-dynamic strategies)
        if timing_strategy != "dynamic":
            if property_weights is not None:
                steering_tensor = self.get_composed_tensor(property_weights, composition_strategy)
            else:
                steering_tensor = self.steering_tensor
        else:
            # Dynamic steering will handle tensor composition internally
            steering_tensor = None

        logger.info(
            f"Generating with DAC steering (strategy={timing_strategy}, strength={steering_strength}, max_tokens={max_new_tokens})"
        )
        if property_weights:
            logger.info(f"Using multi-property composition: {property_weights}")

        return self._generate_with_tensor_steering(
            prompt=prompt,
            steering_tensor=steering_tensor,
            max_new_tokens=max_new_tokens,
            steering_strength=steering_strength,
            timing_strategy=timing_strategy,
            property_weights=property_weights,
            dynamic_config=dynamic_config,
        )

    def _generate_with_tensor_steering(
        self,
        prompt: str,
        steering_tensor: torch.Tensor,
        max_new_tokens: int,
        steering_strength: float = 1.0,
        timing_strategy: str = "normal",
        property_weights: Optional[Dict[str, float]] = None,
        dynamic_config: Optional[Dict] = None,
    ) -> str:
        """
        Internal method to generate text with tensor steering using hooks.

        Args:
            prompt: Input text prompt
            steering_tensor: Tensor to use for steering [steps, layers, heads, d_head]
            max_new_tokens: Maximum tokens to generate
            steering_strength: Strength multiplier
            timing_strategy: When to apply steering ("normal", "start_only", "diminishing", "dynamic")
            property_weights: For dynamic steering - weights for each property
            dynamic_config: Configuration for dynamic steering

        Returns:
            Generated text with steering applied
        """
        # Tokenize the prompt
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)

        if timing_strategy == "dynamic":
            # Use dynamic steering with multi-property composition
            if property_weights is None:
                raise ValueError("Dynamic steering requires property_weights to be specified")

            return self._generate_step_by_step_with_dynamic_steering(
                input_ids=input_ids,
                property_weights=property_weights,
                max_new_tokens=max_new_tokens,
                dynamic_config=dynamic_config or {},
            )[0]  # Return just the text, not the statistics
        # Use standard hook-based generation
        return self._generate_step_by_step_with_hooks(
            input_ids=input_ids,
            steering_tensor=steering_tensor,
            max_new_tokens=max_new_tokens,
            steering_strength=steering_strength,
            timing_strategy=timing_strategy,
        )

    def _generate_step_by_step_with_dynamic_steering(
        self,
        input_ids: torch.Tensor,
        property_weights: Dict[str, float],
        max_new_tokens: int,
        dynamic_config: Dict = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text step-by-step with dynamic multi-property steering.

        This implements the main dynamic generation loop from the original DAC.

        Args:
            input_ids: Tokenized input [1, seq_len]
            property_weights: Weights for each property (used as starting weights)
            max_new_tokens: Maximum new tokens
            dynamic_config: Configuration for dynamic steering

        Returns:
            Tuple of (generated_text, dynamic_statistics)
        """
        config = {"starting_alpha": 2.0, "top_p": 0.95, "alpha_bounds": [0.0, 2.0], **(dynamic_config or {})}

        initial_length = input_ids.shape[1]
        current_ids = input_ids.clone()

        # Prepare property tensors based on weights
        property_tensors = {}
        for prop_name in property_weights:
            if prop_name not in self.property_tensors:
                available = list(self.property_tensors.keys())
                raise ValueError(f"Property '{prop_name}' not found. Available: {available}")
            property_tensors[prop_name] = self.property_tensors[prop_name]

        # Track dynamic statistics
        alpha_history = {prop_name: [] for prop_name in property_weights}
        kl_history = {prop_name: [] for prop_name in property_weights}

        logger.debug(f"Starting dynamic steering generation with {len(property_tensors)} properties")
        logger.debug(f"Dynamic config: {config}")

        for step in range(max_new_tokens):
            # Generate next token with dynamic steering
            next_token, step_alphas = self._generate_next_token_with_dynamic_steering(
                current_ids=current_ids,
                property_tensors=property_tensors,
                step=step,
                initial_length=initial_length,
                alpha_history=alpha_history,
                starting_alpha=config["starting_alpha"],
                top_p=config["top_p"],
            )

            # Update alpha history
            for prop_name, alpha in step_alphas.items():
                alpha_history[prop_name].append(alpha)

            # Check stopping conditions
            if next_token.item() == self._tokenizer.eos_token_id:
                break

            # Add token to sequence
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        # Decode generated text
        generated_ids = current_ids[0, initial_length:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compile statistics
        dynamic_stats = {
            "alpha_history": alpha_history,
            "kl_history": kl_history,  # Will be empty for now, can add KL tracking later
            "properties_used": list(property_weights.keys()),
            "starting_alpha": config["starting_alpha"],
            "top_p": config["top_p"],
            "adaptation_effectiveness": self._compute_adaptation_effectiveness(alpha_history, config["starting_alpha"]),
            "tokens_generated": len(generated_ids),
        }

        return generated_text, dynamic_stats

    def _compute_adaptation_effectiveness(self, alpha_history: Dict[str, List[float]], starting_alpha: float) -> float:
        """
        Compute how much the alphas adapted from their starting values.

        Returns value between 0.0 (no adaptation) and 1.0 (maximum adaptation).
        """
        if not alpha_history:
            return 0.0

        total_variation = 0.0
        total_steps = 0

        for prop_name, alphas in alpha_history.items():
            if not alphas:
                continue

            # Compute variation from starting alpha
            for alpha in alphas:
                variation = abs(alpha - starting_alpha) / starting_alpha if starting_alpha > 0 else 0
                total_variation += variation
                total_steps += 1

        if total_steps == 0:
            return 0.0

        avg_variation = total_variation / total_steps
        # Clamp to [0, 1] range
        return min(avg_variation, 1.0)

    def _generate_step_by_step_with_hooks(
        self,
        input_ids: torch.Tensor,
        steering_tensor: torch.Tensor,
        max_new_tokens: int,
        steering_strength: float = 1.0,
        timing_strategy: str = "normal",
    ) -> str:
        """
        Generate text step-by-step with hook-based steering application.

        Args:
            input_ids: Tokenized input [1, seq_len]
            steering_tensor: Steering tensor [steps, layers, heads, d_head]
            max_new_tokens: Maximum new tokens
            steering_strength: Strength multiplier
            timing_strategy: Steering timing strategy

        Returns:
            Generated text with steering
        """
        initial_length = input_ids.shape[1]
        current_ids = input_ids.clone()

        # Prepare steering strengths for different timing strategies
        step_strengths = self._prepare_step_strengths(max_new_tokens, steering_strength, timing_strategy)

        logger.info(f"Starting step-by-step generation with {timing_strategy} steering")

        for step in range(max_new_tokens):
            # Apply steering for this generation step
            next_token = self._generate_next_token_with_steering(
                current_ids=current_ids,
                steering_tensor=steering_tensor,
                step=step,
                initial_length=initial_length,
                step_strength=step_strengths[step],
                timing_strategy=timing_strategy,
            )

            # Check stopping conditions
            if next_token.item() == self._tokenizer.eos_token_id:
                break

            # Add token to sequence
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        # Decode only the generated part
        generated_ids = current_ids[0, initial_length:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def _prepare_step_strengths(self, max_new_tokens: int, base_strength: float, timing_strategy: str) -> List[float]:
        """Prepare steering strengths for each generation step."""
        if timing_strategy == "normal":
            return [base_strength] * max_new_tokens
        if timing_strategy == "start_only":
            # Only apply steering to the first token
            strengths = [0.0] * max_new_tokens
            if max_new_tokens > 0:
                strengths[0] = base_strength
            return strengths
        if timing_strategy == "diminishing":
            # Linearly decrease steering strength
            strengths = []
            for step in range(max_new_tokens):
                strength = base_strength * (1.0 - step / max_new_tokens)
                strengths.append(max(0.0, strength))
            return strengths
        if timing_strategy == "dynamic":
            # Dynamic strengths are computed on-the-fly based on KL divergence
            # Return placeholder - actual dynamic generation doesn't use this
            return [base_strength] * max_new_tokens
        logger.warning(f"Unknown timing strategy '{timing_strategy}', using normal")
        return [base_strength] * max_new_tokens

    def _generate_next_token_with_steering(
        self,
        current_ids: torch.Tensor,
        steering_tensor: torch.Tensor,
        step: int,
        initial_length: int,
        step_strength: float,
        timing_strategy: str,
    ) -> torch.Tensor:
        """
        Generate next token with steering applied via hooks.

        Args:
            current_ids: Current token sequence [1, seq_len]
            steering_tensor: Steering tensor [steps, layers, heads, d_head]
            step: Current generation step
            initial_length: Length of original prompt
            step_strength: Steering strength for this step
            timing_strategy: Timing strategy

        Returns:
            Next token [1]
        """
        if step_strength == 0.0:
            # No steering needed
            with torch.no_grad():
                outputs = self._model(current_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                return torch.argmax(logits)

        # Register steering hooks for this forward pass
        hooks = []
        try:
            for layer_idx in range(self.model_config["n_layers"]):
                hook = self._create_steering_hook(
                    layer_idx=layer_idx,
                    steering_tensor=steering_tensor,
                    step=step,
                    initial_length=initial_length,
                    step_strength=step_strength,
                    timing_strategy=timing_strategy,
                )

                # Get the attention output projection module (matching original DAC)
                layer_name = f"model.layers.{layer_idx}.self_attn.o_proj"
                layer_module = self._get_layer_module(layer_name)

                # Register pre-hook to modify inputs
                handle = layer_module.register_forward_pre_hook(hook)
                hooks.append(handle)

            # Forward pass with steering
            with torch.no_grad():
                outputs = self._model(current_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                next_token = torch.argmax(logits)

            return next_token

        finally:
            # Clean up hooks
            for handle in hooks:
                handle.remove()

    def _create_steering_hook(
        self,
        layer_idx: int,
        steering_tensor: torch.Tensor,
        step: int,
        initial_length: int,
        step_strength: float,
        timing_strategy: str,
    ):
        """
        Create a steering hook for a specific layer.

        Args:
            layer_idx: Index of the layer
            steering_tensor: Full steering tensor
            step: Current generation step
            initial_length: Original prompt length
            step_strength: Steering strength for this step
            timing_strategy: Timing strategy

        Returns:
            Hook function
        """

        def steering_hook(module, input_tensors):
            """Pre-hook function to apply steering to layer input activations."""
            # Get input activations [batch, seq, d_model]
            activations = input_tensors[0].clone()  # Clone to avoid in-place modification
            batch_size, seq_len, d_model = activations.shape

            if timing_strategy == "start_only":
                # Only modify the last prompt token for start_only
                if step < steering_tensor.shape[0]:
                    # Reshape steering: [heads, d_head] -> [d_model]
                    steering_vec = steering_tensor[0, layer_idx, :, :].reshape(-1).to(activations.device)
                    activations[0, initial_length - 1, :] += step_strength * steering_vec
            else:
                # Apply steering to appropriate positions
                positions_to_steer = (
                    range(initial_length, seq_len) if timing_strategy == "normal" else [initial_length - 1]
                )

                for pos_idx, pos in enumerate(positions_to_steer):
                    steering_step = min(pos_idx, steering_tensor.shape[0] - 1)
                    if steering_step < steering_tensor.shape[0]:
                        # Reshape steering: [heads, d_head] -> [d_model]
                        steering_vec = (
                            steering_tensor[steering_step, layer_idx, :, :].reshape(-1).to(activations.device)
                        )
                        activations[0, pos, :] += step_strength * steering_vec

            # Return modified input tuple for pre-hook
            return (activations,) + input_tensors[1:]

        return steering_hook

    def _get_layer_module(self, layer_name: str):
        """Get layer module by name for hook registration."""
        parts = layer_name.split(".")
        module = self._model
        for part in parts:
            module = getattr(module, part)
        return module

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
        """
        Apply top-p (nucleus) filtering to logits.

        Args:
            logits: Raw logits tensor [vocab_size]
            top_p: Cumulative probability threshold (0.0 to 1.0)

        Returns:
            Boolean mask of tokens to keep [vocab_size]
        """
        if top_p >= 1.0:
            # Keep all tokens
            return torch.ones_like(logits, dtype=torch.bool)

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Create mask to filter out tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the mask to the right to keep also the first token above top_p
        # This ensures we always keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Map the sorted mask back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        indices_to_keep = ~indices_to_remove

        return indices_to_keep

    def _compute_kl_divergence(
        self, steered_logits: torch.Tensor, original_logits: torch.Tensor, top_p: float = 0.9
    ) -> float:
        """
        Compute KL divergence between steered and original logit distributions.

        Args:
            steered_logits: Logits from steered model [vocab_size]
            original_logits: Logits from original model [vocab_size]
            top_p: Top-p filtering threshold for computational efficiency

        Returns:
            KL divergence value (scalar)
        """
        # Apply top-p filtering to both distributions
        original_mask = self._top_p_filtering(original_logits, top_p)
        steered_mask = self._top_p_filtering(steered_logits, top_p)

        # Merge masks - compute KL on union of top-p tokens
        combined_mask = original_mask | steered_mask

        if not combined_mask.any():
            # Fallback: if no tokens selected, use top-1 from each
            original_top1 = torch.zeros_like(original_logits, dtype=torch.bool)
            steered_top1 = torch.zeros_like(steered_logits, dtype=torch.bool)
            original_top1[torch.argmax(original_logits)] = True
            steered_top1[torch.argmax(steered_logits)] = True
            combined_mask = original_top1 | steered_top1

        # Filter logits to selected tokens only
        original_filtered = original_logits.masked_select(combined_mask)
        steered_filtered = steered_logits.masked_select(combined_mask)

        # Compute log probabilities
        original_log_probs = F.log_softmax(original_filtered, dim=-1)
        steered_log_probs = F.log_softmax(steered_filtered, dim=-1)

        # Compute KL divergence: KL(steered || original)
        kl_div = F.kl_div(
            steered_log_probs,  # input (steered distribution)
            original_log_probs,  # target (original distribution)
            reduction="batchmean",
            log_target=True,
        )

        return kl_div.item()

    def _kl_to_alpha(self, kl_divergence: float, alpha_bounds: Tuple[float, float] = (0.0, 2.0)) -> float:
        """
        Transform KL divergence to steering alpha value.

        Args:
            kl_divergence: KL divergence value
            alpha_bounds: (min_alpha, max_alpha) tuple

        Returns:
            Alpha value clamped to [min_alpha, max_alpha]
        """
        min_alpha, max_alpha = alpha_bounds

        # DEBUG: Log the KL value to understand the transformation
        # Remove this after debugging
        if hasattr(self, "_debug_kl_count"):
            self._debug_kl_count += 1
        else:
            self._debug_kl_count = 1

        # Direct transformation: alpha = clamp(kl, [min, max])
        # This matches the original DAC implementation
        alpha = max(min(kl_divergence, max_alpha), min_alpha)

        return alpha

    def _compute_multi_property_kl_divergences(
        self, original_logits: torch.Tensor, steered_logits_per_property: Dict[str, torch.Tensor], top_p: float = 0.9
    ) -> Dict[str, float]:
        """
        Compute KL divergences for multiple properties simultaneously.

        Args:
            original_logits: Baseline logits [vocab_size]
            steered_logits_per_property: Dict of property_name -> steered_logits
            top_p: Top-p filtering threshold

        Returns:
            Dictionary mapping property names to their KL divergence values
        """
        kl_divergences = {}

        for property_name, steered_logits in steered_logits_per_property.items():
            kl_div = self._compute_kl_divergence(
                steered_logits=steered_logits, original_logits=original_logits, top_p=top_p
            )
            kl_divergences[property_name] = kl_div

        return kl_divergences

    def _generate_next_token_with_dynamic_steering(
        self,
        current_ids: torch.Tensor,
        property_tensors: Dict[str, torch.Tensor],
        step: int,
        initial_length: int,
        alpha_history: Dict[str, List[float]],
        starting_alpha: float = 2.0,
        top_p: float = 0.95,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generate next token with dynamic multi-property steering.

        This implements the core dynamic algorithm from the original DAC:
        1. Forward pass: get original (unsteered) logits
        2. Forward pass: get steered logits for each property
        3. Compute KL divergences between steered and original
        4. Transform KL divergences to dynamic alphas
        5. Forward pass: generate with dynamic composition

        Args:
            current_ids: Current token sequence [1, seq_len]
            property_tensors: Dict of property tensors
            step: Current generation step
            initial_length: Original prompt length
            alpha_history: History of alphas used for each property
            starting_alpha: Initial alpha for KL computation
            top_p: Top-p filtering for KL computation

        Returns:
            Tuple of (next_token, dynamic_alphas_used)
        """
        # Step 1: Forward pass for original (unsteered) logits
        with torch.no_grad():
            outputs = self._model(current_ids)
            original_logits = outputs.logits[0, -1, :].detach()  # Last token logits

        # Step 2: Forward pass for each property with starting_alpha
        steered_logits_per_property = {}

        for prop_name, prop_tensor in property_tensors.items():
            # Register hooks for this property
            hooks = []
            try:
                for layer_idx in range(self.model_config["n_layers"]):
                    hook = self._create_dynamic_steering_hook(
                        layer_idx=layer_idx,
                        property_tensor=prop_tensor,
                        step=step,
                        initial_length=initial_length,
                        alpha_history=alpha_history.get(prop_name, []),
                        current_alpha=starting_alpha,
                    )

                    layer_name = f"model.layers.{layer_idx}.self_attn.o_proj"
                    layer_module = self._get_layer_module(layer_name)
                    handle = layer_module.register_forward_pre_hook(hook)
                    hooks.append(handle)

                # Forward pass with property steering
                with torch.no_grad():
                    outputs = self._model(current_ids)
                    steered_logits = outputs.logits[0, -1, :].detach()
                    steered_logits_per_property[prop_name] = steered_logits

            finally:
                # Clean up hooks
                for handle in hooks:
                    handle.remove()

        # Step 3: Compute KL divergences for each property
        kl_divergences = self._compute_multi_property_kl_divergences(
            original_logits=original_logits, steered_logits_per_property=steered_logits_per_property, top_p=top_p
        )

        # Step 4: Transform KL divergences to dynamic alphas
        dynamic_alphas = {}
        alpha_bounds = (0.0, 2.0)  # Default bounds matching original DAC
        for prop_name, kl_div in kl_divergences.items():
            alpha = self._kl_to_alpha(kl_div, alpha_bounds)
            dynamic_alphas[prop_name] = alpha

        # Step 5: Final forward pass with dynamic composition
        composed_tensor = self._create_dynamic_composition(
            property_tensors=property_tensors, dynamic_alphas=dynamic_alphas, alpha_history=alpha_history, step=step
        )

        # Apply composed steering and generate token
        hooks = []
        try:
            for layer_idx in range(self.model_config["n_layers"]):
                hook = self._create_composed_steering_hook(
                    layer_idx=layer_idx, composed_tensor=composed_tensor, step=step, initial_length=initial_length
                )

                layer_name = f"model.layers.{layer_idx}.self_attn.o_proj"
                layer_module = self._get_layer_module(layer_name)
                handle = layer_module.register_forward_pre_hook(hook)
                hooks.append(handle)

            # Forward pass with dynamic composition
            with torch.no_grad():
                outputs = self._model(current_ids)
                logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(logits)

        finally:
            # Clean up hooks
            for handle in hooks:
                handle.remove()

        return next_token, dynamic_alphas

    def _create_dynamic_steering_hook(
        self,
        layer_idx: int,
        property_tensor: torch.Tensor,
        step: int,
        initial_length: int,
        alpha_history: List[float],
        current_alpha: float,
    ):
        """Create hook for dynamic steering with a single property."""

        def hook_fn(module, input_tensors):
            activations = input_tensors[0].clone()  # Clone to avoid in-place modification

            # Apply cumulative steering from previous steps + current step
            for i in range(step + 1):
                alpha_to_use = alpha_history[i] if i < len(alpha_history) else current_alpha
                steering_step = min(i, property_tensor.shape[0] - 1)

                if steering_step < property_tensor.shape[0]:
                    steering_vec = property_tensor[steering_step, layer_idx, :, :].reshape(-1).to(activations.device)
                    position = initial_length + i - 1
                    if position < activations.shape[1]:
                        activations[0, position, :] += alpha_to_use * steering_vec

            return (activations,) + input_tensors[1:]

        return hook_fn

    def _create_dynamic_composition(
        self,
        property_tensors: Dict[str, torch.Tensor],
        dynamic_alphas: Dict[str, float],
        alpha_history: Dict[str, List[float]],
        step: int,
    ) -> torch.Tensor:
        """
        Create dynamically composed tensor using current and historical alphas.

        This implements the core composition from original DAC:
        current_step_steering = sum(alpha_history[prop][i] * property_tensors[prop][i]
                                   for prop in properties for i in range(step+1))
        """
        # Get reference tensor for shape
        first_tensor = next(iter(property_tensors.values()))
        composed = torch.zeros_like(first_tensor)

        for prop_name, prop_tensor in property_tensors.items():
            prop_alpha_history = alpha_history.get(prop_name, [])
            current_alpha = dynamic_alphas[prop_name]

            # Add contribution for each previous step + current step
            for i in range(step + 1):
                alpha_to_use = prop_alpha_history[i] if i < len(prop_alpha_history) else current_alpha
                # Use bounds checking to prevent index errors
                tensor_step = min(i, prop_tensor.shape[0] - 1)
                composed_step = min(i, composed.shape[0] - 1)
                composed[composed_step, :, :, :] += alpha_to_use * prop_tensor[tensor_step, :, :, :]

        return composed

    def _create_composed_steering_hook(
        self, layer_idx: int, composed_tensor: torch.Tensor, step: int, initial_length: int
    ):
        """Create hook that applies the pre-composed dynamic steering tensor."""

        def hook_fn(module, input_tensors):
            activations = input_tensors[0].clone()  # Clone to avoid in-place modification

            # Apply composed steering for all steps
            for i in range(step + 1):
                if i < composed_tensor.shape[0]:
                    steering_vec = composed_tensor[i, layer_idx, :, :].reshape(-1).to(activations.device)
                    position = initial_length + i - 1
                    if position < activations.shape[1]:
                        activations[0, position, :] += steering_vec

            return (activations,) + input_tensors[1:]

        return hook_fn

    def generate_with_multi_property_steering(
        self,
        prompt: str,
        property_weights: Dict[str, float],
        max_new_tokens: Optional[int] = None,
        steering_strength: float = 1.0,
        composition_strategy: str = "linear",
    ) -> Dict[str, Any]:
        """
        Generate text with multi-property steering and return detailed results.

        Args:
            prompt: Input text prompt
            property_weights: Weights for each property
            max_new_tokens: Maximum tokens to generate
            steering_strength: Overall steering strength
            composition_strategy: Composition strategy

        Returns:
            Dictionary with generation results and metadata
        """
        if not property_weights:
            raise ValueError("Property weights cannot be empty for multi-property steering")

        # Validate properties exist
        for prop_name in property_weights:
            if prop_name not in self.property_tensors:
                available = list(self.property_tensors.keys())
                raise ValueError(f"Property '{prop_name}' not found. Available: {available}")

        # Generate with steering
        generated_text = self.generate_with_steering(
            prompt=prompt,
            property_weights=property_weights,
            max_new_tokens=max_new_tokens,
            steering_strength=steering_strength,
            composition_strategy=composition_strategy,
        )

        # Return detailed results
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "properties_used": list(property_weights.keys()),
            "property_weights": property_weights,
            "steering_strength": steering_strength,
            "composition_strategy": composition_strategy,
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
        }

    def generate_with_dynamic_steering(
        self,
        prompt: str,
        property_weights: Dict[str, float],
        max_new_tokens: Optional[int] = None,
        starting_alpha: float = 2.0,
        top_p: float = 0.95,
        alpha_bounds: Tuple[float, float] = (0.0, 2.0),
    ) -> Dict[str, Any]:
        """
        Generate text with dynamic steering and return detailed statistics.

        This method exposes the full dynamic steering functionality with
        comprehensive statistics about the adaptation process.

        Args:
            prompt: Input text prompt
            property_weights: Properties to use for dynamic steering
            max_new_tokens: Maximum tokens to generate
            starting_alpha: Initial alpha for KL computation
            top_p: Top-p filtering threshold for KL computation
            alpha_bounds: (min_alpha, max_alpha) bounds for alpha values

        Returns:
            Dictionary with generation results and dynamic statistics
        """
        if not self.is_trained:
            raise ValueError("DAC must be trained before generating with dynamic steering")

        if not property_weights:
            raise ValueError("Dynamic steering requires at least one property")

        # Load model if needed
        self._load_model()

        # Use class defaults if not specified
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Prepare dynamic configuration
        dynamic_config = {"starting_alpha": starting_alpha, "top_p": top_p, "alpha_bounds": list(alpha_bounds)}

        logger.info(f"Generating with dynamic steering: {len(property_weights)} properties")
        logger.info(f"Dynamic config: {dynamic_config}")

        # Tokenize the prompt
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)

        # Generate with dynamic steering
        generated_text, dynamic_stats = self._generate_step_by_step_with_dynamic_steering(
            input_ids=input_ids,
            property_weights=property_weights,
            max_new_tokens=max_new_tokens,
            dynamic_config=dynamic_config,
        )

        # Return comprehensive results
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "properties_used": dynamic_stats["properties_used"],
            "property_weights": property_weights,
            "starting_alpha": starting_alpha,
            "top_p": top_p,
            "alpha_bounds": alpha_bounds,
            "max_new_tokens": max_new_tokens,
            "tokens_generated": dynamic_stats["tokens_generated"],
            "alpha_history": dynamic_stats["alpha_history"],
            "kl_history": dynamic_stats["kl_history"],
            "adaptation_effectiveness": dynamic_stats["adaptation_effectiveness"],
            "method": "dynamic_DAC",
        }

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
        device = vectors[0].device if hasattr(vectors[0], 'device') else 'cpu'
        if device == 'cpu':
            # If no device attribute, vectors are likely numpy arrays or lists
            vectors = [torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in vectors]
            device = vectors[0].device
        vectors = [v.to(device) if hasattr(v, 'to') else torch.tensor(v).to(device) for v in vectors]

        # Combine vectors using weighted sum
        combined = torch.zeros_like(vectors[0])
        for vector, weight in zip(vectors, weights):
            combined = combined + weight * vector

        return combined
    
    def apply_steering(self, hidden_states, strength=1.0, active_properties=None):
        """Apply steering to hidden states (compatibility method for multi-steer)."""
        if self.steering_vector is None:
            return hidden_states
        
        # Apply the steering vector
        steering = self.steering_vector.to(hidden_states.device)
        
        # Reshape if needed
        if len(steering.shape) == 1:
            steering = steering.unsqueeze(0)
        
        # Apply steering
        return hidden_states + strength * steering
