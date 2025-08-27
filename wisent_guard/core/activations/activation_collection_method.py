"""
Activation Collection Method for extracting activations from contrastive pairs.

This module contains the core logic for collecting activations from language models
using different prompt construction strategies and token targeting approaches. It handles:
- Multiple prompt construction methodologies (multiple choice, role-playing, etc.)
- Different token targeting strategies within each methodology
- Mapping from (question, good_response, bad_response) to activation pairs
- Batch processing of multiple contrastive pairs

The activation collection system is designed to be flexible in how it constructs
prompts and targets tokens, allowing for experimentation with different approaches
to capturing model internal representations.
"""

import logging
from typing import Dict, List, Tuple

import torch

from wisent_guard.core.contrastive_pairs import ContrastivePair
from wisent_guard.core.layer import Layer

from .core import ActivationAggregationStrategy, Activations
from .prompts import PromptConstructionStrategy, PromptPair


class ActivationCollectionLogic:
    """
    Logic for collecting activations from contrastive pairs using configurable
    prompt construction and token targeting strategies.
    """

    def __init__(self, model: "Model"):
        """
        Initialize the activation collection logic.

        Args:
            model: Model primitive that handles proper formatting
        """
        self.model = model

    def _construct_multiple_choice_prompts(
        self, question: str, correct_answer: str, incorrect_answer: str
    ) -> PromptPair:
        """Original multiple choice format."""
        # Create the multiple choice question
        mc_question = f"Which is better: {question} A. {incorrect_answer} B. {correct_answer}"

        # Use the model's proper formatting (no response yet, just the prompt)
        base_prompt = self.model.format_prompt(mc_question)

        return PromptPair(
            positive_prompt=f"{base_prompt}B",  # Chooses correct answer
            negative_prompt=f"{base_prompt}A",  # Chooses incorrect answer
            target_token="B",  # Target the choice tokens
        )

    def _construct_role_playing_prompts(self, question: str, correct_answer: str, incorrect_answer: str) -> PromptPair:
        """Role-playing format: make model behave like someone who would give that answer."""
        # Construct role-playing prompts
        positive_instruction = f"Behave like a person that would answer {question} with {correct_answer}"
        negative_instruction = f"Behave like a person that would answer {question} with {incorrect_answer}"

        # Format with model's chat format
        positive_base = self.model.format_prompt(positive_instruction)
        negative_base = self.model.format_prompt(negative_instruction)

        return PromptPair(
            positive_prompt=f"{positive_base}I",  # Start continuation
            negative_prompt=f"{negative_base}I",  # Start continuation
            target_token="I",  # Target the continuation token
        )

    def _construct_direct_completion_prompts(
        self, question: str, correct_answer: str, incorrect_answer: str
    ) -> PromptPair:
        """Direct completion: question -> answer."""
        base_prompt = self.model.format_prompt(question)

        return PromptPair(
            positive_prompt=f"{base_prompt}{correct_answer}",
            negative_prompt=f"{base_prompt}{incorrect_answer}",
            target_token=None,  # Will use targeting strategy to determine position
        )

    def _construct_instruction_following_prompts(
        self, question: str, correct_answer: str, incorrect_answer: str
    ) -> PromptPair:
        """Instruction following format with explicit tags."""
        # Use model's instruction format if available, otherwise use simple format
        if hasattr(self.model, "format_instruction"):
            instruction_prompt = self.model.format_instruction(question)
        else:
            instruction_prompt = f"[INST] {question} [/INST]"

        return PromptPair(
            positive_prompt=f"{instruction_prompt} {correct_answer}",
            negative_prompt=f"{instruction_prompt} {incorrect_answer}",
            target_token=None,  # Will use targeting strategy
        )

    def construct_prompt_pair(
        self,
        question: str,
        correct_answer: str,
        incorrect_answer: str,
        prompt_strategy: PromptConstructionStrategy = PromptConstructionStrategy.MULTIPLE_CHOICE,
    ) -> PromptPair:
        """
        Construct a prompt pair using the specified strategy.

        Args:
            question: The question to ask
            correct_answer: The correct answer
            incorrect_answer: The incorrect answer
            prompt_strategy: Strategy for constructing prompts

        Returns:
            PromptPair object with positive and negative prompts
        """
        if prompt_strategy == PromptConstructionStrategy.MULTIPLE_CHOICE:
            return self._construct_multiple_choice_prompts(question, correct_answer, incorrect_answer)
        if prompt_strategy == PromptConstructionStrategy.ROLE_PLAYING:
            return self._construct_role_playing_prompts(question, correct_answer, incorrect_answer)
        if prompt_strategy == PromptConstructionStrategy.DIRECT_COMPLETION:
            return self._construct_direct_completion_prompts(question, correct_answer, incorrect_answer)
        if prompt_strategy == PromptConstructionStrategy.INSTRUCTION_FOLLOWING:
            return self._construct_instruction_following_prompts(question, correct_answer, incorrect_answer)
        # Fallback to multiple choice
        return self._construct_multiple_choice_prompts(question, correct_answer, incorrect_answer)

    def create_contrastive_pair(
        self,
        question: str,
        correct_answer: str,
        incorrect_answer: str,
        prompt_strategy: PromptConstructionStrategy = PromptConstructionStrategy.MULTIPLE_CHOICE,
    ) -> ContrastivePair:
        """
        Create a contrastive pair from a question and two answers using specified strategy.

        Args:
            question: The question to ask
            correct_answer: The correct answer
            incorrect_answer: The incorrect answer
            prompt_strategy: Strategy for constructing prompts

        Returns:
            ContrastivePair object with positive and negative responses
        """
        prompt_pair = self.construct_prompt_pair(question, correct_answer, incorrect_answer, prompt_strategy)

        # Create ContrastivePair with the constructed prompts
        # For backward compatibility, we'll store the prompts in a way that works with existing code
        if prompt_strategy == PromptConstructionStrategy.MULTIPLE_CHOICE:
            # Original format: base prompt + response
            base_prompt = prompt_pair.positive_prompt[:-1]  # Remove the "B"
            positive_response = "B"
            negative_response = "A"
        else:
            # New formats: store full prompts, use empty base
            base_prompt = ""
            positive_response = prompt_pair.positive_prompt
            negative_response = prompt_pair.negative_prompt

        pair = ContrastivePair(
            prompt=base_prompt,
            positive_response=positive_response,
            negative_response=negative_response,
            label=f"Q: {question}",
        )

        # Store additional metadata for new extraction logic
        pair._prompt_pair = prompt_pair
        pair._prompt_strategy = prompt_strategy

        return pair

    def create_batch_contrastive_pairs(
        self,
        qa_pairs: List[Dict[str, str]],
        prompt_strategy: PromptConstructionStrategy = PromptConstructionStrategy.MULTIPLE_CHOICE,
    ) -> List[ContrastivePair]:
        """
        Create multiple contrastive pairs from a list of QA pairs.

        Args:
            qa_pairs: List of dictionaries with keys:
                - 'question': The question
                - 'correct_answer': The correct answer
                - 'incorrect_answer': The incorrect answer
            prompt_strategy: Strategy for constructing prompts

        Returns:
            List of ContrastivePair objects
        """
        pairs = []
        for qa_pair in qa_pairs:
            # Handle both field names for backward compatibility
            incorrect = qa_pair.get("incorrect_answer") or qa_pair.get("incorrect_choice")
            if not incorrect:
                # For code generation tasks, provide a default incorrect answer if missing
                if any(
                    key in qa_pair.get("metadata", {}).get("benchmark_type", "")
                    for key in ["mbpp", "humaneval", "code"]
                ):
                    incorrect = "# Incorrect or incomplete code implementation\npass"  # TODO
                else:
                    raise KeyError(f"Missing 'incorrect_answer' field in qa_pair: {qa_pair.keys()}")

            pair = self.create_contrastive_pair(
                question=qa_pair["question"],
                correct_answer=qa_pair["correct_answer"],
                incorrect_answer=incorrect,
                prompt_strategy=prompt_strategy,
            )
            pairs.append(pair)
        return pairs

    def _get_token_position_choice_token(self, tokens: List[str], target_token: str) -> int:
        """Look for the target token (A, B, I, etc.) in the sequence."""
        target_position = -1  # Default to last token

        # First try to find exact match
        for i in range(len(tokens) - 1, -1, -1):
            token_str = str(tokens[i]).lower().strip()
            if (target_token and target_token.lower() == token_str) or target_token.lower() in token_str:
                target_position = i
                break

        # If not found, use last token position
        if target_position == -1:
            target_position = len(tokens) - 1

        return target_position

    def _get_token_position_continuation_token(self, tokens: List[str], target_token: str) -> int:
        """Look for continuation tokens like 'I', 'The', etc. - usually early in response."""
        if not target_token:
            return len(tokens) - 1  # Fallback to last token

        # Search for target token, preferring earlier positions for continuation
        for i in range(len(tokens)):
            token_str = str(tokens[i]).lower().strip()
            if target_token.lower() == token_str or target_token.lower() in token_str:
                return i

        # If not found, use last token
        return len(tokens) - 1

    def _get_token_position_last_token(self, tokens: List[str], target_token: str) -> int:
        """Always use the last token position."""
        return len(tokens) - 1

    def _get_token_position_first_token(self, tokens: List[str], target_token: str) -> int:
        """Always use the first token position."""
        return 0

    def _get_activation_with_strategy(
        self, hidden_states: torch.Tensor, tokens: List[str], target_token: str, strategy: ActivationAggregationStrategy
    ) -> torch.Tensor:
        """
        Get activation based on the specified targeting strategy.

        Args:
            hidden_states: Hidden states tensor [batch_size, seq_len, hidden_dim]
            tokens: List of token strings
            target_token: The target token we're looking for
            strategy: Token targeting strategy to use

        Returns:
            Activation tensor [hidden_dim]
        """
        logging.debug("_get_activation_with_strategy called:")
        logging.debug(f"Strategy: {strategy.value}")
        logging.debug(f"Target token: {target_token}")
        logging.debug(f"Tokens: {tokens[:5]}..." if len(tokens) > 5 else f"Tokens: {tokens}")
        logging.debug(f"Hidden states shape: {hidden_states.shape}")
        if strategy == ActivationAggregationStrategy.CHOICE_TOKEN:
            # Look for A/B choice tokens (backward search)
            position = self._get_token_position_choice_token(tokens, target_token)
            logging.debug(f"CHOICE_TOKEN: Using position {position}")
            return hidden_states[0, position, :]

        if strategy == ActivationAggregationStrategy.CONTINUATION_TOKEN:
            # Look for continuation tokens like "I" (forward search)
            position = self._get_token_position_continuation_token(tokens, target_token)
            logging.debug(f"CONTINUATION_TOKEN: Using position {position}")
            return hidden_states[0, position, :]

        if strategy == ActivationAggregationStrategy.LAST_TOKEN:
            # Always use last token
            position = self._get_token_position_last_token(tokens, target_token)
            logging.debug(f"LAST_TOKEN: Using position {position}")
            return hidden_states[0, position, :]

        if strategy == ActivationAggregationStrategy.FIRST_TOKEN:
            # Always use first token
            position = self._get_token_position_first_token(tokens, target_token)
            logging.debug(f"FIRST_TOKEN: Using position {position}")
            return hidden_states[0, position, :]

        if strategy == ActivationAggregationStrategy.MEAN_POOLING:
            # Use mean pooling across all tokens
            logging.debug(f"MEAN_POOLING: Using mean across all {hidden_states.shape[1]} tokens")
            return hidden_states[0].mean(dim=0)  # [hidden_dim]

        if strategy == ActivationAggregationStrategy.MAX_POOLING:
            # Use max pooling across all tokens
            logging.debug(f"MAX_POOLING: Using max across all {hidden_states.shape[1]} tokens")
            return hidden_states[0].max(dim=0)[0]  # [hidden_dim]

        # Fallback to choice token strategy
        position = self._get_token_position_choice_token(tokens, target_token)
        logging.debug(f"FALLBACK: Using position {position}")
        return hidden_states[0, position, :]

    def extract_activations_from_pair(
        self,
        pair: ContrastivePair,
        layer_index: int,
        device: str = "cuda",
        token_targeting_strategy: ActivationAggregationStrategy = ActivationAggregationStrategy.CHOICE_TOKEN,
    ) -> ContrastivePair:
        """
        Extract activations from a contrastive pair using the appropriate prompt format and targeting strategy.

        Args:
            pair: ContrastivePair object (with potential prompt strategy metadata)
            layer_index: Which layer to extract activations from (0-indexed)
            device: Device to run on
            token_targeting_strategy: Strategy for targeting tokens

        Returns:
            The same ContrastivePair object with activations populated
        """

        def get_activation_at_target_token(full_prompt: str, target_token: str) -> torch.Tensor:
            """Get activation at the target token position using the specified strategy."""
            # Use the model's device instead of forcing a specific device
            model_device = next(self.model.hf_model.parameters()).device

            # Tokenize the prompt
            inputs = self.model.tokenizer(full_prompt, return_tensors="pt")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Get model outputs with hidden states
            with torch.no_grad():
                outputs = self.model.hf_model(**inputs, output_hidden_states=True)

            # Get hidden states from the specified layer (add 1 because hidden_states[0] is embeddings)
            if layer_index + 1 < len(outputs.hidden_states):
                hidden_states = outputs.hidden_states[layer_index + 1]  # [batch_size, seq_len, hidden_dim]
            else:
                # Fallback to last layer if index is too high
                hidden_states = outputs.hidden_states[-1]

            # Find the position of the target token using the specified strategy
            tokens = self.model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            # Extract activation using the specified strategy
            activation = self._get_activation_with_strategy(
                hidden_states, tokens, target_token, token_targeting_strategy
            )

            return activation.cpu()  # Move to CPU for storage

        try:
            # Determine prompts and target tokens based on prompt strategy
            if hasattr(pair, "_prompt_pair") and pair._prompt_pair:
                # New format: use stored prompt pair
                prompt_pair = pair._prompt_pair
                positive_full_prompt = prompt_pair.positive_prompt
                negative_full_prompt = prompt_pair.negative_prompt
                target_token = prompt_pair.target_token
            else:
                # Legacy format: reconstruct from pair attributes
                if hasattr(pair.positive_response, "text"):
                    positive_resp = pair.positive_response.text
                else:
                    positive_resp = str(pair.positive_response)

                if hasattr(pair.negative_response, "text"):
                    negative_resp = pair.negative_response.text
                else:
                    negative_resp = str(pair.negative_response)

                positive_full_prompt = f"{pair.prompt}{positive_resp}"
                negative_full_prompt = f"{pair.prompt}{negative_resp}"
                target_token = positive_resp  # Use response as target token

            # Extract activations for both positive and negative cases
            positive_activation = get_activation_at_target_token(positive_full_prompt, target_token)
            negative_activation = get_activation_at_target_token(negative_full_prompt, target_token)

            # Store activations in the pair object
            pair.positive_activations = positive_activation
            pair.negative_activations = negative_activation

        except Exception as e:
            logging.info(f"Error extracting activations: {e}")
            # Create dummy activations to prevent crashes
            dummy_size = 4096  # Common hidden size
            pair.positive_activations = torch.zeros(dummy_size)
            pair.negative_activations = torch.zeros(dummy_size)

        return pair

    def collect_activations_batch(
        self,
        pairs: List[ContrastivePair],
        layer_index: int,
        device: str = "cuda",
        token_targeting_strategy: ActivationAggregationStrategy = ActivationAggregationStrategy.CHOICE_TOKEN,
    ) -> List[ContrastivePair]:
        """
        Collect activations from multiple contrastive pairs.

        Args:
            pairs: List of ContrastivePair objects
            layer_index: Which layer to extract activations from (0-indexed)
            device: Device to run on (will use model's actual device)
            token_targeting_strategy: Strategy for targeting tokens

        Returns:
            List of ContrastivePair objects with activations populated
        """
        processed_pairs = []

        logging.info(f"Processing {len(pairs)} contrastive pairs...")
        logging.info(f"Token targeting strategy: {token_targeting_strategy.value}")
        logging.info("ACTIVATION COLLECTION DEBUG:")
        logging.info(f"Strategy passed to method: {token_targeting_strategy}")
        logging.info(f"Strategy value: {token_targeting_strategy.value}")
        logging.info(f"Strategy type: {type(token_targeting_strategy)}")

        # Determine prompt strategy from first pair if available
        prompt_strategy = "unknown"
        if pairs and hasattr(pairs[0], "_prompt_strategy"):
            prompt_strategy = pairs[0]._prompt_strategy.value
        logging.info(f"Prompt construction strategy: {prompt_strategy}")

        # Debug first pair details
        if pairs:
            first_pair = pairs[0]
            logging.info("FIRST PAIR DEBUG:")
            logging.info(f"Pair type: {type(first_pair).__name__}")
            logging.info(f"Has _prompt_strategy: {hasattr(first_pair, '_prompt_strategy')}")
            logging.info(f"Has _prompt_pair: {hasattr(first_pair, '_prompt_pair')}")
            if hasattr(first_pair, "_prompt_strategy"):
                logging.info(f"Prompt strategy: {first_pair._prompt_strategy}")
            if hasattr(first_pair, "_prompt_pair"):
                logging.info(f"Target token: {first_pair._prompt_pair.target_token}")
            logging.info(f"Prompt: {first_pair.prompt[:50]}..." if hasattr(first_pair, "prompt") else "No prompt attr")
            logging.info(
                f"Positive response: {first_pair.positive_response}"
                if hasattr(first_pair, "positive_response")
                else "No positive_response attr"
            )

        for i, pair in enumerate(pairs):
            logging.debug(f"Processing pair {i + 1}/{len(pairs)}")
            processed_pair = self.extract_activations_from_pair(pair, layer_index, device, token_targeting_strategy)
            processed_pairs.append(processed_pair)

        logging.info(f"Successfully processed {len(processed_pairs)} pairs")
        return processed_pairs

    def create_activations_from_pairs(
        self, pairs: List[ContrastivePair], layer: Layer
    ) -> Tuple[List[Activations], List[Activations]]:
        """
        Convert ContrastivePair objects with activations to Activations objects.

        Args:
            pairs: List of ContrastivePair objects with activations
            layer: Layer object

        Returns:
            Tuple of (positive_activations_list, negative_activations_list)
        """
        positive_activations = []
        negative_activations = []

        for pair in pairs:
            if pair.positive_activations is not None:
                pos_act = Activations(
                    tensor=pair.positive_activations,
                    layer=layer,
                    aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                )
                positive_activations.append(pos_act)

            if pair.negative_activations is not None:
                neg_act = Activations(
                    tensor=pair.negative_activations,
                    layer=layer,
                    aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                )
                negative_activations.append(neg_act)

        return positive_activations, negative_activations
