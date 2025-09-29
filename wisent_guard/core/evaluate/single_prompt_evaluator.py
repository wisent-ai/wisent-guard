"""
Single-Prompt Evaluation System

This module provides focused single-prompt evaluation for steering effectiveness.
Designed for real-time evaluation to determine if model output should be regenerated.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..model import Model
from ..utils.device import empty_device_cache, resolve_default_device


@dataclass
class EvaluationResult:
    """Container for single-prompt evaluation results."""

    trait_quality: float  # -1 to 1 scale (-1 = strongly opposite, 0 = neutral, 1 = strongly demonstrates)
    answer_quality: float  # 0 to 1 scale (0 = broken/incoherent, 1 = high quality)
    steered_vs_unsteered_similarity: float  # 0 to 1 scale (0 = completely different, 1 = nearly identical)
    response: str  # The generated steered response
    unsteered_response: str  # The generated unsteered baseline response
    prompt: str  # The original prompt

    def to_dict(self) -> Dict[str, Union[float, str]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trait_quality": self.trait_quality,
            "answer_quality": self.answer_quality,
            "steered_vs_unsteered_similarity": self.steered_vs_unsteered_similarity,
            "response": self.response,
            "unsteered_response": self.unsteered_response,
            "prompt": self.prompt,
        }


@dataclass
class MultiTraitEvaluationResult:
    """Container for multi-trait evaluation results."""

    trait_scores: Dict[str, float]  # Mapping of trait names to scores (-1 to 1 scale)
    answer_quality: float  # 0 to 1 scale (0 = broken/incoherent, 1 = high quality)
    steered_vs_unsteered_similarity: float  # 0 to 1 scale (0 = completely different, 1 = nearly identical)
    response: str  # The generated steered response
    unsteered_response: str  # The generated unsteered baseline response
    prompt: str  # The original prompt
    steering_method_name: str  # Name/identifier of the steering method applied

    def to_dict(self) -> Dict[str, Union[float, str, Dict[str, float]]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trait_scores": self.trait_scores,
            "answer_quality": self.answer_quality,
            "steered_vs_unsteered_similarity": self.steered_vs_unsteered_similarity,
            "response": self.response,
            "unsteered_response": self.unsteered_response,
            "prompt": self.prompt,
            "steering_method_name": self.steering_method_name,
        }

    def get_trait_score(self, trait_name: str) -> Optional[float]:
        """Get score for a specific trait."""
        return self.trait_scores.get(trait_name)

    def get_primary_trait_score(self) -> float:
        """Get the highest trait score (assumes primary intended trait)."""
        if not self.trait_scores:
            return 0.0
        return max(self.trait_scores.values())

    def get_average_trait_score(self) -> float:
        """Get average of all trait scores."""
        if not self.trait_scores:
            return 0.0
        return sum(self.trait_scores.values()) / len(self.trait_scores)


class SinglePromptEvaluator:
    """
    Single-prompt evaluation system for steering effectiveness.

    Uses a single model for both generation and evaluation to optimize memory usage.
    Loads/unloads models sequentially to minimize CUDA memory requirements.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the single-prompt evaluator.

        Args:
            model_name: Model to use for both generation and evaluation
            device: Device to run on (cuda/cpu/mps)
            verbose: Whether to print progress
        """
        self.verbose = verbose
        self.device = device or resolve_default_device()

        # Single model used for both generation and evaluation
        self.model = None
        self.tokenizer = None
        self.model_name = model_name

        if self.verbose:
            print(f"SinglePromptEvaluator initialized for device: {self.device}")
            print(f"Using single model for both generation and evaluation: {model_name}")

    def _load_model(self):
        """Load the model using Model class for both generation and evaluation."""
        if self.model is None:
            if self.verbose:
                print(f"Loading model: {self.model_name}")

            # Use Model class for both generation and evaluation
            self.model = Model(self.model_name)
            # Store tokenizer for consistency
            self.tokenizer = self.model.tokenizer

            if self.verbose:
                print(f"✓ Model loaded on {self.model.device}")

    def _unload_model(self):
        """Unload the current model to free memory."""
        if self.model is not None:
            if self.verbose:
                print("🧹 Unloading model to free memory...")

            # Delete model and tokenizer
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            # Clear device cache
            empty_device_cache(self.device)

            if self.verbose:
                print("✓ Model unloaded and memory cleared")

    def load_steering_vector(self, vector_path: str) -> Tuple["SteeringMethod", int]:
        """
        Load steering method from vector file based on method field.

        Args:
            vector_path: Path to the steering vector file (.pt)

        Returns:
            Tuple of (steering_method_instance, layer_index)
        """
        try:
            data = torch.load(vector_path, map_location=self.device)

            # Get method type
            method = data.get("method")
            if method is None:
                raise ValueError(f"Could not find method field in {vector_path}")

            # Get layer index
            layer_index = data.get("layer_index")
            if layer_index is None:
                raise ValueError(f"Could not find layer_index in {vector_path}")

            if self.verbose:
                print(f"✓ Loading steering method: {method} from {vector_path}")

            # Load appropriate steering method (CAA and DAC supported)
            if method == "CAA":
                from ..aggregation import ControlVectorAggregationMethod
                from ..steering_methods.caa import CAA

                steering_method = CAA(device=self.device)

                # Manual loading with case fix for aggregation method
                try:
                    # Set the required fields manually to handle case sensitivity issues
                    steering_method.steering_vector = data["steering_vector"].to(self.device)

                    # Fix aggregation method case: 'CAA' -> 'caa'
                    agg_method = data.get("aggregation_method", "CAA").lower()
                    steering_method.aggregation_method = ControlVectorAggregationMethod(agg_method)

                    steering_method.normalization_method = data.get("normalization_method", "none")
                    steering_method.target_norm = data.get("target_norm")
                    steering_method.legacy_behavior = data.get("legacy_behavior", False)
                    steering_method.layer_index = data.get("layer_index")
                    steering_method.training_stats = data.get("training_stats", {})
                    steering_method.is_trained = True

                    if self.verbose:
                        vector_shape = steering_method.get_steering_vector().shape
                        vector_norm = torch.norm(steering_method.get_steering_vector()).item()
                        print("  ✓ CAA method loaded successfully")
                        print(f"  Vector shape: {vector_shape}")
                        print(f"  Vector norm: {vector_norm:.4f}")
                        print(f"  Layer index: {layer_index}")

                    return steering_method, layer_index

                except Exception as e:
                    raise RuntimeError(f"Failed to manually load CAA steering data: {e}")

            elif method == "DAC":
                from ..steering_methods.dac import DAC

                # Create DAC instance with default parameters
                steering_method = DAC(
                    device=self.device,
                    dynamic_control=data.get("dynamic_control", True),
                    entropy_threshold=data.get("entropy_threshold", 1.0),
                )

                # Manual loading for DAC
                try:
                    # Set the required fields manually
                    steering_method.steering_vector = data["steering_vector"].to(self.device)
                    steering_method.layer_index = data.get("layer_index")
                    steering_method.training_stats = data.get("training_stats", {})
                    steering_method.is_trained = True

                    if self.verbose:
                        vector_shape = steering_method.steering_vector.shape
                        vector_norm = torch.norm(steering_method.steering_vector).item()
                        print("  ✓ DAC method loaded successfully")
                        print(f"  Vector shape: {vector_shape}")
                        print(f"  Vector norm: {vector_norm:.4f}")
                        print(f"  Layer index: {layer_index}")
                        print(f"  Dynamic control: {steering_method.dynamic_control}")
                        print(f"  Entropy threshold: {steering_method.entropy_threshold}")

                    return steering_method, layer_index

                except Exception as e:
                    raise RuntimeError(f"Failed to manually load DAC steering data: {e}")

            else:
                raise ValueError(f"Unsupported steering method: {method}. Currently CAA and DAC are supported.")

        except Exception as e:
            raise RuntimeError(f"Failed to load steering method from {vector_path}: {e}")

    def _generate_steered_response(
        self,
        prompt: str,
        steering_method: "SteeringMethod",
        layer: int,
        steering_strength: float = 2.0,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate a steered response using the proper steering method."""
        self._load_model()

        hooks = []

        try:
            # Set up steering hook using the proper steering method
            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Apply steering using the steering method's apply_steering method
                steered = steering_method.apply_steering(hidden_states, strength=steering_strength, verbose=False)

                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            # Apply hook to the specified layer
            if hasattr(self.model.hf_model, "model") and hasattr(self.model.hf_model.model, "layers"):
                layer_module = self.model.hf_model.model.layers[layer]
            elif hasattr(self.model.hf_model, "transformer") and hasattr(self.model.hf_model.transformer, "h"):
                layer_module = self.model.hf_model.transformer.h[layer]
            else:
                raise ValueError(f"Could not find layer {layer} in model architecture")

            hook_handle = layer_module.register_forward_hook(steering_hook)
            hooks.append(hook_handle)

            # Generate response
            formatted_prompt = self.model.format_prompt(prompt)
            inputs = self.model.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.model.tokenizer.pad_token_id,
                )

            response = self.model.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            return response

        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

    def _generate_unsteered_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate an unsteered (baseline) response without any steering applied."""
        self._load_model()

        # Generate unsteered response
        formatted_prompt = self.model.format_prompt(prompt)
        inputs = self.model.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.model.tokenizer.pad_token_id,
            )

        response = self.model.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def _evaluate_response(
        self,
        response: str,
        prompt: str,
        trait_name: str,
        trait_description: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Evaluate response for trait quality and answer quality.

        Returns:
            Tuple of (trait_quality, answer_quality) on -1 to 1 and 0 to 1 scales
        """
        # Ensure model is loaded - we can use the same Model instance for evaluation
        if self.model is None:
            self._load_model()

        trait_desc = trait_description or trait_name

        # Evaluate trait quality (0-10 scale)
        trait_prompt = f"""Rate how strongly this response demonstrates the trait "{trait_desc}" on a scale from 0 to 10.

0 = Strongly opposite to the trait (e.g., for "catholic" this would be atheistic/secular)
5 = Neutral, no clear expression of the trait
10 = Strongly demonstrates the trait

Consider the language, values, references, and overall perspective in the response.

Response to evaluate:
"{response}"

Provide only a single number from 0 to 10 as your rating. Do not include any explanation.

Rating:"""

        trait_rating = self._get_llm_rating(trait_prompt)

        # Evaluate answer quality (0-10 scale)
        quality_prompt = f"""Rate the quality of this response on a scale from 0 to 10.

Consider:
- Coherence and clarity
- Relevance to the question
- Absence of repetition or nonsensical text
- Overall meaningfulness

0 = Completely broken (repetitive, nonsensical, or incoherent)
5 = Mediocre but understandable
10 = High quality, clear, and coherent

Original question: "{prompt}"

Response to evaluate:
"{response}"

Provide only a single number from 0 to 10 as your rating. Do not include any explanation.

Rating:"""

        quality_rating = self._get_llm_rating(quality_prompt)

        # Convert ratings to target scales
        # Trait quality: 0-10 -> -1 to 1
        trait_quality = (trait_rating - 5) / 5  # Maps 0->-1, 5->0, 10->1

        # Answer quality: 0-10 -> 0 to 1
        answer_quality = quality_rating / 10  # Maps 0->0, 10->1

        return trait_quality, answer_quality

    def _evaluate_similarity(
        self,
        prompt: str,
        unsteered_response: str,
        steered_response: str,
    ) -> float:
        """
        Evaluate similarity between unsteered and steered responses.

        Args:
            prompt: The original question/prompt
            unsteered_response: The baseline response without steering
            steered_response: The response with steering applied

        Returns:
            Similarity score on 0 to 1 scale (0 = completely different, 1 = nearly identical)
        """
        # Ensure model is loaded
        if self.model is None:
            self._load_model()

        # Create similarity evaluation prompt
        similarity_prompt = f"""Compare these two responses to the same question. Rate how similar they are on a scale from 0 to 10.

0 = Completely different (different topics, tone, structure, content)
5 = Somewhat similar (same general topic but different details/approach)
10 = Nearly identical (same ideas, similar wording, minimal differences)

Consider: content similarity, tone, structure, key points covered, and overall meaning.

Question: "{prompt}"

Response 1: "{unsteered_response}"

Response 2: "{steered_response}"

Provide only a single number from 0 to 10 as your similarity rating. Do not include any explanation.

Rating:"""

        similarity_rating = self._get_llm_rating(similarity_prompt)

        # Convert 0-10 scale to 0-1 scale
        # Maps 0->0 (very different), 10->1 (very similar)
        similarity_score = similarity_rating / 10

        return similarity_score

    def _get_llm_rating(self, prompt: str) -> float:
        """Get numerical rating from evaluator LLM."""
        # Format prompt with chat template if available
        if hasattr(self.model.tokenizer, "chat_template") and self.model.tokenizer.chat_template is not None:
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        inputs = self.model.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.hf_model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.model.tokenizer.pad_token_id,
            )

        response = self.model.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return self._extract_rating(response)

    def _extract_rating(self, model_output: str) -> float:
        """Extract numerical rating from model output."""
        # Try to find a number between 0 and 10
        numbers = re.findall(r"\b([0-9]|10)\b", model_output.strip())

        if numbers:
            # Take the first valid number found
            rating = float(numbers[0])
            return max(0.0, min(10.0, rating))  # Clamp to [0, 10]

        # Default to neutral if we can't extract a rating
        if self.verbose:
            print(f"Warning: Could not extract rating from: {model_output}")
        return 5.0

    def generate_and_evaluate(
        self,
        prompt: str,
        steering_method: "SteeringMethod",
        layer: int,
        trait_name: str,
        steering_strength: float = 2.0,
        trait_description: Optional[str] = None,
        max_new_tokens: int = 100,
    ) -> EvaluationResult:
        """
        Generate steered response and evaluate it against unsteered baseline.

        Args:
            prompt: Input prompt to evaluate
            steering_method: The steering method instance to apply
            layer: Layer index to apply steering
            trait_name: Name of the trait (e.g., "catholic", "cynical")
            steering_strength: Strength of steering to apply
            trait_description: Optional description of the trait
            max_new_tokens: Maximum tokens to generate

        Returns:
            EvaluationResult with trait_quality, answer_quality, and similarity scores
        """
        if self.verbose:
            print(f"\n🎯 Evaluating prompt: '{prompt[:50]}...'")
            print(f"   Trait: {trait_name}")
            print(f"   Steering strength: {steering_strength}")

        # Phase 1: Generate unsteered baseline response
        if self.verbose:
            print("   📝 Phase 1: Generating unsteered baseline...")

        unsteered_response = self._generate_unsteered_response(prompt, max_new_tokens)

        if self.verbose:
            print(f"   Unsteered: '{unsteered_response[:80]}...'")

        # Phase 2: Generate steered response
        if self.verbose:
            print("   📝 Phase 2: Generating steered response...")

        steered_response = self._generate_steered_response(
            prompt, steering_method, layer, steering_strength, max_new_tokens
        )

        if self.verbose:
            print(f"   Steered: '{steered_response[:80]}...'")

        # Phase 3: Evaluate trait and answer quality
        if self.verbose:
            print("   🧠 Phase 3: Evaluating quality...")

        trait_quality, answer_quality = self._evaluate_response(steered_response, prompt, trait_name, trait_description)

        # Phase 4: Evaluate similarity between unsteered and steered responses
        if self.verbose:
            print("   🔍 Phase 4: Evaluating similarity...")

        similarity_score = self._evaluate_similarity(prompt, unsteered_response, steered_response)

        if self.verbose:
            print(f"   Trait quality: {trait_quality:.3f}")
            print(f"   Answer quality: {answer_quality:.3f}")
            print(f"   Similarity: {similarity_score:.3f}")

        return EvaluationResult(
            trait_quality=trait_quality,
            answer_quality=answer_quality,
            steered_vs_unsteered_similarity=similarity_score,
            response=steered_response,
            unsteered_response=unsteered_response,
            prompt=prompt,
        )

    def _evaluate_multiple_traits(
        self,
        response: str,
        prompt: str,
        traits: List[str],
        trait_descriptions: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Evaluate response for multiple traits and answer quality.

        Args:
            response: Generated response to evaluate
            prompt: Original prompt
            traits: List of trait names to evaluate
            trait_descriptions: Optional descriptions for each trait

        Returns:
            Tuple of (trait_scores_dict, answer_quality)
        """
        # Ensure model is loaded
        if self.model is None:
            self._load_model()

        trait_descriptions = trait_descriptions or {}
        trait_scores = {}

        # Evaluate each trait individually
        for trait_name in traits:
            trait_desc = trait_descriptions.get(trait_name, trait_name)

            # Evaluate trait quality (0-10 scale)
            trait_prompt = f"""Rate how strongly this response demonstrates the trait "{trait_desc}" on a scale from 0 to 10.

0 = Strongly opposite to the trait (e.g., for "catholic" this would be atheistic/secular)
5 = Neutral, no clear expression of the trait
10 = Strongly demonstrates the trait

Consider the language, values, references, and overall perspective in the response.

Response to evaluate:
"{response}"

Provide only a single number from 0 to 10 as your rating. Do not include any explanation.

Rating:"""

            trait_rating = self._get_llm_rating(trait_prompt)

            # Convert 0-10 scale to -1 to 1 scale
            trait_quality = (trait_rating - 5) / 5  # Maps 0->-1, 5->0, 10->1
            trait_scores[trait_name] = trait_quality

        # Evaluate answer quality once (0-10 scale)
        quality_prompt = f"""Rate the quality of this response on a scale from 0 to 10.

Consider:
- Coherence and clarity
- Relevance to the question
- Absence of repetition or nonsensical text
- Overall meaningfulness

0 = Completely broken (repetitive, nonsensical, or incoherent)
5 = Mediocre but understandable
10 = High quality, clear, and coherent

Original question: "{prompt}"

Response to evaluate:
"{response}"

Provide only a single number from 0 to 10 as your rating. Do not include any explanation.

Rating:"""

        quality_rating = self._get_llm_rating(quality_prompt)
        answer_quality = quality_rating / 10  # Maps 0->0, 10->1

        return trait_scores, answer_quality

    def evaluate_multiple_traits(
        self,
        prompt: str,
        steering_method: "SteeringMethod",
        layer: int,
        traits: List[str],
        trait_descriptions: Optional[Dict[str, str]] = None,
        steering_strength: float = 2.0,
        max_new_tokens: int = 100,
        steering_method_name: Optional[str] = None,
    ) -> MultiTraitEvaluationResult:
        """
        Generate steered response and evaluate it for multiple traits.

        Args:
            prompt: Input prompt to evaluate
            steering_method: The steering method instance to apply
            layer: Layer index to apply steering
            traits: List of trait names to evaluate (e.g., ["italian", "honest"])
            trait_descriptions: Optional descriptions for each trait
            steering_strength: Strength of steering to apply
            max_new_tokens: Maximum tokens to generate
            steering_method_name: Optional name/identifier for the steering method

        Returns:
            MultiTraitEvaluationResult with trait scores, answer quality, and similarity
        """
        if self.verbose:
            print(f"\n🎯 Multi-trait evaluation: '{prompt[:50]}...'")
            print(f"   Traits: {', '.join(traits)}")
            print(f"   Steering: {steering_method_name or 'Unknown'}")
            print(f"   Strength: {steering_strength}")

        # Phase 1: Generate unsteered baseline response
        if self.verbose:
            print("   📝 Phase 1: Generating unsteered baseline...")

        unsteered_response = self._generate_unsteered_response(prompt, max_new_tokens)

        if self.verbose:
            print(f"   Unsteered: '{unsteered_response[:80]}...'")

        # Phase 2: Generate steered response
        if self.verbose:
            print("   📝 Phase 2: Generating steered response...")

        steered_response = self._generate_steered_response(
            prompt, steering_method, layer, steering_strength, max_new_tokens
        )

        if self.verbose:
            print(f"   Steered: '{steered_response[:80]}...'")

        # Phase 3: Evaluate multiple traits and answer quality
        if self.verbose:
            print("   🧠 Phase 3: Evaluating multiple traits...")

        trait_scores, answer_quality = self._evaluate_multiple_traits(
            steered_response, prompt, traits, trait_descriptions
        )

        # Phase 4: Evaluate similarity
        if self.verbose:
            print("   🔍 Phase 4: Evaluating similarity...")

        similarity_score = self._evaluate_similarity(prompt, unsteered_response, steered_response)

        if self.verbose:
            for trait, score in trait_scores.items():
                print(f"   {trait.capitalize()}: {score:+.3f}")
            print(f"   Answer quality: {answer_quality:.3f}")
            print(f"   Similarity: {similarity_score:.3f}")

        return MultiTraitEvaluationResult(
            trait_scores=trait_scores,
            answer_quality=answer_quality,
            steered_vs_unsteered_similarity=similarity_score,
            response=steered_response,
            unsteered_response=unsteered_response,
            prompt=prompt,
            steering_method_name=steering_method_name or "Unknown",
        )


def is_answer_above_thresholds(
    evaluation_result: EvaluationResult,
    trait_threshold: float,
    answer_threshold: float,
) -> bool:
    """
    Determine if evaluation result meets quality thresholds.

    Args:
        evaluation_result: Result from generate_and_evaluate()
        trait_threshold: Minimum trait quality (-1 to 1 scale)
        answer_threshold: Minimum answer quality (0 to 1 scale)

    Returns:
        True if both thresholds are met, False otherwise
    """
    return evaluation_result.trait_quality >= trait_threshold and evaluation_result.answer_quality >= answer_threshold


def is_multi_trait_answer_above_thresholds(
    evaluation_result: MultiTraitEvaluationResult,
    trait_thresholds: Dict[str, float],
    answer_threshold: float,
) -> bool:
    """
    Determine if multi-trait evaluation result meets quality thresholds.

    Args:
        evaluation_result: Result from evaluate_multiple_traits()
        trait_thresholds: Minimum trait quality for each trait (-1 to 1 scale)
        answer_threshold: Minimum answer quality (0 to 1 scale)

    Returns:
        True if all thresholds are met, False otherwise
    """
    # Check answer quality threshold
    if evaluation_result.answer_quality < answer_threshold:
        return False

    # Check each trait threshold
    for trait_name, threshold in trait_thresholds.items():
        trait_score = evaluation_result.get_trait_score(trait_name)
        if trait_score is None or trait_score < threshold:
            return False

    return True
