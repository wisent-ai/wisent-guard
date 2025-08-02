"""
Steering optimization module for improving benchmark performance.

This module handles training and optimizing different steering methods that can
improve model performance on benchmarks by steering internal activations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.response import Response
from wisent_guard.core.steering_methods.dac import DAC

from . import data_utils, metrics

logger = logging.getLogger(__name__)


@dataclass
class SteeringMethodConfig(ABC):
    """Base configuration for steering methods."""

    method_name: str = "base"
    layers: List[int] = None
    strengths: List[float] = None

    def __post_init__(self):
        if self.layers is None:
            self.layers = []
        if self.strengths is None:
            self.strengths = [1.0]


@dataclass
class DACConfig(SteeringMethodConfig):
    """Configuration for DAC (Dynamic Activation Composition) steering method."""

    method_name: str = "dac"
    entropy_thresholds: List[float] = None
    ptop_values: List[float] = None
    max_alpha_values: List[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.entropy_thresholds is None:
            self.entropy_thresholds = [1.0]
        if self.ptop_values is None:
            self.ptop_values = [0.4]
        if self.max_alpha_values is None:
            self.max_alpha_values = [2.0]


@dataclass
class SteeringResult:
    """Results from training and evaluating a steering method configuration."""

    method_name: str
    layer: int
    hyperparameters: Dict[str, Any]
    benchmark_metrics: Dict[str, float]
    training_success: bool
    training_stats: Dict[str, Any] = None


class SteeringMethodTrainer(ABC):
    """Abstract base class for training different steering methods."""

    @abstractmethod
    def create_method_instance(self, hyperparams: Dict[str, Any], device: str) -> Any:
        """Create an instance of the steering method with given hyperparameters."""

    @abstractmethod
    def train_method(
        self,
        method_instance: Any,
        train_samples: List[Dict],
        layer: int,
        model,
        tokenizer,
        device: str,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Train the steering method on training data."""

    @abstractmethod
    def apply_steering_and_evaluate(
        self,
        method_instance: Any,
        evaluation_samples: List[Dict],
        layer: int,
        strength: float,
        model,
        tokenizer,
        device: str,
        batch_size: int,
        max_length: int,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[List[str], List[str]]:
        """Apply steering and generate predictions for evaluation."""


class DACTrainer(SteeringMethodTrainer):
    """Trainer for DAC (Dynamic Activation Composition) steering method."""

    def create_method_instance(self, hyperparams: Dict[str, Any], device: str) -> DAC:
        """Create DAC instance with specified hyperparameters."""
        return DAC(
            device=device,
            dynamic_control=True,
            entropy_threshold=hyperparams.get("entropy_threshold", 1.0),
            ptop=hyperparams.get("ptop", 0.4),
            max_alpha=hyperparams.get("max_alpha", 2.0),
        )

    def train_method(
        self,
        dac_instance: DAC,
        train_samples: List[Dict],
        layer: int,
        model,
        tokenizer,
        device: str,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Train DAC on training data to create steering vectors."""
        try:
            # Set model reference for KL computation
            dac_instance.set_model_reference(model)

            # Extract contrastive pairs from training data using task's extractor
            contrastive_pairs = data_utils.get_task_contrastive_pairs(train_samples, task_name)

            if not contrastive_pairs:
                logger.warning(f"No contrastive pairs extracted from {task_name} training data")
                return False, {"error": "No contrastive pairs"}

            # Convert to ContrastivePairSet format
            pair_set = self._create_pair_set_from_extracted_pairs(contrastive_pairs, layer, model, tokenizer, device)

            # Train DAC
            training_result = dac_instance.train(pair_set, layer)

            success = training_result.get("success", False)
            logger.debug(f"DAC training on layer {layer}: {'Success' if success else 'Failed'}")

            return success, training_result

        except Exception as e:
            logger.error(f"DAC training failed on layer {layer}: {e}")
            return False, {"error": str(e)}

    def apply_steering_and_evaluate(
        self,
        dac_instance: DAC,
        evaluation_samples: List[Dict],
        layer: int,
        strength: float,
        model,
        tokenizer,
        device: str,
        batch_size: int,
        max_length: int,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[List[str], List[str]]:
        """Apply DAC steering and generate predictions using task extractor."""
        from ...task_interface import get_task

        predictions = []
        ground_truths = []

        # Get the task and its extractor
        task = get_task(task_name)
        extractor = task.get_extractor()

        # Pre-extract all questions and answers (optimization)
        questions = []
        answers = []

        for sample in evaluation_samples:
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                logger.warning(f"Skipping sample - extractor couldn't extract QA pair: {sample.keys()}")
                continue
            questions.append(qa_pair["formatted_question"])
            answers.append(qa_pair["correct_answer"])

        # Process questions with steering in batches (optimized approach)
        ground_truths.extend(answers)

        # Handle different model architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA-style models
            layer_module = model.model.layers[layer]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT2-style models
            layer_module = model.transformer.h[layer]
        else:
            raise ValueError("Unsupported model architecture for DAC steering")

        # Process in batches with steering
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating predictions with steering"):
            batch_questions = questions[i : i + batch_size]

            # First, get actual lengths (before padding) for proper steering
            actual_lengths = []
            for question in batch_questions:
                tokens = tokenizer(question, return_tensors="pt")
                actual_lengths.append(tokens["input_ids"].shape[1])

            # Create batched steering hook that handles variable lengths
            def create_batched_steering_hook(actual_lengths):
                def steering_hook(module, input, output):
                    hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]

                    # Apply steering to each sample's actual last token
                    for j, actual_length in enumerate(actual_lengths):
                        if j < hidden_states.shape[0]:  # Safety check for batch size
                            # Get the actual last token (before padding)
                            last_token = hidden_states[j : j + 1, actual_length - 1 : actual_length, :]
                            steered = dac_instance.apply_steering(last_token, strength=strength)
                            hidden_states[j : j + 1, actual_length - 1 : actual_length, :] = steered

                    return (hidden_states,) + output[1:]

                return steering_hook

            # Register the batched hook
            batched_hook = create_batched_steering_hook(actual_lengths)
            handle = layer_module.register_forward_hook(batched_hook)

            try:
                # Tokenize batch with padding for generation
                inputs = tokenizer(
                    batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=False,  # Disable cache to avoid cache_position errors
                    )

                # Decode responses for each item in batch
                for j, (output, question) in enumerate(zip(outputs, batch_questions)):
                    response = tokenizer.decode(output, skip_special_tokens=True)
                    prediction = response[len(question) :].strip()
                    predictions.append(prediction)

            finally:
                handle.remove()

        return predictions, ground_truths

    def _create_pair_set_from_extracted_pairs(
        self, extracted_pairs: List[Dict], layer_index: int, model, tokenizer, device: str
    ) -> ContrastivePairSet:
        """Convert extracted pairs to ContrastivePairSet format with proper activation extraction."""
        pair_set = ContrastivePairSet(name="dac_training", task_type="mathematical_reasoning")

        logger.info(f"Creating {len(extracted_pairs)} contrastive pairs for layer {layer_index}")

        for pair_data in tqdm(extracted_pairs, desc="Creating contrastive pairs"):
            # Extract data from GSM8K format
            try:
                question = pair_data["question"]
                correct_answer = pair_data["correct_answer"]
                incorrect_answer = pair_data["incorrect_answer"]

                # Extract activations for correct and incorrect responses
                correct_activations = self._extract_activations_for_text(
                    f"{question} {correct_answer}", layer_index, model, tokenizer, device
                )
                incorrect_activations = self._extract_activations_for_text(
                    f"{question} {incorrect_answer}", layer_index, model, tokenizer, device
                )

                # Create Response objects
                positive_response = Response(text=correct_answer, activations=correct_activations)
                negative_response = Response(text=incorrect_answer, activations=incorrect_activations)

                # Create ContrastivePair
                contrastive_pair = ContrastivePair(
                    prompt=question, positive_response=positive_response, negative_response=negative_response
                )

                pair_set.pairs.append(contrastive_pair)

            except Exception as e:
                logger.warning(f"Failed to create contrastive pair: {e}")
                continue

        logger.info(f"Successfully created ContrastivePairSet with {len(pair_set.pairs)} pairs")
        return pair_set

    def _extract_activations_for_text(self, text: str, layer_index: int, model, tokenizer, device: str) -> torch.Tensor:
        """Extract activations from a specific layer for given text."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        activations = []

        def hook(module, input, output):
            # Extract the last token's activations
            hidden_states = output[0]
            last_token_activations = hidden_states[:, -1, :]
            activations.append(last_token_activations.detach().cpu())

        # Handle different model architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA-style models
            layer_module = model.model.layers[layer_index]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT2-style models
            layer_module = model.transformer.h[layer_index]
        else:
            raise ValueError("Unsupported model architecture for activation extraction")

        handle = layer_module.register_forward_hook(hook)

        with torch.no_grad():
            model(**inputs)

        handle.remove()
        return activations[0].squeeze(0)


class SteeringOptimizer:
    """
    Optimizes steering methods for improving benchmark performance.

    The steering optimization process:
    1. Train steering methods on training data
    2. Evaluate steering performance on validation data using benchmark metrics
    3. Select best configuration based on benchmark performance
    4. Test final steering method on test data
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.trainers = {"dac": DACTrainer()}

    def register_trainer(self, method_name: str, trainer: SteeringMethodTrainer):
        """Register a new steering method trainer."""
        self.trainers[method_name] = trainer
        self.logger.info(f"Registered trainer for steering method: {method_name}")

    def optimize_steering_hyperparameters(
        self,
        config: SteeringMethodConfig,
        train_samples: List[Dict],
        validation_samples: List[Dict],
        model,
        tokenizer,
        device: str,
        batch_size: int = 32,
        max_length: int = 512,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[Dict[str, Any], List[SteeringResult]]:
        """
        Optimize hyperparameters for a steering method using grid search.

        Args:
            config: Steering method configuration with hyperparameter ranges
            train_samples: Training samples for method training
            validation_samples: Validation samples for evaluation
            model: Language model
            tokenizer: Model tokenizer
            device: Device to run on
            batch_size: Batch size for processing
            max_length: Maximum sequence length

        Returns:
            Tuple of (best_config, all_results)
        """
        method_name = config.method_name

        if method_name not in self.trainers:
            raise ValueError(f"No trainer registered for method: {method_name}")

        trainer = self.trainers[method_name]

        # Generate all hyperparameter combinations
        hyperparameter_combinations = self._generate_hyperparameter_combinations(config)

        self.logger.info(f"Starting {method_name} optimization with {len(hyperparameter_combinations)} configurations")

        best_config = None
        best_score = -1
        all_results = []

        for i, (layer, strength, hyperparams) in enumerate(
            tqdm(hyperparameter_combinations, desc="Optimizing steering hyperparameters")
        ):
            self.logger.debug(
                f"Testing {method_name} config {i + 1}/{len(hyperparameter_combinations)}: "
                f"layer={layer}, strength={strength}, hyperparams={hyperparams}"
            )

            try:
                # Create method instance
                method_instance = trainer.create_method_instance(hyperparams, device)

                # Train the method
                training_success, training_stats = trainer.train_method(
                    method_instance, train_samples, layer, model, tokenizer, device, task_name, max_new_tokens
                )

                if not training_success:
                    self.logger.warning(f"Training failed for config {i + 1}")
                    result = SteeringResult(
                        method_name=method_name,
                        layer=layer,
                        hyperparameters={**hyperparams, "strength": strength},
                        benchmark_metrics={"accuracy": 0.0},
                        training_success=False,
                        training_stats=training_stats,
                    )
                    all_results.append(result)
                    continue

                # Evaluate on validation data
                predictions, ground_truths = trainer.apply_steering_and_evaluate(
                    method_instance,
                    validation_samples,
                    layer,
                    strength,
                    model,
                    tokenizer,
                    device,
                    batch_size,
                    max_length,
                    task_name,
                    max_new_tokens,
                )

                # Calculate benchmark metrics
                benchmark_metrics = metrics.evaluate_benchmark_performance(predictions, ground_truths, task_name)

                result = SteeringResult(
                    method_name=method_name,
                    layer=layer,
                    hyperparameters={**hyperparams, "strength": strength},
                    benchmark_metrics=benchmark_metrics,
                    training_success=True,
                    training_stats=training_stats,
                )
                all_results.append(result)

                # Check if this is the best configuration
                score = benchmark_metrics.get("accuracy", 0.0)
                if score > best_score:
                    best_score = score
                    best_config = {
                        "method": method_name,
                        "layer": layer,
                        "strength": strength,
                        **hyperparams,
                        "benchmark_metrics": benchmark_metrics,
                        "method_instance": method_instance,
                    }

                self.logger.debug(f"Config {i + 1} - Benchmark accuracy: {score:.3f}")

            except Exception as e:
                self.logger.error(f"Failed to evaluate config {i + 1}: {e}")
                result = SteeringResult(
                    method_name=method_name,
                    layer=layer,
                    hyperparameters={**hyperparams, "strength": strength},
                    benchmark_metrics={"accuracy": 0.0},
                    training_success=False,
                    training_stats={"error": str(e)},
                )
                all_results.append(result)
                continue

        if best_config is None:
            self.logger.warning("No successful steering configuration found")
            # Return a default configuration
            best_config = {
                "method": method_name,
                "layer": config.layers[0] if config.layers else 0,
                "strength": config.strengths[0] if config.strengths else 1.0,
                "benchmark_metrics": {"accuracy": 0.0},
                "method_instance": None,
            }
        else:
            self.logger.info(
                f"Best {method_name} config: layer={best_config['layer']}, "
                f"accuracy={best_config['benchmark_metrics']['accuracy']:.3f}"
            )

        return best_config, all_results

    def _generate_hyperparameter_combinations(
        self, config: SteeringMethodConfig
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Generate all combinations of hyperparameters for grid search."""
        combinations = []

        if isinstance(config, DACConfig):
            # Generate DAC hyperparameter combinations
            for layer in config.layers:
                for strength in config.strengths:
                    for entropy_threshold in config.entropy_thresholds:
                        for ptop in config.ptop_values:
                            for max_alpha in config.max_alpha_values:
                                hyperparams = {
                                    "entropy_threshold": entropy_threshold,
                                    "ptop": ptop,
                                    "max_alpha": max_alpha,
                                }
                                combinations.append((layer, strength, hyperparams))
        else:
            # Generic handling for other steering methods
            for layer in config.layers:
                for strength in config.strengths:
                    combinations.append((layer, strength, {}))

        return combinations
