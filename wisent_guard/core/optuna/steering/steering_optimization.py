"""
Steering optimization module for improving benchmark performance.

This module handles training and optimizing different steering methods that can
improve model performance on benchmarks by steering internal activations.
"""

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from wisent_guard.core.activations.core import ActivationAggregationStrategy
from wisent_guard.core.classifier.classifier import Classifier
from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.optuna.classifier import (
    CacheConfig,
    ClassifierCache,
    ClassifierOptimizationConfig,
    GenerationConfig,
    OptunaClassifierOptimizer,
)
from wisent_guard.core.optuna.steering import data_utils, metrics
from wisent_guard.core.response import Response
from wisent_guard.core.steering_methods.dac import DAC
from wisent_guard.core.task_interface import get_task

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
    baseline_metrics: Dict[str, float] = None
    comparative_metrics: Dict[str, Any] = None


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

    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.trainers = {"dac": DACTrainer()}

        # Initialize classifier cache for reusing trained classifiers
        if cache_config is None:
            cache_config = CacheConfig(cache_dir="./steering_classifier_cache")
        self.classifier_cache = ClassifierCache(cache_config)

        # Session-level classifier caching for current optimization run
        self._session_classifier = None  # Best classifier for current session
        self._session_classifier_metadata = {}  # Layer, model_type, performance, etc.
        self._session_cache_key = None  # Track current session

    def register_trainer(self, method_name: str, trainer: SteeringMethodTrainer):
        """Register a new steering method trainer."""
        self.trainers[method_name] = trainer
        self.logger.info(f"Registered trainer for steering method: {method_name}")

    def optimize_steering_hyperparameters(
        self,
        config: SteeringMethodConfig,
        classifier_optimization_config: ClassifierOptimizationConfig,
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
            classifier_optimization_config: Configuration for classifier optimization
            train_samples: Training samples for method training
            validation_samples: Validation samples for evaluation
            model: Language model
            tokenizer: Model tokenizer
            device: Device to run on
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            task_name: Task name for evaluation
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (best_config, all_results)
        """
        method_name = config.method_name

        if method_name not in self.trainers:
            raise ValueError(f"No trainer registered for method: {method_name}")

        trainer = self.trainers[method_name]

        # Load best classifier once at the start of optimization
        self.logger.info("Loading/training classifier for evaluation...")
        contrastive_pairs = data_utils.get_task_contrastive_pairs(train_samples, task_name)

        classifier = self.load_or_find_best_classifier(
            model=model, optimization_config=classifier_optimization_config, contrastive_pairs=contrastive_pairs
        )

        if classifier is None:
            raise ValueError(
                f"Could not load or train classifier for {classifier_optimization_config.model_name}/{task_name}"
            )

        self.logger.info(f"Using classifier: {self._session_classifier_metadata}")

        # Collect baseline predictions once for all trials
        self.logger.info("Collecting baseline predictions for comparison...")
        baseline_predictions, ground_truths = self.collect_baseline_predictions(
            validation_samples, model, tokenizer, classifier, device, batch_size, max_length, task_name, max_new_tokens
        )

        # Calculate baseline metrics with integrated classifier scoring
        classifier_scorer = lambda predictions, description: self.score_predictions_with_classifier(
            predictions, model, tokenizer, device, max_length, description
        )
        baseline_benchmark_metrics = metrics.evaluate_benchmark_performance(
            baseline_predictions, ground_truths, task_name, classifier_scorer=classifier_scorer
        )
        self.logger.info(f"Baseline performance: {baseline_benchmark_metrics}")

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

                # Evaluate on validation data with steering
                steered_predictions, steered_ground_truths = trainer.apply_steering_and_evaluate(
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

                # Compare baseline vs steered predictions using enhanced metrics
                enhanced_metrics = self.compare_predictions(
                    baseline_predictions,
                    steered_predictions,
                    ground_truths,
                    model,
                    tokenizer,
                    device,
                    max_length,
                    task_name,
                )

                # Extract steered metrics for compatibility
                benchmark_metrics = enhanced_metrics["steered"]
                baseline_metrics_for_result = enhanced_metrics["baseline"]
                comparative_metrics = enhanced_metrics["improvement"]

                result = SteeringResult(
                    method_name=method_name,
                    layer=layer,
                    hyperparameters={**hyperparams, "strength": strength},
                    benchmark_metrics=benchmark_metrics,
                    baseline_metrics=baseline_metrics_for_result,
                    comparative_metrics=comparative_metrics,
                    training_success=True,
                    training_stats=training_stats,
                )
                all_results.append(result)

                # Standard Optuna practice: optimize steered accuracy directly
                steered_accuracy = benchmark_metrics.get("accuracy", 0.0)
                baseline_accuracy = baseline_metrics_for_result.get("accuracy", 0.0)
                improvement_delta = steered_accuracy - baseline_accuracy

                if steered_accuracy > best_score:
                    best_score = steered_accuracy
                    best_config = {
                        "method": method_name,
                        "layer": layer,
                        "strength": strength,
                        **hyperparams,
                        "benchmark_metrics": benchmark_metrics,
                        "baseline_metrics": baseline_metrics_for_result,
                        "method_instance": method_instance,
                    }

                self.logger.debug(
                    f"Config {i + 1} - Baseline: {baseline_accuracy:.3f}, "
                    f"Steered: {steered_accuracy:.3f}, Delta: {improvement_delta:+.3f}"
                )

            except Exception as e:
                self.logger.error(f"Failed to evaluate config {i + 1}: {e}")
                result = SteeringResult(
                    method_name=method_name,
                    layer=layer,
                    hyperparameters={**hyperparams, "strength": strength},
                    benchmark_metrics={"accuracy": 0.0},
                    baseline_metrics=baseline_benchmark_metrics,
                    comparative_metrics={"accuracy_delta": 0.0, "improvement_rate": 0.0},
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
            steered_acc = best_config["benchmark_metrics"]["accuracy"]
            baseline_acc = best_config.get("baseline_metrics", {}).get("accuracy", 0.0)
            improvement = steered_acc - baseline_acc

            self.logger.info(
                f"Best {method_name} config (optimized for steered accuracy): "
                f"layer={best_config['layer']}, steered={steered_acc:.3f} "
                f"(baseline={baseline_acc:.3f}, Δ={improvement:+.3f})"
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

    def collect_baseline_predictions(
        self,
        evaluation_samples: List[Dict],
        model,
        tokenizer,
        classifier: Classifier,
        device: str,
        batch_size: int,
        max_length: int,
        task_name: str,
        max_new_tokens: int = 200,
    ) -> Tuple[List[str], List[str]]:
        """
        Collect unsteered model predictions for baseline comparison.
        Uses the same evaluation logic as steered evaluation but without steering hooks.

        Args:
            evaluation_samples: Samples to evaluate
            model: Language model
            tokenizer: Model tokenizer
            classifier: Trained classifier for evaluation
            device: Device to run on
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            task_name: Task name for evaluation
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (predictions, ground_truths)
        """
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
                self.logger.warning(f"Skipping sample - extractor couldn't extract QA pair: {sample.keys()}")
                continue
            questions.append(qa_pair["formatted_question"])
            answers.append(qa_pair["correct_answer"])

        # Process questions WITHOUT steering in batches
        ground_truths.extend(answers)

        # Process in batches without steering
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating baseline predictions"):
            batch_questions = questions[i : i + batch_size]

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

        return predictions, ground_truths

    def _extract_activation_for_text(
        self,
        text: str,
        layer_index: int,
        aggregation_strategy: str,
        model,
        tokenizer,
        device: str,
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Extract activation from text at specified layer with aggregation.

        Args:
            text: Input text to extract activation from
            layer_index: Layer index to extract from
            aggregation_strategy: Aggregation strategy string (e.g., "mean_pooling")
            model: Language model
            tokenizer: Model tokenizer
            device: Device to run on
            max_length: Maximum sequence length

        Returns:
            Aggregated activation tensor
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        activations = []

        def hook(module, input, output):
            # Extract hidden states from the layer
            hidden_states = output[0] if isinstance(output, tuple) else output
            activations.append(hidden_states.detach().cpu())

        # Handle different model architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA-style models
            layer_module = model.model.layers[layer_index]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT2-style models
            layer_module = model.transformer.h[layer_index]
        else:
            raise ValueError("Unsupported model architecture for activation extraction")

        # Register hook and run forward pass
        handle = layer_module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                _ = model(**inputs)
        finally:
            handle.remove()

        if not activations:
            raise ValueError("No activations extracted")

        # Get the activation tensor [1, seq_len, hidden_dim]
        activation_tensor = activations[0]

        # Apply aggregation strategy
        if (
            aggregation_strategy == "mean_pooling"
            or aggregation_strategy == ActivationAggregationStrategy.MEAN_POOLING.value
        ):
            aggregated = torch.mean(activation_tensor, dim=1)  # [1, hidden_dim]
        elif (
            aggregation_strategy == "last_token"
            or aggregation_strategy == ActivationAggregationStrategy.LAST_TOKEN.value
        ):
            aggregated = activation_tensor[:, -1, :]  # [1, hidden_dim]
        elif (
            aggregation_strategy == "first_token"
            or aggregation_strategy == ActivationAggregationStrategy.FIRST_TOKEN.value
        ):
            aggregated = activation_tensor[:, 0, :]  # [1, hidden_dim]
        elif (
            aggregation_strategy == "max_pooling"
            or aggregation_strategy == ActivationAggregationStrategy.MAX_POOLING.value
        ):
            aggregated = torch.max(activation_tensor, dim=1)[0]  # [1, hidden_dim]
        else:
            # Default to mean pooling if unknown
            self.logger.warning(f"Unknown aggregation strategy {aggregation_strategy}, using mean pooling")
            aggregated = torch.mean(activation_tensor, dim=1)

        return aggregated.squeeze(0)  # Return [hidden_dim] tensor

    def score_predictions_with_classifier(
        self,
        predictions: List[str],
        model,
        tokenizer,
        device: str,
        max_length: int = 512,
        description: str = "predictions",
    ) -> List[float]:
        """
        Score predictions using the cached classifier.

        This is the core feature that was requested - using the optimized classifier
        to score unsteered vs steered generations.

        Args:
            predictions: Text predictions to score
            model: Language model for activation extraction
            tokenizer: Model tokenizer
            device: Device to run on
            max_length: Maximum sequence length
            description: Description for logging

        Returns:
            List of classifier scores/probabilities for each prediction
        """
        if self._session_classifier is None:
            self.logger.warning("No cached classifier available for scoring")
            return [0.5] * len(predictions)  # Return neutral scores

        if not predictions:
            self.logger.debug("No predictions to score")
            return []

        # Get classifier metadata
        layer = self._session_classifier_metadata.get("layer", 12)
        aggregation = self._session_classifier_metadata.get("aggregation", "mean_pooling")

        self.logger.info(
            f"Scoring {len(predictions)} {description} with cached classifier (layer={layer}, aggregation={aggregation})"
        )

        confidence_scores = []

        # Process predictions in batches for efficiency
        batch_size = 8  # Smaller batch size to avoid OOM
        for i in range(0, len(predictions), batch_size):
            batch_predictions = predictions[i : i + batch_size]
            batch_activations = []

            # Extract activations for each prediction in the batch
            for pred_text in batch_predictions:
                try:
                    # Extract activation for this prediction text
                    activation = self._extract_activation_for_text(
                        text=pred_text,
                        layer_index=layer,
                        aggregation_strategy=aggregation,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        max_length=max_length,
                    )
                    batch_activations.append(activation)

                except Exception as e:
                    self.logger.debug(f"Failed to extract activation for prediction: {e}")
                    # Use neutral score for failed extractions
                    confidence_scores.append(0.5)
                    continue

            if batch_activations:
                try:
                    # Stack activations into batch tensor
                    batch_tensor = torch.stack(batch_activations)

                    # Convert to numpy for sklearn classifier
                    batch_numpy = batch_tensor.detach().cpu().numpy()

                    # Get prediction probabilities from classifier
                    probabilities = self._session_classifier.predict_proba(batch_numpy)

                    # Extract confidence scores (probability for positive class)
                    # Assuming binary classification with class 1 as positive
                    if probabilities.shape[1] > 1:
                        batch_scores = probabilities[:, 1].tolist()  # Probability of positive class
                    else:
                        batch_scores = probabilities[:, 0].tolist()  # Single class probability

                    confidence_scores.extend(batch_scores)

                except Exception as e:
                    self.logger.warning(f"Failed to score batch of activations: {e}")
                    # Add neutral scores for failed batch
                    confidence_scores.extend([0.5] * len(batch_activations))

        # Ensure we have scores for all predictions
        while len(confidence_scores) < len(predictions):
            confidence_scores.append(0.5)  # Pad with neutral scores if needed

        # Truncate if we have too many scores (shouldn't happen)
        confidence_scores = confidence_scores[: len(predictions)]

        # Log statistics
        avg_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        self.logger.debug(
            f"Generated {len(confidence_scores)} classifier confidence scores for {description} (avg={avg_score:.3f})"
        )

        return confidence_scores

    def compare_predictions(
        self,
        baseline_predictions: List[str],
        steered_predictions: List[str],
        ground_truths: List[str],
        model,
        tokenizer,
        device: str,
        max_length: int = 512,
        task_name: str = "gsm8k",
    ) -> Dict[str, Any]:
        """
        Compare baseline vs steered predictions using benchmark metrics and classifier scores.

        Args:
            baseline_predictions: Unsteered model predictions
            steered_predictions: Steered model predictions
            ground_truths: Ground truth answers
            model: Language model for classifier scoring
            tokenizer: Model tokenizer
            device: Device to run on
            max_length: Maximum sequence length
            task_name: Task name for evaluation metrics

        Returns:
            Enhanced metrics with baseline vs steered comparison including classifier scores
        """
        # Create classifier scorer function for metrics integration
        classifier_scorer = lambda predictions, description: self.score_predictions_with_classifier(
            predictions, model, tokenizer, device, max_length, description
        )

        # Calculate standard benchmark metrics with integrated classifier confidence scores
        baseline_metrics = metrics.evaluate_benchmark_performance(
            baseline_predictions, ground_truths, task_name, classifier_scorer=classifier_scorer
        )
        steered_metrics = metrics.evaluate_benchmark_performance(
            steered_predictions, ground_truths, task_name, classifier_scorer=classifier_scorer
        )

        # Extract classifier scores from integrated metrics
        baseline_scores = [
            detail.get("classifier_confidence", 0.5) for detail in baseline_metrics.get("evaluation_details", [])
        ]
        steered_scores = [
            detail.get("classifier_confidence", 0.5) for detail in steered_metrics.get("evaluation_details", [])
        ]

        # Calculate improvement metrics
        accuracy_delta = steered_metrics.get("accuracy", 0) - baseline_metrics.get("accuracy", 0)
        f1_delta = steered_metrics.get("f1", 0) - baseline_metrics.get("f1", 0)

        # Calculate classifier score improvements
        avg_baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        avg_steered_score = sum(steered_scores) / len(steered_scores) if steered_scores else 0.0
        classifier_score_delta = avg_steered_score - avg_baseline_score

        return {
            "baseline": {
                "accuracy": baseline_metrics.get("accuracy", 0.0),
                "f1": baseline_metrics.get("f1", 0.0),
                "classifier_scores": baseline_scores,
                "avg_classifier_score": avg_baseline_score,
                "predictions": baseline_predictions,
            },
            "steered": {
                "accuracy": steered_metrics.get("accuracy", 0.0),
                "f1": steered_metrics.get("f1", 0.0),
                "classifier_scores": steered_scores,
                "avg_classifier_score": avg_steered_score,
                "predictions": steered_predictions,
            },
            "improvement": {
                "accuracy_delta": accuracy_delta,
                "f1_delta": f1_delta,
                "classifier_score_delta": classifier_score_delta,
            },
        }

    def load_or_find_best_classifier(
        self,
        model,
        optimization_config: Optional[ClassifierOptimizationConfig] = None,
        model_name: Optional[str] = None,
        task_name: Optional[str] = None,
        contrastive_pairs: Optional[List] = None,
        force_reoptimize: bool = False,
    ) -> Optional[Classifier]:
        """
        Load or train the best classifier for current steering session.

        On first call: Run full classifier optimization and cache result for session
        On subsequent calls: Return cached classifier from current session

        Args:
            model: Language model (wisent_guard Model wrapper)
            optimization_config: Primary configuration source
            model_name: Fallback model name if optimization_config not provided
            task_name: Fallback task name if optimization_config not provided
            contrastive_pairs: Training data for classifier optimization
            force_reoptimize: Force reoptimization even if session classifier exists

        Returns:
            Best trained classifier or None if optimization failed
        """
        # Extract configuration
        if optimization_config is not None:
            model_name = optimization_config.model_name
            task_name = getattr(optimization_config, "task_name", task_name)
            limit = getattr(optimization_config, "data_limit", 100)
        else:
            limit = 100  # Default data limit

        if not model_name or not task_name:
            raise ValueError("model_name and task_name must be provided either via optimization_config or directly")

        # Create session cache key
        session_cache_key = f"{model_name}_{task_name}"

        # Check if we already have a classifier for this session
        if (
            not force_reoptimize
            and self._session_classifier is not None
            and self._session_cache_key == session_cache_key
        ):
            self.logger.info("Using cached classifier from current session")
            return self._session_classifier

        # First call or forced reoptimization - run classifier optimization
        self.logger.info("Running classifier optimization (first trial in session)")

        if not contrastive_pairs:
            self.logger.error("contrastive_pairs required for classifier optimization")
            return None

        try:
            # Create configuration for classifier optimization if not provided
            if optimization_config is None:
                optimization_config = ClassifierOptimizationConfig(
                    model_name=model_name,
                    device="auto",
                    n_trials=20,  # Reasonable number for steering optimization
                    model_types=["logistic", "mlp"],
                    primary_metric="f1",
                )

            # Create generation config for activation pre-generation
            generation_config = GenerationConfig(
                layer_search_range=(0, 23),  # Will be auto-detected from model
                aggregation_methods=[
                    ActivationAggregationStrategy.MEAN_POOLING,
                    ActivationAggregationStrategy.LAST_TOKEN,
                    ActivationAggregationStrategy.FIRST_TOKEN,
                    ActivationAggregationStrategy.MAX_POOLING,
                ],
                cache_dir="./cache/steering_activations",
                device=optimization_config.device,
                batch_size=32,
            )

            # Create classifier optimizer
            classifier_optimizer = OptunaClassifierOptimizer(
                optimization_config=optimization_config,
                generation_config=generation_config,
                cache_config=self.classifier_cache.config,
            )

            # Run classifier optimization
            self.logger.info(f"Optimizing classifier for {model_name}/{task_name} with {len(contrastive_pairs)} pairs")
            result = classifier_optimizer.optimize(
                model=model,
                contrastive_pairs=contrastive_pairs,
                task_name=task_name,
                model_name=model_name,
                limit=limit,
            )

            if result.best_value > 0:
                # Get the best configuration and classifier
                best_config = result.get_best_config()
                best_classifier = result.best_classifier

                # Cache for current session
                self._session_classifier = best_classifier
                self._session_classifier_metadata = {
                    "layer": best_config["layer"],
                    "aggregation": best_config["aggregation"],
                    "model_type": best_config["model_type"],
                    "threshold": best_config["threshold"],
                    "f1_score": result.best_value,
                    "hyperparameters": best_config.get("hyperparameters", {}),
                }
                self._session_cache_key = session_cache_key

                self.logger.info(
                    f"Cached best classifier for session: layer_{best_config['layer']} "
                    f"{best_config['model_type']} (F1: {result.best_value:.3f})"
                )

                return best_classifier
            self.logger.warning("Classifier optimization failed - no successful trials")
            return None

        except Exception as e:
            self.logger.error(f"Failed to run classifier optimization: {e}")
            traceback.print_exc()
            return None

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached classifiers."""
        return self.classifier_cache.get_cache_info()

    def clear_classifier_cache(self, keep_recent_hours: float = 24.0) -> int:
        """Clear old cached classifiers."""
        return self.classifier_cache.clear_cache(keep_recent_hours=keep_recent_hours)
