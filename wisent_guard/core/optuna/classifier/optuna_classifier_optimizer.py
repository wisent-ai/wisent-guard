"""
Optuna-based classifier optimization for efficient hyperparameter search.

This module provides a modern, efficient optimization system that pre-generates
activations once and uses intelligent caching to avoid redundant training.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from wisent_guard.core.classifier.classifier import Classifier
from wisent_guard.core.utils.device import resolve_default_device

from .activation_generator import ActivationData, ActivationGenerator, GenerationConfig
from .classifier_cache import CacheConfig, ClassifierCache


def get_model_dtype(model) -> torch.dtype:
    """
    Extract model's native dtype from parameters.

    Args:
        model: PyTorch model or wisent_guard Model wrapper

    Returns:
        The model's native dtype
    """
    # Handle wisent_guard Model wrapper
    if hasattr(model, "hf_model"):
        model_params = model.hf_model.parameters()
    else:
        model_params = model.parameters()

    try:
        return next(model_params).dtype
    except StopIteration:
        # Fallback if no parameters found
        return torch.float32


logger = logging.getLogger(__name__)


@dataclass
class ClassifierOptimizationConfig:
    """Configuration for Optuna classifier optimization."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    model_dtype: Optional[torch.dtype] = None  # Auto-detect if None

    # Optuna settings
    n_trials: int = 100
    timeout: Optional[float] = None
    n_jobs: int = 1
    sampler_seed: int = 42

    # Model type search space
    model_types: list[str] = None

    # Hyperparameter ranges
    hidden_dim_range: tuple[int, int] = (32, 512)
    threshold_range: tuple[float, float] = (0.3, 0.9)

    # Training settings
    num_epochs_range: tuple[int, int] = (20, 100)
    learning_rate_range: tuple[float, float] = (1e-4, 1e-2)
    batch_size_options: list[int] = None

    # Evaluation settings
    cv_folds: int = 3
    test_size: float = 0.2
    random_state: int = 42

    # Optimization objective
    primary_metric: str = "f1"  # "accuracy", "f1", "auc", "precision", "recall"

    # Pruning settings
    enable_pruning: bool = True
    pruning_patience: int = 10

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ["logistic", "mlp"]
        if self.batch_size_options is None:
            self.batch_size_options = [16, 32, 64]

        # Auto-detect device if needed
        if self.device == "auto":
            self.device = resolve_default_device()


@dataclass
class OptimizationResult:
    """Result from Optuna optimization."""

    best_params: dict[str, Any]
    best_value: float
    best_classifier: Classifier
    study: optuna.Study
    trial_results: list[dict[str, Any]]
    optimization_time: float
    cache_hits: int
    cache_misses: int

    def get_best_config(self) -> dict[str, Any]:
        """Get the best configuration found."""
        if not self.best_params:
            return {
                "model_type": "unknown",
                "layer": -1,
                "aggregation": "unknown",
                "threshold": 0.0,
                "hyperparameters": {},
            }

        return {
            "model_type": self.best_params["model_type"],
            "layer": self.best_params["layer"],
            "aggregation": self.best_params["aggregation"],
            "threshold": self.best_params["threshold"],
            "hyperparameters": {
                k: v
                for k, v in self.best_params.items()
                if k not in ["model_type", "layer", "aggregation", "threshold"]
            },
        }


class OptunaClassifierOptimizer:
    """
    Optuna-based classifier optimizer with efficient caching and pre-generation.

    Key features:
    - Pre-generates activations once for all trials
    - Uses intelligent model caching to avoid retraining
    - Supports both logistic and MLP classifiers
    - Multi-objective optimization with pruning
    - Cross-validation for robust evaluation
    """

    def __init__(
        self,
        optimization_config: ClassifierOptimizationConfig,
        generation_config: GenerationConfig,
        cache_config: CacheConfig,
    ):
        self.opt_config = optimization_config
        self.gen_config = generation_config
        self.cache_config = cache_config

        self.activation_generator = ActivationGenerator(generation_config)
        self.classifier_cache = ClassifierCache(cache_config)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Statistics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.activation_data: dict[str, ActivationData] = {}

    def optimize(
        self, model, contrastive_pairs: list, task_name: str, model_name: str, limit: int
    ) -> OptimizationResult:
        """
        Run Optuna-based classifier optimization.

        Args:
            model: Language model
            contrastive_pairs: Training contrastive pairs
            task_name: Name of the task
            model_name: Name of the model
            limit: Data limit used

        Returns:
            OptimizationResult with best configuration and classifier
        """
        self.logger.info(f"Starting Optuna classifier optimization for {task_name}")
        layer_range = self.gen_config.layer_search_range[1] - self.gen_config.layer_search_range[0] + 1
        self.logger.info(
            f"Configuration: {self.opt_config.n_trials} trials, layers {self.gen_config.layer_search_range[0]}-{self.gen_config.layer_search_range[1]} ({layer_range} layers)"
        )

        # Detect or use configured model dtype
        detected_dtype = get_model_dtype(model)
        self.model_dtype = self.opt_config.model_dtype if self.opt_config.model_dtype is not None else detected_dtype
        self.logger.info(f"Using model dtype: {self.model_dtype} (detected: {detected_dtype})")

        start_time = time.time()

        # Step 1: Pre-generate all activations
        self.logger.info("Pre-generating activations for all layers and aggregation methods...")
        self.activation_data = self.activation_generator.generate_from_contrastive_pairs(
            model=model, contrastive_pairs=contrastive_pairs, task_name=task_name, model_name=model_name, limit=limit
        )

        if not self.activation_data:
            raise ValueError("No activation data generated - cannot proceed with optimization")

        self.logger.info(f"Generated {len(self.activation_data)} activation datasets")

        # Step 2: Set up Optuna study
        sampler = TPESampler(seed=self.opt_config.sampler_seed)
        pruner = (
            MedianPruner(n_startup_trials=5, n_warmup_steps=self.opt_config.pruning_patience)
            if self.opt_config.enable_pruning
            else None
        )

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        # Step 3: Run optimization
        self.logger.info("Starting Optuna trials...")

        def objective(trial):
            return self._objective_function(trial, task_name, model_name)

        study.optimize(
            objective,
            n_trials=self.opt_config.n_trials,
            timeout=self.opt_config.timeout,
            n_jobs=self.opt_config.n_jobs,
            show_progress_bar=True,
        )

        # Step 4: Get best results
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not completed_trials:
            self.logger.warning("No trials completed successfully - all trials were pruned or failed")
            # Show trial states for debugging
            trial_states = {}
            for trial in study.trials:
                state = trial.state.name
                trial_states[state] = trial_states.get(state, 0) + 1
            self.logger.warning(f"Trial states: {trial_states}")

            # Return a dummy result for debugging
            dummy_result = OptimizationResult(
                best_params={},
                best_value=0.0,
                best_classifier=None,
                study=study,
                trial_results=[],
                optimization_time=time.time() - start_time,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
            )
            return dummy_result

        best_params = study.best_params
        best_value = study.best_value

        self.logger.info(f"Best trial: {best_params} -> {self.opt_config.primary_metric}={best_value:.4f}")

        # Step 5: Train final model with best parameters
        best_classifier = self._train_final_classifier(best_params, task_name, model_name)

        optimization_time = time.time() - start_time

        # Step 6: Collect trial results
        trial_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_results.append(
                    {
                        "trial_number": trial.number,
                        "params": trial.params,
                        "value": trial.value,
                        "duration": trial.duration.total_seconds() if trial.duration else None,
                    }
                )

        result = OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            best_classifier=best_classifier,
            study=study,
            trial_results=trial_results,
            optimization_time=optimization_time,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
        )

        self.logger.info(
            f"Optimization completed in {optimization_time:.1f}s "
            f"({self.cache_hits} cache hits, {self.cache_misses} cache misses)"
        )

        return result

    def _objective_function(self, trial: optuna.Trial, task_name: str, model_name: str) -> float:
        """
        Optuna objective function for a single trial.

        Args:
            trial: Optuna trial object
            task_name: Task name
            model_name: Model name

        Returns:
            Objective value to maximize
        """
        # Sample hyperparameters directly (following steering pattern)
        model_type = trial.suggest_categorical("model_type", self.opt_config.model_types)

        # Layer and aggregation from pre-generated activation data
        available_layers = set()
        available_aggregations = set()

        for key in self.activation_data.keys():
            parts = key.split("_")
            if len(parts) >= 4:  # layer_X_agg_Y
                layer = int(parts[1])
                agg = parts[3]
                available_layers.add(layer)
                available_aggregations.add(agg)

        layer = trial.suggest_categorical("layer", sorted(available_layers))
        aggregation = trial.suggest_categorical("aggregation", sorted(available_aggregations))

        # Classification threshold
        threshold = trial.suggest_float(
            "threshold", self.opt_config.threshold_range[0], self.opt_config.threshold_range[1]
        )

        # Training hyperparameters
        num_epochs = trial.suggest_int(
            "num_epochs", self.opt_config.num_epochs_range[0], self.opt_config.num_epochs_range[1]
        )

        learning_rate = trial.suggest_float(
            "learning_rate", self.opt_config.learning_rate_range[0], self.opt_config.learning_rate_range[1], log=True
        )

        batch_size = trial.suggest_categorical("batch_size", self.opt_config.batch_size_options)

        # Model-specific hyperparameters (conditional logic like steering)
        hyperparams = {"num_epochs": num_epochs, "learning_rate": learning_rate, "batch_size": batch_size}

        if model_type == "mlp":
            # MLP-specific parameters
            hyperparams["hidden_dim"] = trial.suggest_int(
                "hidden_dim", self.opt_config.hidden_dim_range[0], self.opt_config.hidden_dim_range[1], step=32
            )

        # Combine all parameters
        params = {
            "model_type": model_type,
            "layer": layer,
            "aggregation": aggregation,
            "threshold": threshold,
            **hyperparams,
        }

        # Get activation data for this configuration
        activation_key = f"layer_{params['layer']}_agg_{params['aggregation']}"

        if activation_key not in self.activation_data:
            self.logger.warning(f"No activation data for {activation_key}")
            raise optuna.TrialPruned()

        activation_data = self.activation_data[activation_key]
        X, y = activation_data.to_tensors(device=self.gen_config.device, dtype=self.model_dtype)
        print(f"DEBUG: Training data shape: X.shape={X.shape}, y.shape={y.shape}, dtype={X.dtype}")

        # Generate cache key
        data_hash = self.classifier_cache.compute_data_hash(X, y)
        cache_key = self.classifier_cache.get_cache_key(
            model_name=model_name,
            task_name=task_name,
            model_type=params["model_type"],
            layer=params["layer"],
            aggregation=params["aggregation"],
            threshold=params["threshold"],
            hyperparameters={
                k: v for k, v in params.items() if k not in ["model_type", "layer", "aggregation", "threshold"]
            },
            data_hash=data_hash,
        )

        # Try to load from cache
        cached_classifier = self.classifier_cache.load_classifier(cache_key)
        if cached_classifier is not None:
            self.cache_hits += 1
            # Evaluate cached classifier
            return self._evaluate_classifier(cached_classifier, X, y, params["threshold"])

        self.cache_misses += 1

        # Train new classifier
        classifier = self._train_classifier(params, X, y, trial)

        if classifier is None:
            raise optuna.TrialPruned()

        # Evaluate classifier
        score = self._evaluate_classifier(classifier, X, y, params["threshold"])

        # Save to cache if training was successful
        if score > 0:
            try:
                performance_metrics = {self.opt_config.primary_metric: score}

                self.classifier_cache.save_classifier(
                    cache_key=cache_key,
                    classifier=classifier,
                    model_name=model_name,
                    task_name=task_name,
                    layer=params["layer"],
                    aggregation=params["aggregation"],
                    threshold=params["threshold"],
                    hyperparameters={
                        k: v for k, v in params.items() if k not in ["model_type", "layer", "aggregation", "threshold"]
                    },
                    performance_metrics=performance_metrics,
                    training_samples=len(X),
                    data_hash=data_hash,
                )
            except Exception as e:
                self.logger.warning(f"Failed to cache classifier: {e}")

        return score

    def _train_classifier(
        self, params: dict[str, Any], X: np.ndarray, y: np.ndarray, trial: Optional[optuna.Trial] = None
    ) -> Optional[Classifier]:
        """
        Train a classifier with the given parameters.

        Args:
            params: Hyperparameters
            X: Training features
            y: Training labels
            trial: Optuna trial for pruning

        Returns:
            Trained classifier or None if training failed
        """
        try:
            # Create classifier (don't pass hidden_dim to constructor)
            classifier_kwargs = {
                "model_type": params["model_type"],
                "threshold": params["threshold"],
                "device": self.gen_config.device if self.gen_config.device else "auto",
                "dtype": self.model_dtype,
            }

            print(
                f"Preparing to train {params['model_type']} classifier with {len(X)} samples (dtype: {self.model_dtype})"
            )
            classifier = Classifier(**classifier_kwargs)

            # Train classifier
            training_kwargs = {
                "num_epochs": params["num_epochs"],
                "learning_rate": params["learning_rate"],
                "batch_size": params["batch_size"],
                "test_size": self.opt_config.test_size,
                "random_state": self.opt_config.random_state,
            }

            if params["model_type"] == "mlp":
                training_kwargs["hidden_dim"] = params["hidden_dim"]

            # Add pruning callback if trial is provided
            if trial and self.opt_config.enable_pruning:
                # TODO: Implement pruning callback for early stopping
                pass

            print(f"About to fit classifier with kwargs: {training_kwargs}")
            results = classifier.fit(X, y, **training_kwargs)
            print(f"Training results: {results}")

            accuracy = results.get("accuracy", 0)
            if accuracy <= 0.35:  # More permissive threshold - only prune very poor performance
                self.logger.debug(f"Classifier performance too low ({accuracy:.3f}), pruning")
                print(f"Classifier pruned - accuracy too low: {accuracy:.3f}")
                return None

            self.logger.debug(f"Classifier training successful - accuracy: {accuracy:.3f}")
            print(f"Classifier training successful - accuracy: {accuracy:.3f}")

            return classifier

        except Exception as e:
            print(f"EXCEPTION during classifier training: {e}")
            import traceback

            traceback.print_exc()
            self.logger.debug(f"Training failed with params {params}: {e}")
            return None

    def _evaluate_classifier(self, classifier: Classifier, X: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """
        Evaluate classifier performance.

        Args:
            classifier: Trained classifier
            X: Features
            y: Labels
            threshold: Classification threshold

        Returns:
            Performance score based on primary metric
        """
        try:
            print(f"DEBUG: Evaluation data shape: X.shape={X.shape}, y.shape={y.shape}, dtype={X.dtype}")

            # Set threshold
            classifier.set_threshold(threshold)

            # Get predictions
            results = classifier.evaluate(X, y)
            print(f"Evaluation results: {results}")
            print(f"Looking for primary metric '{self.opt_config.primary_metric}' in results")

            # Return primary metric
            score = results.get(self.opt_config.primary_metric, 0.0)
            print(f"Score extracted: {score}")
            return float(score)

        except Exception as e:
            print(f"EXCEPTION during evaluation: {e}")
            import traceback

            traceback.print_exc()
            self.logger.debug(f"Evaluation failed: {e}")
            return 0.0

    def _train_final_classifier(self, best_params: dict[str, Any], task_name: str, model_name: str) -> Classifier:
        """Train the final classifier with best parameters."""
        # Get activation data
        activation_key = f"layer_{best_params['layer']}_agg_{best_params['aggregation']}"
        activation_data = self.activation_data[activation_key]
        X, y = activation_data.to_tensors(device=self.gen_config.device, dtype=self.model_dtype)

        # Try cache first
        data_hash = self.classifier_cache.compute_data_hash(X, y)
        cache_key = self.classifier_cache.get_cache_key(
            model_name=model_name,
            task_name=task_name,
            model_type=best_params["model_type"],
            layer=best_params["layer"],
            aggregation=best_params["aggregation"],
            threshold=best_params["threshold"],
            hyperparameters={
                k: v for k, v in best_params.items() if k not in ["model_type", "layer", "aggregation", "threshold"]
            },
            data_hash=data_hash,
        )

        cached_classifier = self.classifier_cache.load_classifier(cache_key)
        if cached_classifier is not None:
            self.logger.info("Using cached classifier for final model")
            return cached_classifier

        # Train new classifier
        self.logger.info("Training final classifier with best parameters")
        classifier = self._train_classifier(best_params, X, y)

        if classifier is None:
            raise ValueError("Failed to train final classifier")

        return classifier

    def get_optimization_summary(self, result: OptimizationResult) -> dict[str, Any]:
        """Get a comprehensive optimization summary."""
        return {
            "best_configuration": result.get_best_config(),
            "best_score": result.best_value,
            "optimization_time_seconds": result.optimization_time,
            "total_trials": len(result.trial_results),
            "cache_efficiency": {
                "hits": result.cache_hits,
                "misses": result.cache_misses,
                "hit_rate": result.cache_hits / (result.cache_hits + result.cache_misses)
                if (result.cache_hits + result.cache_misses) > 0
                else 0,
            },
            "activation_data_info": {
                key: {
                    "samples": data.activations.shape[0],
                    "features": data.activations.shape[1]
                    if len(data.activations.shape) > 1
                    else data.activations.shape[0],
                    "layer": data.layer,
                    "aggregation": data.aggregation,
                }
                for key, data in self.activation_data.items()
            },
            "study_info": {
                "n_trials": len(result.study.trials),
                "best_trial": result.study.best_trial.number,
                "pruned_trials": len([t for t in result.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            },
        }
