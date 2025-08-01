"""
Refactored comprehensive evaluation pipeline with clean separation of concerns.

This module orchestrates the comprehensive evaluation process:
1. Probe optimization: Train and optimize probes for detecting correctness
2. Steering optimization: Train and optimize steering methods for better performance
3. Final evaluation: Test optimized configurations on held-out test data
"""

import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import wandb

from . import data_utils, metrics
from .config import ComprehensiveEvaluationConfig
from .probe_optimization import ProbeConfig, ProbeOptimizer
from .steering_optimization import DACConfig, SteeringOptimizer

logger = logging.getLogger(__name__)


class ComprehensiveEvaluationPipeline:
    """
    Refactored comprehensive evaluation pipeline with clean architecture.

    This pipeline properly separates:
    1. Probe training and optimization (for detecting problems)
    2. Steering training and optimization (for improving performance)
    3. Final evaluation on test data
    4. Results collection and reporting
    """

    def __init__(self, config: ComprehensiveEvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Set up output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model and tokenizer (loaded once)
        self.model = None
        self.tokenizer = None

        # Initialize optimizers
        self.probe_optimizer = None
        self.steering_optimizer = SteeringOptimizer()

        # Results storage
        self.results = {"config": config.to_dict(), "timestamp": datetime.now().isoformat()}

        # Wandb setup
        self.wandb_run = None
        if config.enable_wandb:
            self._setup_wandb()

    def _setup_wandb(self):
        """Initialize wandb experiment."""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.to_dict(),
                tags=self.config.wandb_tags,
                entity=self.config.wandb_entity,
            )
            self.logger.info("Wandb experiment initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.config.enable_wandb = False

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline with proper separation of concerns."""
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING COMPREHENSIVE EVALUATION PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Train: {self.config.train_dataset}")
        self.logger.info(f"Validation: {self.config.val_dataset}")
        self.logger.info(f"Test: {self.config.test_dataset}")
        self.logger.info(f"Model: {self.config.model_name}")

        # Phase 0: Setup
        self._load_model_and_data()

        # Phase 1: Probe Training and Optimization
        probe_results = self._run_probe_optimization()

        # Phase 2: Steering Training and Optimization
        steering_results = self._run_steering_optimization()

        # Phase 3: Final Evaluation on Test Data
        test_results = self._run_final_evaluation(probe_results, steering_results)

        # Phase 4: Collect and Save Results
        final_results = self._collect_final_results(probe_results, steering_results, test_results)

        self.logger.info("✅ Comprehensive evaluation completed successfully!")
        return final_results

    def _load_model_and_data(self):
        """Load model and datasets once for reuse throughout evaluation."""
        self.logger.info("📊 Loading model and datasets...")

        # Load model once
        self.model, self.tokenizer = data_utils.load_model_and_tokenizer(self.config.model_name, self.device)

        # Load datasets
        self.train_samples = data_utils.load_dataset_samples(self.config.train_dataset, self.config.train_limit)
        self.val_samples = data_utils.load_dataset_samples(self.config.val_dataset, self.config.val_limit)
        self.test_samples = data_utils.load_dataset_samples(self.config.test_dataset, self.config.test_limit)

        self.logger.info(
            f"Loaded {len(self.train_samples)} train, {len(self.val_samples)} val, {len(self.test_samples)} test samples"
        )

    def _run_probe_optimization(self) -> Dict[str, Any]:
        """Run probe training and optimization phase."""
        self.logger.info("\n🎯 Phase 1: Probe Training and Optimization...")

        # Skip if no probe layers configured
        if not self.config.probe_layers:
            self.logger.info("No probe layers configured, skipping probe training")
            return {}

        # Initialize probe optimizer
        probe_config = ProbeConfig(layers=self.config.probe_layers, c_values=self.config.probe_c_values)
        self.probe_optimizer = ProbeOptimizer(probe_config)

        # Train probes on all layers
        self.logger.info(f"Training probes on {len(self.config.probe_layers)} layers...")
        probe_training_results = self.probe_optimizer.train_probes(
            self.model,
            self.tokenizer,
            self.train_samples,
            self.config.device,
            self.config.batch_size,
            self.config.max_length,
            self.config.train_dataset,
            self.config.max_new_tokens,
        )

        # Evaluate probes on validation data
        self.logger.info("Evaluating probes on validation data...")
        best_probe_config, validation_metrics = self.probe_optimizer.evaluate_probes(
            probe_training_results,
            self.model,
            self.tokenizer,
            self.val_samples,
            self.config.device,
            self.config.batch_size,
            self.config.max_length,
            self.config.val_dataset,
            self.config.max_new_tokens,
        )

        probe_results = {
            "training_results": probe_training_results,
            "validation_metrics": validation_metrics,
            "best_config": best_probe_config,
        }

        self.logger.info(
            f"✅ Probe optimization completed. Best: Layer {best_probe_config['layer']}, AUC {best_probe_config['metrics']['auc']:.3f}"
        )

        # Log to wandb
        if self.config.enable_wandb:
            wandb.log(
                {"probe_best_layer": best_probe_config["layer"], "probe_best_auc": best_probe_config["metrics"]["auc"]}
            )

        return probe_results

    def _run_steering_optimization(self) -> Dict[str, Any]:
        """Run steering training and optimization phase."""
        self.logger.info("\n⚙️ Phase 2: Steering Training and Optimization...")

        # Currently supports DAC, but extensible for other methods
        steering_configs = []

        # DAC configuration
        if "dac" in self.config.steering_methods:
            dac_config = DACConfig(
                layers=self.config.steering_layers,
                strengths=self.config.steering_strengths,
                entropy_thresholds=self.config.dac_entropy_thresholds,
                ptop_values=self.config.dac_ptop_values,
                max_alpha_values=self.config.dac_max_alpha_values,
            )
            steering_configs.append(dac_config)

        best_steering_config = None
        best_steering_score = -1
        all_steering_results = []

        # Optimize each steering method
        for config in steering_configs:
            self.logger.info(f"Optimizing {config.method_name} steering method...")

            best_config, method_results = self.steering_optimizer.optimize_steering_hyperparameters(
                config,
                self.train_samples,
                self.val_samples,
                self.model,
                self.tokenizer,
                self.config.device,
                self.config.batch_size,
                self.config.max_length,
                self.config.train_dataset,
                self.config.max_new_tokens,
            )

            all_steering_results.extend(method_results)

            # Check if this is the best steering method so far
            score = best_config["benchmark_metrics"].get("accuracy", 0.0)
            if score > best_steering_score:
                best_steering_score = score
                best_steering_config = best_config

        steering_results = {
            "best_config": best_steering_config,
            "all_results": all_steering_results,
            "best_score": best_steering_score,
        }

        if best_steering_config:
            self.logger.info(
                f"✅ Steering optimization completed. Best: {best_steering_config['method']} "
                f"on layer {best_steering_config['layer']}, accuracy {best_steering_score:.3f}"
            )

            # Log to wandb
            if self.config.enable_wandb:
                wandb.log(
                    {
                        "steering_best_method": best_steering_config["method"],
                        "steering_best_layer": best_steering_config["layer"],
                        "steering_best_accuracy": best_steering_score,
                    }
                )
        else:
            self.logger.warning("No successful steering configuration found")

        return steering_results

    def _run_final_evaluation(self, probe_results: Dict[str, Any], steering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run final evaluation on test data using optimized configurations."""
        self.logger.info("\n🏆 Phase 3: Final Test Evaluation...")

        # Get best configurations
        best_probe_config = probe_results.get("best_config")
        best_steering_config = steering_results.get("best_config")

        # 1. Baseline model performance (no steering)
        self.logger.info("Evaluating baseline model performance...")
        base_predictions, ground_truths = data_utils.generate_benchmark_predictions(
            self.model,
            self.tokenizer,
            self.test_samples,
            self.config.batch_size,
            self.config.max_length,
            self.device,
            self.config.test_dataset,
            self.config.max_new_tokens,
        )
        base_benchmark_metrics = metrics.evaluate_benchmark_performance(
            base_predictions, ground_truths, self.config.test_dataset
        )

        # Store predictions and ground truths for probe evaluation optimization
        stored_test_data = {
            "base_predictions": base_predictions,
            "ground_truths": ground_truths,
            "test_samples": self.test_samples,
        }

        # 2. Steered model performance (if steering available)
        steered_benchmark_metrics = None
        if best_steering_config and best_steering_config.get("method_instance"):
            self.logger.info("Evaluating steered model performance...")

            method_instance = best_steering_config["method_instance"]
            trainer = self.steering_optimizer.trainers[best_steering_config["method"]]

            steered_predictions, _ = trainer.apply_steering_and_evaluate(
                method_instance,
                self.test_samples,
                best_steering_config["layer"],
                best_steering_config["strength"],
                self.model,
                self.tokenizer,
                self.device,
                self.config.batch_size,
                self.config.max_length,
                self.config.test_dataset,
                self.config.max_new_tokens,
            )
            steered_benchmark_metrics = metrics.evaluate_benchmark_performance(
                steered_predictions, ground_truths, self.config.test_dataset
            )
            stored_test_data["steered_predictions"] = steered_predictions

        # 3. Probe performance evaluation (if probes available)
        base_probe_metrics = None
        steered_probe_metrics = None

        if best_probe_config and best_probe_config.get("probe"):
            self.logger.info("Evaluating probe performance...")

            # Evaluate probe on base model activations (using stored data)
            base_probe_metrics = self._evaluate_probe_on_test(
                best_probe_config, steered=False, stored_data=stored_test_data
            )

            # Evaluate probe on steered model activations (if steering available)
            if best_steering_config and "steered_predictions" in stored_test_data:
                steered_probe_metrics = self._evaluate_probe_on_test(
                    best_probe_config, steered=True, steering_config=best_steering_config, stored_data=stored_test_data
                )

        test_results = {
            "base_benchmark_metrics": base_benchmark_metrics,
            "steered_benchmark_metrics": steered_benchmark_metrics,
            "base_probe_metrics": base_probe_metrics,
            "steered_probe_metrics": steered_probe_metrics,
            "best_probe_config": best_probe_config,
            "best_steering_config": best_steering_config,
        }

        # Log final results
        self._log_final_results(test_results)

        return test_results

    def _create_probe_data_from_predictions(
        self, test_samples: List[Dict], predictions: List[str], ground_truths: List[str], layer: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create probe training data efficiently using pre-computed predictions."""
        from ...task_interface import get_task

        # Get the task and its extractor
        task = get_task(self.config.test_dataset)
        extractor = task.get_extractor()

        texts = []
        labels = []

        for sample, prediction, ground_truth in zip(test_samples, predictions, ground_truths):
            # Use the task's extractor to get QA pair
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                continue

            question = qa_pair["formatted_question"]

            # Create examples with actual prediction and correct answer
            correct_text = f"{question} {ground_truth}"
            incorrect_text = f"{question} {prediction}"

            texts.extend([correct_text, incorrect_text])

            # Evaluate if prediction is correct using our evaluation logic
            is_correct = metrics.evaluate_response_correctness(prediction, ground_truth, self.config.test_dataset)
            labels.extend([1, 1 if is_correct else 0])

        # Extract activations for the texts
        activations = data_utils.extract_activations_with_hook(
            self.model, self.tokenizer, texts, layer, self.config.batch_size, self.config.max_length, self.device
        )

        return activations, np.array(labels)

    def _evaluate_probe_on_test(
        self,
        probe_config: Dict[str, Any],
        steered: bool = False,
        steering_config: Dict[str, Any] = None,
        stored_data: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """Evaluate probe on test data."""
        if not probe_config or not probe_config.get("probe"):
            return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.5}

        layer = probe_config["layer"]
        probe = probe_config["probe"]

        # Optimization: Use stored predictions and create activations efficiently
        if stored_data:
            predictions = stored_data["steered_predictions"] if steered else stored_data["base_predictions"]
            ground_truths = stored_data["ground_truths"]
            test_samples = stored_data["test_samples"]

            # Create probe data efficiently using pre-computed predictions
            X_test, y_test = self._create_probe_data_from_predictions(test_samples, predictions, ground_truths, layer)
        else:
            # Fallback to original method if no stored data
            self.logger.warning("No stored data available, using slower probe data creation")
            X_test, y_test = data_utils.create_probe_training_data(
                self.model,
                self.tokenizer,
                self.test_samples,
                layer,
                self.config.batch_size,
                self.config.max_length,
                self.device,
                self.config.test_dataset,
                self.config.max_new_tokens,
            )

        if len(X_test) < 2:
            self.logger.warning(f"Insufficient test data for probe evaluation: {len(X_test)} samples")
            return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.5}

        # Evaluate probe
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

            y_pred = probe.predict(X_test)
            y_pred_proba = probe.predict_proba(X_test)[:, 1]

            probe_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "auc": roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
            }

            return probe_metrics

        except Exception as e:
            self.logger.error(f"Failed to evaluate probe: {e}")
            return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.5}

    def _log_final_results(self, test_results: Dict[str, Any]):
        """Log final test results."""
        base_benchmark = test_results.get("base_benchmark_metrics", {})
        steered_benchmark = test_results.get("steered_benchmark_metrics", {})
        base_probe = test_results.get("base_probe_metrics", {})
        steered_probe = test_results.get("steered_probe_metrics", {})

        self.logger.info("🏆 FINAL TEST RESULTS:")
        self.logger.info("-" * 40)

        # Benchmark results
        base_acc = base_benchmark.get("accuracy", 0.0)
        self.logger.info(f"📊 Base Model Benchmark Accuracy: {base_acc:.3f}")

        if steered_benchmark:
            steered_acc = steered_benchmark.get("accuracy", 0.0)
            improvement = steered_acc - base_acc
            self.logger.info(f"🎯 Steered Model Benchmark Accuracy: {steered_acc:.3f}")
            self.logger.info(f"📈 Steering Improvement: {improvement:+.3f}")

        # Probe results
        if base_probe:
            base_auc = base_probe.get("auc", 0.5)
            self.logger.info(f"🔍 Base Model Probe AUC: {base_auc:.3f}")

        if steered_probe:
            steered_auc = steered_probe.get("auc", 0.5)
            self.logger.info(f"🔍 Steered Model Probe AUC: {steered_auc:.3f}")

        # Log to wandb
        if self.config.enable_wandb:
            log_data = {
                "test_base_benchmark_accuracy": base_acc,
                "test_base_probe_auc": base_probe.get("auc", 0.5) if base_probe else 0.5,
            }

            if steered_benchmark:
                log_data["test_steered_benchmark_accuracy"] = steered_benchmark.get("accuracy", 0.0)
                log_data["test_steering_improvement"] = steered_benchmark.get("accuracy", 0.0) - base_acc

            if steered_probe:
                log_data["test_steered_probe_auc"] = steered_probe.get("auc", 0.5)

            wandb.log(log_data)

    def _collect_final_results(
        self, probe_results: Dict[str, Any], steering_results: Dict[str, Any], test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect and organize all results for return."""
        final_results = {
            **self.results,  # Include config and timestamp
            "probe_optimization_results": probe_results,
            "steering_optimization_results": steering_results,
            "test_results": test_results,
        }

        # Save results to file
        self._save_results(final_results)

        # Clean up wandb
        if self.config.enable_wandb and self.wandb_run:
            wandb.finish()

        return final_results

    def _save_results(self, results: Dict[str, Any]):
        """Save essential results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"comprehensive_evaluation_results_{timestamp}.json"

        try:
            import json

            # Extract only essential information
            essential_results = {
                "experiment_info": {
                    "timestamp": timestamp,
                    "experiment_name": self.config.experiment_name,
                    "model_name": self.config.model_name,
                    "datasets": {
                        "train": self.config.train_dataset,
                        "val": self.config.val_dataset,
                        "test": self.config.test_dataset,
                    },
                    "sample_limits": {
                        "train": self.config.train_limit,
                        "val": self.config.val_limit,
                        "test": self.config.test_limit,
                    },
                },
                "final_performance": {},
                "best_configurations": {},
                "hyperparameter_search": {},
                "technical_config": {
                    "batch_size": self.config.batch_size,
                    "max_length": self.config.max_length,
                    "max_new_tokens": self.config.max_new_tokens,
                    "probe_layers": self.config.probe_layers,
                    "steering_layers": self.config.steering_layers,
                    "steering_methods": self.config.steering_methods,
                },
            }

            # Extract final performance metrics
            if "test_results" in results:
                test_results = results["test_results"]
                essential_results["final_performance"] = {
                    "base_benchmark_accuracy": test_results.get("base_benchmark_metrics", {}).get("accuracy", 0.0),
                    "steered_benchmark_accuracy": test_results.get("steered_benchmark_metrics", {}).get(
                        "accuracy", 0.0
                    ),
                    "benchmark_improvement": (
                        test_results.get("steered_benchmark_metrics", {}).get("accuracy", 0.0)
                        - test_results.get("base_benchmark_metrics", {}).get("accuracy", 0.0)
                    ),
                    "base_probe_auc": test_results.get("base_probe_metrics", {}).get("auc", 0.5),
                    "steered_probe_auc": test_results.get("steered_probe_metrics", {}).get("auc", 0.5),
                }

                # Extract best configurations (without objects)
                if test_results.get("best_probe_config"):
                    probe_config = test_results["best_probe_config"]
                    essential_results["best_configurations"]["probe"] = {
                        "layer": probe_config.get("layer"),
                        "c_value": probe_config.get("c_value"),
                        "metrics": probe_config.get("metrics", {}),
                    }

                if test_results.get("best_steering_config"):
                    steering_config = test_results["best_steering_config"]
                    essential_results["best_configurations"]["steering"] = {
                        "method": steering_config.get("method"),
                        "layer": steering_config.get("layer"),
                        "strength": steering_config.get("strength"),
                        "hyperparameters": {
                            k: v
                            for k, v in steering_config.items()
                            if k not in ["method_instance", "benchmark_metrics"]
                        },
                        "benchmark_metrics": steering_config.get("benchmark_metrics", {}),
                    }

            # Extract hyperparameter search results
            if "steering_optimization_results" in results:
                steering_results = results["steering_optimization_results"]
                if "all_results" in steering_results:
                    essential_results["hyperparameter_search"]["steering_results"] = [
                        {
                            "method_name": result.method_name,
                            "layer": result.layer,
                            "hyperparameters": result.hyperparameters,
                            "benchmark_metrics": result.benchmark_metrics,
                            "training_success": result.training_success,
                        }
                        for result in steering_results["all_results"]
                    ]

            with open(results_file, "w") as f:
                json.dump(essential_results, f, indent=2, default=str)

            self.logger.info(f"💾 Essential results saved to: {results_file}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def __del__(self):
        """Cleanup resources."""
        # Clean up GPU memory
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
