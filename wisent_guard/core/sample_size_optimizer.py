"""
Sample Size Optimizer for finding the optimal training sample size for classifiers.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from wisent_guard.core.classifier.classifier import Classifier

from .activations import ActivationAggregationStrategy
from .contrastive_pairs import ContrastivePairSet
from .model import Model
from .model_config_manager import ModelConfigManager

logger = logging.getLogger(__name__)


class SampleSizeOptimizer:
    """Optimizes training sample size for classifiers."""

    def __init__(
        self,
        model_name: str,
        task_name: str = "truthfulqa_mc1",
        layer: int = 0,
        token_aggregation: str = "average",
        threshold: float = 0.5,
        test_split: float = 0.2,
        sample_sizes: Optional[List[int]] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the sample size optimizer.

        Args:
            model_name: Name of the model to optimize
            task_name: Task to optimize for
            layer: Layer index to optimize
            token_aggregation: Token aggregation method (average, final, first, max, min)
            threshold: Detection threshold for classification
            test_split: Fraction of data to use for testing
            sample_sizes: List of sample sizes to test
            device: Device to use for computation
            verbose: Enable verbose output
        """
        self.model_name = model_name
        self.task_name = task_name
        self.layer = layer
        self.token_aggregation = token_aggregation
        self.threshold = threshold
        self.test_split = test_split
        self.verbose = verbose

        # Default sample sizes if not provided
        if sample_sizes is None:
            self.sample_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        else:
            self.sample_sizes = sorted(sample_sizes)

        # Initialize model
        self.model = Model(name=model_name, device=device)
        self.device = self.model.device

        # Storage for results
        self.results = []
        self.optimal_sample_size = None

        logger.info(f"Initialized SampleSizeOptimizer for {model_name}")
        logger.info(f"Task: {task_name}, Layer: {layer}")
        logger.info(f"Sample sizes to test: {self.sample_sizes}")

    def load_and_split_data(self, limit: Optional[int] = None) -> Tuple[ContrastivePairSet, ContrastivePairSet]:
        """
        Load task data and split into train/test sets.

        Args:
            limit: Maximum number of samples to load (None for all)

        Returns:
            Tuple of (train_pairs, test_pairs)
        """
        logger.info(f"Loading data for task: {self.task_name}")

        # Load task data using the model
        max_samples = limit or 1000  # Default to 1000 if not specified

        # Try to use cached benchmark data first
        qa_pairs = None
        try:
            from .managed_cached_benchmarks import get_managed_cache

            cache = get_managed_cache()
            logger.info(f"Attempting to load from cache with limit={max_samples}")

            # Load samples from cache (it will download if needed)
            samples = cache.get_task_samples(self.task_name, limit=max_samples)

            if samples:
                logger.info(f"Loaded {len(samples)} samples from cache")
                # Convert cached samples to QA pairs format
                qa_pairs = []
                for sample in samples:
                    # The cached sample has 'normalized' field with the QA pair
                    normalized = sample.get("normalized", {})
                    # Handle both formats: good_response/bad_response and correct_answer
                    if "good_response" in normalized and "bad_response" in normalized:
                        qa_pair = {
                            "question": normalized.get("context", normalized.get("question", "")),
                            "correct_answer": normalized.get("good_response", ""),
                            "incorrect_answer": normalized.get("bad_response", ""),
                            "metadata": normalized.get("metadata", {}),
                        }
                    else:
                        # For truthfulqa_mc1, we need to get incorrect answers from mc1_targets
                        raw_data = sample.get("raw_data", {})
                        mc1_targets = raw_data.get("mc1_targets", {})
                        choices = mc1_targets.get("choices", [])
                        labels = mc1_targets.get("labels", [])

                        # Find first incorrect answer
                        incorrect_answer = None
                        for i, label in enumerate(labels):
                            if label == 0 and i < len(choices):
                                incorrect_answer = choices[i]
                                break

                        if not incorrect_answer:
                            incorrect_answer = "This is incorrect"

                        qa_pair = {
                            "question": normalized.get("question", ""),
                            "correct_answer": normalized.get("correct_answer", ""),
                            "incorrect_answer": incorrect_answer,
                            "metadata": normalized.get("metadata", {}),
                        }
                    qa_pairs.append(qa_pair)
                logger.info(f"Converted {len(qa_pairs)} cached samples to QA pairs")
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            qa_pairs = None

        # Fallback to loading from lm-eval if cache failed
        if not qa_pairs:
            logger.info("Loading from lm-eval harness...")
            # Load lm-eval task
            task_data = self.model.load_lm_eval_task(self.task_name, shots=0, limit=max_samples)

            # Split into train/test docs
            docs, _ = self.model.split_task_data(task_data, split_ratio=1.0)  # Use all for now

            if not docs:
                raise ValueError(f"No documents loaded for task {self.task_name}")

            logger.info(f"Loaded {len(docs)} documents from {self.task_name}")

            # Extract QA pairs from task docs
            qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(self.task_name, task_data, docs)

        if not qa_pairs:
            raise ValueError(f"No QA pairs could be extracted from task {self.task_name}")

        logger.info(f"Extracted {len(qa_pairs)} QA pairs")

        # Create contrastive pairs from QA pairs
        from wisent_guard.core.activations.activation_collection_method import ActivationCollectionLogic

        collector = ActivationCollectionLogic(model=self.model)

        # Import token aggregation function

        # Create contrastive pairs
        all_pairs = []
        for qa_pair in qa_pairs:
            # Create prompts for positive and negative cases
            question = qa_pair["question"]
            correct_answer = qa_pair["correct_answer"]
            incorrect_answer = qa_pair["incorrect_answer"]

            # Generate with model to get activations
            # Positive case (correct answer)
            pos_prompt = self.model.format_prompt(question)
            pos_response = correct_answer

            # Negative case (incorrect answer)
            neg_prompt = self.model.format_prompt(question)
            neg_response = incorrect_answer

            # Create contrastive pair
            from .contrastive_pairs import ContrastivePair
            from .response import NegativeResponse, PositiveResponse

            pair = ContrastivePair(
                prompt=question,
                positive_response=PositiveResponse(text=pos_response),
                negative_response=NegativeResponse(text=neg_response),
            )
            all_pairs.append(pair)

        if not all_pairs:
            raise ValueError(f"No contrastive pairs created for task {self.task_name}")

        # Extract activations for all pairs at the specified layer
        logger.info(f"Extracting activations at layer {self.layer}")

        # Use the collector to extract activations
        # For MULTIPLE_CHOICE, we use CHOICE_TOKEN targeting
        all_pairs = collector.collect_activations_batch(
            all_pairs,
            layer_index=self.layer,
            device=self.device,
            token_targeting_strategy=ActivationAggregationStrategy.CHOICE_TOKEN,
        )

        # Filter out any pairs without activations
        all_pairs = [p for p in all_pairs if p.positive_activations is not None and p.negative_activations is not None]

        logger.info(f"Loaded {len(all_pairs)} contrastive pairs")

        # Calculate split index
        n_test = int(len(all_pairs) * self.test_split)
        n_train = len(all_pairs) - n_test

        # Create train and test sets
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(len(all_pairs))

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_pairs = [all_pairs[i] for i in train_indices]
        test_pairs = [all_pairs[i] for i in test_indices]

        # Create ContrastivePairSet objects
        train_set = ContrastivePairSet(name=f"{self.task_name}_train", pairs=train_pairs)
        test_set = ContrastivePairSet(name=f"{self.task_name}_test", pairs=test_pairs)

        logger.info(f"Split data: {len(train_pairs)} train, {len(test_pairs)} test")

        return train_set, test_set

    def _aggregate_activations(self, activations):
        """
        Apply token aggregation to activations based on configured method.

        Since we're using CHOICE_TOKEN strategy, activations should be a single vector.
        This method is here for consistency with the main CLI approach.

        Args:
            activations: Activation vector or tensor

        Returns:
            Aggregated activation vector
        """
        # For CHOICE_TOKEN strategy, activations are already a single vector
        # No aggregation needed
        return activations

    def train_classifier_with_sample_size(
        self, train_set: ContrastivePairSet, sample_size: int
    ) -> Tuple[Classifier, float]:
        """
        Train a classifier with a specific sample size.

        Args:
            train_set: Full training set
            sample_size: Number of samples to use for training

        Returns:
            Tuple of (trained_classifier, training_time)
        """
        # Limit training set to sample_size
        if sample_size >= len(train_set.pairs):
            train_pairs = train_set.pairs
        else:
            # Use first sample_size pairs (already shuffled)
            train_pairs = train_set.pairs[:sample_size]

        logger.info(f"Training classifier with {len(train_pairs)} samples")

        # Ensure we have enough samples for training
        if len(train_pairs) < 2:
            logger.warning(f"Not enough training samples ({len(train_pairs)}). Skipping.")
            return None, 0.0

        # Extract activations
        X_train = []
        y_train = []

        for pair in train_pairs:
            # Positive example (correct answer)
            X_train.append(pair.positive_activations)
            y_train.append(0)  # 0 for correct/truthful

            # Negative example (incorrect answer)
            X_train.append(pair.negative_activations)
            y_train.append(1)  # 1 for incorrect/untruthful

        # Create and train classifier
        classifier = Classifier(model_type="logistic", device=self.device)

        start_time = time.time()
        classifier.fit(X_train, y_train)
        training_time = time.time() - start_time

        return classifier, training_time

    def evaluate_classifier(self, classifier: Classifier, test_set: ContrastivePairSet) -> Dict[str, float]:
        """
        Evaluate a classifier on the test set.

        Args:
            classifier: Trained classifier
            test_set: Test set to evaluate on

        Returns:
            Dictionary of metrics
        """
        X_test = []
        y_test = []

        for pair in test_set.pairs:
            # Positive example
            X_test.append(pair.positive_activations)
            y_test.append(0)

            # Negative example
            X_test.append(pair.negative_activations)
            y_test.append(1)

        # Get predictions
        y_pred = []
        for x in X_test:
            pred = classifier.predict(x)
            y_pred.append(1 if pred > 0.5 else 0)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        return metrics

    def find_optimal_sample_size(self) -> int:
        """
        Determine the optimal sample size based on diminishing returns.

        Returns:
            Optimal sample size
        """
        if len(self.results) < 2:
            return self.sample_sizes[-1]

        # Extract accuracies and times
        accuracies = [r["metrics"]["accuracy"] for r in self.results]
        times = [r["training_time"] for r in self.results]
        sizes = [r["sample_size"] for r in self.results]

        # Calculate accuracy gains
        gains = []
        for i in range(1, len(accuracies)):
            gain = accuracies[i] - accuracies[i - 1]
            gains.append(gain)

        # Find where gain drops below threshold (2% improvement)
        threshold = 0.02
        optimal_idx = len(sizes) - 1  # Default to largest

        for i, gain in enumerate(gains):
            if gain < threshold and accuracies[i + 1] > 0.7:  # Ensure reasonable accuracy
                optimal_idx = i + 1
                break

        # Also consider training time - if time increases dramatically, prefer smaller
        if optimal_idx < len(sizes) - 1 and times[optimal_idx] > 0:
            time_ratio = times[optimal_idx + 1] / times[optimal_idx]
            if time_ratio > 2.0 and gains[optimal_idx] < 0.01:
                # Training time doubled for < 1% gain, stick with current
                pass
            elif accuracies[optimal_idx + 1] - accuracies[optimal_idx] > 0.05:
                # Significant accuracy improvement, use larger size
                optimal_idx += 1

        return sizes[optimal_idx]

    def run_optimization(self) -> Dict[str, Any]:
        """
        Run the complete sample size optimization process.

        Returns:
            Dictionary containing results and optimal sample size
        """
        logger.info("Starting sample size optimization...")

        # Load and split data
        dataset_limit = getattr(self, "dataset_limit", None)
        train_set, test_set = self.load_and_split_data(limit=dataset_limit)

        # Ensure we don't test sample sizes larger than training set
        max_train_size = len(train_set.pairs)
        valid_sample_sizes = [s for s in self.sample_sizes if s <= max_train_size]

        if not valid_sample_sizes:
            raise ValueError(f"No valid sample sizes. Training set has only {max_train_size} samples.")

        logger.info(f"Testing sample sizes: {valid_sample_sizes}")

        # Test each sample size
        for sample_size in valid_sample_sizes:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Testing sample size: {sample_size}")

            # Train classifier
            classifier, training_time = self.train_classifier_with_sample_size(train_set, sample_size)

            # Skip if classifier training failed
            if classifier is None:
                logger.warning(f"Skipping sample size {sample_size} - not enough samples for training")
                continue

            # Evaluate on test set
            metrics = self.evaluate_classifier(classifier, test_set)

            # Store results
            result = {"sample_size": sample_size, "training_time": training_time, "metrics": metrics}
            self.results.append(result)

            logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"F1 Score: {metrics['f1']:.3f}")
            logger.info(f"Training time: {training_time:.3f}s")

        # Find optimal sample size
        self.optimal_sample_size = self.find_optimal_sample_size()

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Optimal sample size: {self.optimal_sample_size}")

        # Create summary
        summary = {
            "model": self.model_name,
            "task": self.task_name,
            "layer": self.layer,
            "test_split": self.test_split,
            "results": self.results,
            "optimal_sample_size": self.optimal_sample_size,
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def save_results(self, output_dir: Optional[str] = None) -> str:
        """
        Save optimization results to file.

        Args:
            output_dir: Directory to save results (uses default if None)

        Returns:
            Path to saved results file
        """
        if output_dir is None:
            output_dir = "./sample_size_optimization_results"

        os.makedirs(output_dir, exist_ok=True)

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = self.model_name.replace("/", "_")
        filename = f"sample_size_{model_safe}_{self.task_name}_layer{self.layer}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # Prepare data for saving
        save_data = {
            "model": self.model_name,
            "task": self.task_name,
            "layer": self.layer,
            "test_split": self.test_split,
            "results": self.results,
            "optimal_sample_size": self.optimal_sample_size,
            "timestamp": datetime.now().isoformat(),
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Results saved to: {filepath}")
        return filepath

    def plot_results(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot accuracy vs sample size curve.

        Args:
            save_path: Path to save plot (optional)
            show: Whether to display the plot
        """
        if not self.results:
            logger.warning("No results to plot")
            return

        # Extract data
        sizes = [r["sample_size"] for r in self.results]
        accuracies = [r["metrics"]["accuracy"] for r in self.results]
        f1_scores = [r["metrics"]["f1"] for r in self.results]
        times = [r["training_time"] for r in self.results]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 1: Accuracy and F1 vs Sample Size
        ax1.plot(sizes, accuracies, "b-o", label="Accuracy", linewidth=2, markersize=8)
        ax1.plot(sizes, f1_scores, "g--s", label="F1 Score", linewidth=2, markersize=8)

        # Mark optimal sample size
        if self.optimal_sample_size:
            ax1.axvline(
                self.optimal_sample_size, color="r", linestyle=":", label=f"Optimal: {self.optimal_sample_size}"
            )

        ax1.set_xlabel("Sample Size")
        ax1.set_ylabel("Score")
        ax1.set_title(
            f"Classifier Performance vs Sample Size\n{self.model_name} - {self.task_name} - Layer {self.layer}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Use linear scale for x-axis
        ax1.set_xticks(sizes)
        ax1.set_xticklabels([str(s) for s in sizes])

        # Plot 2: Training Time vs Sample Size
        ax2.plot(sizes, times, "r-^", linewidth=2, markersize=8)
        ax2.set_xlabel("Sample Size")
        ax2.set_ylabel("Training Time (seconds)")
        ax2.set_title("Training Time vs Sample Size")
        ax2.grid(True, alpha=0.3)
        # Use linear scale for x-axis
        ax2.set_xticks(sizes)
        ax2.set_xticklabels([str(s) for s in sizes])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to: {save_path}")

        if show:
            plt.show()

        plt.close()


def run_sample_size_optimization(
    model_name: str,
    task_name: str = "truthfulqa_mc1",
    layer: int = 0,
    token_aggregation: str = "average",
    threshold: float = 0.5,
    test_split: float = 0.2,
    sample_sizes: Optional[List[int]] = None,
    dataset_limit: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    save_plot: bool = True,
    save_to_config: bool = True,
) -> Dict[str, Any]:
    """
    Run sample size optimization and optionally save to model config.

    Args:
        model_name: Name of the model
        task_name: Task to optimize for
        layer: Layer index
        token_aggregation: Token aggregation method
        threshold: Detection threshold
        test_split: Test split ratio
        sample_sizes: Sample sizes to test
        dataset_limit: Maximum number of samples to load from dataset
        device: Computation device
        verbose: Verbose output
        save_plot: Whether to save the plot
        save_to_config: Whether to save to model config

    Returns:
        Optimization results dictionary
    """
    # Create optimizer
    optimizer = SampleSizeOptimizer(
        model_name=model_name,
        task_name=task_name,
        layer=layer,
        token_aggregation=token_aggregation,
        threshold=threshold,
        test_split=test_split,
        sample_sizes=sample_sizes,
        device=device,
        verbose=verbose,
    )

    # Run optimization with dataset limit
    optimizer.dataset_limit = dataset_limit
    results = optimizer.run_optimization()

    # Save results
    results_path = optimizer.save_results()

    # Create plot
    if save_plot:
        plot_path = results_path.replace(".json", ".png")
        optimizer.plot_results(save_path=plot_path, show=False)

    # Save to model config if requested
    if save_to_config and optimizer.optimal_sample_size:
        config_manager = ModelConfigManager()

        # Load existing config or create new
        existing_config = config_manager.load_model_config(model_name)

        if existing_config:
            # Update existing config
            if "optimal_sample_sizes" not in existing_config:
                existing_config["optimal_sample_sizes"] = {}

            if task_name not in existing_config["optimal_sample_sizes"]:
                existing_config["optimal_sample_sizes"][task_name] = {}

            existing_config["optimal_sample_sizes"][task_name][str(layer)] = optimizer.optimal_sample_size

            # Save updated config
            config_manager.update_model_config(model_name, existing_config)
            logger.info(f"Updated model config with optimal sample size: {optimizer.optimal_sample_size}")
        else:
            logger.warning("No existing model config found. Run optimize-classification first.")

    return results
