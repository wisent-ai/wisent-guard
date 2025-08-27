"""
Evaluation metrics for comprehensive evaluation pipeline.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from wisent_guard.core.bigcode_extractors import MBPPExtractor

# Import LMEvalHarnessGroundTruth for intelligent evaluation (newer approach used by CLI)
from wisent_guard.core.lm_eval_harness_ground_truth import LMEvalHarnessGroundTruth
from wisent_guard.core.task_interface import get_task
from wisent_guard.core.tasks.file_task import FileTask
from wisent_guard.parameters.task_config import CODING_TASKS

from .bigcode_evaluator_wrapper import OptunaBigCodeEvaluator

logger = logging.getLogger(__name__)


def evaluate_response_correctness(response: str, expected_answer: str, task_name: str) -> bool:
    """
    Evaluate if a response is correct using LMEvalHarnessGroundTruth (same approach as CLI).
    Note: For coding tasks, response should already be extracted code before calling this function.

    Args:
        response: Model's response (pre-extracted code for coding tasks)
        expected_answer: Expected correct answer
        task_name: Name of the task for proper evaluation

    Returns:
        True if response is correct, False otherwise
    """
    # Check if this is a file-based task (custom dataset loaded from JSON)
    # For file-based tasks, use exact string matching to avoid false positives
    try:
        task = get_task(task_name, limit=1)
        if isinstance(task, FileTask):
            logger.debug(f"Using exact match for file-based task '{task_name}'")
            return response.strip().lower() == expected_answer.strip().lower()
    except:
        pass  # Continue with normal evaluation if task lookup fails

    try:
        # Use the same evaluation approach as the CLI
        evaluator = LMEvalHarnessGroundTruth(task_name)

        # Create response data format expected by _evaluate_with_lm_eval_metrics
        response_data = [
            {
                "generated_response": response,
                "ground_truth": expected_answer,
                "question": "evaluation_question",  # Required field for evaluation
            }
        ]

        # Use the same evaluation logic as CLI
        eval_results = evaluator._evaluate_with_lm_eval_metrics(task_name, response_data, None)

        # Extract the result - accuracy > 0 means at least one correct
        return eval_results.get("accuracy", 0.0) > 0.0

    except Exception as e:
        logger.warning(f"LMEvalHarnessGroundTruth failed, using exact match fallback: {e}")
        # Fallback to simple string matching
        return response.strip().lower() == expected_answer.strip().lower()


def evaluate_benchmark_performance(
    predictions: List[str],
    ground_truths: List[str],
    task_name: str = None,
    task_docs: List[Dict] = None,
    classifier_scorer: Optional[Callable[[List[str], str], List[float]]] = None,
) -> Dict[str, float]:
    """
    Evaluate benchmark performance using LMEvalHarnessGroundTruth (same approach as CLI).
    For coding tasks, uses BigCode execution-based evaluation instead of string comparison.

    Args:
        predictions: List of model predictions
        ground_truths: List of correct answers
        task_name: Name of the task for intelligent evaluation
        task_docs: List of original task documents (required for coding tasks)
        classifier_scorer: Optional function to score predictions with classifier for confidence scores

    Returns:
        Dictionary containing benchmark performance metrics
    """
    if task_name:
        # Check if this is a coding task that requires code execution evaluation
        is_coding_task = task_name.lower() in CODING_TASKS

        # Calculate classifier confidence scores if classifier_scorer provided
        classifier_confidences = None
        if classifier_scorer is not None:
            try:
                logger.debug(f"Calculating classifier confidence scores for {len(predictions)} predictions")
                classifier_confidences = classifier_scorer(predictions, f"metrics_evaluation_{task_name}")
                logger.debug(f"Calculated {len(classifier_confidences)} confidence scores")
            except Exception as e:
                logger.warning(f"Failed to calculate classifier confidence scores: {e}")
                classifier_confidences = None

        if is_coding_task:
            # Use BigCode execution-based evaluation for coding tasks
            logger.info(f"Using BigCode execution-based evaluation for coding task: {task_name}")

            try:
                bigcode_evaluator = OptunaBigCodeEvaluator()

                # Validate task docs are provided for coding tasks
                if task_docs is None or len(task_docs) == 0:
                    logger.error(
                        f"No task docs provided for coding task {task_name}. BigCode evaluation requires original task documents with test cases."
                    )
                    raise ValueError(f"Task documents required for coding task evaluation: {task_name}")

                # Ensure we have the right number of task docs
                if len(task_docs) != len(predictions):
                    logger.error(f"Task docs length mismatch: {len(task_docs)} docs vs {len(predictions)} predictions")
                    raise ValueError(
                        f"Number of task documents ({len(task_docs)}) must match number of predictions ({len(predictions)})"
                    )

                # Evaluate using BigCode execution
                evaluation_results, accuracy_metrics = bigcode_evaluator.evaluate_and_calculate_accuracy(
                    predictions, task_docs, task_name
                )

                # Create evaluation details in the expected format
                evaluation_details = []
                for i, (pred, result) in enumerate(zip(predictions, evaluation_results)):
                    eval_detail = {
                        "prediction": result.get("extracted_code", pred),
                        "ground_truth": ground_truths[i] if i < len(ground_truths) else "unknown",
                        "is_correct": result.get("passed", False),
                        "classifier_confidence": classifier_confidences[i]
                        if classifier_confidences and i < len(classifier_confidences)
                        else 1.0,
                        "method": "bigcode_execution",
                        "original_prediction": pred,
                        "code_extracted": result.get("extracted_code", "") != pred,
                        "execution_error": result.get("error"),
                    }
                    evaluation_details.append(eval_detail)

                return {
                    "accuracy": accuracy_metrics["accuracy"],
                    "total_samples": accuracy_metrics["total_samples"],
                    "correct": accuracy_metrics["correct"],
                    "incorrect": accuracy_metrics["incorrect"],
                    "evaluation_method": "bigcode_execution",
                    "task_name": task_name,
                    "evaluation_details": evaluation_details,
                    "pass_count": accuracy_metrics.get("pass_count", 0),
                    "fail_count": accuracy_metrics.get("fail_count", 0),
                    "error_count": accuracy_metrics.get("error_count", 0),
                }

            except Exception as e:
                logger.error(f"BigCode evaluation failed for {task_name}, falling back to string-based evaluation: {e}")
                # Fall through to string-based evaluation

        # String-based evaluation for non-coding tasks or BigCode fallback
        extracted_predictions = predictions

        if is_coding_task:
            # Extract code from predictions for coding tasks (fallback mode)
            extractor = MBPPExtractor()  # Works for all coding tasks, not just MBPP
            extracted_predictions = []

            for pred in predictions:
                extracted_code = extractor.extract_code_from_answer(pred)
                extracted_predictions.append(extracted_code)

            logger.debug(f"Code extraction applied for {task_name}: {len(predictions)} predictions processed")

        # Use intelligent evaluation with LMEvalHarnessGroundTruth (same as CLI)
        correct_predictions = []
        evaluation_details = []

        for i, (orig_pred, extracted_pred, gt) in enumerate(zip(predictions, extracted_predictions, ground_truths)):
            try:
                # Use the extracted prediction for evaluation
                is_correct = evaluate_response_correctness(extracted_pred, gt, task_name)
                correct_predictions.append(is_correct)

                # Include both original and extracted predictions in details for debugging
                eval_detail = {
                    "prediction": extracted_pred,
                    "ground_truth": gt,
                    "is_correct": is_correct,
                    "classifier_confidence": classifier_confidences[i]
                    if classifier_confidences and i < len(classifier_confidences)
                    else 1.0,
                    "method": "lm_eval_harness_ground_truth",
                }

                # Add original prediction for coding tasks to help with debugging
                if is_coding_task and orig_pred != extracted_pred:
                    eval_detail["original_prediction"] = orig_pred
                    eval_detail["code_extracted"] = True

                evaluation_details.append(eval_detail)

            except Exception as e:
                logger.warning(f"LMEvalHarnessGroundTruth failed for prediction '{extracted_pred}' vs '{gt}': {e}")
                # Fallback to simple string matching
                is_correct = extracted_pred.strip().lower() == gt.strip().lower()
                correct_predictions.append(is_correct)

                eval_detail = {
                    "prediction": extracted_pred,
                    "ground_truth": gt,
                    "is_correct": is_correct,
                    "classifier_confidence": classifier_confidences[i]
                    if classifier_confidences and i < len(classifier_confidences)
                    else 1.0,
                    "method": "fallback_exact_match",
                }

                if is_coding_task and orig_pred != extracted_pred:
                    eval_detail["original_prediction"] = orig_pred
                    eval_detail["code_extracted"] = True

                evaluation_details.append(eval_detail)

        accuracy = np.mean(correct_predictions)
        total_correct = sum(correct_predictions)

        return {
            "accuracy": accuracy,
            "total_samples": len(predictions),
            "correct": total_correct,
            "incorrect": len(predictions) - total_correct,
            "evaluation_method": "lm_eval_harness_ground_truth",
            "task_name": task_name,
            "evaluation_details": evaluation_details[:5],  # Include first 5 for debugging
        }
    # Fallback to simple exact match
    logger.info("No task_name provided, using simple exact match evaluation")
    exact_matches = [pred.strip().lower() == gt.strip().lower() for pred, gt in zip(predictions, ground_truths)]
    accuracy = np.mean(exact_matches)

    return {
        "accuracy": accuracy,
        "total_samples": len(predictions),
        "correct": sum(exact_matches),
        "incorrect": len(predictions) - sum(exact_matches),
        "evaluation_method": "exact_match",
    }


def evaluate_probe_performance(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Evaluate probe performance with comprehensive metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for positive class)

    Returns:
        Dictionary containing probe performance metrics
    """
    if len(y_true) == 0:
        # Return default metrics if no data
        return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.5, "total_samples": 0}

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.5  # Default for cases where AUC can't be computed

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "total_samples": len(y_true),
    }


def calculate_combined_score(
    benchmark_metrics: Dict[str, float],
    probe_metrics: Dict[str, float],
    benchmark_weight: float = 0.7,
    probe_weight: float = 0.3,
) -> float:
    """
    Calculate combined score from benchmark and probe performance.

    Args:
        benchmark_metrics: Benchmark performance metrics
        probe_metrics: Probe performance metrics
        benchmark_weight: Weight for benchmark performance
        probe_weight: Weight for probe performance

    Returns:
        Combined score (0-1)
    """
    benchmark_score = benchmark_metrics.get("accuracy", 0.0)
    probe_score = probe_metrics.get("auc", 0.5)  # Use AUC as primary probe metric

    combined_score = benchmark_weight * benchmark_score + probe_weight * probe_score
    return combined_score


def calculate_comprehensive_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics from evaluation results.

    Args:
        results: Complete evaluation results

    Returns:
        Dictionary with comprehensive metrics and analysis
    """
    comprehensive_metrics = {}

    if "test_results" in results:
        test_results = results["test_results"]

        # Extract key metrics
        base_benchmark_acc = test_results.get("base_benchmark_metrics", {}).get("accuracy", 0.0)
        steered_benchmark_acc = test_results.get("steered_benchmark_metrics", {}).get("accuracy", 0.0)
        base_probe_auc = test_results.get("base_probe_metrics", {}).get("auc", 0.5)
        steered_probe_auc = test_results.get("steered_probe_metrics", {}).get("auc", 0.5)

        # Calculate improvements
        benchmark_improvement = steered_benchmark_acc - base_benchmark_acc
        probe_improvement = steered_probe_auc - base_probe_auc

        comprehensive_metrics.update(
            {
                "base_benchmark_accuracy": base_benchmark_acc,
                "steered_benchmark_accuracy": steered_benchmark_acc,
                "benchmark_improvement": benchmark_improvement,
                "benchmark_improvement_percent": (benchmark_improvement / max(base_benchmark_acc, 0.001)) * 100,
                "base_probe_auc": base_probe_auc,
                "steered_probe_auc": steered_probe_auc,
                "probe_improvement": probe_improvement,
                "probe_improvement_percent": (probe_improvement / max(base_probe_auc, 0.001)) * 100,
                "overall_effectiveness": (benchmark_improvement + probe_improvement) / 2,
                "validation_score": test_results.get("validation_combined_score", 0.0),
            }
        )

    # Add training statistics
    if "probe_training_results" in results:
        training_results = results["probe_training_results"]

        # Calculate training performance statistics
        all_training_aucs = []
        for layer_key, layer_results in training_results.items():
            for c_key, metrics in layer_results.items():
                if isinstance(metrics, dict) and "auc" in metrics:
                    all_training_aucs.append(metrics["auc"])

        if all_training_aucs:
            comprehensive_metrics.update(
                {
                    "training_probe_auc_mean": np.mean(all_training_aucs),
                    "training_probe_auc_std": np.std(all_training_aucs),
                    "training_probe_auc_max": np.max(all_training_aucs),
                    "training_probe_auc_min": np.min(all_training_aucs),
                }
            )

    # Add optimization statistics
    if "steering_optimization_results" in results:
        optimization_results = results["steering_optimization_results"]

        all_configs = optimization_results.get("all_configs", [])
        if all_configs:
            combined_scores = [config.get("combined_score", 0.0) for config in all_configs]
            benchmark_scores = [config.get("benchmark_metrics", {}).get("accuracy", 0.0) for config in all_configs]

            comprehensive_metrics.update(
                {
                    "optimization_configs_tested": len(all_configs),
                    "optimization_score_mean": np.mean(combined_scores),
                    "optimization_score_std": np.std(combined_scores),
                    "optimization_benchmark_mean": np.mean(benchmark_scores),
                    "optimization_benchmark_std": np.std(benchmark_scores),
                }
            )

    return comprehensive_metrics


def generate_performance_summary(comprehensive_metrics: Dict[str, Any]) -> str:
    """
    Generate a human-readable performance summary.

    Args:
        comprehensive_metrics: Comprehensive metrics dictionary

    Returns:
        String summary of performance
    """
    summary = []
    summary.append("=" * 60)
    summary.append("COMPREHENSIVE EVALUATION PERFORMANCE SUMMARY")
    summary.append("=" * 60)

    # Benchmark Performance
    if "base_benchmark_accuracy" in comprehensive_metrics:
        base_acc = comprehensive_metrics["base_benchmark_accuracy"]
        steered_acc = comprehensive_metrics["steered_benchmark_accuracy"]
        improvement = comprehensive_metrics["benchmark_improvement"]

        summary.append("\n📊 BENCHMARK PERFORMANCE:")
        summary.append(f"  Base Model Accuracy:    {base_acc:.3f} ({base_acc * 100:.1f}%)")
        summary.append(f"  Steered Model Accuracy: {steered_acc:.3f} ({steered_acc * 100:.1f}%)")
        summary.append(f"  Improvement:            {improvement:+.3f} ({improvement * 100:+.1f}%)")

    # Probe Performance
    if "base_probe_auc" in comprehensive_metrics:
        base_auc = comprehensive_metrics["base_probe_auc"]
        steered_auc = comprehensive_metrics["steered_probe_auc"]
        improvement = comprehensive_metrics["probe_improvement"]

        summary.append("\n🔍 PROBE PERFORMANCE:")
        summary.append(f"  Base Model Probe AUC:    {base_auc:.3f}")
        summary.append(f"  Steered Model Probe AUC: {steered_auc:.3f}")
        summary.append(f"  Improvement:             {improvement:+.3f}")

    # Training Statistics
    if "training_probe_auc_mean" in comprehensive_metrics:
        mean_auc = comprehensive_metrics["training_probe_auc_mean"]
        std_auc = comprehensive_metrics["training_probe_auc_std"]
        max_auc = comprehensive_metrics["training_probe_auc_max"]

        summary.append("\n🎯 TRAINING STATISTICS:")
        summary.append(f"  Probe Training AUC:      {mean_auc:.3f} ± {std_auc:.3f}")
        summary.append(f"  Best Training AUC:       {max_auc:.3f}")

    # Optimization Statistics
    if "optimization_configs_tested" in comprehensive_metrics:
        num_configs = comprehensive_metrics["optimization_configs_tested"]
        best_score = comprehensive_metrics.get("validation_score", 0.0)

        summary.append("\n⚙️ OPTIMIZATION STATISTICS:")
        summary.append(f"  Configurations Tested:   {num_configs}")
        summary.append(f"  Best Validation Score:   {best_score:.3f}")

    # Overall Assessment
    if "overall_effectiveness" in comprehensive_metrics:
        effectiveness = comprehensive_metrics["overall_effectiveness"]

        summary.append("\n🏆 OVERALL ASSESSMENT:")
        if effectiveness > 0.1:
            assessment = "Highly Effective"
        elif effectiveness > 0.05:
            assessment = "Moderately Effective"
        elif effectiveness > 0.01:
            assessment = "Slightly Effective"
        else:
            assessment = "Minimal Effect"

        summary.append(f"  Steering Effectiveness:  {assessment} ({effectiveness:+.3f})")

    summary.append("=" * 60)

    return "\n".join(summary)
