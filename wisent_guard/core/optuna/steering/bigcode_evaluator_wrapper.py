"""
BigCode evaluator wrapper for optuna pipeline integration.

This module provides a clean interface for integrating BigCode code execution
evaluation with the optuna optimization pipeline.
"""

import logging
from typing import Any, Dict, List, Tuple

from wisent_guard.core.bigcode_extractors import get_bigcode_extractor
from wisent_guard.core.bigcode_integration import BigCodeEvaluator, is_bigcode_task
from wisent_guard.parameters.task_config import CODING_TASKS

logger = logging.getLogger(__name__)


class OptunaBigCodeEvaluator:
    """
    Wrapper for BigCode evaluation in optuna pipeline.

    This class provides a clean interface for evaluating coding tasks
    using actual code execution instead of string comparison.
    """

    def __init__(self, docker_executor=None):
        """
        Initialize the evaluator.

        Args:
            docker_executor: Optional Docker executor for secure code execution
        """
        self.bigcode_evaluator = BigCodeEvaluator(docker_executor)
        self.code_extractor = None  # Will be set per task

    def is_coding_task(self, task_name: str) -> bool:
        """
        Check if a task requires code execution evaluation.

        Args:
            task_name: Name of the task

        Returns:
            True if task requires code execution, False otherwise
        """
        if not task_name:
            return False
        return task_name.lower() in CODING_TASKS or is_bigcode_task(task_name)

    def evaluate_predictions(
        self, predictions: List[str], task_docs: List[Dict[str, Any]], task_name: str
    ) -> List[Dict[str, Any]]:
        """
        Evaluate model predictions using code execution.

        Args:
            predictions: List of model-generated code predictions
            task_docs: List of original task documents with test cases
            task_name: Name of the task

        Returns:
            List of evaluation results for each prediction
        """
        if not self.is_coding_task(task_name):
            raise ValueError(f"Task {task_name} is not a coding task")

        if len(predictions) != len(task_docs):
            raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(task_docs)} task docs")

        results = []

        for i, (prediction, task_doc) in enumerate(zip(predictions, task_docs)):
            try:
                # Get the appropriate extractor for this task
                code_extractor = get_bigcode_extractor(task_name)

                # Extract code from the prediction
                extracted_code = code_extractor.extract_code_from_answer(prediction)

                if not extracted_code.strip():
                    logger.warning(f"No code extracted from prediction {i}: {prediction[:100]}...")
                    result = {
                        "passed": False,
                        "error": "No code extracted from prediction",
                        "extracted_code": extracted_code,
                        "original_prediction": prediction,
                    }
                else:
                    # Execute the code against test cases
                    result = self.bigcode_evaluator._execute_and_test(task_doc, extracted_code, task_name)
                    result["extracted_code"] = extracted_code
                    result["original_prediction"] = prediction

                results.append(result)

            except Exception as e:
                logger.warning(f"Error evaluating prediction {i}: {e}")
                results.append(
                    {"passed": False, "error": str(e), "extracted_code": "", "original_prediction": prediction}
                )

        return results

    def calculate_accuracy(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate accuracy metrics from evaluation results.

        Args:
            evaluation_results: List of evaluation results from evaluate_predictions

        Returns:
            Dictionary with accuracy metrics
        """
        total_samples = len(evaluation_results)
        if total_samples == 0:
            return {
                "accuracy": 0.0,
                "total_samples": 0,
                "correct": 0,
                "incorrect": 0,
                "pass_count": 0,
                "fail_count": 0,
                "error_count": 0,
            }

        pass_count = sum(1 for result in evaluation_results if result.get("passed", False))
        fail_count = sum(
            1 for result in evaluation_results if result.get("passed", False) == False and not result.get("error")
        )
        error_count = sum(1 for result in evaluation_results if result.get("error"))

        accuracy = pass_count / total_samples

        return {
            "accuracy": accuracy,
            "total_samples": total_samples,
            "correct": pass_count,
            "incorrect": fail_count + error_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "error_count": error_count,
            "evaluation_method": "bigcode_execution",
        }

    def evaluate_and_calculate_accuracy(
        self, predictions: List[str], task_docs: List[Dict[str, Any]], task_name: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Convenience method to evaluate predictions and calculate accuracy.

        Args:
            predictions: List of model-generated code predictions
            task_docs: List of original task documents with test cases
            task_name: Name of the task

        Returns:
            Tuple of (evaluation_results, accuracy_metrics)
        """
        evaluation_results = self.evaluate_predictions(predictions, task_docs, task_name)
        accuracy_metrics = self.calculate_accuracy(evaluation_results)

        logger.info(
            f"BigCode evaluation for {task_name}: "
            f"{accuracy_metrics['pass_count']}/{accuracy_metrics['total_samples']} passed "
            f"({accuracy_metrics['accuracy']:.3f} accuracy)"
        )

        return evaluation_results, accuracy_metrics


# Global instance for easy access
_optuna_bigcode_evaluator = None


def get_optuna_bigcode_evaluator(docker_executor=None) -> OptunaBigCodeEvaluator:
    """
    Get global OptunaBigCodeEvaluator instance.

    Args:
        docker_executor: Optional Docker executor

    Returns:
        OptunaBigCodeEvaluator instance
    """
    global _optuna_bigcode_evaluator
    if _optuna_bigcode_evaluator is None:
        _optuna_bigcode_evaluator = OptunaBigCodeEvaluator(docker_executor)
    return _optuna_bigcode_evaluator
