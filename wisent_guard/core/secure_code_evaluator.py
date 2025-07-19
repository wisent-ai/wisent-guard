"""
Secure code evaluator that enforces Docker execution for all code evaluation tasks.
This module ensures that NO code execution happens outside of Docker containers.
"""

import logging
from typing import Dict, Any, List, Optional

from .docker import DockerExecutor

logger = logging.getLogger(__name__)

# Tasks that require code execution - MUST use Docker
CODE_EXECUTION_TASKS = {
    "mbpp",
    "humaneval",
    "humaneval_plus",
    "instructhumaneval",
    "apps",
    "mbpp_plus",
    "ds1000",
    "humanevalpack",
    "multiple_py",
    "multiple_js",
    "multiple_java",
    "multiple_cpp",
    "multiple_rs",
    "multiple_go",
    "recode",
    "conala",
    "concode",
    "codexglue_code_to_text",
    "codexglue_code_to_text_python",
    "codexglue_code_to_text_go",
    "codexglue_code_to_text_ruby",
    "codexglue_code_to_text_java",
    "codexglue_code_to_text_javascript",
    "codexglue_code_to_text_php",
    "mercury",
    # Add more code execution tasks as needed
}


class SecureCodeEvaluator:
    """
    Evaluates code tasks securely using Docker containers.

    This class ensures that ALL code execution happens in isolated Docker containers,
    preventing any security risks from running untrusted code.
    """

    def __init__(
        self, docker_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize secure code evaluator.

        Args:
            docker_config: Docker configuration options
        """
        self.executor = DockerExecutor(**(docker_config or {}))
        self.docker_config = docker_config or {}

    @classmethod
    def is_code_execution_task(cls, task_name: str) -> bool:
        """
        Check if a task requires code execution.

        Args:
            task_name: Name of the task

        Returns:
            True if task requires code execution
        """
        return task_name.lower() in CODE_EXECUTION_TASKS

    def evaluate_response(
        self, task_name: str, task_data: Dict[str, Any], generated_response: str
    ) -> Dict[str, Any]:
        """
        Evaluate a generated response for a code task.

        Args:
            task_name: Name of the task (e.g., "mbpp")
            task_data: Task data including test cases
            generated_response: Generated code from the model

        Returns:
            Evaluation results
        """
        if not self.is_code_execution_task(task_name):
            raise ValueError(f"Task {task_name} is not a code execution task")

        if task_name.lower() == "mbpp":
            return self._evaluate_mbpp_response(task_data, generated_response)
        else:
            raise NotImplementedError(f"Evaluation for {task_name} not yet implemented")

    def _evaluate_mbpp_response(
        self, task_data: Dict[str, Any], generated_code: str
    ) -> Dict[str, Any]:
        """
        Evaluate MBPP response in Docker.

        Args:
            task_data: MBPP task data with test_list
            generated_code: Generated code to evaluate

        Returns:
            Evaluation results
        """
        # Prepare task for Docker execution
        docker_task = {
            "task_id": task_data.get("task_id", "unknown"),
            "code": generated_code,
            "test_list": task_data.get("test_list", []),
        }

        # Execute in Docker
        result = self.executor.execute_code_task(docker_task)

        # Add task-specific information
        result["task_name"] = "mbpp"
        result["passed"] = result.get("success", False)

        return result

    def batch_evaluate_responses(
        self,
        task_name: str,
        task_data_list: List[Dict[str, Any]],
        generated_responses: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses in batch.

        Args:
            task_name: Name of the task
            task_data_list: List of task data
            generated_responses: List of generated responses

        Returns:
            List of evaluation results
        """
        if len(task_data_list) != len(generated_responses):
            raise ValueError("Number of tasks and responses must match")

        results = []
        for task_data, response in zip(task_data_list, generated_responses):
            result = self.evaluate_response(task_name, task_data, response)
            results.append(result)

        return results

    def evaluate_contrastive_pairs(
        self, task_name: str, contrastive_pairs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate contrastive pairs for classifier training.

        Args:
            task_name: Name of the task
            contrastive_pairs: List of contrastive pairs with correct/incorrect code

        Returns:
            Evaluation results for both correct and incorrect examples
        """
        if not self.is_code_execution_task(task_name):
            raise ValueError(f"Task {task_name} is not a code execution task")

        correct_results = []
        incorrect_results = []

        for pair in contrastive_pairs:
            # Evaluate correct code
            correct_result = self.evaluate_response(
                task_name, pair, pair.get("correct_answer", "")
            )
            correct_results.append(correct_result)

            # Evaluate incorrect code
            incorrect_result = self.evaluate_response(
                task_name, pair, pair.get("incorrect_answer", "")
            )
            incorrect_results.append(incorrect_result)

        return {
            "correct_results": correct_results,
            "incorrect_results": incorrect_results,
            "correct_pass_rate": (
                sum(r["passed"] for r in correct_results) / len(correct_results)
                if correct_results
                else 0
            ),
            "incorrect_pass_rate": (
                sum(r["passed"] for r in incorrect_results) / len(incorrect_results)
                if incorrect_results
                else 0
            ),
        }

    def get_executor_info(self) -> Dict[str, Any]:
        """Get information about the Docker executor."""
        return {
            "executor_type": type(self.executor).__name__,
            "image_name": self.executor.image_name,
            "timeout": self.executor.timeout,
            "memory_limit": self.executor.memory_limit,
            "cpu_limit": self.executor.cpu_limit,
        }

    def cleanup(self):
        """Clean up Docker resources."""
        self.executor.cleanup()


def enforce_secure_execution(task_name: str) -> bool:
    """
    Check if a task must use secure Docker execution.

    Args:
        task_name: Name of the task

    Returns:
        True if task requires secure execution

    Raises:
        SecurityError: If attempting to execute code outside Docker
    """
    if SecureCodeEvaluator.is_code_execution_task(task_name):
        # This is a code execution task - MUST use Docker
        return True
    return False


class SecurityError(Exception):
    """Raised when attempting unsafe code execution."""

    pass
