"""
Secure code evaluator that enforces Docker execution for all code evaluation tasks.
This module ensures that NO code execution happens outside of Docker containers.
"""

import logging
from typing import Any, Dict, List, Optional

from .docker import OptimizedDockerExecutor

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
    "livecodebench",  # LiveCodeBench task
    # Add more code execution tasks as needed
}


class SecureCodeEvaluator:
    """
    Evaluates code tasks securely using Docker containers.

    This class ensures that ALL code execution happens in isolated Docker containers,
    preventing any security risks from running untrusted code.
    """

    def __init__(self, docker_config: Optional[Dict[str, Any]] = None):
        """
        Initialize secure code evaluator.

        Args:
            docker_config: Docker configuration options

        Raises:
            RuntimeError: If Docker is not available or not running
        """
        # Filter out configs that OptimizedDockerExecutor doesn't accept
        executor_config = docker_config or {}
        valid_params = {"image_name", "build_if_missing", "enable_batching", "enable_resource_optimization"}
        filtered_config = {k: v for k, v in executor_config.items() if k in valid_params}

        try:
            self.executor = OptimizedDockerExecutor(
                **filtered_config, enable_batching=True, enable_resource_optimization=True
            )
        except RuntimeError as e:
            # Docker is not available - fail hard with clear message
            logger.error(f"Docker is required for code execution tasks: {e}")
            raise RuntimeError(
                f"\n{'=' * 60}\n"
                f"ERROR: Docker is required for code execution tasks\n"
                f"{'=' * 60}\n"
                f"{e!s}\n\n"
                f"Please ensure Docker is:\n"
                f"1. Installed on your system\n"
                f"2. Running (start Docker Desktop or daemon)\n"
                f"3. Accessible to the current user\n"
                f"{'=' * 60}\n"
            )

        # Store additional config separately if needed
        self.runtime_config = {k: v for k, v in executor_config.items() if k not in valid_params}
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

    def evaluate_response(self, task_name: str, task_data: Dict[str, Any], generated_response: str) -> Dict[str, Any]:
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
        if task_name.lower() == "livecodebench":
            return self._evaluate_livecodebench_response(task_data, generated_response)
        raise ValueError(f"Unsupported code execution task: {task_name}")

    def _evaluate_mbpp_response(self, task_data: Dict[str, Any], generated_code: str) -> Dict[str, Any]:
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

        # Execute in Docker - convert MBPP task to code execution format
        code = docker_task["code"]
        # Create test execution code
        test_code = "\n".join(docker_task["test_list"])
        full_code = f"{code}\n\n# Run tests\n{test_code}"

        result = self.executor.execute_single(full_code)

        # Add task-specific information
        result["task_name"] = "mbpp"
        result["passed"] = result.get("success", False)

        return result

    def _evaluate_livecodebench_response(self, task_data: Dict[str, Any], generated_code: str) -> Dict[str, Any]:
        """
        Evaluate LiveCodeBench response in Docker.

        Args:
            task_data: LiveCodeBench task data with test cases
            generated_code: Generated code to evaluate

        Returns:
            Evaluation results
        """
        # Extract test cases from task data
        public_test_cases = task_data.get("public_test_cases", [])
        private_test_cases = task_data.get("private_test_cases", [])
        all_test_cases = public_test_cases + private_test_cases

        if not all_test_cases:
            # If no test cases, can't evaluate
            return {"task_name": "livecodebench", "passed": False, "error": "No test cases available", "success": False}

        # Create test execution code for LiveCodeBench
        # LiveCodeBench uses stdin/stdout format
        test_results = []
        for i, test_case in enumerate(all_test_cases):
            test_input = test_case.get("input", "")
            expected_output = test_case.get("output", "").strip()

            # Create a test script that runs the code with the input
            # Use repr() to properly escape the strings
            test_script = f"""
import sys
import io

# Test input and expected output
test_input = {test_input!r}
expected_output = {expected_output!r}

# Redirect stdin
sys.stdin = io.StringIO(test_input)

# Capture stdout
old_stdout = sys.stdout
sys.stdout = io.StringIO()

# Generated code
{generated_code}

# Get output
output = sys.stdout.getvalue()
sys.stdout = old_stdout

# Check if output matches expected
actual = output.strip()
expected = expected_output.strip()

if actual == expected:
    print("TEST PASSED")
else:
    print("TEST FAILED")
    print(f"Expected:\\n{{expected}}")
    print(f"Actual:\\n{{actual}}")
"""

            # Execute test in Docker
            result = self.executor.execute_single(test_script)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            success = result.get("success", False)

            # Check if test passed by looking for TEST PASSED in stdout
            passed = success and "TEST PASSED" in stdout

            test_results.append({"test_id": i, "passed": passed, "output": stdout, "error": stderr})

        # Aggregate results
        passed_count = sum(1 for r in test_results if r["passed"])
        total_count = len(test_results)

        return {
            "task_name": "livecodebench",
            "passed": passed_count == total_count,
            "success": passed_count == total_count,
            "passed_tests": passed_count,
            "total_tests": total_count,
            "test_results": test_results,
            "pass_rate": passed_count / total_count if total_count > 0 else 0.0,
        }

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

    def evaluate_contrastive_pairs(self, task_name: str, contrastive_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            correct_result = self.evaluate_response(task_name, pair, pair.get("correct_answer", ""))
            correct_results.append(correct_result)

            # Evaluate incorrect code
            incorrect_result = self.evaluate_response(task_name, pair, pair.get("incorrect_answer", ""))
            incorrect_results.append(incorrect_result)

        return {
            "correct_results": correct_results,
            "incorrect_results": incorrect_results,
            "correct_pass_rate": (
                sum(r["passed"] for r in correct_results) / len(correct_results) if correct_results else 0
            ),
            "incorrect_pass_rate": (
                sum(r["passed"] for r in incorrect_results) / len(incorrect_results) if incorrect_results else 0
            ),
        }

    def get_executor_info(self) -> Dict[str, Any]:
        """Get information about the Docker executor."""
        return {
            "executor_type": type(self.executor).__name__,
            "image_name": getattr(self.executor, "image_name", "wisent-guard-codeexec:latest"),
            "timeout": getattr(self.executor, "timeout", 10),
            "memory_limit": getattr(self.executor, "memory_limit", "256m"),
            "cpu_limit": getattr(self.executor, "cpu_limit", 0.5),
        }

    def cleanup(self):
        """Clean up Docker resources."""
        self.executor.cleanup()


def enforce_secure_execution(task_name: str, trust_code_execution: bool = False) -> bool:
    """
    Check if a task must use secure Docker execution.

    Args:
        task_name: Name of the task
        trust_code_execution: If True, allows bypassing Docker requirement (UNSAFE - use only in trusted environments)

    Returns:
        True if task requires secure execution

    Raises:
        SecurityError: If attempting to execute code outside Docker
    """
    if SecureCodeEvaluator.is_code_execution_task(task_name):
        if trust_code_execution:
            # UNSAFE: User explicitly trusts this environment for code execution
            print(f"⚠️  WARNING: Running code task '{task_name}' WITHOUT Docker security!")
            print("   • Code will execute directly in current environment")
            print("   • This is UNSAFE unless you're in a secure sandbox (e.g., RunPod container)")
            print("   • Use --trust-code-execution only in isolated environments")

            # Set required environment variable for HuggingFace code_eval metric
            import os

            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            if "TRUST_REMOTE_CODE" not in os.environ:
                os.environ["TRUST_REMOTE_CODE"] = "1"
            print("   • Set HF_ALLOW_CODE_EVAL=1 for code evaluation")

            return False  # Skip Docker requirement
        # This is a code execution task - MUST use Docker
        return True
    return False


class SecurityError(Exception):
    """Raised when attempting unsafe code execution."""
