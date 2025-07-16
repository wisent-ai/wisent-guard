"""
Tests for secure code evaluator with Docker enforcement.
"""

import pytest
from unittest.mock import MagicMock

from wisent_guard.core.secure_code_evaluator import (
    SecureCodeEvaluator,
    enforce_secure_execution,
    CODE_EXECUTION_TASKS,
)
from wisent_guard.core.docker import MockDockerExecutor


class TestSecureCodeEvaluator:
    """Test suite for SecureCodeEvaluator."""

    def test_code_execution_task_detection(self):
        """Test detection of code execution tasks."""
        # Test known code execution tasks
        assert SecureCodeEvaluator.is_code_execution_task("mbpp") is True
        assert SecureCodeEvaluator.is_code_execution_task("MBPP") is True
        assert SecureCodeEvaluator.is_code_execution_task("humaneval") is True
        assert SecureCodeEvaluator.is_code_execution_task("apps") is True

        # Test non-code execution tasks
        assert SecureCodeEvaluator.is_code_execution_task("gsm8k") is False
        assert SecureCodeEvaluator.is_code_execution_task("hellaswag") is False
        assert SecureCodeEvaluator.is_code_execution_task("truthfulqa_mc1") is False

    def test_secure_evaluator_initialization_with_mock(self):
        """Test SecureCodeEvaluator initialization with mock executor."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        assert isinstance(evaluator.executor, MockDockerExecutor)
        assert evaluator.docker_config == {}

    def test_secure_evaluator_initialization_with_config(self):
        """Test SecureCodeEvaluator initialization with custom config."""
        config = {"timeout": 30, "memory_limit": "512m", "cpu_limit": 1.0}

        evaluator = SecureCodeEvaluator(use_mock=True, docker_config=config)

        assert evaluator.docker_config == config
        assert evaluator.executor.timeout == 30
        assert evaluator.executor.memory_limit == "512m"
        assert evaluator.executor.cpu_limit == 1.0

    def test_evaluate_mbpp_response(self):
        """Test evaluating MBPP response."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        task_data = {
            "task_id": "test_1",
            "text": "Write a function to add two numbers",
            "test_list": [
                "assert add_numbers(2, 3) == 5",
                "assert add_numbers(0, 0) == 0",
            ],
        }

        generated_code = """
def add_numbers(a, b):
    return a + b
"""

        result = evaluator.evaluate_response("mbpp", task_data, generated_code)

        assert result is not None
        assert result["task_name"] == "mbpp"
        assert "success" in result
        assert "passed" in result
        assert result["passed"] == result["success"]

    def test_evaluate_non_code_task_raises_error(self):
        """Test that non-code tasks raise ValueError."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        with pytest.raises(ValueError, match="is not a code execution task"):
            evaluator.evaluate_response("gsm8k", {}, "some response")

    def test_batch_evaluate_responses(self):
        """Test batch evaluation of responses."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        tasks = [
            {"task_id": "test_1", "test_list": ["assert func1() == 1"]},
            {"task_id": "test_2", "test_list": ["assert func2() == 2"]},
        ]

        responses = ["def func1(): return 1", "def func2(): return 2"]

        results = evaluator.batch_evaluate_responses("mbpp", tasks, responses)

        assert len(results) == 2
        assert all(r["task_name"] == "mbpp" for r in results)
        assert all("success" in r for r in results)

    def test_batch_evaluate_mismatched_lengths(self):
        """Test batch evaluation with mismatched lengths raises error."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        tasks = [{"task_id": "test_1"}]
        responses = ["code1", "code2"]  # Different length

        with pytest.raises(ValueError, match="must match"):
            evaluator.batch_evaluate_responses("mbpp", tasks, responses)

    def test_evaluate_contrastive_pairs(self):
        """Test evaluation of contrastive pairs."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        contrastive_pairs = [
            {
                "task_id": "test_1",
                "correct_answer": "def func(): return 1",
                "incorrect_answer": "def func(): return 2",
                "test_list": ["assert func() == 1"],
            }
        ]

        result = evaluator.evaluate_contrastive_pairs("mbpp", contrastive_pairs)

        assert "correct_results" in result
        assert "incorrect_results" in result
        assert "correct_pass_rate" in result
        assert "incorrect_pass_rate" in result
        assert len(result["correct_results"]) == 1
        assert len(result["incorrect_results"]) == 1

    def test_get_executor_info(self):
        """Test getting executor information."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        info = evaluator.get_executor_info()

        assert "executor_type" in info
        assert "image_name" in info
        assert "timeout" in info
        assert "memory_limit" in info
        assert "cpu_limit" in info
        assert info["executor_type"] == "MockDockerExecutor"

    def test_cleanup(self):
        """Test cleanup method."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        # Mock the executor cleanup method
        evaluator.executor.cleanup = MagicMock()

        evaluator.cleanup()

        evaluator.executor.cleanup.assert_called_once()


class TestEnforceSecureExecution:
    """Test suite for enforce_secure_execution function."""

    def test_enforce_secure_execution_for_code_tasks(self):
        """Test that code execution tasks require security."""
        assert enforce_secure_execution("mbpp") is True
        assert enforce_secure_execution("humaneval") is True
        assert enforce_secure_execution("apps") is True

    def test_enforce_secure_execution_for_non_code_tasks(self):
        """Test that non-code tasks don't require security."""
        assert enforce_secure_execution("gsm8k") is False
        assert enforce_secure_execution("hellaswag") is False
        assert enforce_secure_execution("truthfulqa_mc1") is False

    def test_code_execution_tasks_constant(self):
        """Test that CODE_EXECUTION_TASKS contains expected values."""
        assert "mbpp" in CODE_EXECUTION_TASKS
        assert "humaneval" in CODE_EXECUTION_TASKS
        assert "apps" in CODE_EXECUTION_TASKS
        assert "codexglue_code_to_text" in CODE_EXECUTION_TASKS
        assert "concode" in CODE_EXECUTION_TASKS
        assert "conala" in CODE_EXECUTION_TASKS

        # Non-code tasks should not be included
        assert "gsm8k" not in CODE_EXECUTION_TASKS
        assert "hellaswag" not in CODE_EXECUTION_TASKS


@pytest.mark.integration
class TestSecureCodeEvaluatorIntegration:
    """Integration tests for SecureCodeEvaluator."""

    def test_mbpp_evaluation_with_correct_code(self):
        """Test MBPP evaluation with correct code."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        task_data = {
            "task_id": "mbpp_1",
            "text": "Write a function to calculate factorial",
            "test_list": [
                "assert factorial(5) == 120",
                "assert factorial(0) == 1",
                "assert factorial(1) == 1",
            ],
        }

        correct_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

        result = evaluator.evaluate_response("mbpp", task_data, correct_code)

        # With mock executor, syntactically correct code should pass
        assert result["success"] is True
        assert result["passed"] is True
        assert result["task_name"] == "mbpp"

    def test_mbpp_evaluation_with_incorrect_code(self):
        """Test MBPP evaluation with syntactically incorrect code."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        task_data = {
            "task_id": "mbpp_2",
            "text": "Write a function to calculate factorial",
            "test_list": ["assert factorial(5) == 120"],
        }

        incorrect_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1  # Missing closing parenthesis
"""

        result = evaluator.evaluate_response("mbpp", task_data, incorrect_code)

        # With mock executor, syntactically incorrect code should fail
        assert result["success"] is False
        assert result["passed"] is False
        assert "SyntaxError" in result["error"] or "syntax" in result["error"].lower()

    def test_end_to_end_contrastive_evaluation(self):
        """Test end-to-end contrastive evaluation."""
        evaluator = SecureCodeEvaluator(use_mock=True)

        pairs = [
            {
                "task_id": "test_1",
                "correct_answer": "def add(a, b): return a + b",
                "incorrect_answer": "def add(a, b): return a - b",  # Wrong operation
                "test_list": ["assert add(2, 3) == 5"],
            },
            {
                "task_id": "test_2",
                "correct_answer": "def multiply(a, b): return a * b",
                "incorrect_answer": "def multiply(a, b): return a +",  # Syntax error
                "test_list": ["assert multiply(2, 3) == 6"],
            },
        ]

        results = evaluator.evaluate_contrastive_pairs("mbpp", pairs)

        assert len(results["correct_results"]) == 2
        assert len(results["incorrect_results"]) == 2

        # All correct code should pass (syntactically)
        assert all(r["success"] for r in results["correct_results"])

        # Incorrect code with syntax error should fail
        assert not results["incorrect_results"][1]["success"]

        # Pass rates should be calculated correctly
        assert results["correct_pass_rate"] == 1.0  # All correct code passes
        assert results["incorrect_pass_rate"] < 1.0  # Some incorrect code fails
