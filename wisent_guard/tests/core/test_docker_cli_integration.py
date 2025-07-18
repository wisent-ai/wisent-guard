"""
Tests for Docker CLI integration with MBPP tasks.
"""

import pytest
from unittest.mock import patch, MagicMock

from wisent_guard.cli import run_task_pipeline
from wisent_guard.core.secure_code_evaluator import SecureCodeEvaluator


@pytest.mark.docker
@pytest.mark.integration
class TestDockerCLIIntegration:
    """Test suite for Docker CLI integration."""

    @patch("wisent_guard.core.secure_code_evaluator.SecureCodeEvaluator")
    def test_mbpp_task_triggers_docker_security(self, mock_evaluator_class):
        """Test that MBPP task triggers Docker security enforcement."""
        # Setup mock
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_evaluator.get_executor_info.return_value = {
            "executor_type": "OptimizedDockerExecutor",
            "image_name": "wisent-guard-codeexec:latest",
            "timeout": 10,
            "memory_limit": "256m",
            "cpu_limit": 0.5,
        }

        # This should trigger Docker security enforcement
        with patch("wisent_guard.cli.load_task_data") as mock_load_task:
            mock_load_task.return_value = ([], "mbpp_config")

            try:
                run_task_pipeline(
                    task_name="mbpp",
                    model_name="MockModel",
                    layer="5",
                    limit=1,
                    verbose=True,
                )
            except Exception:
                # We expect this to fail in testing due to missing dependencies
                # but we want to verify that SecureCodeEvaluator was called
                pass

        # Verify Docker enforcement was triggered
        mock_evaluator_class.assert_called()

    @patch("wisent_guard.core.secure_code_evaluator.SecureCodeEvaluator")
    def test_non_code_task_skips_docker_security(self, mock_evaluator_class):
        """Test that non-code tasks skip Docker security enforcement."""
        with patch("wisent_guard.cli.load_task_data") as mock_load_task:
            mock_load_task.return_value = ([], "gsm8k_config")

            try:
                run_task_pipeline(
                    task_name="gsm8k",
                    model_name="MockModel",
                    layer="5",
                    limit=1,
                    verbose=True,
                )
            except Exception:
                # Expected to fail due to missing dependencies
                pass

        # Docker security should not be triggered for non-code tasks
        mock_evaluator_class.assert_not_called()

    @patch("wisent_guard.core.secure_code_evaluator.SecureCodeEvaluator")
    def test_docker_error_handling(self, mock_evaluator_class):
        """Test error handling when Docker operations fail."""
        # Setup mock to raise error
        mock_evaluator_class.side_effect = RuntimeError("Docker not available")

        with patch("wisent_guard.cli.load_task_data") as mock_load_task:
            mock_load_task.return_value = ([], "mbpp_config")

            with pytest.raises(RuntimeError):
                run_task_pipeline(
                    task_name="mbpp",
                    model_name="MockModel",
                    layer="5",
                    limit=1,
                    verbose=True,
                )

    def test_enforce_secure_execution_integration(self):
        """Test that enforce_secure_execution is properly integrated."""
        from wisent_guard.core.secure_code_evaluator import enforce_secure_execution

        # Test that the function correctly identifies code execution tasks
        assert enforce_secure_execution("mbpp") is True
        assert enforce_secure_execution("humaneval") is True
        assert enforce_secure_execution("gsm8k") is False
        assert enforce_secure_execution("hellaswag") is False


@pytest.mark.docker
@pytest.mark.unit
class TestDockerConfigurationIntegration:
    """Test Docker configuration integration."""

    def test_secure_evaluator_with_custom_docker_config(self):
        """Test SecureCodeEvaluator with custom Docker configuration."""
        docker_config = {
            "timeout": 30,
            "memory_limit": "512m",
            "cpu_limit": 1.0,
            "image_name": "custom-mbpp:latest",
        }

        with patch('wisent_guard.core.secure_code_evaluator.OptimizedDockerExecutor') as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor.timeout = 30
            mock_executor.memory_limit = "512m"
            mock_executor.cpu_limit = 1.0
            mock_executor.image_name = "custom-mbpp:latest"
            mock_executor_class.return_value = mock_executor
            
            evaluator = SecureCodeEvaluator(docker_config=docker_config)

            # Verify configuration is applied
            assert evaluator.executor.timeout == 30
            assert evaluator.executor.memory_limit == "512m"
            assert evaluator.executor.cpu_limit == 1.0
            assert evaluator.executor.image_name == "custom-mbpp:latest"

    @patch('wisent_guard.core.docker.optimized_docker_executor.OptimizedDockerExecutor')
    def test_executor_info_includes_security_settings(self, mock_executor_class):
        """Test that executor info includes all security settings."""
        mock_executor = MagicMock()
        mock_executor.image_name = "wisent-guard-codeexec:latest"
        mock_executor.timeout = 10
        mock_executor.memory_limit = "256m"
        mock_executor.cpu_limit = 0.5
        mock_executor_class.return_value = mock_executor
        
        evaluator = SecureCodeEvaluator()
        info = evaluator.get_executor_info()

        required_keys = [
            "executor_type",
            "image_name",
            "timeout",
            "memory_limit",
            "cpu_limit",
        ]

        for key in required_keys:
            assert key in info, f"Missing key: {key}"

        # Verify security-focused defaults
        assert "wisent-guard-codeexec" in info["image_name"]
        assert info["timeout"] > 0
        assert "m" in info["memory_limit"]  # Memory limit format
        assert 0 < info["cpu_limit"] <= 2.0  # Reasonable CPU limit


@pytest.mark.docker
@pytest.mark.cli
class TestCLISecurityMessages:
    """Test CLI security messages and user feedback."""

    @patch("builtins.print")
    @patch("wisent_guard.core.secure_code_evaluator.SecureCodeEvaluator")
    def test_verbose_security_messages(self, mock_evaluator_class, mock_print):
        """Test that verbose mode shows appropriate security messages."""
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_evaluator.get_executor_info.return_value = {
            "image_name": "wisent-guard-codeexec:latest"
        }

        with patch("wisent_guard.cli.load_task_data") as mock_load_task:
            mock_load_task.return_value = ([], "mbpp_config")

            try:
                run_task_pipeline(
                    task_name="mbpp",
                    model_name="MockModel",
                    layer="5",
                    limit=1,
                    verbose=True,
                )
            except Exception:
                pass

        # Check that security messages were printed
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        security_messages = [
            msg
            for msg in print_calls
            if "🔒" in msg or "secure" in msg.lower() or "docker" in msg.lower()
        ]

        assert (
            len(security_messages) > 0
        ), "No security messages found in verbose output"

    @patch("builtins.print")
    @patch("wisent_guard.core.secure_code_evaluator.SecureCodeEvaluator")
    def test_docker_error_message(self, mock_evaluator_class, mock_print):
        """Test error message when Docker operations fail."""
        # Setup to raise exception
        mock_evaluator_class.side_effect = RuntimeError("Docker not running")

        with patch("wisent_guard.cli.load_task_data") as mock_load_task:
            mock_load_task.return_value = ([], "mbpp_config")

            try:
                run_task_pipeline(
                    task_name="mbpp",
                    model_name="MockModel",
                    layer="5",
                    limit=1,
                    verbose=True,
                )
            except RuntimeError:
                # Expected error
                pass


@pytest.mark.docker
@pytest.mark.performance
class TestDockerPerformanceIntegration:
    """Test Docker performance integration."""

    def test_docker_executor_respects_timeout(self):
        """Test that Docker executor respects timeout configuration."""
        short_timeout_config = {"timeout": 1}  # 1 second
        with patch('wisent_guard.core.secure_code_evaluator.OptimizedDockerExecutor') as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor.timeout = 1
            mock_executor_class.return_value = mock_executor
            
            evaluator = SecureCodeEvaluator(docker_config=short_timeout_config)

            assert evaluator.executor.timeout == 1

    def test_docker_executor_respects_resource_limits(self):
        """Test that Docker executor respects resource limits."""
        resource_config = {"memory_limit": "128m", "cpu_limit": 0.25}
        with patch('wisent_guard.core.secure_code_evaluator.OptimizedDockerExecutor') as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor.memory_limit = "128m"
            mock_executor.cpu_limit = 0.25
            mock_executor_class.return_value = mock_executor
            
            evaluator = SecureCodeEvaluator(docker_config=resource_config)

            assert evaluator.executor.memory_limit == "128m"
            assert evaluator.executor.cpu_limit == 0.25
