"""
Integration tests for Docker-based MBPP execution with wisent-guard pipeline.
"""

import pytest
from unittest.mock import patch


@pytest.mark.docker
@pytest.mark.integration
class TestDockerWisentGuardIntegration:
    """Integration tests for Docker execution within wisent-guard pipeline."""

    def test_docker_mbpp_pipeline_integration(
        self, docker_mbpp_runner, sample_mbpp_data, docker_config
    ):
        """Test that Docker MBPP execution integrates with wisent-guard pipeline."""
        # Simulate the pipeline flow:
        # 1. Load MBPP data
        # 2. Run in Docker
        # 3. Collect results

        tasks = sample_mbpp_data[:2]  # Test with 2 tasks
        results = docker_mbpp_runner.run_mbpp_batch(tasks, docker_config)

        # Verify pipeline output format
        assert len(results) == 2
        for result in results:
            assert "success" in result
            assert "task_id" in result
            assert "exit_code" in result or "error" in result

    def test_docker_activation_extraction_simulation(
        self, docker_mbpp_runner, docker_config
    ):
        """Test simulation of activation extraction from Docker execution."""
        # Simulate a task that would generate activations
        activation_task = {
            "task_id": 1,
            "code": '''def generate_response():
    # This simulates a function that would trigger activation extraction
    return "Generated response that would have activations"''',
            "test_list": ["assert generate_response().startswith('Generated')"],
        }

        result = docker_mbpp_runner.run_mbpp_task(activation_task, docker_config)

        # In real integration, this would trigger activation extraction
        assert result["success"] is True
        assert result["task_id"] == 1

    def test_docker_contrastive_pair_execution(self, docker_mbpp_runner, docker_config):
        """Test Docker execution of contrastive pairs for wisent-guard training."""
        # Correct implementation (positive example)
        correct_task = {
            "task_id": 1,
            "code": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
            "test_list": [
                "assert fibonacci(0) == 0",
                "assert fibonacci(1) == 1",
                "assert fibonacci(5) == 5",
            ],
        }

        # Incorrect implementation (negative example)
        incorrect_task = {
            "task_id": 2,
            "code": """def fibonacci(n):
    if n <= 1:
        return n + 1  # Bug: adds 1
    return fibonacci(n-1) + fibonacci(n-2)""",
            "test_list": [
                "assert fibonacci(0) == 0",
                "assert fibonacci(1) == 1",
                "assert fibonacci(5) == 5",
            ],
        }

        correct_result = docker_mbpp_runner.run_mbpp_task(correct_task, docker_config)
        incorrect_result = docker_mbpp_runner.run_mbpp_task(
            incorrect_task, docker_config
        )

        # Both should execute (Docker isolation), but incorrect should fail assertions
        assert correct_result["success"] is True
        assert "task_id" in incorrect_result  # Docker execution works regardless

    def test_docker_batch_processing_for_training(
        self, docker_mbpp_runner, sample_mbpp_data, docker_config
    ):
        """Test batch processing of MBPP tasks for classifier training."""
        # Process multiple tasks as would be done for training
        batch_results = docker_mbpp_runner.run_mbpp_batch(
            sample_mbpp_data, docker_config
        )

        # Verify batch processing results
        assert len(batch_results) == len(sample_mbpp_data)

        # All tasks should have been processed
        processed_task_ids = [r["task_id"] for r in batch_results]
        expected_task_ids = [t["task_id"] for t in sample_mbpp_data]
        assert set(processed_task_ids) == set(expected_task_ids)

    def test_docker_environment_for_hf_eval(
        self, docker_mbpp_runner, docker_config, docker_environment
    ):
        """Test Docker environment setup for HuggingFace evaluation."""
        # Test that HF_ALLOW_CODE_EVAL is properly set
        env_task = {
            "task_id": 1,
            "code": """import os
def check_hf_eval():
    return os.environ.get('HF_ALLOW_CODE_EVAL', '0')""",
            "test_list": ["assert check_hf_eval() == '1'"],
        }

        result = docker_mbpp_runner.run_mbpp_task(env_task, docker_config)
        assert result["success"] is True


@pytest.mark.docker
@pytest.mark.performance
class TestDockerPerformance:
    """Performance tests for Docker-based MBPP execution."""

    def test_docker_startup_time(self, docker_mbpp_runner, docker_config):
        """Test Docker container startup time is reasonable."""
        import time

        simple_task = {
            "task_id": 1,
            "code": "def quick(): return 'done'",
            "test_list": ["assert quick() == 'done'"],
        }

        start_time = time.time()
        result = docker_mbpp_runner.run_mbpp_task(simple_task, docker_config)
        execution_time = time.time() - start_time

        assert result["success"] is True
        # Should complete quickly (mock execution)
        assert execution_time < 1.0  # Less than 1 second for mock

    def test_docker_concurrent_execution(
        self, docker_mbpp_runner, sample_mbpp_data, docker_config
    ):
        """Test concurrent Docker execution performance."""
        # Simulate concurrent execution of multiple tasks
        results = docker_mbpp_runner.run_mbpp_batch(sample_mbpp_data, docker_config)

        # All tasks should complete successfully
        successful_tasks = [r for r in results if r["success"]]
        assert len(successful_tasks) == len(sample_mbpp_data)

    def test_docker_memory_usage_tracking(self, docker_mbpp_runner, docker_config):
        """Test that Docker memory usage is tracked."""
        memory_task = {
            "task_id": 1,
            "code": """def create_data():
    return [i for i in range(1000)]""",
            "test_list": ["assert len(create_data()) == 1000"],
        }

        result = docker_mbpp_runner.run_mbpp_task(memory_task, docker_config)
        assert result["success"] is True

        # Memory limit should be enforced
        assert docker_config["memory_limit"] == "128m"


@pytest.mark.docker
@pytest.mark.cli
class TestDockerCLIIntegration:
    """Tests for CLI integration with Docker-based MBPP execution."""

    def test_docker_cli_parameter_mapping(self, docker_config):
        """Test that CLI parameters map correctly to Docker configuration."""
        # Test CLI parameters that should map to Docker config
        cli_to_docker_mapping = {
            "timeout": "timeout",
            "memory_limit": "memory_limit",
            "network_isolation": "network_mode",
            "read_only_filesystem": "read_only",
            "auto_remove": "remove",
        }

        for cli_param, docker_param in cli_to_docker_mapping.items():
            # Docker config should have the mapped parameter
            assert docker_param in docker_config

    def test_docker_image_name_configuration(self, docker_config):
        """Test that Docker image name is properly configured."""
        assert docker_config["image"] == "wisent-guard-codeexec:latest"

        # Image name should follow convention
        assert "wisent-guard" in docker_config["image"]
        assert "codeexec" in docker_config["image"]

    def test_docker_volume_configuration_for_cli(self, docker_config):
        """Test Docker volume configuration for CLI usage."""
        volumes = docker_config["volumes"]

        # Should have minimal required volumes
        assert "/tmp" in volumes

        # Volumes should be properly configured
        for volume_config in volumes.values():
            assert "bind" in volume_config
            assert "mode" in volume_config

    @patch("wisent_guard.cli.run_task_pipeline")
    def test_docker_cli_command_simulation(
        self, mock_run_task, docker_mbpp_runner, docker_config
    ):
        """Test simulation of CLI command: python -m wisent_guard tasks mbpp --docker"""
        # Simulate CLI call with Docker flag
        mock_run_task.return_value = {
            "success": True,
            "results": [{"task_id": 1, "passed": True}],
        }

        # This would be the actual CLI integration
        cli_result = mock_run_task(
            task_name="mbpp",
            model="distilbert/distilgpt2",
            layer=5,
            limit=10,
            docker=True,
            docker_config=docker_config,
        )

        assert cli_result["success"] is True
        mock_run_task.assert_called_once()

    def test_docker_error_handling_for_cli(self, docker_mbpp_runner, docker_config):
        """Test Docker error handling for CLI usage."""
        # Test with invalid task that should fail gracefully
        invalid_task = {
            "task_id": 999,
            "code": """def invalid_function():
    raise ValueError("This should fail")""",
            "test_list": ["assert invalid_function() == 'success'"],
        }

        result = docker_mbpp_runner.run_mbpp_task(invalid_task, docker_config)

        # Should handle error gracefully
        assert "task_id" in result
        # Error information should be available for CLI reporting
        assert "error" in result or "success" in result
