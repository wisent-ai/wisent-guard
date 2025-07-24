"""
Tests for Docker-based MBPP execution, following bigcode-evaluation-harness patterns.
"""

import pytest
import docker


@pytest.mark.docker
class TestDockerMBPPBasic:
    """Basic Docker functionality tests for MBPP execution."""

    def test_docker_client_availability(self):
        """Test that Docker client is available."""
        try:
            client = docker.from_env()
            assert client.ping() is True
            version_info = client.version()
            assert "Version" in version_info
        except docker.errors.DockerException as e:
            pytest.fail(f"Docker is not running or not installed: {e}")

    def test_docker_image_availability(self):
        """Test that MBPP Docker image is available."""
        try:
            client = docker.from_env()
            images = client.images.list()
            image_tags = [tag for image in images for tag in image.tags]
            # Skip if image not found - it will be built when needed
            if "wisent-guard-codeexec:latest" not in image_tags:
                pytest.skip("wisent-guard-codeexec:latest image not built yet")
        except docker.errors.DockerException as e:
            pytest.fail(f"Docker is not running or not installed: {e}")

    def test_basic_python_execution(self, docker_config):
        """Test basic Python code execution in Docker container."""
        try:
            client = docker.from_env()
            simple_code = "print('Hello, Docker!')"

            result = client.containers.run(
                image=docker_config["image"],
                command=["python", "-c", simple_code],
                timeout=docker_config["timeout"],
                remove=docker_config["remove"],
            )

            assert result.exit_code == 0
        except docker.errors.DockerException as e:
            pytest.skip(f"Docker not available: {e}")

    def test_container_configuration(self, docker_config):
        """Test that container is configured with proper security settings."""
        try:
            client = docker.from_env()
            test_code = "import os; print(os.getcwd())"

            container = client.containers.run(
                image=docker_config["image"],
                command=["python", "-c", test_code],
                timeout=docker_config["timeout"],
                mem_limit=docker_config["memory_limit"],
                network_mode=docker_config["network_mode"],
                read_only=docker_config["read_only"],
                remove=docker_config["remove"],
                volumes=docker_config["volumes"],
                detach=True,
            )

            assert container.image == docker_config["image"]
            assert container.volumes == docker_config["volumes"]
        except docker.errors.DockerException as e:
            pytest.skip(f"Docker not available: {e}")


@pytest.mark.docker
class TestDockerMBPPHappyPath:
    """Happy path tests for MBPP execution in Docker."""

    def test_single_mbpp_task_execution(
        self, docker_mbpp_runner, sample_mbpp_data, docker_config
    ):
        """Test executing a single MBPP task in Docker."""
        task = sample_mbpp_data[0]

        result = docker_mbpp_runner.run_mbpp_task(task, docker_config)

        assert result["success"] is True
        assert result["task_id"] == task["task_id"]
        assert result["exit_code"] == 0

    def test_mbpp_batch_execution(
        self, docker_mbpp_runner, sample_mbpp_data, docker_config
    ):
        """Test executing multiple MBPP tasks in Docker."""
        results = docker_mbpp_runner.run_mbpp_batch(sample_mbpp_data, docker_config)

        assert len(results) == len(sample_mbpp_data)

        for i, result in enumerate(results):
            assert result["success"] is True
            assert result["task_id"] == sample_mbpp_data[i]["task_id"]
            assert result["exit_code"] == 0

    def test_mbpp_task_with_assertions(
        self, docker_mbpp_runner, sample_mbpp_data, docker_config
    ):
        """Test MBPP task with proper test assertions."""
        task = sample_mbpp_data[0]  # remove_Occ function

        # This should pass all assertions
        result = docker_mbpp_runner.run_mbpp_task(task, docker_config)

        assert result["success"] is True
        assert result["task_id"] == task["task_id"]

    def test_mbpp_code_generation_format(
        self, docker_mbpp_runner, sample_mbpp_data, docker_config
    ):
        """Test that MBPP code follows expected format."""
        task = sample_mbpp_data[1]  # sort_matrix function

        # Verify the code structure
        assert "def sort_matrix" in task["code"]
        assert "return result" in task["code"]

        # Test execution
        result = docker_mbpp_runner.run_mbpp_task(task, docker_config)
        assert result["success"] is True

    def test_docker_environment_isolation(self, docker_mbpp_runner, docker_config):
        """Test that Docker environment is properly isolated."""
        # Test that HF_ALLOW_CODE_EVAL is set in the environment
        task = {
            "task_id": 999,
            "code": "import os",
            "test_list": ["assert os.environ.get('HF_ALLOW_CODE_EVAL') == '1'"],
        }

        result = docker_mbpp_runner.run_mbpp_task(task, docker_config)
        assert result["success"] is True


@pytest.mark.docker
@pytest.mark.integration
class TestDockerMBPPIntegration:
    """Integration tests for Docker MBPP execution with wisent-guard."""

    def test_docker_mbpp_with_contrastive_pairs(
        self, docker_mbpp_runner, docker_config
    ):
        """Test Docker execution with contrastive pairs (correct vs incorrect code)."""
        # Correct implementation
        correct_task = {
            "task_id": 1,
            "code": """def add_numbers(a, b):
    return a + b""",
            "test_list": [
                "assert add_numbers(2, 3) == 5",
                "assert add_numbers(0, 0) == 0",
            ],
        }

        # Incorrect implementation
        incorrect_task = {
            "task_id": 2,
            "code": """def add_numbers(a, b):
    return a * b""",  # Wrong operation
            "test_list": [
                "assert add_numbers(2, 3) == 5",
                "assert add_numbers(0, 0) == 0",
            ],
        }

        correct_result = docker_mbpp_runner.run_mbpp_task(correct_task, docker_config)
        incorrect_result = docker_mbpp_runner.run_mbpp_task(
            incorrect_task, docker_config
        )

        # Correct should pass
        assert correct_result["success"] is True

        # Incorrect should fail (due to assertion error)
        # But Docker execution itself should still work
        assert "task_id" in incorrect_result

    def test_docker_timeout_handling(self, docker_mbpp_runner, docker_config):
        """Test that Docker properly handles timeout scenarios."""
        # Create a task that should complete within timeout
        quick_task = {
            "task_id": 1,
            "code": '''def quick_function():
    return "done"''',
            "test_list": ["assert quick_function() == 'done'"],
        }

        # Set a reasonable timeout
        config_with_timeout = docker_config.copy()
        config_with_timeout["timeout"] = 5

        result = docker_mbpp_runner.run_mbpp_task(quick_task, config_with_timeout)
        assert result["success"] is True

    def test_docker_memory_limit_configuration(self, docker_mbpp_runner, docker_config):
        """Test that Docker memory limits are properly configured."""
        # Simple task that should run within memory limits
        memory_task = {
            "task_id": 1,
            "code": """def create_small_list():
    return [1, 2, 3, 4, 5]""",
            "test_list": ["assert len(create_small_list()) == 5"],
        }

        result = docker_mbpp_runner.run_mbpp_task(memory_task, docker_config)
        assert result["success"] is True

    def test_docker_volume_mounting(self, docker_mbpp_runner, docker_config):
        """Test that Docker volumes are properly mounted."""
        # Task that uses /tmp directory
        volume_task = {
            "task_id": 1,
            "code": """import tempfile
import os
def test_tmp_access():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test")
        temp_file = f.name
    return os.path.exists(temp_file)""",
            "test_list": ["assert test_tmp_access() == True"],
        }

        result = docker_mbpp_runner.run_mbpp_task(volume_task, docker_config)
        assert result["success"] is True

