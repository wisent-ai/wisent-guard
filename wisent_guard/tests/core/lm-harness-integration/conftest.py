"""
Pytest configuration for lm-harness integration tests.
"""

import pytest
import os
import tempfile
from typing import Dict, Any, List


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for all tests."""
    # Set environment variable for code evaluation
    os.environ['HF_ALLOW_CODE_EVAL'] = '1'
    
    # Set other test-specific environment variables if needed
    original_env = dict(os.environ)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def docker_environment():
    """Setup Docker environment variables for safe code execution."""
    # Set up the environment variable required for code execution
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Store original environment
    original_env = dict(os.environ)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Removed mock_docker_client fixture - Docker is required for code execution tests
# Tests should use real Docker or be marked with @pytest.mark.skip if Docker is not available


@pytest.fixture
def sample_mbpp_data():
    """Sample MBPP data for testing."""
    return [
        {
            "task_id": 1,
            "text": "Write a function to remove first and last occurrence of a given character from the string.",
            "code": """def remove_Occ(s,ch): 
    for i in range(len(s)): 
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    for i in range(len(s) - 1,-1,-1):  
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    return s""",
            "test_list": [
                'assert remove_Occ("hello","l") == "heo"',
                'assert remove_Occ("abcda","a") == "bcd"',
                'assert remove_Occ("PHP","P") == "H"',
            ],
        },
        {
            "task_id": 2,
            "text": "Write a function to sort a given matrix in ascending order according to the sum of its rows.",
            "code": """def sort_matrix(M):
    result = sorted(M, key=sum)
    return result""",
            "test_list": [
                "assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]",
                "assert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]",
                "assert sort_matrix([[5, 8, 9], [6, 4, 3], [2, 1, 4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]",
            ],
        },
    ]


@pytest.fixture
def docker_config():
    """Docker configuration for MBPP execution."""
    return {
        "image": "wisent-guard-codeexec:latest",
        "timeout": 10,
        "memory_limit": "128m",
        "cpu_limit": "0.5",
        "network_mode": "none",
        "read_only": True,
        "remove": True,
        "volumes": {"/tmp": {"bind": "/tmp", "mode": "rw"}},
    }


class DockerMBPPRunner:
    """Helper class for running MBPP tasks in Docker."""

    def __init__(self, docker_client):
        self.docker_client = docker_client

    def run_mbpp_task(
        self, task_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single MBPP task in Docker container."""
        try:
            # Create test program
            test_program = task_data["code"] + "\n" + "\n".join(task_data["test_list"])

            result = self.docker_client.containers.run(
                image=config["image"],
                command=["python", "-c", test_program],
                timeout=config["timeout"],
                mem_limit=config["memory_limit"],
                network_mode=config["network_mode"],
                read_only=config["read_only"],
                remove=config["remove"],
                volumes=config["volumes"],
            )

            return {
                "success": True,
                "exit_code": result.exit_code,
                "output": result.decode() if hasattr(result, "decode") else str(result),
                "task_id": task_data["task_id"],
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "task_id": task_data["task_id"],
            }

    def run_mbpp_batch(
        self, tasks: List[Dict[str, Any]], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run multiple MBPP tasks in Docker containers."""
        results = []
        for task in tasks:
            result = self.run_mbpp_task(task, config)
            results.append(result)
        return results


@pytest.fixture
def docker_mbpp_runner():
    """Create a Docker MBPP runner instance with real Docker."""
    try:
        import docker
        client = docker.from_env()
        # Test if Docker is actually running
        client.ping()
        return DockerMBPPRunner(client)
    except Exception as e:
        pytest.skip(f"Docker is not available: {e}")