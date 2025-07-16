"""
Configuration for Docker integration tests for MBPP execution.
"""
import os
import pytest
import tempfile
from typing import Dict, Any, List


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


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for testing without actual Docker."""
    
    class MockDockerClient:
        def __init__(self):
            self.containers = MockContainerManager()
            self.images = MockImageManager()
            
        def ping(self):
            return True
            
        def version(self):
            return {"Version": "20.10.0"}
    
    class MockContainerManager:
        def __init__(self):
            self.running_containers = []
            
        def run(self, image, command=None, volumes=None, environment=None, 
                remove=True, detach=False, timeout=None, **kwargs):
            """Mock container run method."""
            container = MockContainer(image, command, volumes, environment)
            if not detach:
                # Simulate immediate execution
                return container.wait_and_get_output()
            else:
                self.running_containers.append(container)
                return container
                
        def list(self, all=False):
            return self.running_containers
            
        def prune(self):
            self.running_containers.clear()
    
    class MockImageManager:
        def list(self):
            return [{"RepoTags": ["wisent-guard-mbpp:latest"]}]
            
        def build(self, path, tag, **kwargs):
            return {"Id": "sha256:12345", "stream": "Successfully built"}
    
    class MockContainer:
        def __init__(self, image, command, volumes, environment):
            self.image = image
            self.command = command
            self.volumes = volumes
            self.environment = environment
            self.id = "mock_container_id"
            self.status = "running"
            
        def wait_and_get_output(self):
            """Mock successful execution."""
            return MockContainerOutput()
            
        def kill(self):
            self.status = "killed"
            
        def remove(self):
            self.status = "removed"
    
    class MockContainerOutput:
        def __init__(self):
            self.exit_code = 0
            
        def decode(self):
            return '{"pass@1": 0.5, "results": {"0": [{"passed": true}]}}'
    
    return MockDockerClient()


@pytest.fixture
def sample_mbpp_data():
    """Sample MBPP data for testing."""
    return [
        {
            "task_id": 1,
            "text": "Write a function to remove first and last occurrence of a given character from the string.",
            "code": '''def remove_Occ(s,ch): 
    for i in range(len(s)): 
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    for i in range(len(s) - 1,-1,-1):  
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    return s''',
            "test_list": [
                'assert remove_Occ("hello","l") == "heo"',
                'assert remove_Occ("abcda","a") == "bcd"',
                'assert remove_Occ("PHP","P") == "H"'
            ]
        },
        {
            "task_id": 2,
            "text": "Write a function to sort a given matrix in ascending order according to the sum of its rows.",
            "code": '''def sort_matrix(M):
    result = sorted(M, key=sum)
    return result''',
            "test_list": [
                'assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]',
                'assert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]',
                'assert sort_matrix([[5, 8, 9], [6, 4, 3], [2, 1, 4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]'
            ]
        }
    ]


@pytest.fixture
def docker_config():
    """Docker configuration for MBPP execution."""
    return {
        "image": "wisent-guard-mbpp:latest",
        "timeout": 10,
        "memory_limit": "128m",
        "cpu_limit": "0.5",
        "network_mode": "none",
        "read_only": True,
        "remove": True,
        "volumes": {
            "/tmp": {"bind": "/tmp", "mode": "rw"}
        }
    }


class DockerMBPPRunner:
    """Helper class for running MBPP tasks in Docker."""
    
    def __init__(self, docker_client):
        self.docker_client = docker_client
        
    def run_mbpp_task(self, task_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
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
                volumes=config["volumes"]
            )
            
            return {
                "success": True,
                "exit_code": result.exit_code,
                "output": result.decode() if hasattr(result, 'decode') else str(result),
                "task_id": task_data["task_id"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "task_id": task_data["task_id"]
            }
    
    def run_mbpp_batch(self, tasks: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run multiple MBPP tasks in Docker containers."""
        results = []
        for task in tasks:
            result = self.run_mbpp_task(task, config)
            results.append(result)
        return results


@pytest.fixture
def docker_mbpp_runner(mock_docker_client):
    """Create a Docker MBPP runner instance."""
    return DockerMBPPRunner(mock_docker_client)