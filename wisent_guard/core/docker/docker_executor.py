"""
Docker-based code executor for secure code evaluation.
Ensures all code execution happens in isolated Docker containers.

⚠️ DEPRECATED: Use OptimizedDockerExecutor instead for 3.1x better performance.
This module is kept for backward compatibility and will be removed in future versions.
"""

import tempfile
import subprocess
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class DockerExecutor:
    """
    Executes code safely in Docker containers.
    Based on bigcode-evaluation-harness security patterns.
    """

    def __init__(
        self,
        image_name: str = "wisent-guard-codeexec:latest",
        timeout: int = 10,
        memory_limit: str = "256m",
        cpu_limit: float = 0.5,
        build_if_missing: bool = True,
    ):
        """
        Initialize Docker executor.

        Args:
            image_name: Docker image to use
            timeout: Maximum execution time in seconds
            memory_limit: Memory limit (e.g., "256m", "1g")
            cpu_limit: CPU limit (0.5 = 50% of one CPU)
            build_if_missing: Whether to build image if it doesn't exist
        """
        self.image_name = image_name
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.build_if_missing = build_if_missing

        # Check if Docker is available
        self._check_docker_available()

        # Ensure image exists
        if build_if_missing:
            self._ensure_image_exists()

    def _check_docker_available(self):
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(
                ["docker", "version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not running. Please start Docker.")
        except FileNotFoundError:
            raise RuntimeError("Docker is not installed. Please install Docker.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker is not responding. Please check Docker daemon.")

    def _ensure_image_exists(self):
        """Ensure Docker image exists, build if necessary."""
        try:
            # Check if image exists
            result = subprocess.run(
                ["docker", "images", "-q", self.image_name],
                capture_output=True,
                text=True,
            )

            if not result.stdout.strip():
                logger.info(f"Docker image {self.image_name} not found. Building...")
                self._build_image()
        except Exception as e:
            logger.error(f"Error checking Docker image: {e}")
            raise

    def _build_image(self):
        """Build Docker image from Dockerfile."""
        dockerfile_path = Path(__file__).parent / "Dockerfile"

        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

        try:
            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    self.image_name,
                    "-f",
                    str(dockerfile_path),
                    str(dockerfile_path.parent),
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for build
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to build Docker image: {result.stderr}")

            logger.info(f"Successfully built Docker image {self.image_name}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker build timed out after 5 minutes")

    def execute_code(
        self, code: str, test_code: str = "", setup_code: str = ""
    ) -> Dict[str, Any]:
        """
        Execute code in Docker container.

        Args:
            code: The code to execute
            test_code: Optional test code to run after the main code
            setup_code: Optional setup code to run before the main code

        Returns:
            Dictionary with execution results
        """
        # Create temporary directory for code files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write code to file
            code_file = Path(temp_dir) / "solution.py"

            # Combine setup, code, and test
            full_code = []
            if setup_code:
                full_code.append(setup_code)
            full_code.append(code)
            if test_code:
                full_code.append("\n# Tests")
                full_code.append(test_code)

            code_file.write_text("\n".join(full_code))

            # Run in Docker
            try:
                result = self._run_in_docker(code_file, temp_dir)
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "output": "",
                    "exit_code": -1,
                }

    def _run_in_docker(self, code_file: Path, temp_dir: str) -> Dict[str, Any]:
        """Run code file in Docker container."""
        # Docker run command with enhanced security settings
        docker_cmd = [
            "docker",
            "run",
            "--rm",  # Remove container after execution
            "--network",
            "none",  # No network access
            "--memory",
            self.memory_limit,
            "--cpus",
            str(self.cpu_limit),
            "--read-only",  # Read-only root filesystem
            "--tmpfs",
            "/tmp:size=50M,noexec,nosuid,nodev",  # Limited temp space with security flags
            "--tmpfs",
            "/home/coderunner/workspace:size=50M,noexec,nosuid,nodev",
            "-v",
            f"{code_file}:/home/coderunner/workspace/solution.py:ro",
            "--security-opt",
            "no-new-privileges",  # Prevent privilege escalation
            "--cap-drop",
            "ALL",  # Drop all capabilities
            "--user",
            "1001:1001",  # Explicit non-root user
            "--pids-limit",
            "64",  # Limit number of processes
            "--ulimit",
            "fsize=10485760",  # Limit file size to 10MB
            "--ulimit",
            "nproc=32",  # Limit number of processes
            "--hostname",
            "sandbox",  # Set predictable hostname
            "--ipc",
            "none",  # Disable IPC
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
            self.image_name,
            "python",
            "/home/coderunner/workspace/solution.py",
        ]

        start_time = time.time()

        try:
            # Run with timeout
            result = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=self.timeout
            )

            execution_time = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.returncode,
                "execution_time": execution_time,
            }

        except subprocess.TimeoutExpired:
            # Kill the container if it's still running
            container_id = self._get_running_container()
            if container_id:
                subprocess.run(["docker", "kill", container_id], capture_output=True)

            return {
                "success": False,
                "output": "",
                "error": f"Code execution timed out after {self.timeout} seconds",
                "exit_code": -1,
                "execution_time": self.timeout,
            }

    def _get_running_container(self) -> Optional[str]:
        """Get ID of running container with our image."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"ancestor={self.image_name}"],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() if result.stdout else None
        except Exception:
            return None

    def execute_code_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code execution task (MBPP, HumanEval, etc.) with code and tests.

        Args:
            task_data: Dictionary with 'code' and 'test_list' keys

        Returns:
            Execution results
        """
        code = task_data.get("code", "")
        test_list = task_data.get("test_list", [])

        # Combine code and tests
        test_code = "\n".join(test_list)

        return self.execute_code(code, test_code)

    def batch_execute_code(
        self, tasks: List[Dict[str, Any]], max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple code execution tasks in parallel.

        Args:
            tasks: List of task dictionaries
            max_workers: Maximum parallel executions

        Returns:
            List of execution results
        """
        results = []

        # For now, execute sequentially
        # TODO: Implement parallel execution with worker pool
        for i, task in enumerate(tasks):
            logger.info(f"Executing task {i+1}/{len(tasks)}")
            result = self.execute_code_task(task)
            result["task_id"] = task.get("task_id", i)
            results.append(result)

        return results

    def cleanup(self):
        """Clean up Docker resources."""
        try:
            # Remove any stopped containers
            subprocess.run(["docker", "container", "prune", "-f"], capture_output=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup Docker containers: {e}")


