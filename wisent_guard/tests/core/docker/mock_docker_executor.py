"""
Mock Docker executor for testing purposes.
"""

from typing import Dict, Any


class MockDockerExecutor:
    """
    Mock Docker executor for testing.
    Simulates Docker execution without actually running containers.
    """

    def __init__(
        self,
        image_name: str = "wisent-guard-codeexec:latest",
        timeout: int = 10,
        memory_limit: str = "256m",
        cpu_limit: float = 0.5,
        **kwargs
    ):
        """
        Initialize mock Docker executor.

        Args:
            image_name: Docker image name (ignored)
            timeout: Timeout in seconds
            memory_limit: Memory limit string
            cpu_limit: CPU limit as float
            **kwargs: Additional ignored arguments
        """
        self.image_name = image_name
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.execution_count = 0

    def execute_code(self, code: str, input_data: str = "") -> Dict[str, Any]:
        """
        Mock code execution.
        
        Args:
            code: Python code to execute
            input_data: Input data for the code
            
        Returns:
            Mock execution result
        """
        self.execution_count += 1
        
        # Simple mock: return success with fake output
        return {
            "success": True,
            "stdout": f"Mock execution result for code: {code[:50]}...",
            "stderr": "",
            "return_code": 0,
            "execution_time": 0.1,
            "timeout": self.timeout,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "mock": True
        }

    def execute_code_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock code task execution.
        
        Args:
            task_data: Task data containing code and test information
            
        Returns:
            Mock execution result
        """
        self.execution_count += 1
        
        code = task_data.get("code", "")
        
        # Simple heuristic: code with obvious syntax errors should fail
        has_syntax_error = (
            "def add(a, b):" in code and "return a +" in code and code.count("return a +") > 0 and not code.strip().endswith("return a + b")
        ) or (
            "syntax error" in code.lower() or 
            "indentation" in code.lower() or
            code.count("(") != code.count(")") or
            code.strip().endswith("return a +") or  # Incomplete expression
            code.strip().endswith("return a *") or  # Incomplete expression
            code.strip().endswith("return a -") or  # Incomplete expression
            code.strip().endswith("return a /")     # Incomplete expression
        )
        
        success = not has_syntax_error
        
        return {
            "success": success,
            "stdout": f"Mock task execution result for: {code[:50]}..." if success else "",
            "stderr": "Mock syntax error" if not success else "",
            "return_code": 0 if success else 1,
            "execution_time": 0.1,
            "timeout": self.timeout,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "tests_passed": success,
            "total_tests": len(task_data.get("test_list", [])),
            "error": "Mock SyntaxError: invalid syntax" if not success else "",
            "mock": True
        }

    def get_info(self) -> Dict[str, Any]:
        """
        Get executor information.
        
        Returns:
            Dictionary with executor details
        """
        return {
            "executor_type": "MockDockerExecutor",
            "image_name": self.image_name,
            "timeout": self.timeout,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "execution_count": self.execution_count,
            "mock": True
        }

    def cleanup(self):
        """Mock cleanup method."""
        pass