"""
Task-agnostic interface for benchmark integration.

This module provides a unified interface for integrating different benchmarks
without depending on lm-evaluation-harness.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from .benchmark_extractors import BenchmarkExtractor


class TaskInterface(ABC):
    """Abstract interface for benchmark tasks."""
    
    @abstractmethod
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load task data."""
        pass
    
    @abstractmethod
    def get_extractor(self) -> BenchmarkExtractor:
        """Get the benchmark extractor for this task."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the task name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get the task description."""
        pass
    
    @abstractmethod
    def get_categories(self) -> List[str]:
        """Get the task categories (e.g., ['coding', 'reasoning'])."""
        pass


class TaskRegistry:
    """Registry for managing available tasks."""
    
    def __init__(self):
        self._tasks: Dict[str, Type[TaskInterface]] = {}
    
    def register_task(self, name: str, task_class: Type[TaskInterface]):
        """Register a new task."""
        self._tasks[name] = task_class
    
    def get_task(self, name: str) -> TaskInterface:
        """Get a task instance by name."""
        if name not in self._tasks:
            raise ValueError(f"Task '{name}' not found. Available tasks: {list(self._tasks.keys())}")
        return self._tasks[name]()
    
    def list_tasks(self) -> List[str]:
        """List all available task names."""
        return list(self._tasks.keys())
    
    def get_task_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific task."""
        task = self.get_task(name)
        return {
            "name": task.get_name(),
            "description": task.get_description(),
            "categories": task.get_categories()
        }
    
    def list_task_info(self) -> List[Dict[str, Any]]:
        """List information about all available tasks."""
        return [self.get_task_info(name) for name in self.list_tasks()]


# Global task registry instance
_task_registry = TaskRegistry()


def register_task(name: str, task_class: Type[TaskInterface]):
    """Register a new task globally."""
    _task_registry.register_task(name, task_class)


def get_task(name: str) -> TaskInterface:
    """Get a task instance by name."""
    return _task_registry.get_task(name)


def list_tasks() -> List[str]:
    """List all available task names."""
    return _task_registry.list_tasks()


def get_task_info(name: str) -> Dict[str, Any]:
    """Get information about a specific task."""
    return _task_registry.get_task_info(name)


def list_task_info() -> List[Dict[str, Any]]:
    """List information about all available tasks."""
    return _task_registry.list_task_info()