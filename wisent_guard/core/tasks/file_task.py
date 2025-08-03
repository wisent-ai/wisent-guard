"""
File-based task implementation for loading custom datasets from JSON files.

This allows users to easily test the optimization pipeline with their own datasets
without needing to implement complex task classes or modify the core system.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..benchmark_extractors import GSM8KExtractor
from ..task_interface import TaskInterface


class FileTask(TaskInterface):
    """Task that loads data from a JSON file."""

    def __init__(self, file_path: str, task_name: Optional[str] = None, limit: Optional[int] = None):
        """
        Initialize a file-based task.

        Args:
            file_path: Path to JSON file containing the dataset
            task_name: Optional custom name for the task (defaults to filename)
            limit: Optional limit on number of samples to load
        """
        self.file_path = Path(file_path)
        self._limit = limit
        self._data = None  # Cache for loaded data
        self._extractor = GSM8KExtractor()  # Reuse GSM8K extractor for QA format

        # Set task name
        if task_name:
            self._task_name = task_name
        else:
            self._task_name = self.file_path.stem.lower()

    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data from the JSON file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        try:
            with open(self.file_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {self.file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load file {self.file_path}: {e}")

        # Ensure data is a list
        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain a list of objects, got {type(data).__name__}")

        # Validate samples
        for i, sample in enumerate(data):
            if not self.validate_sample(sample):
                raise ValueError(f"Invalid sample at index {i}: {sample}")

        # Apply limit
        effective_limit = limit or self._limit
        if effective_limit:
            data = data[: min(effective_limit, len(data))]

        return data

    def get_extractor(self) -> GSM8KExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor

    def get_name(self) -> str:
        """Get the task name."""
        return self._task_name

    def get_description(self) -> str:
        """Get the task description."""
        return f"Custom dataset loaded from {self.file_path.name}"

    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["custom", "file_based", "text_generation"]

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample has the required format.

        Expected format:
        {
            "question": "Question text",
            "answer": "Expected answer"
        }

        Optional fields:
        - "problem": Alternative to "question"
        - Any other fields will be preserved but ignored
        """
        if not isinstance(sample, dict):
            return False

        # Check for question field (or alternative names)
        question = sample.get("question") or sample.get("problem")
        if not question or not isinstance(question, str):
            return False

        # Check for answer field
        answer = sample.get("answer")
        if answer is None:
            return False

        return True

    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # File tasks don't have separate validation sets

    def has_test_docs(self) -> bool:
        """Check if task has test documents."""
        return True  # All samples are considered test docs

    def test_docs(self) -> List[Dict[str, Any]]:
        """Get test documents."""
        if self._data is None:
            self._data = self.load_data()
        return self._data

    def validation_docs(self) -> List[Dict[str, Any]]:
        """Get validation documents."""
        return []  # No separate validation set

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to text prompt."""
        question = doc.get("question") or doc.get("problem", "")
        return f"Question: {question}\nAnswer:"

    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the file task."""
        return {
            "task_name": self._task_name,
            "description": self.get_description(),
            "source": str(self.file_path),
            "task_type": "text_generation",
            "evaluation_method": "exact_match",
            "num_samples": len(self.test_docs()) if self._data else "unknown",
        }


def create_file_task(file_path: str, task_name: Optional[str] = None) -> callable:
    """
    Create a task factory function for a file-based task.

    This is the recommended way to create file tasks for registration.

    Args:
        file_path: Path to the JSON dataset file
        task_name: Optional custom name for the task

    Returns:
        A factory function that creates FileTask instances
    """

    def task_factory(limit: Optional[int] = None) -> FileTask:
        return FileTask(file_path=file_path, task_name=task_name, limit=limit)

    return task_factory


def register_file_task(task_name: str, file_path: str, registry=None):
    """
    Register a file-based task with the global task registry.

    Args:
        task_name: Name to register the task under
        file_path: Path to the JSON dataset file
        registry: Optional registry to use (defaults to global registry)
    """
    from ..task_interface import register_task

    task_factory = create_file_task(file_path, task_name)
    register_task(task_name, task_factory)


def load_tasks_from_directory(directory: str, pattern: str = "*.json", prefix: str = ""):
    """
    Load all JSON files in a directory as tasks.

    Args:
        directory: Directory to search for JSON files
        pattern: File pattern to match (default: "*.json")
        prefix: Optional prefix to add to task names
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    loaded_tasks = []

    for json_file in directory_path.glob(pattern):
        try:
            task_name = f"{prefix}{json_file.stem}".lower()
            register_file_task(task_name, str(json_file))
            loaded_tasks.append(task_name)
        except Exception as e:
            print(f"Warning: Failed to load task from {json_file}: {e}")

    return loaded_tasks
