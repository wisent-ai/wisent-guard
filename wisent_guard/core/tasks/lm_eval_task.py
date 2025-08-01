"""
LM-Evaluation-Harness task wrapper for task-agnostic architecture.
"""

from typing import Any, Dict, List, Optional

from ..benchmark_extractors import BenchmarkExtractor, get_extractor
from ..task_interface import TaskInterface


class LMEvalTask(TaskInterface):
    """Wrapper for lm-evaluation-harness tasks."""

    def __init__(self, task_name: str, description: str, categories: List[str]):
        self.task_name = task_name
        self._description = description
        self._categories = categories
        self._extractor = get_extractor(task_name)

    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data directly from lm-eval without Model dependency."""
        try:
            # Load data directly from lm-eval without creating a Model instance
            from lm_eval.tasks import get_task_dict

            # Get task directly from lm-eval
            task_dict = get_task_dict([self.task_name])
            if self.task_name not in task_dict:
                print(f"Warning: Task '{self.task_name}' not found in lm-eval")
                return []

            task = task_dict[self.task_name]

            # Get the task's test documents
            docs = []
            if hasattr(task, "test_docs"):
                # For lm-eval versions with test_docs method
                docs = list(task.test_docs())
            elif hasattr(task, "dataset"):
                # For newer lm-eval versions
                dataset = task.dataset
                if hasattr(dataset, "test"):
                    docs = list(dataset.test)
                elif hasattr(dataset, "validation"):
                    docs = list(dataset.validation)
                else:
                    # Fallback to the main dataset
                    docs = list(dataset)

            # Ensure docs are in dictionary format
            processed_docs = []
            for doc in docs:
                if isinstance(doc, dict):
                    processed_docs.append(doc)
                elif isinstance(doc, str):
                    # Handle string documents by wrapping them
                    processed_docs.append({"text": doc})
                else:
                    # Try to convert to dict if possible
                    try:
                        processed_docs.append(dict(doc))
                    except:
                        processed_docs.append({"data": str(doc)})

            docs = processed_docs

            # Apply limit if specified
            if limit and len(docs) > limit:
                docs = docs[:limit]

            return docs

        except Exception as e:
            print(f"Warning: Could not load lm-eval task '{self.task_name}': {e}")
            return []

    def get_extractor(self) -> BenchmarkExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor

    def get_name(self) -> str:
        """Get the task name."""
        return self.task_name

    def get_description(self) -> str:
        """Get the task description."""
        return self._description

    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return self._categories


class MBPPTask(LMEvalTask):
    """MBPP task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mbpp",
            description="MBPP: Mostly Basic Python Problems coding benchmark",
            categories=["coding", "reasoning", "python"],
        )


class GSM8KTask(LMEvalTask):
    """GSM8K task implementation."""

    def __init__(self):
        super().__init__(
            task_name="gsm8k",
            description="GSM8K: Grade School Math 8K problems",
            categories=["mathematics", "reasoning", "arithmetic"],
        )


class TruthfulQATask(LMEvalTask):
    """TruthfulQA task implementation."""

    def __init__(self):
        super().__init__(
            task_name="truthfulqa_mc1",
            description="TruthfulQA: Truthfulness evaluation benchmark",
            categories=["hallucination", "general-knowledge", "reasoning"],
        )


class MMLUTask(LMEvalTask):
    """MMLU task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mmlu",
            description="MMLU: Massive Multitask Language Understanding",
            categories=["general-knowledge", "science", "reasoning"],
        )
