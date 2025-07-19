"""
LM-Evaluation-Harness task wrapper for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from ..task_interface import TaskInterface
from ..benchmark_extractors import BenchmarkExtractor, get_extractor


class LMEvalTask(TaskInterface):
    """Wrapper for lm-evaluation-harness tasks."""
    
    def __init__(self, task_name: str, description: str, categories: List[str]):
        self.task_name = task_name
        self._description = description
        self._categories = categories
        self._extractor = get_extractor(task_name)
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data using lm-eval."""
        try:
            # Import here to avoid circular dependencies
            from ..model import Model
            
            # Create a temporary model instance to load task data
            # In a real implementation, this would be optimized
            model = Model("dummy")  # We don't actually need the model for data loading
            task_data = model.load_lm_eval_task(self.task_name, limit=limit)
            
            # Convert lm-eval task data to our format
            if hasattr(task_data, 'docs'):
                docs = task_data.docs
                if limit:
                    docs = docs[:limit]
                return docs
            else:
                return []
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
            categories=["coding", "reasoning", "python"]
        )


class GSM8KTask(LMEvalTask):
    """GSM8K task implementation."""
    
    def __init__(self):
        super().__init__(
            task_name="gsm8k",
            description="GSM8K: Grade School Math 8K problems",
            categories=["mathematics", "reasoning", "arithmetic"]
        )


class TruthfulQATask(LMEvalTask):
    """TruthfulQA task implementation."""
    
    def __init__(self):
        super().__init__(
            task_name="truthfulqa_mc1",
            description="TruthfulQA: Truthfulness evaluation benchmark",
            categories=["hallucination", "general-knowledge", "reasoning"]
        )


class MMLUTask(LMEvalTask):
    """MMLU task implementation."""
    
    def __init__(self):
        super().__init__(
            task_name="mmlu",
            description="MMLU: Massive Multitask Language Understanding",
            categories=["general-knowledge", "science", "reasoning"]
        )