"""
MATH-500 task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from ..task_interface import TaskInterface
from ..benchmark_extractors import GSM8KExtractor
import datasets


class Math500Task(TaskInterface):
    """MATH-500 mathematical reasoning task implementation."""
    
    def __init__(self, limit: Optional[int] = None):
        self._limit = limit
        self._data = None  # Cache for loaded data
        self._extractor = GSM8KExtractor()  # Reuse GSM8K extractor
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load MATH-500 data from HuggingFace."""
        dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        
        # Apply limit
        effective_limit = limit or self._limit
        if effective_limit:
            dataset = dataset.select(range(min(effective_limit, len(dataset))))
        
        # Convert to list of dictionaries
        return [dict(item) for item in dataset]
            
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the MATH-500 task."""
        return {
            "task_name": "math500",
            "description": "500 mathematical reasoning problems from OpenAI's MATH dataset",
            "source": "HuggingFaceH4/MATH-500",
            "task_type": "text_generation",
            "evaluation_method": "mathematical_equivalence"
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has required MATH-500 fields."""
        required_fields = ["problem", "answer"]
        return all(field in sample for field in required_fields)
    
    def get_extractor(self) -> GSM8KExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor
    
    def get_name(self) -> str:
        """Get the task name."""
        return "math500"
    
    def get_description(self) -> str:
        """Get the task description."""
        return "500 mathematical reasoning problems from OpenAI's MATH dataset requiring multi-step solutions"
    
    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["mathematics", "reasoning", "text_generation"]
    
    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # MATH-500 doesn't have separate validation sets
    
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
        return doc.get('problem', '')