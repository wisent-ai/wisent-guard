"""
AIME 2024 task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from ..task_interface import TaskInterface
from ..benchmark_extractors import GSM8KExtractor
import datasets


class AIME2024Task(TaskInterface):
    """AIME 2024 mathematical contest task implementation."""
    
    def __init__(self, limit: Optional[int] = None):
        self._limit = limit
        self._data = None  # Cache for loaded data
        self._extractor = GSM8KExtractor()  # Reuse GSM8K extractor
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load AIME 2024 data from HuggingFace."""
        # Load AIME 2024 from HuggingFace
        dataset = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")
        
        # Apply limit
        effective_limit = limit or self._limit
        if effective_limit:
            dataset = dataset.select(range(min(effective_limit, len(dataset))))
        
        # Convert to list of dictionaries
        return [dict(item) for item in dataset]
            
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the AIME 2024 task."""
        return {
            "task_name": "aime2024",
            "description": "30 high-difficulty AIME contest problems from 2024",
            "source": "Maxwell-Jia/AIME_2024",
            "task_type": "text_generation",
            "evaluation_method": "mathematical_equivalence"
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has required AIME 2024 fields."""
        required_fields = ["Problem", "Answer"]
        return all(field in sample for field in required_fields)
    
    def get_extractor(self) -> GSM8KExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor
    
    def get_name(self) -> str:
        """Get the task name."""
        return "aime2024"
    
    def get_description(self) -> str:
        """Get the task description."""
        return "30 high-difficulty AIME contest problems from 2024 requiring advanced mathematical reasoning"
    
    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["mathematics", "reasoning", "contest", "text_generation"]