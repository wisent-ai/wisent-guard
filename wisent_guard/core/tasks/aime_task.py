"""
AIME task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from ..task_interface import TaskInterface
from ..benchmark_extractors import GSM8KExtractor
import datasets


class AIMETask(TaskInterface):
    """AIME mathematical contest task implementation."""
    
    def __init__(self, year: str = "2025", config_name: Optional[str] = None, limit: Optional[int] = None):
        self._year = year
        self._config_name = config_name
        self._limit = limit
        self._data = None  # Cache for loaded data
        self._extractor = GSM8KExtractor()  # Reuse GSM8K extractor
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load AIME data from HuggingFace."""
        # Load AIME dataset based on year
        if self._year == "2024":
            dataset = datasets.load_dataset("HuggingFaceH4/aime_2024", split="train")
        elif self._year == "2025":
            if not self._config_name:
                raise ValueError("Config name is missing for AIME2025. Please pick one among the available configs: ['AIME2025-I', 'AIME2025-II']")
            dataset = datasets.load_dataset("opencompass/AIME2025", self._config_name, split="test")
        else:
            # Fallback for other years
            dataset = datasets.load_dataset(f"opencompass/AIME{self._year}", split="train")
        
        # Apply limit
        effective_limit = limit or self._limit
        if effective_limit:
            dataset = dataset.select(range(min(effective_limit, len(dataset))))
        
        # Convert to list of dictionaries
        return [dict(item) for item in dataset]
            
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the AIME task."""
        return {
            "task_name": f"aime{self._year}",
            "description": f"30 high-difficulty AIME contest problems from {self._year}",
            "source": f"Maxwell-Jia/AIME_{self._year}",
            "task_type": "text_generation",
            "evaluation_method": "mathematical_equivalence"
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has required AIME fields."""
        required_fields = ["Problem", "Answer"]
        return all(field in sample for field in required_fields)
    
    def get_extractor(self) -> GSM8KExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor
    
    def get_name(self) -> str:
        """Get the task name."""
        return f"aime{self._year}"
    
    def get_description(self) -> str:
        """Get the task description."""
        return f"30 high-difficulty AIME contest problems from {self._year} requiring advanced mathematical reasoning"
    
    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["mathematics", "reasoning", "contest", "text_generation"]
