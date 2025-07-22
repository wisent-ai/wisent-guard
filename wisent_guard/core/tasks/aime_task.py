"""
AIME task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from ..task_interface import TaskInterface
from ..benchmark_extractors import GSM8KExtractor
import datasets


class AIMETask(TaskInterface):
    """General AIME mathematical contest task implementation."""
    
    # Dataset configurations for different years
    DATASET_CONFIGS = {
        "2024": {
            "source": "Maxwell-Jia/AIME_2024",
            "split": "train",
            "fields": {"problem": "Problem", "answer": "Answer"},
            "description": "30 high-difficulty AIME contest problems from 2024"
        },
        "2025": {
            "source": "MathArena/aime_2025", 
            "split": "train",
            "fields": {"problem": "problem", "answer": "answer"},
            "description": "30 high-difficulty AIME contest problems from 2025 (MathArena)"
        }
    }
    
    def __init__(self, year: str = "2025", limit: Optional[int] = None):
        """
        Initialize AIME task for specified year.
        
        Args:
            year: AIME year to load ("2024", "2025"). Default: "2025" (latest)
            limit: Maximum number of samples to load
        """
        if year not in self.DATASET_CONFIGS:
            available = list(self.DATASET_CONFIGS.keys())
            raise ValueError(f"AIME year '{year}' not supported. Available: {available}")
            
        self.year = year
        self.config = self.DATASET_CONFIGS[year]
        self._limit = limit
        self._data = None  # Cache for loaded data
        self._extractor = GSM8KExtractor()  # Reuse enhanced GSM8K extractor
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load AIME data from HuggingFace for specified year."""
        # Load dataset based on year configuration
        dataset = datasets.load_dataset(
            self.config["source"], 
            split=self.config["split"]
        )
        
        # Apply limit
        effective_limit = limit or self._limit
        if effective_limit:
            dataset = dataset.select(range(min(effective_limit, len(dataset))))
        
        # Convert to list and normalize field names
        data = [dict(item) for item in dataset]
        
        # Normalize field names for consistent processing
        normalized_data = []
        problem_field = self.config["fields"]["problem"]
        answer_field = self.config["fields"]["answer"]
        
        for item in data:
            normalized_item = dict(item)  # Keep all original fields
            
            # Ensure consistent field names for extractor
            if problem_field in item:
                normalized_item["Problem"] = item[problem_field]
                normalized_item["question"] = item[problem_field]  # For question/answer format
            
            if answer_field in item:
                normalized_item["Answer"] = item[answer_field]
                normalized_item["answer"] = item[answer_field]  # For question/answer format
            
            normalized_data.append(normalized_item)
        
        return normalized_data
            
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the AIME task."""
        return {
            "task_name": f"aime{self.year}" if self.year != "2025" else "aime",
            "year": self.year,
            "description": self.config["description"],
            "source": self.config["source"],
            "task_type": "text_generation",
            "evaluation_method": "mathematical_equivalence"
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has required AIME fields."""
        problem_field = self.config["fields"]["problem"]
        answer_field = self.config["fields"]["answer"]
        
        return all(field in sample for field in [problem_field, answer_field])
    
    def get_extractor(self) -> GSM8KExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor
    
    def get_name(self) -> str:
        """Get the task name."""
        return f"aime{self.year}" if self.year != "2025" else "aime"
    
    def get_description(self) -> str:
        """Get the task description."""
        return f"AIME {self.year} contest problems requiring advanced mathematical reasoning"
    
    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["mathematics", "reasoning", "contest", "text_generation"]
    
    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # AIME doesn't have separate validation sets
    
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
        return doc.get('Problem', '')
