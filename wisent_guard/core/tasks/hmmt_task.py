"""
HMMT (Harvard-MIT Math Tournament) task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from ..task_interface import TaskInterface
from ..benchmark_extractors import GSM8KExtractor
import datasets


class HMMTTask(TaskInterface):
    """HMMT (Harvard-MIT Math Tournament) mathematical contest task implementation."""
    
    # Dataset configurations for different HMMT competitions
    DATASET_CONFIGS = {
        "feb_2025": {
            "source": "MathArena/hmmt_feb_2025",
            "split": "train",
            "fields": {"problem": "problem", "answer": "answer"},
            "description": "30 high-difficulty HMMT February 2025 contest problems"
        }
    }
    
    def __init__(self, competition: str = "feb_2025", limit: Optional[int] = None):
        """
        Initialize HMMT task for specified competition.
        
        Args:
            competition: HMMT competition to load ("feb_2025"). Default: "feb_2025" (latest)
            limit: Maximum number of samples to load
        """
        if competition not in self.DATASET_CONFIGS:
            available = list(self.DATASET_CONFIGS.keys())
            raise ValueError(f"HMMT competition '{competition}' not supported. Available: {available}")
            
        self.competition = competition
        self.config = self.DATASET_CONFIGS[competition]
        self._limit = limit
        self._data = None  # Cache for loaded data
        self._extractor = GSM8KExtractor()  # Reuse enhanced GSM8K extractor
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load HMMT data from HuggingFace for specified competition."""
        # Load dataset based on competition configuration
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
        """Get information about the HMMT task."""
        return {
            "task_name": f"hmmt_{self.competition}" if self.competition != "feb_2025" else "hmmt",
            "competition": self.competition,
            "description": self.config["description"],
            "source": self.config["source"],
            "task_type": "text_generation",
            "evaluation_method": "mathematical_equivalence"
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has required HMMT fields."""
        problem_field = self.config["fields"]["problem"]
        answer_field = self.config["fields"]["answer"]
        
        return all(field in sample for field in [problem_field, answer_field])
    
    def get_extractor(self) -> GSM8KExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor
    
    def get_name(self) -> str:
        """Get the task name."""
        return f"hmmt_{self.competition}" if self.competition != "feb_2025" else "hmmt"
    
    def get_description(self) -> str:
        """Get the task description."""
        return f"HMMT {self.competition.replace('_', ' ').title()} contest problems requiring advanced mathematical reasoning"
    
    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["mathematics", "reasoning", "contest", "text_generation"]
    
    @classmethod
    def get_supported_competitions(cls) -> List[str]:
        """Get list of supported HMMT competitions."""
        return list(cls.DATASET_CONFIGS.keys())


