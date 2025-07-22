"""
PolyMath multilingual mathematical reasoning task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from ..task_interface import TaskInterface
from ..benchmark_extractors import GSM8KExtractor
import datasets


class PolyMathTask(TaskInterface):
    """PolyMath multilingual mathematical reasoning task implementation."""
    
    # Dataset configurations for different language-difficulty combinations
    DATASET_CONFIGS = {
        "zh_medium": {
            "source": "Qwen/PolyMath",
            "language": "zh",
            "split": "medium",
            "fields": {"problem": "question", "answer": "answer"},
            "description": "125 medium-difficulty mathematical problems in Chinese"
        },
        "en_medium": {
            "source": "Qwen/PolyMath",
            "language": "en", 
            "split": "medium",
            "fields": {"problem": "question", "answer": "answer"},
            "description": "125 medium-difficulty mathematical problems in English"
        },
        "zh_high": {
            "source": "Qwen/PolyMath",
            "language": "zh",
            "split": "high",
            "fields": {"problem": "question", "answer": "answer"},
            "description": "125 high-difficulty mathematical problems in Chinese"
        },
        "en_high": {
            "source": "Qwen/PolyMath",
            "language": "en",
            "split": "high", 
            "fields": {"problem": "question", "answer": "answer"},
            "description": "125 high-difficulty mathematical problems in English"
        }
    }
    
    def __init__(self, language: str = "en", difficulty: str = "medium", limit: Optional[int] = None):
        """
        Initialize PolyMath task for specified language and difficulty.
        
        Args:
            language: Language code ("en" for English, "zh" for Chinese). Default: "en"
            difficulty: Difficulty level ("medium", "high"). Default: "medium" 
            limit: Maximum number of samples to load
        """
        config_key = f"{language}_{difficulty}"
        if config_key not in self.DATASET_CONFIGS:
            available = list(self.DATASET_CONFIGS.keys())
            raise ValueError(f"PolyMath config '{config_key}' not supported. Available: {available}")
            
        self.language = language
        self.difficulty = difficulty
        self.config_key = config_key
        self.config = self.DATASET_CONFIGS[config_key]
        self._limit = limit
        self._data = None  # Cache for loaded data
        self._extractor = GSM8KExtractor()  # Reuse enhanced GSM8K extractor
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load PolyMath data from HuggingFace for specified language and difficulty."""
        # Load dataset based on language and difficulty configuration
        dataset = datasets.load_dataset(
            self.config["source"],
            self.config["language"], 
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
        """Get information about the PolyMath task."""
        return {
            "task_name": f"polymath_{self.config_key}",
            "language": self.language,
            "difficulty": self.difficulty,
            "description": self.config["description"],
            "source": self.config["source"],
            "task_type": "text_generation",
            "evaluation_method": "mathematical_equivalence"
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has required PolyMath fields."""
        problem_field = self.config["fields"]["problem"]
        answer_field = self.config["fields"]["answer"]
        
        return all(field in sample for field in [problem_field, answer_field])
    
    def get_extractor(self) -> GSM8KExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor
    
    def get_name(self) -> str:
        """Get the task name."""
        return f"polymath_{self.config_key}"
    
    def get_description(self) -> str:
        """Get the task description."""
        lang_name = "Chinese" if self.language == "zh" else "English"
        return f"PolyMath {self.difficulty}-difficulty mathematical problems in {lang_name}"
    
    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["mathematics", "reasoning", "multilingual", "text_generation"]
    
    @classmethod
    def get_supported_configs(cls) -> List[str]:
        """Get list of supported PolyMath language-difficulty configurations."""
        return list(cls.DATASET_CONFIGS.keys())