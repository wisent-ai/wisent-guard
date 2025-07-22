"""
LiveMathBench CNMO 2024 mathematical reasoning task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from ..task_interface import TaskInterface
from ..benchmark_extractors import GSM8KExtractor
import datasets


class LiveMathBenchTask(TaskInterface):
    """LiveMathBench CNMO 2024 mathematical reasoning task implementation."""
    
    # Dataset configurations for CNMO 2024 Chinese and English
    DATASET_CONFIGS = {
        "cnmo_en": {
            "source": "opencompass/LiveMathBench",
            "config": "v202412_CNMO_en",
            "split": "test",
            "fields": {"problem": "question", "answer": "answer"},
            "description": "18 CNMO 2024 mathematical problems in English"
        },
        "cnmo_zh": {
            "source": "opencompass/LiveMathBench", 
            "config": "v202412_CNMO_cn",
            "split": "test",
            "fields": {"problem": "question", "answer": "answer"},
            "description": "18 CNMO 2024 mathematical problems in Chinese"
        }
    }
    
    def __init__(self, language: str = "en", limit: Optional[int] = None):
        """
        Initialize LiveMathBench task for specified language.
        
        Args:
            language: Language code ("en" for English, "zh" for Chinese). Default: "en"
            limit: Maximum number of samples to load
        """
        config_key = f"cnmo_{language}" if language in ["en", "zh"] else "cnmo_en"
        if config_key not in self.DATASET_CONFIGS:
            available = list(self.DATASET_CONFIGS.keys())
            raise ValueError(f"LiveMathBench config '{config_key}' not supported. Available: {available}")
            
        self.language = language
        self.config_key = config_key
        self.config = self.DATASET_CONFIGS[config_key]
        self._limit = limit
        self._data = None  # Cache for loaded data
        self._extractor = GSM8KExtractor()  # Reuse enhanced GSM8K extractor
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load LiveMathBench CNMO 2024 data from HuggingFace for specified language."""
        # Load dataset based on language configuration
        dataset = datasets.load_dataset(
            self.config["source"],
            self.config["config"], 
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
        """Get information about the LiveMathBench task."""
        return {
            "task_name": f"livemathbench_{self.config_key}",
            "language": self.language,
            "contest": "CNMO 2024",
            "description": self.config["description"],
            "source": self.config["source"],
            "task_type": "text_generation",
            "evaluation_method": "mathematical_equivalence"
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has required LiveMathBench fields."""
        problem_field = self.config["fields"]["problem"]
        answer_field = self.config["fields"]["answer"]
        
        return all(field in sample for field in [problem_field, answer_field])
    
    def get_extractor(self) -> GSM8KExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor
    
    def get_name(self) -> str:
        """Get the task name."""
        return f"livemathbench_{self.config_key}"
    
    def get_description(self) -> str:
        """Get the task description."""
        lang_name = "Chinese" if self.language == "zh" else "English"
        return f"LiveMathBench CNMO 2024 mathematical olympiad problems in {lang_name}"
    
    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["mathematics", "reasoning", "olympiad", "multilingual", "text_generation"]
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages for CNMO 2024."""
        return ["en", "zh"]