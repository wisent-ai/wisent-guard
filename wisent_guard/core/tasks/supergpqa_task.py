"""
SuperGPQA task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from datasets import load_dataset
from ..task_interface import TaskInterface
from ..benchmark_extractors import SuperGPQAExtractor


class SuperGPQATask(TaskInterface):
    """SuperGPQA scientific reasoning task implementation."""
    
    def __init__(self, discipline_filter: Optional[str] = None, difficulty_filter: Optional[str] = None, 
                 calculation_only: Optional[bool] = None, limit: Optional[int] = None):
        """Initialize SuperGPQA task.
        
        Args:
            discipline_filter: Filter by discipline (Science, Engineering, etc.)
            difficulty_filter: Filter by difficulty level
            calculation_only: If True, only include calculation problems; if False, exclude them
            limit: Maximum number of examples to load
        """
        self.dataset_name = "m-a-p/SuperGPQA"
        self.discipline_filter = discipline_filter
        self.difficulty_filter = difficulty_filter
        self.calculation_only = calculation_only
        self.limit = limit
        self.field_filter = None  # Can be set by subclasses
        self._extractor = SuperGPQAExtractor()
        self._data = None  # Cache for loaded data
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load SuperGPQA data from HuggingFace datasets."""
        dataset = load_dataset(self.dataset_name, split="train")
        
        # Apply filters
        filtered_data = self._filter_and_process(dataset)
        
        # Apply limit
        effective_limit = limit or self.limit
        if effective_limit:
            filtered_data = filtered_data[:effective_limit]
        
        return filtered_data
    
    def _filter_and_process(self, dataset) -> List[Dict[str, Any]]:
        """Filter data by discipline, field, difficulty, and calculation type, then convert to internal format."""
        filtered_data = []
        
        for item in dataset:
            # Apply discipline filter
            if self.discipline_filter and item.get('discipline') != self.discipline_filter:
                continue
            
            # Apply field filter (for subject-specific tasks)
            if self.field_filter and item.get('field') != self.field_filter:
                continue
            
            # Apply difficulty filter
            if self.difficulty_filter and item.get('difficulty') != self.difficulty_filter:
                continue
            
            # Apply calculation filter
            if self.calculation_only is not None:
                if self.calculation_only and not item.get('is_calculation', False):
                    continue
                elif not self.calculation_only and item.get('is_calculation', False):
                    continue
            
            # Convert to internal format
            processed_item = {
                'uuid': item.get('uuid', ''),
                'question': item.get('question', ''),
                'options': item.get('options', []),
                'answer': item.get('answer', ''),
                'answer_letter': item.get('answer_letter', ''),
                'discipline': item.get('discipline', ''),
                'field': item.get('field', ''),
                'subfield': item.get('subfield', ''),
                'difficulty': item.get('difficulty', ''),
                'is_calculation': item.get('is_calculation', False),
                'metadata': {
                    'dataset': self.dataset_name
                }
            }
            
            filtered_data.append(processed_item)
        
        return filtered_data
    
    def get_extractor(self) -> SuperGPQAExtractor:
        """Get the SuperGPQA benchmark extractor."""
        return self._extractor
    
    def get_name(self) -> str:
        """Get the task name."""
        name = "supergpqa"
        if self.discipline_filter:
            name += f"_{self.discipline_filter.lower()}"
        if self.difficulty_filter:
            name += f"_{self.difficulty_filter.lower()}"
        if self.calculation_only is not None:
            name += "_calc" if self.calculation_only else "_nocalc"
        return name
    
    def get_description(self) -> str:
        """Get the task description."""
        desc = "SuperGPQA: Large-scale dataset of scientific multiple-choice questions across disciplines"
        filters = []
        if self.discipline_filter:
            filters.append(f"discipline: {self.discipline_filter}")
        if self.difficulty_filter:
            filters.append(f"difficulty: {self.difficulty_filter}")
        if self.calculation_only is not None:
            filters.append("calculation problems only" if self.calculation_only else "non-calculation problems only")
        
        if filters:
            desc += f" (filtered: {', '.join(filters)})"
        return desc
    
    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["science", "reasoning", "multiple_choice", "knowledge"]
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the SuperGPQA task."""
        return {
            "task_name": self.get_name(),
            "description": self.get_description(),
            "source": self.dataset_name,
            "task_type": "multiple_choice",
            "evaluation_method": "exact_match",
            "filters": {
                "discipline": self.discipline_filter,
                "difficulty": self.difficulty_filter,
                "calculation_only": self.calculation_only
            }
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has required SuperGPQA fields."""
        required_fields = ["question", "options", "answer", "answer_letter"]
        return all(field in sample for field in required_fields)
    
    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # SuperGPQA doesn't have separate validation sets
    
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
        question = doc.get('question', '')
        options = doc.get('options', [])
        
        # Format as multiple choice question
        if options:
            choices = []
            for i, option in enumerate(options):
                letter = chr(ord('A') + i)
                choices.append(f"{letter}. {option}")
            return f"{question}\n\n" + "\n".join(choices)
        else:
            return question


class SuperGPQAPhysicsTask(SuperGPQATask):
    """SuperGPQA task filtered to Physics questions only."""
    
    def __init__(self, difficulty_filter: Optional[str] = None, calculation_only: Optional[bool] = None, 
                 limit: Optional[int] = None):
        # Filter by discipline=Science and field=Physics
        super().__init__(discipline_filter="Science", difficulty_filter=difficulty_filter, 
                        calculation_only=calculation_only, limit=limit)
        self.field_filter = "Physics"  # Add field filtering
    
    def get_name(self) -> str:
        return "supergpqa_physics"


class SuperGPQAChemistryTask(SuperGPQATask):
    """SuperGPQA task filtered to Chemistry questions only."""
    
    def __init__(self, difficulty_filter: Optional[str] = None, calculation_only: Optional[bool] = None, 
                 limit: Optional[int] = None):
        # Filter by discipline=Science and field=Chemistry  
        super().__init__(discipline_filter="Science", difficulty_filter=difficulty_filter, 
                        calculation_only=calculation_only, limit=limit)
        self.field_filter = "Chemistry"  # Add field filtering
    
    def get_name(self) -> str:
        return "supergpqa_chemistry"


class SuperGPQABiologyTask(SuperGPQATask):
    """SuperGPQA task filtered to Biology questions only."""
    
    def __init__(self, difficulty_filter: Optional[str] = None, calculation_only: Optional[bool] = None, 
                 limit: Optional[int] = None):
        # Filter by discipline=Science and field=Biology
        super().__init__(discipline_filter="Science", difficulty_filter=difficulty_filter, 
                        calculation_only=calculation_only, limit=limit)
        self.field_filter = "Biology"  # Add field filtering
    
    def get_name(self) -> str:
        return "supergpqa_biology"