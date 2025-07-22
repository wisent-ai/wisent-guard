"""
HLE (Human-Level Evaluation) task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from datasets import load_dataset
from ..task_interface import TaskInterface
from ..benchmark_extractors import HLEExtractor


class HLETask(TaskInterface):
    """HLE (Human-Level Evaluation) task implementation."""
    
    def __init__(self, category_filter: Optional[str] = None, answer_type_filter: Optional[str] = None, 
                 limit: Optional[int] = None):
        """Initialize HLE task.
        
        Args:
            category_filter: Filter by category (Math, Physics, CS, etc.)
            answer_type_filter: Filter by answer type ('exactMatch' or 'multipleChoice')
            limit: Maximum number of examples to load
        """
        self.dataset_name = "cais/hle"
        self.category_filter = category_filter
        self.answer_type_filter = answer_type_filter
        self.limit = limit
        self._extractor = HLEExtractor()
        self._data = None  # Cache for loaded data
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load HLE data from HuggingFace datasets."""
        dataset = load_dataset(self.dataset_name, split="test")
        
        # Filter out multimodal examples for initial implementation
        text_only_data = [
            item for item in dataset 
            if not item.get('image') and not item.get('image_1') and not item.get('image_2')
        ]
        
        # Apply additional filters
        filtered_data = self._filter_and_process(text_only_data)
        
        # Apply limit
        effective_limit = limit or self.limit
        if effective_limit:
            filtered_data = filtered_data[:effective_limit]
        
        return filtered_data
    
    def _filter_and_process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data by category and answer type, and convert to internal format."""
        filtered_data = []
        
        for item in data:
            # Apply category filter
            if self.category_filter and item.get('category') != self.category_filter:
                continue
            
            # Apply answer type filter
            if self.answer_type_filter and item.get('answer_type') != self.answer_type_filter:
                continue
            
            # Convert to internal format
            processed_item = {
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'answer_type': item.get('answer_type', ''),
                'category': item.get('category', ''),
                'raw_subject': item.get('raw_subject', ''),
                'rationale': item.get('rationale', ''),
                'author_name': item.get('author_name', ''),
                'id': item.get('id', ''),
                'metadata': {
                    'canary': item.get('canary', ''),
                    'dataset': self.dataset_name
                }
            }
            
            # For multiple choice, parse choices from question text if needed
            if item.get('answer_type') == 'multipleChoice':
                processed_item['parsed_choices'] = self._parse_choices_from_question(item.get('question', ''))
            
            filtered_data.append(processed_item)
        
        return filtered_data
    
    def _parse_choices_from_question(self, question: str) -> List[str]:
        """Parse multiple choice options from question text."""
        # Look for patterns like "A. ", "B. ", etc.
        import re
        choices = []
        patterns = [
            r'([A-E])\.\s+(.+?)(?=\n[A-E]\.|$)',  # "A. option" format
            r'([A-E])\)\s+(.+?)(?=\n[A-E]\)|$)',  # "A) option" format
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.MULTILINE | re.DOTALL)
            if matches:
                choices = [f"{letter}. {text.strip()}" for letter, text in matches]
                break
        
        return choices
    
    def get_extractor(self) -> HLEExtractor:
        """Get the HLE benchmark extractor."""
        return self._extractor
    
    def get_name(self) -> str:
        """Get the task name."""
        return "hle"
    
    def get_description(self) -> str:
        """Get the task description."""
        desc = "HLE (Human-Level Evaluation): Multimodal benchmark for human-level reasoning across multiple domains"
        if self.category_filter:
            desc += f" (filtered to {self.category_filter})"
        if self.answer_type_filter:
            desc += f" ({self.answer_type_filter} questions only)"
        return desc
    
    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["reasoning", "knowledge", "multimodal", "evaluation"]
    
    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # HLE doesn't have separate validation sets
    
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
        # For HLE, the question already contains the choices for multiple choice
        return doc.get('question', '')


class HLEExactMatchTask(HLETask):
    """HLE task filtered to exact match questions only."""
    
    def __init__(self, category_filter: Optional[str] = None, limit: Optional[int] = None):
        super().__init__(category_filter=category_filter, answer_type_filter='exactMatch', limit=limit)
    
    def get_name(self) -> str:
        return "hle_exact_match"
    
    def get_description(self) -> str:
        desc = "HLE Exact Match: Text-based questions requiring exact string matching"
        if self.category_filter:
            desc += f" (filtered to {self.category_filter})"
        return desc


class HLEMultipleChoiceTask(HLETask):
    """HLE task filtered to multiple choice questions only."""
    
    def __init__(self, category_filter: Optional[str] = None, limit: Optional[int] = None):
        super().__init__(category_filter=category_filter, answer_type_filter='multipleChoice', limit=limit)
    
    def get_name(self) -> str:
        return "hle_multiple_choice"
    
    def get_description(self) -> str:
        desc = "HLE Multiple Choice: Questions with A/B/C/D/E answer options"
        if self.category_filter:
            desc += f" (filtered to {self.category_filter})"
        return desc