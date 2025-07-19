"""Base benchmark extractor class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BenchmarkExtractor(ABC):
    """Base class for benchmark-specific extractors."""
    
    def __init__(self):
        """Initialize the extractor."""
        pass
    
    @abstractmethod
    def can_extract(self, task_name: str) -> bool:
        """Check if this extractor can handle the given task."""
        pass
    
    @abstractmethod
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract a single QA pair from a document."""
        pass
    
    def extract_contrastive_pairs(self, docs: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract contrastive pairs from documents."""
        # Default implementation - can be overridden
        pairs = []
        for doc in docs:
            if limit and len(pairs) >= limit:
                break
            pair = self.extract_qa_pair(doc)
            if pair:
                pairs.append(pair)
        return pairs
    
    def format_question(self, doc: Dict[str, Any]) -> str:
        """Format the question for display."""
        return doc.get('question', str(doc))