"""
LiveCodeBench extractor that uses pre-annotated model outputs for contrastive pairs.
"""

from typing import Dict, Any, List, Optional
from .base import BenchmarkExtractor
from ..data_loaders.livecodebench_model_outputs import LiveCodeBenchModelOutputsLoader
import logging

logger = logging.getLogger(__name__)


class LiveCodeBenchModelOutputsExtractor(BenchmarkExtractor):
    """
    Extractor for LiveCodeBench that uses pre-annotated model outputs
    to create contrastive pairs from actual passing/failing solutions.
    """
    
    def __init__(self):
        """Initialize the extractor with model outputs loader."""
        super().__init__()
        self.outputs_loader = LiveCodeBenchModelOutputsLoader()
        self._problem_mapping = {}  # Map task_id to problem_idx
        
    def can_extract(self, task_name: str) -> bool:
        """Check if this extractor can handle the given task."""
        return task_name == "livecodebench"
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        Extract QA pair from LiveCodeBench document using model outputs.
        
        This doesn't extract a single QA pair but rather prepares for
        contrastive pair extraction from model outputs.
        """
        # Store mapping for later use
        task_id = doc.get("task_id", "")
        if task_id and task_id not in self._problem_mapping:
            # Extract problem index from task_id if possible
            # LiveCodeBench task IDs often have format like "lcb_release_v1_1873_A"
            # We need to map this to the problem index in model outputs
            self._problem_mapping[task_id] = len(self._problem_mapping)
            
        # Return None as we'll use extract_contrastive_pairs instead
        return None
    
    def extract_contrastive_pairs(self, docs: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract contrastive pairs from LiveCodeBench using model outputs.
        
        Args:
            docs: List of LiveCodeBench documents
            limit: Optional limit on number of pairs
            
        Returns:
            List of contrastive pairs with actual passing/failing code
        """
        contrastive_pairs = []
        
        # Get pairs from model outputs
        # The problem indices in model outputs should correspond to the document order
        for i, doc in enumerate(docs):
            if limit and len(contrastive_pairs) >= limit:
                break
                
            # Get contrastive pairs for this problem
            pairs = self.outputs_loader.get_contrastive_pairs_for_problem(i)
            
            # Add document context to each pair
            for pair in pairs:
                # Create properly formatted contrastive pair
                contrastive_pair = {
                    "question": doc.get("question_content", ""),
                    "task_id": doc.get("task_id", f"problem_{i}"),
                    "correct_answer": pair["good_example"]["code"],
                    "incorrect_answer": pair["bad_example"]["code"],
                    "metadata": {
                        "good_model": pair["good_example"]["model"],
                        "bad_model": pair["bad_example"]["model"],
                        "problem_idx": i,
                        "difficulty": doc.get("difficulty", "unknown"),
                        "platform": doc.get("platform", "unknown")
                    }
                }
                contrastive_pairs.append(contrastive_pair)
                
                if limit and len(contrastive_pairs) >= limit:
                    break
        
        logger.info(f"Extracted {len(contrastive_pairs)} contrastive pairs from {len(docs)} LiveCodeBench documents")
        return contrastive_pairs
    
    def format_question(self, doc: Dict[str, Any]) -> str:
        """Format the question for display."""
        question = doc.get("question_content", "")
        title = doc.get("question_title", "")
        
        if title:
            return f"{title}\n\n{question}"
        return question