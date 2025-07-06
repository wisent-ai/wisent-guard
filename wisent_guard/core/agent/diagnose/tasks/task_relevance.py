"""
Task Relevance Selection for Wisent Guard.

This module provides functionality to select the most relevant tasks from the
lm-evaluation-harness library based on a user query or issue type.

Uses model-driven decisions instead of hardcoded patterns.
"""

from typing import List, Dict, Set, Tuple
from .task_manager import get_available_tasks


class TaskRelevanceSelector:
    """Selects tasks based on model-driven relevance analysis."""
    
    def __init__(self, model):
        self.model = model
        
    def find_relevant_tasks(
        self, 
        query: str, 
        max_results: int = 20,
        min_relevance_score: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Find tasks most relevant to the given query using model decisions.
        
        Args:
            query: The search query (e.g., "hallucination detection", "bias", "truthfulness")
            max_results: Maximum number of tasks to return
            min_relevance_score: Minimum relevance score threshold (0.0 to 1.0)
            
        Returns:
            List of (task_name, relevance_score) tuples, sorted by relevance
        """
        available_tasks = get_available_tasks()
        
        # Use model to score task relevance
        task_scores = []
        for task_name in available_tasks[:100]:  # Limit for efficiency
            score = self._get_model_relevance_score(query, task_name)
            if score >= min_relevance_score:
                task_scores.append((task_name, score))
        
        # Sort by relevance score (descending)
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        return task_scores[:max_results]
    
    def _get_model_relevance_score(self, query: str, task_name: str) -> float:
        """Get relevance score from the model."""
        prompt = f"""Rate the relevance of this task for the given query.
        
Query: {query}
Task: {task_name}

Rate relevance from 0.0 to 1.0 (1.0 = highly relevant, 0.0 = not relevant).
Respond with only the number:"""
        
        try:
            response = self.model.generate(prompt, layer_index=15, max_new_tokens=10, temperature=0.1)
            score_str = response.strip()
            
            # Extract number from response
            import re
            match = re.search(r'(\d+\.?\d*)', score_str)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))  # Clamp to [0,1]
            return 0.0
        except:
            return 0.0


def find_relevant_tasks(
    query: str, 
    max_results: int = 20,
    min_relevance_score: float = 0.1,
    model=None
) -> List[Tuple[str, float]]:
    """Standalone function for task relevance selection."""
    if model is None:
        from ....model import Model
        model = Model("meta-llama/Llama-3.1-8B-Instruct")
    
    selector = TaskRelevanceSelector(model)
    return selector.find_relevant_tasks(query, max_results, min_relevance_score)


def get_top_relevant_tasks(query: str, count: int, model=None) -> List[str]:
    """Get top N relevant tasks for a query."""
    results = find_relevant_tasks(query, max_results=count, model=model)
    return [task_name for task_name, _ in results]
