"""
Task Selector for intelligent task selection based on issue types.

This module provides functionality to select the most relevant lm-eval tasks
for training classifiers for specific issue types using model-driven decisions.
"""

from typing import List, Dict, Any, Set, Tuple
from .task_manager import get_available_tasks


class TaskSelector:
    """Model-driven task selector for issue-type-specific training."""
    
    def __init__(self, model):
        self.model = model
    
    def find_relevant_tasks_for_issue_type(self, issue_type: str, max_tasks: int = 10) -> List[str]:
        """
        Find the most relevant tasks for a specific issue type using model decisions.
        
        Args:
            issue_type: Type of issue to find tasks for
            max_tasks: Maximum number of tasks to return
            
        Returns:
            List of task names ranked by relevance
        """
        available_tasks = get_available_tasks()
        
        # Use model to score task relevance for the issue type
        task_scores = []
        for task_name in available_tasks[:50]:  # Limit for efficiency
            score = self._get_model_task_relevance(issue_type, task_name)
            if score > 0.0:
                task_scores.append((task_name, score))
        
        # Sort by relevance score (descending) and return top tasks
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return [task_name for task_name, _ in task_scores[:max_tasks]]
    
    def select_best_tasks_for_training(
        self, 
        issue_type: str, 
        min_tasks: int = 1,
        max_tasks: int = 10,
        quality_threshold: float = 1.5
    ) -> List[str]:
        """
        Select the best tasks for training a classifier for the given issue type.
        
        Args:
            issue_type: Type of issue to select tasks for
            min_tasks: Minimum number of tasks to select
            max_tasks: Maximum number of tasks to select
            quality_threshold: Minimum quality score for task inclusion
            
        Returns:
            List of selected task names
        """
        # Get relevant tasks using model decisions
        relevant_tasks = self.find_relevant_tasks_for_issue_type(issue_type, max_tasks * 2)
        
        # Use model to evaluate task quality
        selected_tasks = []
        for task_name in relevant_tasks:
            quality_score = self._get_model_task_quality(task_name)
            if quality_score >= quality_threshold or len(selected_tasks) < min_tasks:
                selected_tasks.append(task_name)
                if len(selected_tasks) >= max_tasks:
                    break
        
        return selected_tasks[:max_tasks]
    
    def _get_model_task_relevance(self, issue_type: str, task_name: str) -> float:
        """Get task relevance score from the model."""
        prompt = f"""Rate how relevant this task is for detecting/training on this issue type.

Issue Type: {issue_type}
Task: {task_name}

Rate relevance from 0.0 to 1.0 (1.0 = highly relevant, 0.0 = not relevant).
Respond with only the number:"""
        
        try:
            response = self.model.generate(prompt, layer_index=15, max_new_tokens=10, temperature=0.1)
            score_str = response.strip()
            
            import re
            match = re.search(r'(\d+\.?\d*)', score_str)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            return 0.0
        except:
            return 0.0
    
    def _get_model_task_quality(self, task_name: str) -> float:
        """Get task quality assessment from the model."""
        prompt = f"""Rate the quality and reliability of this evaluation task for training AI safety classifiers.

Task: {task_name}

Consider factors like:
- Data quality and reliability
- Task design and clarity
- Usefulness for training safety classifiers

Rate quality from 0.0 to 5.0 (5.0 = excellent quality, 0.0 = poor quality).
Respond with only the number:"""
        
        try:
            response = self.model.generate(prompt, layer_index=15, max_new_tokens=10, temperature=0.1)
            score_str = response.strip()
            
            import re
            match = re.search(r'(\d+\.?\d*)', score_str)
            if match:
                score = float(match.group(1))
                return min(5.0, max(0.0, score))
            return 1.0
        except:
            return 1.0


def find_relevant_tasks_for_issue_type(issue_type: str, max_tasks: int = 10, model=None) -> List[str]:
    """Standalone function for finding relevant tasks."""
    if model is None:
        from ....model import Model
        model = Model("meta-llama/Llama-3.1-8B-Instruct")
    
    selector = TaskSelector(model)
    return selector.find_relevant_tasks_for_issue_type(issue_type, max_tasks)


def select_best_tasks_for_training(
    issue_type: str, 
    min_tasks: int = 1,
    max_tasks: int = 10,
    quality_threshold: float = 1.5,
    model=None
) -> List[str]:
    """Standalone function for selecting best training tasks."""
    if model is None:
        from ....model import Model
        model = Model("meta-llama/Llama-3.1-8B-Instruct")
    
    selector = TaskSelector(model)
    return selector.select_best_tasks_for_training(
        issue_type, min_tasks, max_tasks, quality_threshold
    ) 