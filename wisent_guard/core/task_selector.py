"""
Task selector for choosing tasks based on skills and risks tags.
"""

import json
import os
import random
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskSelector:
    """Select tasks based on skills and risks criteria."""
    
    def __init__(self):
        """Initialize the task selector by loading metadata."""
        self.base_path = Path(__file__).parent.parent / "parameters" / "tasks"
        self.skills = self._load_json("skills.json")
        self.risks = self._load_json("risks.json")
        self.tasks_data = self._load_json("tasks.json")
        self.tasks = self.tasks_data.get("tasks", {})
        
    def _load_json(self, filename: str) -> Any:
        """Load JSON file from parameters/tasks directory."""
        filepath = self.base_path / filename
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return {} if filename == "tasks.json" else []
    
    def get_available_skills(self) -> List[str]:
        """Get list of available skills."""
        return self.skills
    
    def get_available_risks(self) -> List[str]:
        """Get list of available risks."""
        return self.risks
    
    def find_tasks_by_tags(
        self, 
        skills: Optional[List[str]] = None, 
        risks: Optional[List[str]] = None,
        min_quality_score: int = 2
    ) -> List[str]:
        """
        Find tasks that match the given skills and/or risks.
        
        Args:
            skills: List of skill tags to match
            risks: List of risk tags to match
            min_quality_score: Minimum quality score for tasks (default: 2)
            
        Returns:
            List of task names that match the criteria
        """
        if not skills and not risks:
            # Return all tasks if no criteria specified
            return [
                task_name for task_name, task_data in self.tasks.items()
                if task_data.get("quality_score", 0) >= min_quality_score
            ]
        
        # Convert to sets for efficient lookup
        required_tags = set()
        if skills:
            required_tags.update(skills)
        if risks:
            required_tags.update(risks)
        
        matched_tasks = []
        for task_name, task_data in self.tasks.items():
            # Check quality score
            if task_data.get("quality_score", 0) < min_quality_score:
                continue
                
            # Check if task has any of the required tags
            task_tags = set(task_data.get("tags", []))
            if task_tags.intersection(required_tags):
                matched_tasks.append(task_name)
        
        return matched_tasks
    
    def select_random_tasks(
        self,
        skills: Optional[List[str]] = None,
        risks: Optional[List[str]] = None,
        num_tasks: Optional[int] = None,
        min_quality_score: int = 2,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Select random tasks based on skills/risks criteria.
        
        Args:
            skills: List of skill tags to match
            risks: List of risk tags to match
            num_tasks: Number of tasks to select (None = all matching tasks)
            min_quality_score: Minimum quality score for tasks
            seed: Random seed for reproducibility
            
        Returns:
            List of randomly selected task names
        """
        # Find matching tasks
        matched_tasks = self.find_tasks_by_tags(skills, risks, min_quality_score)
        
        if not matched_tasks:
            logger.warning(f"No tasks found matching skills={skills}, risks={risks}")
            return []
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Select tasks
        if num_tasks is None or num_tasks >= len(matched_tasks):
            selected = matched_tasks
        else:
            selected = random.sample(matched_tasks, num_tasks)
        
        logger.info(f"Selected {len(selected)} tasks from {len(matched_tasks)} matching tasks")
        return selected
    
    def validate_skills_and_risks(
        self, 
        skills: Optional[List[str]] = None, 
        risks: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Validate provided skills and risks against available options.
        
        Returns:
            Dictionary with 'invalid_skills' and 'invalid_risks' lists
        """
        invalid = {"invalid_skills": [], "invalid_risks": []}
        
        if skills:
            valid_skills = set(self.skills)
            invalid["invalid_skills"] = [s for s in skills if s not in valid_skills]
        
        if risks:
            valid_risks = set(self.risks)
            invalid["invalid_risks"] = [r for r in risks if r not in valid_risks]
        
        return invalid


def get_tasks_for_skills_and_risks(
    skills: Optional[List[str]] = None,
    risks: Optional[List[str]] = None,
    num_tasks: Optional[int] = None,
    min_quality_score: int = 2,
    seed: Optional[int] = None
) -> List[str]:
    """
    Convenience function to get tasks matching skills/risks criteria.
    
    Args:
        skills: List of skill tags to match
        risks: List of risk tags to match
        num_tasks: Number of tasks to select (None = all)
        min_quality_score: Minimum quality score
        seed: Random seed
        
    Returns:
        List of task names
    """
    selector = TaskSelector()
    
    # Validate inputs
    invalid = selector.validate_skills_and_risks(skills, risks)
    if invalid["invalid_skills"]:
        logger.warning(f"Invalid skills: {invalid['invalid_skills']}")
    if invalid["invalid_risks"]:
        logger.warning(f"Invalid risks: {invalid['invalid_risks']}")
    
    # Select tasks
    return selector.select_random_tasks(
        skills=skills,
        risks=risks,
        num_tasks=num_tasks,
        min_quality_score=min_quality_score,
        seed=seed
    )