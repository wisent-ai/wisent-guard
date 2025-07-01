"""
Task Manager for lm-evaluation-harness integration.

This module handles discovery, validation, and loading of tasks from the 
lm-evaluation-harness library.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher


def load_available_tasks() -> List[str]:
    """Load available tasks from local tasks.json file or lm-eval registry."""
    
    # First try to load from local tasks.json file
    try:
        tasks_json_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tasks.json")
        if not os.path.exists(tasks_json_path):
            # Try alternative path
            tasks_json_path = os.path.join(os.path.dirname(__file__), "..", "..", "tasks.json")
        
        if os.path.exists(tasks_json_path):
            with open(tasks_json_path, 'r') as f:
                tasks_data = json.load(f)
                if 'task_list' in tasks_data and tasks_data['task_list']:
                    print(f"Loaded {len(tasks_data['task_list'])} tasks from local tasks.json")
                    return tasks_data['task_list']
                elif 'tasks' in tasks_data:
                    task_names = list(tasks_data['tasks'].keys())
                    print(f"Loaded {len(task_names)} tasks from local tasks.json")
                    return task_names
    except Exception as e:
        print(f"Warning: Could not load from local tasks.json: {e}")
    
    # Fallback to dynamic loading from lm-eval
    try:
        # Try to import lm-eval and get tasks from registry
        from lm_eval.api.registry import ALL_TASKS
        return list(ALL_TASKS)
    except ImportError:
        # If lm-eval not available, try subprocess approach
        try:
            import subprocess
            result = subprocess.run(['lm_eval', '--tasks', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            
            # Extract task names from the formatted output
            task_names = []
            for line in result.stdout.split('\n'):
                if '|' in line and not line.startswith('|---') and not 'Group' in line and not 'Config Location' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        task_name = parts[1].strip()
                        if task_name and not task_name.startswith('-') and task_name != 'Group':
                            task_names.append(task_name)
            
            return task_names
        except Exception:
            # Final fallback - try to discover from lm_eval module
            try:
                import lm_eval.tasks
                # Get all available task names through introspection
                from lm_eval.tasks import get_task_dict
                # This will fail for invalid tasks, so we need another approach
                
                # Try to get task names from lm_eval internals
                try:
                    import lm_eval.tasks.openbookqa  # Import a known task module to trigger loading
                    from lm_eval.api.registry import TASK_REGISTRY
                    return list(TASK_REGISTRY.keys())
                except:
                    pass
                
                # Last resort - scan lm_eval.tasks for modules
                import pkgutil
                import lm_eval.tasks as tasks_pkg
                
                task_names = []
                for importer, modname, ispkg in pkgutil.iter_modules(tasks_pkg.__path__):
                    if not ispkg and not modname.startswith('_'):
                        task_names.append(modname)
                
                return task_names
                
            except Exception as e:
                raise RuntimeError(
                    f"Could not discover tasks from lm-eval or local tasks.json. "
                    f"Please ensure lm-evaluation-harness is installed and accessible. "
                    f"Error: {e}. Try: pip install lm-eval"
                )


def load_docs(task, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load documents from the most appropriate split (validation → test → train).
    
    Args:
        task: Task object from lm_eval
        limit: Optional limit on number of documents to load
        
    Returns:
        List of documents from the most appropriate split
    """
    docs = []
    
    if task.has_validation_docs():
        docs = list(task.validation_docs())
    elif task.has_test_docs():
        docs = list(task.test_docs())
    elif task.has_training_docs():
        docs = list(task.training_docs())
    else:
        raise RuntimeError(f"No labelled docs available for task {task.NAME}")
    
    if limit is not None and limit > 0:
        docs = docs[:limit]
    
    return docs


class TaskManager:
    """Manages lm-eval task discovery, validation, and loading."""
    
    def __init__(self):
        self._available_tasks = None
        self._task_name_mappings = {}
        
    @property 
    def available_tasks(self) -> List[str]:
        """Get list of available tasks, loading if necessary."""
        if self._available_tasks is None:
            self._available_tasks = load_available_tasks()
        return self._available_tasks
    
    def get_available_tasks(self) -> List[str]:
        """Get list of all available tasks."""
        return self.available_tasks
    
    def is_valid_task(self, task_name: str) -> bool:
        """Check if a task name is valid."""
        try:
            resolved_name = self.resolve_task_name(task_name)
            return resolved_name in self.available_tasks
        except ValueError:
            return False
    
    def resolve_task_name(self, task_name: str) -> str:
        """
        Resolve a task name to its canonical form, handling variations and common mistakes.
        
        Args:
            task_name: The task name to resolve
            
        Returns:
            The canonical task name
            
        Raises:
            ValueError: If the task name cannot be resolved
        """
        # Direct match
        if task_name in self.available_tasks:
            return task_name
        
        # Check cached mappings
        if task_name in self._task_name_mappings:
            return self._task_name_mappings[task_name]
        
        # Try fuzzy matching
        best_match = None
        best_similarity = 0.0
        similarity_threshold = 0.6
        
        for available_task in self.available_tasks:
            similarity = self._calculate_task_name_similarity(task_name, available_task)
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = available_task
        
        if best_match:
            # Cache the mapping
            self._task_name_mappings[task_name] = best_match
            return best_match
        
        # List some suggestions if no match found
        suggestions = [task for task in self.available_tasks 
                      if any(word.lower() in task.lower() for word in task_name.split('_'))][:5]
        
        raise ValueError(
            f"Task '{task_name}' not found. "
            f"Available tasks: {len(self.available_tasks)} total. "
            f"Suggestions: {suggestions if suggestions else 'Use get_available_tasks() to see all options'}"
        )
    
    def _calculate_task_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two task names."""
        # Direct similarity
        base_similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        
        # Bonus for word-level matches
        words1 = set(re.split(r'[_\-\s]+', name1.lower()))
        words2 = set(re.split(r'[_\-\s]+', name2.lower()))
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
            return (base_similarity + word_overlap) / 2
        
        return base_similarity
    
    def load_task(self, task_name: str, shots: int = 0, limit: Optional[int] = None):
        """
        Load a task from lm-evaluation-harness with dynamic task name resolution.
        
        Args:
            task_name: Name of the task
            shots: Number of few-shot examples  
            limit: Optional limit on number of documents
            
        Returns:
            Task object from lm_eval
        """
        try:
            from lm_eval.tasks import get_task_dict
            from lm_eval.api.registry import TASK_REGISTRY
        except ImportError as e:
            raise ImportError(
                "lm-evaluation-harness is required. Install with: pip install lm-eval"
            ) from e
        
        # Find the actual task name dynamically
        actual_task_name = self.resolve_task_name(task_name)
        
        # Check if task is in our available tasks list
        if not self.is_valid_task(actual_task_name):
            raise ValueError(
                f"Task '{task_name}' could not be resolved to a valid task. "
                f"Use get_available_tasks() to see all available tasks."
            )
        
        try:
            task_dict = get_task_dict([actual_task_name])
            
            if actual_task_name not in task_dict:
                raise ValueError(
                    f"Task '{task_name}' (resolved to '{actual_task_name}') not found in lm_eval registry."
                )
            
            task = task_dict[actual_task_name]
            task._limit = limit
            
            return task
            
        except Exception as e:
            raise ValueError(f"Failed to load task '{task_name}': {e}") from e
    
    def split_task_data(self, task_data, split_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split task data into training and testing sets.
        
        Args:
            task_data: Task object from lm_eval
            split_ratio: Ratio for training split (0.0 to 1.0)
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (training_docs, testing_docs)
        """
        import random
        
        # Load documents with limit if specified
        limit = getattr(task_data, '_limit', None)
        docs = load_docs(task_data, limit)
        
        # Shuffle with seed for reproducibility
        random.seed(random_seed)
        shuffled_docs = docs.copy()
        random.shuffle(shuffled_docs)
        
        # Split based on ratio
        split_point = int(len(shuffled_docs) * split_ratio)
        training_docs = shuffled_docs[:split_point]
        testing_docs = shuffled_docs[split_point:]
        
        return training_docs, testing_docs
    
    def prepare_prompts_from_docs(self, task, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Prepare prompts from task documents.
        
        Args:
            task: Task object from lm_eval
            docs: List of documents to convert to prompts
            
        Returns:
            List of formatted prompts
        """
        prompts = []
        
        for doc in docs:
            try:
                # Different tasks have different prompt creation methods
                if hasattr(task, 'doc_to_text'):
                    prompt = task.doc_to_text(doc)
                elif hasattr(task, 'doc_format'):
                    prompt = task.doc_format(doc)
                elif 'input' in doc:
                    prompt = doc['input']
                elif 'question' in doc:
                    prompt = doc['question']
                elif 'prompt' in doc:
                    prompt = doc['prompt']
                else:
                    # Fallback: use the first text-like field
                    text_fields = ['text', 'passage', 'context', 'story']
                    prompt = None
                    for field in text_fields:
                        if field in doc and isinstance(doc[field], str):
                            prompt = doc[field]
                            break
                    
                    if prompt is None:
                        prompt = str(doc)
                
                prompts.append(prompt)
                
            except Exception as e:
                # Skip problematic documents
                print(f"Warning: Could not create prompt from document: {e}")
                continue
        
        return prompts
    
    def get_reference_answers(self, task, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Extract reference answers from task documents.
        
        Args:
            task: Task object from lm_eval  
            docs: List of documents to extract answers from
            
        Returns:
            List of reference answers
        """
        answers = []
        
        for doc in docs:
            try:
                # Different tasks store answers differently
                if hasattr(task, 'doc_to_target'):
                    answer = task.doc_to_target(doc)
                elif hasattr(task, 'get_answer'):
                    answer = task.get_answer(doc)
                elif 'answer' in doc:
                    answer = doc['answer']
                elif 'target' in doc:
                    answer = doc['target']
                elif 'label' in doc:
                    answer = doc['label']
                elif 'output' in doc:
                    answer = doc['output']
                else:
                    # Look for likely answer fields
                    answer_fields = ['correct_answer', 'gold', 'truth', 'solution']
                    answer = None
                    for field in answer_fields:
                        if field in doc:
                            answer = doc[field]
                            break
                    
                    if answer is None:
                        answer = "UNKNOWN"
                
                answers.append(str(answer))
                
            except Exception as e:
                print(f"Warning: Could not extract answer from document: {e}")
                answers.append("UNKNOWN")
        
        return answers


# Global instance for convenience
_task_manager = TaskManager()

# Convenience functions that use the global instance
def get_available_tasks() -> List[str]:
    """Get list of all available tasks."""
    return _task_manager.get_available_tasks()

def is_valid_task(task_name: str) -> bool:
    """Check if a task name is valid."""
    return _task_manager.is_valid_task(task_name)

def resolve_task_name(task_name: str) -> str:
    """Resolve a task name to its canonical form."""
    return _task_manager.resolve_task_name(task_name) 