"""
Data loading and splitting functionality for lm-evaluation-harness integration.
"""

import logging
import random
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Try to import sklearn, fallback to manual implementation
try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available, using manual train_test_split implementation")

def manual_train_test_split(docs: List[Any], train_size: float = 0.8, random_state: int = 42, shuffle: bool = True) -> Tuple[List[Any], List[Any]]:
    """Manual implementation of train_test_split without sklearn."""
    if shuffle:
        random.seed(random_state)
        docs_copy = docs.copy()
        random.shuffle(docs_copy)
    else:
        docs_copy = docs
    
    split_idx = int(len(docs_copy) * train_size)
    return docs_copy[:split_idx], docs_copy[split_idx:]

# Task name mappings for common aliases
TASK_NAME_MAPPINGS = {
    'truthfulqa': 'truthfulqa_mc1',  # Default to MC1 variant
    'truthful_qa': 'truthfulqa_mc1',
    'hellaswag': 'hellaswag',
    'mmlu': 'mmlu_abstract_algebra',  # Use a specific MMLU subtask
    'mmlu_easy': 'mmlu_elementary_mathematics',
    'arc_easy': 'arc_easy',
    'arc_challenge': 'arc_challenge',
    'winogrande': 'winogrande',
    'piqa': 'piqa',
    'boolq': 'boolq',
}

def load_docs(task, limit: Optional[int] = None):
    """
    Zwraca listę dokumentów z najodpowiedniejszego splitu
    (kolejność priorytetu: validation → test → train).
    
    Args:
        task: Task object from lm_eval
        limit: Optional limit on number of documents to load
        
    Returns:
        List of documents from the most appropriate split
        
    Raises:
        RuntimeError: If no labelled docs are available
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
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        docs = docs[:limit]
        print(f"✓ Applied limit: using {len(docs)} documents (limit: {limit})")
    
    return docs

def load_task(task_name: str, shots: int = 0, limit: Optional[int] = None) -> Any:
    """
    Load a task from lm-evaluation-harness.
    
    Args:
        task_name: Name of the task (e.g., 'hellaswag', 'mmlu', 'truthfulqa')
        shots: Number of few-shot examples (default: 0)
        limit: Optional limit on number of documents to load
        
    Returns:
        Task object from lm_eval
        
    Raises:
        ImportError: If lm_eval is not installed
        ValueError: If task is not found
    """
    try:
        # Try new API (lm_eval >= 0.4.0)
        from lm_eval.tasks import get_task_dict
        from lm_eval.api.registry import TASK_REGISTRY
    except ImportError as e:
        raise ImportError(
            "lm-evaluation-harness is required for this function. "
            "Install with: pip install lm-eval"
        ) from e
    
    # Map common task name aliases to actual task names
    actual_task_name = TASK_NAME_MAPPINGS.get(task_name, task_name)
    
    try:
        # Get task dictionary
        task_dict = get_task_dict([actual_task_name])
        
        if actual_task_name not in task_dict:
            available_tasks = list(TASK_REGISTRY.keys()) if TASK_REGISTRY else ["Run initialize_tasks() first"]
            raise ValueError(
                f"Task '{task_name}' (mapped to '{actual_task_name}') not found. "
                f"Available tasks: {available_tasks[:10]}..."  # Show first 10
            )
        
        task = task_dict[actual_task_name]
        print(f"✓ Loaded task: {actual_task_name}")
        print(f"  - Task type: {type(task).__name__}")
        print(f"  - Has validation docs: {task.has_validation_docs()}")
        print(f"  - Has test docs: {task.has_test_docs()}")
        print(f"  - Has training docs: {task.has_training_docs()}")
        
        # Store limit on the task object for later use
        task._limit = limit
        
        return task
        
    except Exception as e:
        raise ValueError(f"Failed to load task '{task_name}': {e}") from e


def split_docs(docs: List[Dict[str, Any]], ratio: float = 0.8, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split documents into training and testing sets.
    
    Args:
        docs: List of documents to split
        ratio: Proportion for training set (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_docs, test_docs)
    """
    if not docs:
        logger.warning("No documents provided for splitting")
        return [], []
    
    if len(docs) < 2:
        logger.warning("Less than 2 documents provided, using single doc for both train and test")
        return docs, docs
    
    logger.info(f"Splitting {len(docs)} documents with ratio {ratio} (seed: {seed})")
    
    if HAS_SKLEARN:
        train_docs, test_docs = train_test_split(
            docs, 
            train_size=ratio, 
            random_state=seed,
            shuffle=True
        )
    else:
        train_docs, test_docs = manual_train_test_split(
            docs,
            train_size=ratio,
            random_state=seed,
            shuffle=True
        )
    
    logger.info(f"Split complete: {len(train_docs)} train, {len(test_docs)} test documents")
    
    return train_docs, test_docs


def get_task_metadata(task_name: str) -> Dict[str, Any]:
    """
    Get metadata about a task including its evaluation method and expected format.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Dictionary with task metadata
    """
    try:
        from lm_eval.api.registry import get_task_dict
    except ImportError as e:
        raise ImportError(
            "lm-evaluation-harness is required for this functionality. "
            "Install with: pip install lm-eval"
        ) from e
    
    task_dict = get_task_dict([task_name])
    if task_name not in task_dict:
        raise ValueError(f"Task '{task_name}' not found")
    
    task = task_dict[task_name]
    
    # Extract useful metadata
    metadata = {
        'task_name': task_name,
        'output_type': getattr(task, 'OUTPUT_TYPE', 'unknown'),
        'higher_is_better': getattr(task, 'higher_is_better', True),
        'primary_metric': getattr(task, 'metric', 'accuracy'),
        'is_multiple_choice': hasattr(task, 'choices') or 'multiple_choice' in str(type(task)).lower(),
        'has_references': hasattr(task, 'doc_to_target'),
    }
    
    logger.debug(f"Task metadata for {task_name}: {metadata}")
    
    return metadata


def prepare_prompts_from_docs(task: Any, docs: List[Dict[str, Any]]) -> List[str]:
    """
    Convert task documents to prompts for generation.
    
    Args:
        task: Task object from lm_eval
        docs: List of documents to convert
        
    Returns:
        List of formatted prompts
    """
    prompts = []
    
    for doc in docs:
        try:
            # Different tasks may have different prompt formatting methods
            if hasattr(task, 'doc_to_text'):
                prompt = task.doc_to_text(doc)
                
                # Handle case where doc_to_text returns a list or other types
                if isinstance(prompt, list):
                    # Join list elements with space or take first element
                    prompt = ' '.join(str(p) for p in prompt) if prompt else ""
                elif not isinstance(prompt, str):
                    # Convert to string if it's not already
                    prompt = str(prompt)
                    
            elif hasattr(task, 'fewshot_context'):
                # Handle few-shot prompts
                prompt = task.fewshot_context(doc, 0)  # Use 0 shots for now
                if not isinstance(prompt, str):
                    prompt = str(prompt)
            else:
                # Fallback: try to extract text directly
                prompt = doc.get('text', doc.get('question', str(doc)))
            
            # Ensure we have a valid string
            if not isinstance(prompt, str):
                prompt = str(prompt)
                
            prompts.append(prompt)
            
        except Exception as e:
            logger.warning(f"Failed to format prompt for doc: {e}")
            # Fallback to simple string representation
            prompts.append(str(doc))
    
    logger.info(f"Prepared {len(prompts)} prompts from {len(docs)} documents")
    
    return prompts


def get_reference_answers(task: Any, docs: List[Dict[str, Any]]) -> List[str]:
    """
    Extract reference answers from task documents.
    
    Args:
        task: Task object from lm_eval
        docs: List of documents
        
    Returns:
        List of reference answers
    """
    references = []
    
    for doc in docs:
        try:
            if hasattr(task, 'doc_to_target'):
                ref = task.doc_to_target(doc)
            elif hasattr(task, 'doc_to_choice'):
                # For multiple choice tasks
                choices = task.doc_to_choice(doc)
                if isinstance(choices, list):
                    ref = choices[0] if choices else ""
                else:
                    ref = str(choices)
            else:
                # Fallback
                ref = doc.get('answer', doc.get('target', doc.get('label', "")))
            
            references.append(str(ref))
            
        except Exception as e:
            logger.warning(f"Failed to extract reference for doc: {e}")
            references.append("")
    
    logger.info(f"Extracted {len(references)} reference answers")
    
    return references

def split_data(task_data: Any, split_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    Split task data into train and test sets.
    
    Args:
        task_data: Task object from lm_eval
        split_ratio: Ratio for train split (default: 0.8)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_docs, test_docs)
    """
    # Load documents from the task using appropriate split
    try:
        # Get limit from task object if it was stored there
        limit = getattr(task_data, '_limit', None)
        docs = load_docs(task_data, limit=limit)
        print(f"✓ Loaded {len(docs)} documents from task")
    except RuntimeError as e:
        print(f"✗ Error loading documents: {e}")
        return [], []
    
    if not docs:
        print("✗ No documents found in task")
        return [], []
    
    # Split the data
    try:
        from sklearn.model_selection import train_test_split
        train_docs, test_docs = train_test_split(
            docs, 
            train_size=split_ratio, 
            random_state=random_seed,
            shuffle=True
        )
    except ImportError:
        # Manual fallback if sklearn not available
        import random
        random.seed(random_seed)
        docs_copy = docs.copy()
        random.shuffle(docs_copy)
        
        split_idx = int(len(docs_copy) * split_ratio)
        train_docs = docs_copy[:split_idx]
        test_docs = docs_copy[split_idx:]
    
    print(f"✓ Split data: {len(train_docs)} train, {len(test_docs)} test documents")
    return train_docs, test_docs 