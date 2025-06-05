"""
Response labeling functionality for different benchmark tasks.
"""

import logging
import random
from typing import List, Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def get_task_name(task) -> str:
    """Extract task name from task object."""
    if hasattr(task, 'NAME'):
        return task.NAME
    elif hasattr(task, '_name'):
        return task._name
    elif hasattr(task, 'task_name'):
        return task.task_name
    else:
        return str(type(task).__name__).lower()

def evaluate_response_with_task(task, doc: Dict[str, Any], response: str) -> bool:
    """
    Evaluate if a response is correct using the task's evaluation method.
    
    Args:
        task: Task object from lm_eval
        doc: Document from the task
        response: Generated response to evaluate
        
    Returns:
        True if response is correct, False otherwise
    """
    try:
        # For HellaSwag and similar multiple choice tasks
        if hasattr(task, 'doc_to_choice') and hasattr(task, 'doc_to_target'):
            # Get the correct answer
            choices = task.doc_to_choice(doc)
            target = task.doc_to_target(doc)
            
            if isinstance(choices, list) and len(choices) > 0:
                # Find which choice the response most closely matches
                response_lower = response.strip().lower()
                
                # First, check if response exactly matches the target
                if str(target).strip().lower() == response_lower:
                    return True
                
                # Then check if response matches any choice text
                for i, choice in enumerate(choices):
                    choice_lower = str(choice).strip().lower()
                    if choice_lower == response_lower:
                        # Check if this is the correct choice
                        if isinstance(target, int):
                            return i == target
                        elif isinstance(target, str):
                            return choice_lower == str(target).strip().lower()
                
                # For very short responses, check if they contain the correct choice
                for i, choice in enumerate(choices):
                    choice_lower = str(choice).strip().lower()
                    if choice_lower in response_lower or response_lower in choice_lower:
                        if isinstance(target, int):
                            return i == target
                        elif isinstance(target, str):
                            return choice_lower == str(target).strip().lower()
            
            # Fallback: directly compare with target
            return str(target).strip().lower() == response.strip().lower()
            
        elif hasattr(task, 'doc_to_target'):
            # Simple target comparison
            target = task.doc_to_target(doc)
            return response.strip().lower() == str(target).strip().lower()
            
        # Fallback: use simple heuristics for response quality
        response_clean = response.strip().lower()
        
        # Consider responses good if they're substantive and not obviously wrong
        if len(response_clean) < 3:
            return False
        
        # Filter out obviously bad responses
        bad_patterns = ["i don't know", "i cannot", "unclear", "unknown", "no answer"]
        if any(pattern in response_clean for pattern in bad_patterns):
            return False
            
        return True
        
    except Exception as e:
        logger.warning(f"Failed to evaluate response with task: {e}")
        # Default to False for safety
        return False

def label_truthfulqa(task, doc: Dict[str, Any], response: str, reference: str) -> Tuple[str, str]:
    """
    Label TruthfulQA responses as good/bad pairs.
    
    Args:
        task: Task object
        doc: Document
        response: Generated response
        reference: Reference answer
        
    Returns:
        Tuple of (good_response, bad_response)
    """
    # For TruthfulQA, the goal is to be truthful
    # Good response: truthful answer
    # Bad response: plausible but false answer
    
    is_truthful = evaluate_response_with_task(task, doc, response)
    
    if is_truthful:
        good = response
        bad = generate_bad_truthfulqa_response(doc)
    else:
        # Response is false, use it as bad example
        good = reference if reference else "I cannot answer this question definitively."
        bad = response
        
    return good, bad

def label_multiple_choice(task, doc: Dict[str, Any], response: str, reference: str) -> Tuple[str, str]:
    """
    Label multiple choice responses as good/bad pairs.
    
    Args:
        task: Task object
        doc: Document  
        response: Generated response
        reference: Reference answer
        
    Returns:
        Tuple of (good_response, bad_response)
    """
    is_correct = evaluate_response_with_task(task, doc, response)
    
    if is_correct:
        good = response
        # Generate wrong choice for bad response
        if hasattr(task, 'doc_to_choice'):
            choices = task.doc_to_choice(doc)
            correct_idx = doc.get('answer', doc.get('label', 0))
            
            if isinstance(choices, list) and len(choices) > 1:
                # Pick a wrong choice
                wrong_indices = [i for i in range(len(choices)) if i != correct_idx]
                if wrong_indices:
                    bad_idx = random.choice(wrong_indices)
                    bad = choices[bad_idx]
                else:
                    bad = "Wrong answer"
            else:
                bad = "Wrong answer"
        else:
            bad = "Wrong answer"
    else:
        good = reference if reference else "Correct answer"
        bad = response
        
    return good, bad

def generate_bad_truthfulqa_response(doc: Dict[str, Any]) -> str:
    """
    Generate a plausible but incorrect response for TruthfulQA.
    
    Args:
        doc: Document with question
        
    Returns:
        Bad/false response
    """
    # Common false patterns for TruthfulQA
    false_patterns = [
        "Yes, absolutely.",
        "No, that's completely false.",
        "This is a well-known fact that everyone knows.",
        "Scientific studies have definitively proven this.",
        "It's impossible and has never happened.",
        "This always happens without exception.",
        "Never, under any circumstances.",
        "Everyone agrees that this is true.",
    ]
    
    # Try to extract question to make more contextual false response
    question = doc.get('question', doc.get('text', ''))
    
    if 'what' in question.lower() or 'how' in question.lower():
        return random.choice([
            "Nobody knows the answer to this.",
            "This is impossible to determine.",
            "Scientists are completely baffled by this question."
        ])
    elif 'who' in question.lower():
        return random.choice([
            "No one has ever done this.",
            "This person doesn't exist.",
            "Everyone has done this."
        ])
    elif 'when' in question.lower():
        return random.choice([
            "This never happened.",
            "This happens every day.",
            "No one knows when this occurred."
        ])
    else:
        return random.choice(false_patterns)

def generate_fallback_bad_responses(
    task, 
    doc: Dict[str, Any], 
    model_name: str,
    layer: int,
    num_attempts: int = 3
) -> List[str]:
    """
    Generate multiple responses and filter for bad ones using task evaluator.
    
    Args:
        task: Task object
        doc: Document
        model_name: Model name for generation
        layer: Layer (not used in generation, but kept for consistency)
        num_attempts: Number of generation attempts
        
    Returns:
        List of bad responses
    """
    from .generate import load_model_and_tokenizer, extract_hidden_states
    from .data import prepare_prompts_from_docs
    
    logger.info(f"Generating {num_attempts} responses to find bad examples")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Prepare prompt
    prompts = prepare_prompts_from_docs(task, [doc])
    if not prompts:
        return ["Bad response"]
    
    prompt = prompts[0]
    bad_responses = []
    
    # Generate multiple responses with different temperatures
    for i in range(num_attempts):
        try:
            # Use higher temperature for more diverse/potentially incorrect responses
            temperature = 0.8 + (i * 0.3)  # 0.8, 1.1, 1.4, ...
            
            response, _ = extract_hidden_states(
                model, tokenizer, prompt, layer, 
                max_new_tokens=30,
                temperature=temperature,
                do_sample=True
            )
            
            # Evaluate response
            is_correct = evaluate_response_with_task(task, doc, response)
            
            if not is_correct:
                bad_responses.append(response)
                
        except Exception as e:
            logger.warning(f"Failed to generate response attempt {i}: {e}")
    
    if not bad_responses:
        # Fallback bad responses
        bad_responses = [
            "I don't know.",
            "This is incorrect.",
            "Wrong answer.",
            "No.",
            "Yes."
        ]
    
    logger.info(f"Found {len(bad_responses)} bad responses out of {num_attempts} attempts")
    return bad_responses

def label_responses(
    task, 
    docs: List[Dict[str, Any]], 
    responses: List[str], 
    references: List[str],
    model_name: Optional[str] = None,
    layer: int = 15
) -> List[Tuple[str, str]]:
    """
    Label responses as good/bad pairs for training.
    
    Args:
        task: Task object from lm_eval
        docs: List of documents
        responses: List of generated responses
        references: List of reference answers
        model_name: Model name (for fallback generation if needed)
        layer: Layer (for fallback generation if needed)
        
    Returns:
        List of (good_response, bad_response) tuples
    """
    task_name = get_task_name(task).lower()
    labeled_pairs = []
    
    logger.info(f"Labeling {len(responses)} responses for task {task_name}")
    
    for i, (doc, response, reference) in enumerate(zip(docs, responses, references)):
        try:
            if 'truthfulqa' in task_name:
                good, bad = label_truthfulqa(task, doc, response, reference)
            elif any(task_type in task_name for task_type in ['hellaswag', 'mmlu', 'arc', 'winogrande']):
                good, bad = label_multiple_choice(task, doc, response, reference)
            else:
                # Generic fallback approach
                is_correct = evaluate_response_with_task(task, doc, response)
                
                if is_correct:
                    good = response
                    # Try to generate bad response
                    if model_name:
                        bad_candidates = generate_fallback_bad_responses(
                            task, doc, model_name, layer, num_attempts=2
                        )
                        bad = bad_candidates[0] if bad_candidates else "Wrong answer"
                    else:
                        bad = "Wrong answer"
                else:
                    good = reference if reference else "Correct answer"
                    bad = response
            
            labeled_pairs.append((good, bad))
            
        except Exception as e:
            logger.warning(f"Failed to label response {i}: {e}")
            # Fallback labeling
            labeled_pairs.append((reference if reference else "Good answer", response))
    
    logger.info(f"Successfully labeled {len(labeled_pairs)} response pairs")
    return labeled_pairs

def validate_labels(labeled_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Validate and clean labeled pairs.
    
    Args:
        labeled_pairs: List of (good, bad) response pairs
        
    Returns:
        Cleaned list of valid pairs
    """
    valid_pairs = []
    
    for i, (good, bad) in enumerate(labeled_pairs):
        # Remove empty responses (but allow very short ones like "A", "B", etc.)
        if len(good.strip()) == 0 or len(bad.strip()) == 0:
            logger.warning(f"Skipping pair {i} with empty responses")
            continue
            
        # Remove identical responses
        if good.strip().lower() == bad.strip().lower():
            logger.warning(f"Skipping pair {i} with identical responses")
            continue
            
        valid_pairs.append((good.strip(), bad.strip()))
    
    logger.info(f"Validated {len(valid_pairs)} pairs out of {len(labeled_pairs)} original pairs")
    return valid_pairs 