#!/usr/bin/env python3
"""
Script to populate tasks.json with real information from lm_eval tasks.
This script fetches actual examples, descriptions, and evaluation methods.
"""

import json
import os
from typing import Dict, Any, List, Optional
import sys
import subprocess
import random

def get_task_samples_for_analysis(task_name: str, num_samples: int = 5) -> Dict[str, Any]:
    """
    Retrieve sample questions and answers from a benchmark task for AI analysis.
    
    This function extracts representative examples from a benchmark that can be
    analyzed to determine what cognitive abilities the benchmark tests.
    
    Args:
        task_name: Name of the task to analyze
        num_samples: Number of samples to retrieve (default 5)
    
    Returns:
        Dictionary containing task info and samples for analysis
    """
    try:
        # Load lm_eval
        from lm_eval.tasks import get_task_dict
        
        # Get the task (automatically trust remote code via environment)
        task_dict = get_task_dict([task_name])
        if task_name not in task_dict:
            return {
                "task_name": task_name,
                "error": f"Task '{task_name}' not found",
                "samples": []
            }
        
        task = task_dict[task_name]
        
        # Get task description
        description = getattr(task, 'DESCRIPTION', getattr(task, '__doc__', f"Task: {task_name}"))
        
        # Get documents
        docs = []
        try:
            if hasattr(task, 'validation_docs') and task.has_validation_docs():
                docs = list(task.validation_docs())
            elif hasattr(task, 'test_docs') and task.has_test_docs():
                docs = list(task.test_docs())
            elif hasattr(task, 'training_docs') and task.has_training_docs():
                docs = list(task.training_docs())
        except Exception as e:
            print(f"Warning: Could not get docs for {task_name}: {e}")
        
        if not docs:
            return {
                "task_name": task_name,
                "description": description,
                "error": "No documents found for this task",
                "samples": []
            }
        
        # Sample documents - take up to num_samples, distributed across the dataset
        if len(docs) <= num_samples:
            sample_docs = docs
        else:
            # Use a combination of random and distributed sampling for diversity
            if len(docs) > 100:
                # For large datasets, use random sampling from different sections
                section_size = len(docs) // num_samples
                sample_docs = []
                for i in range(num_samples):
                    start_idx = i * section_size
                    end_idx = min((i + 1) * section_size, len(docs))
                    # Pick a random document from this section
                    random_idx = random.randint(start_idx, min(end_idx - 1, len(docs) - 1))
                    sample_docs.append(docs[random_idx])
            else:
                # For smaller datasets, just take random samples
                sample_docs = random.sample(docs, num_samples)
        
        # Extract samples
        samples = []
        for i, doc in enumerate(sample_docs):
            sample = {"sample_id": i + 1}
            
            # Get question/prompt
            try:
                if hasattr(task, 'doc_to_text'):
                    sample["question"] = str(task.doc_to_text(doc))
                elif 'question' in doc:
                    sample["question"] = str(doc['question'])
                elif 'text' in doc:
                    sample["question"] = str(doc['text'])
                elif 'prompt' in doc:
                    sample["question"] = str(doc['prompt'])
                else:
                    sample["question"] = "Question format not recognized"
            except Exception as e:
                sample["question"] = f"Error extracting question: {e}"
            
            # Get correct answer
            try:
                if hasattr(task, 'doc_to_target'):
                    target = task.doc_to_target(doc)
                    sample["correct_answer"] = str(target)
                elif 'answer' in doc:
                    sample["correct_answer"] = str(doc['answer'])
                elif 'target' in doc:
                    sample["correct_answer"] = str(doc['target'])
                else:
                    sample["correct_answer"] = "Answer format not recognized"
            except Exception as e:
                sample["correct_answer"] = f"Error extracting answer: {e}"
            
            # Get choices if it's multiple choice
            sample["choices"] = []
            try:
                # Check for standard multiple choice format
                if 'choices' in doc and isinstance(doc['choices'], list):
                    sample["choices"] = [str(choice) for choice in doc['choices']]
                    sample["format"] = "multiple_choice"
                    
                    # Get correct choice index
                    gold = doc.get('gold', doc.get('label', None))
                    if isinstance(gold, list) and len(gold) > 0:
                        sample["correct_choice_index"] = gold[0]
                    elif isinstance(gold, int):
                        sample["correct_choice_index"] = gold
                    else:
                        sample["correct_choice_index"] = None
                
                # Special handling for TruthfulQA format
                elif 'mc1_targets' in doc or 'mc2_targets' in doc:
                    sample["format"] = "multiple_choice"
                    
                    # Try to get choices from mc1_targets
                    if 'mc1_targets' in doc:
                        mc1_targets = doc['mc1_targets']
                        if 'choices' in mc1_targets:
                            sample["choices"] = [str(choice) for choice in mc1_targets['choices']]
                            # Find the correct answer
                            if 'labels' in mc1_targets:
                                labels = mc1_targets['labels']
                                if isinstance(labels, list) and len(labels) > 0:
                                    # Find the index of the correct answer (label = 1)
                                    try:
                                        sample["correct_choice_index"] = labels.index(1)
                                    except ValueError:
                                        sample["correct_choice_index"] = 0  # fallback
                
                # If still no choices found, check if it has multiple choice indicators
                elif any(key in doc for key in ['A)', 'B)', 'C)', 'D)']):
                    sample["format"] = "multiple_choice"
                    # Try to extract choices from text
                    sample["choices"] = ["Could not extract choices automatically"]
                else:
                    sample["format"] = "open_ended"
                    
            except Exception as e:
                sample["choices"] = []
                sample["format"] = "unknown"
                print(f"Warning: Error processing choices for {task_name}: {e}")
            
            # Add any additional context from the document
            sample["additional_info"] = {}
            for key, value in doc.items():
                if key not in ['question', 'text', 'answer', 'target', 'choices', 'gold', 'label']:
                    try:
                        # For nested structures, show the structure
                        if isinstance(value, dict):
                            sample["additional_info"][key] = f"Dict with keys: {list(value.keys())}"
                        elif isinstance(value, list):
                            sample["additional_info"][key] = f"List with {len(value)} items"
                        else:
                            sample["additional_info"][key] = str(value)[:200]  # Truncate long values
                    except:
                        sample["additional_info"][key] = "Could not convert to string"
            
            samples.append(sample)
        
        # Get task metadata
        task_info = {
            "task_name": task_name,
            "description": description,
            "total_docs": len(docs),
            "sampled_docs": len(samples),
            "output_type": getattr(task, 'OUTPUT_TYPE', 'unknown'),
            "samples": samples
        }
        
        return task_info
        
    except ImportError as e:
        return {
            "task_name": task_name,
            "error": f"lm-evaluation-harness not installed: {e}",
            "samples": []
        }
    except Exception as e:
        return {
            "task_name": task_name,
            "error": f"Error retrieving samples: {e}",
            "samples": []
        }

def load_lm_eval():
    """Load lm_eval library and handle import errors."""
    try:
        from lm_eval.tasks import get_task_dict
        return get_task_dict, None  # Don't need registry, will use subprocess
    except ImportError as e:
        print(f"Error: lm-evaluation-harness is required. Install with: pip install lm-eval")
        print(f"Import error: {e}")
        sys.exit(1)

def expand_group_task(task_name: str, get_task_dict) -> List[str]:
    """Try to expand a group task to get its individual sub-tasks."""
    try:
        # Try to get the task as a group
        from lm_eval.api.registry import get_group
        group_tasks = get_group(task_name)
        if group_tasks:
            return list(group_tasks)
    except:
        pass
    
    # Alternative: try to infer from task registry
    try:
        from lm_eval.api.registry import ALL_TASKS
        # Look for tasks that start with the group name
        sub_tasks = [t for t in ALL_TASKS if t.startswith(task_name + "_")]
        if sub_tasks:
            return sub_tasks
    except:
        pass
    
    # Fallback: read from our tasks.json to find sub-tasks
    try:
        tasks_file = os.path.join(os.path.dirname(__file__), '..', 'tasks.json')
        if os.path.exists(tasks_file):
            with open(tasks_file, 'r') as f:
                data = json.load(f)
                all_task_names = list(data.get('tasks', {}).keys())
                # Look for tasks that start with the group name + "_"
                sub_tasks = [t for t in all_task_names if t.startswith(task_name + "_")]
                if sub_tasks:
                    return sub_tasks
    except Exception as e:
        print(f"    Error reading tasks.json for group expansion: {e}")
    
    return []

def get_task_info(task_name: str, get_task_dict, task_registry) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific task."""
    try:
        # Try to get task object directly first
        task_dict = get_task_dict([task_name])
        
        if task_name not in task_dict:
            # This might be a group task, try to expand it
            print(f"  Task {task_name} not found directly, checking if it's a group task...")
            sub_tasks = expand_group_task(task_name, get_task_dict)
            
            if not sub_tasks:
                print(f"  Warning: Task {task_name} not found and no sub-tasks found")
                return None
            
            print(f"  Found group task with {len(sub_tasks)} sub-tasks: {sub_tasks[:3]}{'...' if len(sub_tasks) > 3 else ''}")
            
            # Process group task by sampling from sub-tasks
            return process_group_task(task_name, sub_tasks, get_task_dict)
        
        # Process individual task
        task = task_dict[task_name]
        return process_individual_task(task_name, task)
        
    except Exception as e:
        print(f"Error processing task {task_name}: {e}")
        return None

def process_individual_task(task_name: str, task) -> Dict[str, Any]:
    """Process an individual task to extract information."""
    # Get basic info
    info = {
        "name": task_name,
        "description": getattr(task, 'DESCRIPTION', getattr(task, '__doc__', f"Task: {task_name}")),
        "example_question": "",
        "example_good_response": "",
        "example_bad_response": "",
        "evaluation_method": "",
        "category": "",
        "task_type": "individual"
    }
    
    # Try to get a sample document
    try:
        if hasattr(task, 'validation_docs') and task.has_validation_docs():
            docs = list(task.validation_docs())
        elif hasattr(task, 'test_docs') and task.has_test_docs():
            docs = list(task.test_docs())
        elif hasattr(task, 'training_docs') and task.has_training_docs():
            docs = list(task.training_docs())
        else:
            docs = []
        
        if docs:
            sample_doc = docs[0]
            
            # Get example question
            if hasattr(task, 'doc_to_text'):
                info["example_question"] = str(task.doc_to_text(sample_doc))
            elif 'question' in sample_doc:
                info["example_question"] = str(sample_doc['question'])
            elif 'text' in sample_doc:
                info["example_question"] = str(sample_doc['text'])
            
            # Get example answer/target
            if hasattr(task, 'doc_to_target'):
                target = task.doc_to_target(sample_doc)
                info["example_good_response"] = str(target)
            elif 'answer' in sample_doc:
                info["example_good_response"] = str(sample_doc['answer'])
            elif 'target' in sample_doc:
                info["example_good_response"] = str(sample_doc['target'])
            
            # Try to get choices for multiple choice
            if 'choices' in sample_doc and isinstance(sample_doc['choices'], list):
                choices = sample_doc['choices']
                if len(choices) > 1:
                    # Find incorrect choice for bad response
                    gold = sample_doc.get('gold', sample_doc.get('label', [0]))
                    # Handle both list format [0] and integer format 0
                    if isinstance(gold, list) and len(gold) > 0:
                        correct_idx = gold[0]
                    elif isinstance(gold, int):
                        correct_idx = gold
                    else:
                        correct_idx = 0
                    
                    if correct_idx < len(choices):
                        # Set good response to the actual correct choice text
                        info["example_good_response"] = str(choices[correct_idx])
                        
                        # Get a different choice as bad response
                        bad_idx = (correct_idx + 1) % len(choices)
                        info["example_bad_response"] = str(choices[bad_idx])
            
            # If no bad response found, create a generic one based on task type
            if not info["example_bad_response"]:
                if 'bool' in task_name.lower():
                    info["example_bad_response"] = "No" if info["example_good_response"].strip().lower().startswith("yes") else "Yes"
                elif 'math' in task_name.lower() or 'gsm' in task_name.lower():
                    info["example_bad_response"] = "42"  # Generic wrong number
                elif 'truth' in task_name.lower():
                    info["example_bad_response"] = "This is a common misconception"
                else:
                    info["example_bad_response"] = "Incorrect or irrelevant response"
            
    except Exception as e:
        print(f"Warning: Could not get sample for {task_name}: {e}")
    
    # Determine evaluation method by inspecting the actual task
    eval_method = get_evaluation_method(task)
    info["evaluation_method"] = eval_method
    
    # Determine category based on evaluation type
    category = get_category(task)
    info["category"] = category
    
    return info

def process_group_task(group_name: str, sub_tasks: List[str], get_task_dict) -> Dict[str, Any]:
    """Process a group task by sampling from its sub-tasks."""
    # Pick a random sub-task for examples
    random_subtask = random.choice(sub_tasks)
    print(f"  Using random sub-task '{random_subtask}' for examples")
    
    # Get info from the random sub-task
    try:
        subtask_dict = get_task_dict([random_subtask])
        if random_subtask not in subtask_dict:
            print(f"  Warning: Random sub-task {random_subtask} not found")
            return None
        
        subtask = subtask_dict[random_subtask]
        example_info = extract_examples_from_task(random_subtask, subtask)
    except Exception as e:
        print(f"  Error getting examples from {random_subtask}: {e}")
        example_info = {
            "example_question": f"Example from {group_name} group",
            "example_good_response": "Sample correct response",
            "example_bad_response": "Sample incorrect response"
        }
    
    # Collect all evaluation methods and categories from sub-tasks
    all_eval_methods = set()
    all_categories = set()
    
    print(f"  Analyzing all {len(sub_tasks)} sub-tasks for evaluation methods and categories...")
    for i, subtask_name in enumerate(sub_tasks[:10]):  # Limit to first 10 for performance
        try:
            subtask_dict = get_task_dict([subtask_name])
            if subtask_name in subtask_dict:
                subtask = subtask_dict[subtask_name]
                eval_method = get_evaluation_method(subtask)
                category = get_category(subtask)
                
                if eval_method != "Unknown evaluation method":
                    all_eval_methods.add(eval_method)
                if category != "unknown":
                    all_categories.add(category)
        except Exception as e:
            print(f"    Warning: Could not analyze sub-task {subtask_name}: {e}")
            continue
    
    # Create group task info
    info = {
        "name": group_name,
        "description": f"Group task containing {len(sub_tasks)} sub-tasks",
        "example_question": example_info.get("example_question", ""),
        "example_good_response": example_info.get("example_good_response", ""),
        "example_bad_response": example_info.get("example_bad_response", ""),
        "evaluation_method": list(all_eval_methods) if all_eval_methods else ["Unknown evaluation method"],
        "category": list(all_categories) if all_categories else ["unknown"],
        "task_type": "group",
        "sub_tasks": sub_tasks,
        "sub_task_count": len(sub_tasks),
        "example_source": random_subtask
    }
    
    return info

def extract_examples_from_task(task_name: str, task) -> Dict[str, str]:
    """Extract example question and responses from a task."""
    info = {
        "example_question": "",
        "example_good_response": "",
        "example_bad_response": ""
    }
    
    # Try to get a sample document
    try:
        if hasattr(task, 'validation_docs') and task.has_validation_docs():
            docs = list(task.validation_docs())
        elif hasattr(task, 'test_docs') and task.has_test_docs():
            docs = list(task.test_docs())
        elif hasattr(task, 'training_docs') and task.has_training_docs():
            docs = list(task.training_docs())
        else:
            docs = []
        
        if docs:
            sample_doc = docs[0]
            
            # Get example question
            if hasattr(task, 'doc_to_text'):
                info["example_question"] = str(task.doc_to_text(sample_doc))
            elif 'question' in sample_doc:
                info["example_question"] = str(sample_doc['question'])
            elif 'text' in sample_doc:
                info["example_question"] = str(sample_doc['text'])
            
            # Get example answer/target
            if hasattr(task, 'doc_to_target'):
                target = task.doc_to_target(sample_doc)
                info["example_good_response"] = str(target)
            elif 'answer' in sample_doc:
                info["example_good_response"] = str(sample_doc['answer'])
            elif 'target' in sample_doc:
                info["example_good_response"] = str(sample_doc['target'])
            
            # Try to get choices for multiple choice
            if 'choices' in sample_doc and isinstance(sample_doc['choices'], list):
                choices = sample_doc['choices']
                if len(choices) > 1:
                    # Find incorrect choice for bad response
                    gold = sample_doc.get('gold', sample_doc.get('label', [0]))
                    # Handle both list format [0] and integer format 0
                    if isinstance(gold, list) and len(gold) > 0:
                        correct_idx = gold[0]
                    elif isinstance(gold, int):
                        correct_idx = gold
                    else:
                        correct_idx = 0
                    
                    if correct_idx < len(choices):
                        # Set good response to the actual correct choice text
                        info["example_good_response"] = str(choices[correct_idx])
                        
                        # Get a different choice as bad response
                        bad_idx = (correct_idx + 1) % len(choices)
                        info["example_bad_response"] = str(choices[bad_idx])
            
            # If no bad response found, create a generic one based on task type
            if not info["example_bad_response"]:
                if 'bool' in task_name.lower():
                    info["example_bad_response"] = "No" if info["example_good_response"].strip().lower().startswith("yes") else "Yes"
                elif 'math' in task_name.lower() or 'gsm' in task_name.lower():
                    info["example_bad_response"] = "42"  # Generic wrong number
                elif 'truth' in task_name.lower():
                    info["example_bad_response"] = "This is a common misconception"
                else:
                    info["example_bad_response"] = "Incorrect or irrelevant response"
            
    except Exception as e:
        print(f"Warning: Could not get sample for {task_name}: {e}")
    
    return info

def get_evaluation_method(task) -> str:
    """Get evaluation method from a task."""
    eval_method = "Unknown evaluation method"
    
    if hasattr(task, 'process_results'):
        try:
            # Get the output type which determines evaluation method
            if hasattr(task, 'OUTPUT_TYPE'):
                output_type = task.OUTPUT_TYPE
                if output_type == "multiple_choice":
                    eval_method = "Multiple choice accuracy (argmax of log-likelihoods vs gold labels)"
                elif output_type == "generate_until":
                    eval_method = "Text generation (exact match, F1, BLEU, ROUGE depending on task)"
                elif output_type == "loglikelihood":
                    eval_method = "Log-likelihood evaluation (perplexity, accuracy)"
                elif output_type == "loglikelihood_rolling":
                    eval_method = "Rolling log-likelihood (word/byte perplexity)"
                else:
                    eval_method = f"Unknown output type: {output_type}"
            else:
                eval_method = "Has process_results but no OUTPUT_TYPE found"
        except Exception as e:
            eval_method = f"Has process_results but couldn't inspect: {e}"
    else:
        eval_method = "No process_results method found"
    
    return eval_method

def get_category(task) -> str:
    """Get category from a task."""
    category = "unknown"
    if hasattr(task, 'OUTPUT_TYPE'):
        output_type = task.OUTPUT_TYPE
        if output_type == "multiple_choice":
            category = "multiple_choice"
        elif output_type == "generate_until":
            category = "open_ended_generation"
        elif output_type == "loglikelihood":
            category = "log_likelihood"
        elif output_type == "loglikelihood_rolling":
            category = "rolling_log_likelihood"
        else:
            category = f"other_{output_type}"
    else:
        category = "no_output_type"
    
    return category

def test_sample_retrieval(task_name: str = "truthfulqa_mc1"):
    """Test function to demonstrate the get_task_samples_for_analysis function."""
    print(f"\n=== Testing Sample Retrieval for '{task_name}' ===")
    
    # Get samples
    result = get_task_samples_for_analysis(task_name, num_samples=3)
    
    # Print results
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    print(f"✅ Successfully retrieved samples from '{result['task_name']}'")
    
    # Handle None description
    description = result.get('description') or "No description available"
    print(f"📝 Description: {description[:200]}...")
    
    print(f"📊 Total documents: {result['total_docs']}")
    print(f"🎯 Sampled documents: {result['sampled_docs']}")
    print(f"🔧 Output type: {result['output_type']}")
    
    print(f"\n--- Sample Questions ---")
    for i, sample in enumerate(result['samples']):
        print(f"\n📋 Sample {sample['sample_id']}:")
        question = sample.get('question', 'No question available')
        print(f"❓ Question: {question[:300]}...")
        
        answer = sample.get('correct_answer', 'No answer available')
        print(f"✅ Correct Answer: {answer}")
        
        format_type = sample.get('format', 'unknown')
        print(f"📐 Format: {format_type}")
        
        if sample.get('choices'):
            print(f"🔤 Choices:")
            for j, choice in enumerate(sample['choices']):
                marker = "👉" if j == sample.get('correct_choice_index') else "  "
                print(f"  {marker} {j}: {choice}")
        
        if sample.get('additional_info'):
            print(f"ℹ️  Additional info: {list(sample['additional_info'].keys())}")
    
    print(f"\n=== Analysis Summary ===")
    print("Based on these samples, an AI could analyze:")
    print("- Question format and complexity")
    print("- Type of reasoning required")
    print("- Domain knowledge needed")
    print("- Cognitive abilities being tested")
    
    return True

def main():
    """Main function to populate tasks.json."""
    
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        task_name = sys.argv[2] if len(sys.argv) > 2 else "truthfulqa_mc1"
        print(f"🧪 Running in TEST MODE")
        success = test_sample_retrieval(task_name)
        if success:
            print(f"\n✅ Test completed successfully!")
        else:
            print(f"\n❌ Test failed!")
        return
    
    print("Loading lm_eval library...")
    get_task_dict, task_registry = load_lm_eval()
    
    # Get all available tasks from lm_eval using subprocess
    print("Getting all available tasks from lm_eval...")
    
    try:
        # Use subprocess to get task list
        print("Using subprocess to get task list...")
        result = subprocess.run(['lm_eval', '--tasks', 'list'], capture_output=True, text=True, timeout=60)
        all_tasks_output = result.stdout
        
        # Extract just the task names from the formatted output
        task_names = []
        for line in all_tasks_output.split('\n'):
            if '|' in line and not line.startswith('|---') and not 'Group' in line and not 'Config Location' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    task_name = parts[1].strip()
                    if task_name and not task_name.startswith('-') and task_name != 'Group':
                        task_names.append(task_name)
        
        available_tasks = task_names
        print(f"Found {len(available_tasks)} total tasks via subprocess")
        
    except Exception as e:
        print(f"Error getting tasks from lm_eval: {e}")
        # Fallback to known tasks
        available_tasks = ["truthfulqa_mc1", "hellaswag"]
    
    # PHASE 1: Save all task names to tasks.json
    tasks_file = "wisent_guard/core/tasks.json"
    print(f"\nPhase 1: Saving {len(available_tasks)} task names to {tasks_file}")
    
    # Create initial structure with all task names but empty task details
    initial_data = {
        "tasks": {task_name: {} for task_name in available_tasks},
        "task_list": available_tasks
    }
    
    with open(tasks_file, 'w') as f:
        json.dump(initial_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(available_tasks)} task names to {tasks_file}")
    
    # PHASE 2: Populate details for first 2 tasks (for testing)
    print(f"\nPhase 2: Populating detailed information for 2 test tasks...")
    
    # Use known individual tasks that actually exist
    tasks_to_populate = ["truthfulqa_mc1", "hellaswag"]
    print(f"Will populate: {tasks_to_populate}")
    
    # Load current data
    with open(tasks_file, 'r') as f:
        current_data = json.load(f)
    
    processed = 0
    for i, task_name in enumerate(tasks_to_populate):
        print(f"Processing {i+1}/{len(tasks_to_populate)}: {task_name}")
        
        task_info = get_task_info(task_name, get_task_dict, task_registry)
        if task_info:
            current_data["tasks"][task_name] = task_info
            processed += 1
            
            # Save after each task to prevent data loss
            with open(tasks_file, 'w') as f:
                json.dump(current_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Updated {task_name}")
        else:
            print(f"  ✗ Failed to process {task_name}")
    
    print(f"\nCompleted processing. Successfully populated {processed}/{len(tasks_to_populate)} tasks")
    print(f"Total tasks in file: {len(current_data['task_list'])}")
    print(f"Tasks with details: {sum(1 for task in current_data['tasks'].values() if task)}")
    
    # Print summary statistics for populated tasks
    categories = {}
    eval_methods = {}
    
    for task_info in current_data["tasks"].values():
        if task_info:  # Only count tasks with details
            cat = task_info.get('category', 'unknown')
            eval_method = task_info.get('evaluation_method', 'unknown')
            
            # Handle both string and list categories
            if isinstance(cat, list):
                for c in cat:
                    categories[c] = categories.get(c, 0) + 1
            else:
                categories[cat] = categories.get(cat, 0) + 1
            
            # Handle both string and list evaluation methods
            if isinstance(eval_method, list):
                for method in eval_method:
                    eval_methods[method] = eval_methods.get(method, 0) + 1
            else:
                eval_methods[eval_method] = eval_methods.get(eval_method, 0) + 1
    
    if categories:
        print("\nCategory distribution (populated tasks):")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
    
    if eval_methods:
        print("\nEvaluation method distribution (populated tasks):")
        for method, count in sorted(eval_methods.items()):
            print(f"  {method}: {count}")

if __name__ == "__main__":
    main() 