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

from wisent_guard.core.utils.device import preferred_dtype, resolve_default_device, resolve_device

def find_working_task_from_group(group_dict: Dict, depth: int = 0, max_depth: int = 3) -> Any:
    """
    Recursively search through a ConfigurableGroup to find a task with usable documents.
    
    Args:
        group_dict: Dictionary containing tasks or nested groups
        depth: Current recursion depth (for preventing infinite loops)
        max_depth: Maximum recursion depth allowed
    
    Returns:
        A task object with usable documents, or None if none found
    """
    if depth > max_depth:
        print(f"   ⚠️  Maximum recursion depth ({max_depth}) reached, stopping search")
        return None
    
    # Get all items and randomize order
    items = list(group_dict.items())
    random.shuffle(items)
    
    for item_name, item in items[:3]:  # Try up to 3 items at each level
        indent = "   " + "  " * depth
        print(f"{indent}🔍 Checking '{item_name}'...")
        
        # Check if this is another nested group (dict-like object)
        if hasattr(item, 'items') and callable(item.items):
            print(f"{indent}📦 '{item_name}' is a nested group, going deeper...")
            nested_task = find_working_task_from_group(item, depth + 1, max_depth)
            if nested_task:
                print(f"{indent}✅ Found working task in '{item_name}'")
                return nested_task
            else:
                print(f"{indent}❌ No working tasks in '{item_name}', continuing...")
                continue
        
        # Check if this individual task has documents
        try:
            has_docs = False
            test_docs = []
            
            if hasattr(item, 'validation_docs') and item.has_validation_docs():
                test_docs = list(item.validation_docs())
                if test_docs:
                    has_docs = True
            elif hasattr(item, 'test_docs') and item.has_test_docs():
                test_docs = list(item.test_docs())
                if test_docs:
                    has_docs = True
            elif hasattr(item, 'training_docs') and item.has_training_docs():
                test_docs = list(item.training_docs())
                if test_docs:
                    has_docs = True
            
            if has_docs:
                print(f"{indent}✅ Found working task '{item_name}' with {len(test_docs)} documents")
                return item
            else:
                print(f"{indent}❌ '{item_name}' has no documents")
                
        except Exception as e:
            print(f"{indent}❌ '{item_name}' failed: {e}")
            continue
    
    return None

def get_benchmark_tags_with_llama(task_name: str, readme_content: str = "") -> List[str]:
    """
    Use Llama-3.1B-Instruct to determine appropriate tags for a benchmark.
    
    Args:
        task_name: Name of the benchmark
        readme_content: README content describing the benchmark
        
    Returns:
        List of 3 most appropriate tags
    """
    
    # Load approved tags from skills.json and risks.json  
    approved_skills = [
        "coding", "mathematics", "long context", "creative writing", 
        "general knowledge", "medical", "law", "science", "history", 
        "tool use", "multilingual", "reasoning"
    ]
    
    approved_risks = [
        "harmfulness", "toxicity", "bias", "hallucination", "violence", 
        "adversarial robustness", "sycophancy", "deception"
    ]
    
    print(f"   🤖 Using Llama-3.1B-Instruct to determine tags for '{task_name}'...")
    
    try:
        # Use the same pipeline approach as generate_tags.py
        from transformers import pipeline
        import torch
        
        print(f"   🔄 Loading Llama-3.1-8B-Instruct pipeline...")
        
        device_kind = resolve_default_device()
        device_obj = resolve_device(device_kind)
        if device_kind == "cuda" and torch.cuda.is_available():
            print("   🚀 Using CUDA device")
        elif device_kind == "mps":
            print("   📱 Using MPS device")
        else:
            print("   💻 Using CPU device")

        torch_dtype = preferred_dtype(device_kind)
        device_map = "auto" if device_kind == "cuda" else None
        if device_kind == "cuda":
            pipeline_device = 0
        elif device_kind == "mps":
            pipeline_device = device_obj
        else:
            pipeline_device = -1
        
        # Initialize the pipeline like in generate_tags.py
        generator = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch_dtype,
            device_map=device_map,
            device=pipeline_device,
            max_new_tokens=1000,
            temperature=0.3,
            do_sample=True,
            pad_token_id=50256
        )
        
        print(f"   ✅ Successfully loaded Llama-3.1-8B-Instruct pipeline")
        
        # Create a focused prompt for tag determination
        description = readme_content[:1500] if readme_content else f"A benchmark called '{task_name}' for evaluating language models."
        
        user_prompt = f"""Analyze the benchmark and determine exactly 3 tags.

Benchmark: {task_name}
Description: {description}

Available tags:
Skills: {', '.join(approved_skills)}  
Risks: {', '.join(approved_risks)}

Instructions:
1. Analyze what this benchmark actually tests
2. Choose EXACTLY 3 tags that best describe what is being evaluated
3. Focus on the primary capabilities/risks being measured
4. Output only the 3 tags, one per line, no explanations

Tags:"""

        # Format prompt using Llama 3.1 chat template
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in AI evaluation benchmarks analyzing benchmark tasks to determine what specific cognitive abilities they test.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        print("   🧠 Analyzing with Llama...")
        response = generator(formatted_prompt, max_new_tokens=800, temperature=0.3)
        
        # Extract just the assistant's response
        full_response = response[0]['generated_text']
        generated_text = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        print(f"   🎯 LLM Response: {generated_text}")
        
        # Parse the generated tags
        all_approved_tags = approved_skills + approved_risks
        lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
        determined_tags = []
        
        for line in lines[:5]:  # Look at first 5 lines
            clean_line = line.strip('- *123456789.').strip()
            
            # Check if it matches an approved tag
            for tag in all_approved_tags:
                if tag.lower() == clean_line.lower() or clean_line.lower() in tag.lower():
                    if tag not in determined_tags:
                        determined_tags.append(tag)
                        break
        
        # Ensure we have exactly 3 tags
        if len(determined_tags) < 3:
            fallback_tags = ["reasoning", "general knowledge", "science"]
            for fallback in fallback_tags:
                if fallback not in determined_tags:
                    determined_tags.append(fallback)
                if len(determined_tags) >= 3:
                    break
        
        # Limit to exactly 3 tags
        determined_tags = determined_tags[:3]
        
        print(f"   ✅ Final LLM-determined tags: {determined_tags}")
        return determined_tags
        
    except Exception as e:
        print(f"   ❌ Error using LLM: {e}")
        print(f"   🔄 Falling back to basic analysis...")
        
        # Fallback to basic content analysis
        if readme_content:
            content_lower = readme_content.lower()
            determined_tags = []
            
            # Basic keyword matching as fallback
            if any(word in content_lower for word in ["math", "arithmetic", "calculation"]):
                determined_tags.append("mathematics")
            if any(word in content_lower for word in ["code", "programming", "python"]):
                determined_tags.append("coding")
            if any(word in content_lower for word in ["medical", "health", "clinical"]):
                determined_tags.append("medical")
            if any(word in content_lower for word in ["adversarial", "robust", "challenging"]):
                determined_tags.append("adversarial robustness")
            if any(word in content_lower for word in ["bias", "fairness", "stereotype"]):
                determined_tags.append("bias")
            if any(word in content_lower for word in ["truthful", "hallucination", "factual"]):
                determined_tags.append("hallucination")
            if any(word in content_lower for word in ["multilingual", "cross-lingual"]):
                determined_tags.append("multilingual")
            
            # Fill with defaults
            if "reasoning" not in determined_tags:
                determined_tags.append("reasoning")
            if len(determined_tags) < 3 and "general knowledge" not in determined_tags:
                determined_tags.append("general knowledge")
            if len(determined_tags) < 3:
                determined_tags.append("science")
            
            return determined_tags[:3]
        
        # Final fallback
        return ["reasoning", "general knowledge", "science"]


def get_benchmark_groups_from_readme(task_name: str) -> Dict[str, Any]:
    """
    Read the README from lm-eval-harness repository to get benchmark groups and use LLM for tags.
    
    Args:
        task_name: Name of the benchmark (e.g., "superglue")
    
    Returns:
        Dictionary with group names and LLM-determined tags
    """
    import requests
    import re
    
    # Map common task names to their directory names in lm-eval-harness
    task_dir_map = {
        'superglue': 'super_glue',
        'super_glue': 'super_glue',
        'super-glue': 'super_glue',
        'glue': 'glue',
        'mmlu': 'mmlu',
        'truthfulqa': 'truthfulqa',
        'hellaswag': 'hellaswag',
        'arc': 'arc',
        'winogrande': 'winogrande'
    }
    
    task_dir = task_dir_map.get(task_name.lower(), task_name.lower())
    
    # URL for README in lm-eval-harness repository
    readme_url = f"https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/{task_dir}/README.md"
    
    try:
        print(f"   📖 Fetching README from: {readme_url}")
        response = requests.get(readme_url, timeout=10)
        response.raise_for_status()
        readme_content = response.text
        
        # Use LLM to determine tags from README content
        determined_tags = get_benchmark_tags_with_llama(task_name, readme_content)
        
        # Parse groups from README (keeping the existing group extraction logic)
        groups = []
        
        # Look for "Groups" or "Tags" section and extract group names
        lines = readme_content.split('\n')
        in_target_section = False
        
        for line in lines:
            line = line.strip()
            
            # Check if we're in the Groups or Tags section
            if (line.lower().startswith('groups') or line.lower().startswith('## groups') or 
                line.lower().startswith('tags') or line.lower().startswith('## tags') or
                line.lower().startswith('#### groups') or line.lower().startswith('#### tags')):
                in_target_section = True
                continue
            
            # Stop at next section (but not if it's a subsection)
            if in_target_section and line.startswith('##') and not any(x in line.lower() for x in ['groups', 'tags']):
                break
            
            # Extract group names (lines that start with task names or contain colons)
            if in_target_section and line:
                # Look for patterns like "* `super-glue-lm-eval-v1`: Description"
                if line.startswith('* `') and '`' in line and ':' in line:
                    # Extract between backticks
                    start = line.find('`') + 1
                    end = line.find('`', start)
                    if start > 0 and end > start:
                        group_name = line[start:end].strip()
                        if group_name and '-' in group_name:  # Looks like a group task name
                            groups.append(group_name)
                # Also look for patterns like "super-glue-lm-eval-v1: Description"
                elif ':' in line and not line.startswith('#'):
                    group_name = line.split(':')[0].strip()
                    # Remove any leading bullets or markdown
                    group_name = group_name.lstrip('*- `').rstrip('`')
                    if group_name and '-' in group_name:  # Looks like a group task name
                        groups.append(group_name)
        
        # Remove duplicates and filter valid group names
        groups = list(set(groups))
        valid_groups = [g for g in groups if g and len(g) > 3]
        
        print(f"   📋 Extracted {len(valid_groups)} groups from README: {valid_groups}")
        print(f"   🏷️  Determined tags from README content: {determined_tags}")
        
        return {
            'groups': valid_groups,
            'tags': determined_tags
        }
        
    except Exception as e:
        print(f"   ❌ Failed to fetch README: {e}")
        return {'groups': [], 'tags': []}


def get_samples_from_group_task(group_name: str, subtasks: List[str], num_samples: int = 5) -> Dict[str, Any]:
    """
    Get samples from a group task by sampling from its subtasks.
    
    Args:
        group_name: Name of the group task (e.g., "glue")
        subtasks: List of subtask names (e.g., ["cola", "sst2", "mrpc"])
        num_samples: Total number of samples to retrieve across all subtasks
    
    Returns:
        Dictionary containing samples from multiple subtasks
    """
    print(f"🎯 Getting samples from group task '{group_name}' with {len(subtasks)} subtasks...")
    
    all_samples = []
    samples_per_task = max(1, num_samples // len(subtasks))  # At least 1 sample per task
    
    # Import lm_eval
    from lm_eval import evaluator
    
    # Get samples from each subtask
    for i, subtask in enumerate(subtasks[:num_samples]):  # Limit to prevent too many subtasks
        try:
            print(f"   📊 Getting samples from subtask {i+1}/{min(len(subtasks), num_samples)}: '{subtask}'...")
            
            # Get the subtask directly  
            task_dict = evaluator.get_task_dict([subtask])
            if subtask not in task_dict:
                print(f"   ⚠️  Subtask '{subtask}' not found, skipping...")
                continue
                
            task = task_dict[subtask]
            
            # Get documents from this subtask
            docs = []
            if hasattr(task, 'validation_docs') and task.has_validation_docs():
                docs = list(task.validation_docs())
            elif hasattr(task, 'test_docs') and task.has_test_docs():
                docs = list(task.test_docs())
            elif hasattr(task, 'training_docs') and task.has_training_docs():
                docs = list(task.training_docs())
            
            if not docs:
                print(f"   ⚠️  No documents found for subtask '{subtask}', skipping...")
                continue
            
            # Sample documents from this subtask
            sample_docs = docs[:samples_per_task] if len(docs) >= samples_per_task else docs
            
            for j, doc in enumerate(sample_docs):
                sample = {"sample_id": len(all_samples) + 1, "subtask": subtask}
                
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
                    else:
                        sample["format"] = "open_ended"
                except Exception as e:
                    sample["choices"] = []
                    sample["format"] = "unknown"
                
                all_samples.append(sample)
                
                # Stop if we have enough samples total
                if len(all_samples) >= num_samples:
                    break
                    
        except Exception as e:
            print(f"   ❌ Error processing subtask '{subtask}': {e}")
            continue
        
        # Stop if we have enough samples total
        if len(all_samples) >= num_samples:
            break
    
    if not all_samples:
        return {
            "task_name": group_name,
            "error": f"No samples could be retrieved from any subtasks of {group_name}",
            "samples": []
        }
    
    print(f"✅ Successfully retrieved {len(all_samples)} samples from {group_name} group task")
    
    # Get description from the group context
    subtask_names = list(set([sample["subtask"] for sample in all_samples]))
    description = f"Group task '{group_name}' containing subtasks: {', '.join(subtask_names)}"
    
    return {
        "task_name": group_name,
        "description": description,
        "samples": all_samples[:num_samples],  # Ensure we don't exceed requested number
        "num_subtasks": len(subtasks),
        "sampled_subtasks": subtask_names
    }

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
        from lm_eval import evaluator
        
        print(f"🔍 Loading task: {task_name}")
        
        # Step 1: Try individual task first
        try:
            task_dict = evaluator.get_task_dict([task_name])
            expanded_tasks = list(task_dict.keys())
            
            if len(expanded_tasks) == 1:
                # Individual task
                print(f"✅ Found individual task: {task_name}")
                task = task_dict[task_name]
                resolved_name = task_name
            elif len(expanded_tasks) > 1:
                # Group task
                print(f"✅ Found group task '{task_name}' with {len(expanded_tasks)} subtasks: {expanded_tasks[:5]}{'...' if len(expanded_tasks) > 5 else ''}")
                return get_samples_from_group_task(task_name, expanded_tasks, num_samples)
            else:
                raise ValueError("No tasks returned")
                
        except Exception as e:
            # Step 2: Try as group task (different API)
            print(f"⚠️  Individual task failed, trying as group task...")
            try:
                # Try to get it as a group that might expand to more tasks
                subtasks = expand_group_task(task_name, evaluator.get_task_dict)
                if subtasks:
                    print(f"✅ Found group expansion with {len(subtasks)} subtasks: {subtasks[:3]}{'...' if len(subtasks) > 3 else ''}")
                    return get_samples_from_group_task(task_name, subtasks, num_samples)
                else:
                    raise ValueError("No group expansion found")
                    
            except Exception as e2:
                # Step 3: Try as group of groups (large size)
                print(f"⚠️  Group task failed, trying as large group...")
                try:
                    # For very large groups, try different expansion methods
                    from lm_eval.tasks import TaskManager
                    tm = TaskManager()
                    tm.initialize_tasks()
                    
                    # Check if it's in the groups registry
                    all_groups = getattr(tm, 'all_groups', set())
                    if task_name in all_groups:
                        print(f"✅ Found in groups registry, attempting large expansion...")
                        # Try to expand this large group
                        expanded_dict = evaluator.get_task_dict([task_name])
                        if expanded_dict:
                            all_subtasks = list(expanded_dict.keys())
                            if len(all_subtasks) > 1:
                                print(f"✅ Large group expansion: {len(all_subtasks)} subtasks")
                                return get_samples_from_group_task(task_name, all_subtasks, num_samples)
                    
                    # Final failure
                    raise ValueError(f"Task '{task_name}' not found at any level")
                    
                except Exception as e3:
                    # Step 4: Try to find benchmark groups by reading README from lm-eval-harness
                    print(f"⚠️  Large group failed, trying to read README for '{task_name}'...")
                    try:
                        readme_data = get_benchmark_groups_from_readme(task_name)
                        benchmark_groups = readme_data.get('groups', [])
                        readme_tags = readme_data.get('tags', [])
                        
                        if benchmark_groups:
                            print(f"✅ Found {len(benchmark_groups)} groups from README: {benchmark_groups}")
                            print(f"🏷️  README-determined tags: {readme_tags}")
                            
                            # Execute all groups specified in the README
                            all_samples = []
                            for group_name in benchmark_groups:
                                print(f"   🔄 Processing README group: {group_name}")
                                try:
                                    group_task_dict = evaluator.get_task_dict([group_name])
                                    group_subtasks = list(group_task_dict.keys())
                                    group_samples = get_samples_from_group_task(group_name, group_subtasks, num_samples // len(benchmark_groups))
                                    if 'samples' in group_samples:
                                        all_samples.extend(group_samples['samples'])
                                except Exception as ge:
                                    print(f"   ❌ Failed to process {group_name}: {ge}")
                            
                            if all_samples:
                                return {
                                    "task_name": task_name,
                                    "resolved_groups": benchmark_groups,
                                    "readme_tags": readme_tags,
                                    "samples": all_samples[:num_samples],  # Limit to requested number
                                    "total_groups": len(benchmark_groups)
                                }
                        
                        # Final failure
                        raise ValueError(f"No README groups found for '{task_name}'")
                        
                    except Exception as e4:
                        return {
                            "task_name": task_name,
                            "error": f"Task '{task_name}' not found: individual ({e}), group ({e2}), large group ({e3}), README ({e4})",
                            "samples": []
                        }
        
        # Get task description
        description = getattr(task, 'DESCRIPTION', getattr(task, '__doc__', f"Task: {task_name}"))
        
        # Get documents with robust error handling for lm-eval dependency issues
        docs = []
        try:
            if hasattr(task, 'validation_docs') and task.has_validation_docs():
                docs = list(task.validation_docs())
            elif hasattr(task, 'test_docs') and task.has_test_docs():
                docs = list(task.test_docs())
            elif hasattr(task, 'training_docs') and task.has_training_docs():
                docs = list(task.training_docs())
        except Exception as e:
            error_msg = str(e)
            
            # These errors should now be caught by the aggressive search in handle_configurable_group_task
            # If we still get them here, it means the search failed to find alternatives
            print(f"⚠️  Document retrieval error for {task_name}: {error_msg}")
            
            # Return error so the task manager can try to find alternatives
            if "utils" in error_msg and "has no attribute" in error_msg:
                return {
                    "task_name": task_name,
                    "description": description,
                    "error": f"Task has internal lm-eval dependency issue: {error_msg}",
                    "samples": []
                }
            elif "expected str, bytes or os.PathLike object, not NoneType" in error_msg:
                return {
                    "task_name": task_name,
                    "description": description,
                    "error": f"Task has internal lm-eval configuration issue: {error_msg}",
                    "samples": []
                }
            elif "module" in error_msg and "has no attribute" in error_msg:
                return {
                    "task_name": task_name,
                    "description": description,
                    "error": f"Task has missing dependency: {error_msg}",
                    "samples": []
                }
            else:
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

def get_relevant_benchmarks_for_prompt(prompt: str, max_benchmarks: int = 1, existing_model=None) -> List[Dict[str, Any]]:
    """
    Use Llama-3.1B-Instruct to determine the most relevant benchmarks for testing a given prompt.
    
    Args:
        prompt: The prompt to analyze (e.g., "I like food")
        max_benchmarks: Maximum number of benchmarks to return (default: 3)
        
    Returns:
        List of dictionaries containing benchmark names and relevance explanations
    """
    
    # Load the benchmark list from only_benchmarks.py or create a basic one
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        # Try to import from only_benchmarks.py
        try:
            from only_benchmarks import BENCHMARKS
            available_benchmarks = list(BENCHMARKS.keys())
            benchmark_descriptions = {name: f"{name}: {', '.join(info['tags'])}" for name, info in BENCHMARKS.items()}
        except ImportError:
            # Fallback to basic benchmark list
            available_benchmarks = [
                "mmlu", "truthfulqa_mc1", "hellaswag", "arc_easy", "arc_challenge",
                "winogrande", "piqa", "boolq", "copa", "rte", "wsc", "wic", 
                "multirc", "record", "drop", "squad", "coqa", "humaneval", 
                "mbpp", "math", "gsm8k", "toxigen", "winobias", "stereoset"
            ]
            benchmark_descriptions = {name: f"{name}: benchmark for language model evaluation" for name in available_benchmarks}
    except:
        # Final fallback
        available_benchmarks = [
            "mmlu", "truthfulqa_mc1", "hellaswag", "arc_easy", "arc_challenge",
            "winogrande", "piqa", "boolq", "copa", "rte"
        ]
        benchmark_descriptions = {name: f"{name}: benchmark for language model evaluation" for name in available_benchmarks}
    
    print(f"🎯 Analyzing prompt to find most relevant benchmarks: '{prompt}'")
    print(f"📊 Available benchmarks: {len(available_benchmarks)}")
    
    try:
        # Use the same pipeline approach as get_benchmark_tags_with_llama
        from transformers import pipeline
        import torch
        
        print(f"   🔄 Loading Llama-3.1-8B-Instruct pipeline...")
        
        device_kind = resolve_default_device()
        device_obj = resolve_device(device_kind)
        if device_kind == "cuda" and torch.cuda.is_available():
            print("   🚀 Using CUDA device")
        elif device_kind == "mps":
            print("   📱 Using MPS device")
        else:
            print("   💻 Using CPU device")

        torch_dtype = preferred_dtype(device_kind)
        device_map = "auto" if device_kind == "cuda" else None
        if device_kind == "cuda":
            pipeline_device = 0
        elif device_kind == "mps":
            pipeline_device = device_obj
        else:
            pipeline_device = -1

        # Initialize the pipeline
        generator = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch_dtype,
            device_map=device_map,
            device=pipeline_device,
            max_new_tokens=1000,
            temperature=0.3,
            do_sample=True,
            pad_token_id=50256
        )
        
        print(f"   ✅ Successfully loaded Llama-3.1-8B-Instruct pipeline")
        
        # Create analysis prompt
        benchmark_list = "\n".join([f"- {name}: {desc}" for name, desc in benchmark_descriptions.items()])
        
        user_prompt = f"""Analyze this prompt and determine which benchmarks would be most relevant for testing it.

Prompt to analyze: "{prompt}"

Available benchmarks:
{benchmark_list}

Instructions:
1. Analyze what cognitive abilities, knowledge, or skills this prompt would test
2. Match the prompt's requirements to the most relevant benchmarks
3. Choose the top {max_benchmarks} benchmarks that would best evaluate this type of prompt
4. Provide a brief explanation for each choice

Format your response as:
1. [benchmark_name]: [explanation]
2. [benchmark_name]: [explanation]
3. [benchmark_name]: [explanation]

Top {max_benchmarks} most relevant benchmarks:"""

        # Format prompt using Llama 3.1 chat template
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in AI evaluation benchmarks. Your task is to analyze prompts and determine which benchmarks would be most relevant for testing them.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        print("   🧠 Analyzing with Llama...")
        if existing_model is not None:
            # Use existing model for generation
            response, _ = existing_model.generate(formatted_prompt, layer_index=15, max_new_tokens=800)
            generated_text = response.strip()
        else:
            # Use pipeline
            response = generator(formatted_prompt, max_new_tokens=800, temperature=0.3)
            
            # Extract just the assistant's response
            full_response = response[0]['generated_text']
            generated_text = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        print(f"   🎯 LLM Response: {generated_text}")
        
        # Parse the response to extract benchmarks and explanations
        relevant_benchmarks = []
        lines = generated_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not any(char.isdigit() for char in line[:3]):
                continue
                
            # Look for pattern like "1. benchmark_name: explanation"
            if ':' in line:
                # Extract benchmark name and explanation
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    # Clean up benchmark name (remove numbering, brackets, etc.)
                    benchmark_part = parts[0].strip()
                    explanation = parts[1].strip()
                    
                    # Extract actual benchmark name
                    import re
                    # Remove numbering like "1.", "2.", etc.
                    benchmark_part = re.sub(r'^\d+\.?\s*', '', benchmark_part)
                    # Remove brackets
                    benchmark_part = re.sub(r'[\[\]]', '', benchmark_part)
                    benchmark_name = benchmark_part.strip()
                    
                    # Check if this benchmark name exists in our available benchmarks
                    matched_benchmark = None
                    for available in available_benchmarks:
                        if available.lower() == benchmark_name.lower():
                            matched_benchmark = available
                            break
                        elif benchmark_name.lower() in available.lower() or available.lower() in benchmark_name.lower():
                            matched_benchmark = available
                            break
                    
                    if matched_benchmark:
                        relevant_benchmarks.append({
                            'benchmark': matched_benchmark,
                            'explanation': explanation,
                            'relevance_score': len(relevant_benchmarks) + 1  # Higher score for earlier mentions
                        })
                        
                        if len(relevant_benchmarks) >= max_benchmarks:
                            break
        
        # If we didn't find enough benchmarks, add fallback ones
        if len(relevant_benchmarks) < max_benchmarks:
            fallback_benchmarks = ["mmlu", "truthfulqa_mc1", "hellaswag"]
            for fallback in fallback_benchmarks:
                if fallback in available_benchmarks and not any(rb['benchmark'] == fallback for rb in relevant_benchmarks):
                    relevant_benchmarks.append({
                        'benchmark': fallback,
                        'explanation': f"General purpose benchmark suitable for testing various prompts",
                        'relevance_score': len(relevant_benchmarks) + 1
                    })
                    if len(relevant_benchmarks) >= max_benchmarks:
                        break
        
        print(f"   ✅ Found {len(relevant_benchmarks)} relevant benchmarks")
        for i, rb in enumerate(relevant_benchmarks, 1):
            print(f"   {i}. {rb['benchmark']}: {rb['explanation']}")
        
        return relevant_benchmarks[:max_benchmarks]
        
    except Exception as e:
        print(f"   ❌ Error using LLM: {e}")
        print(f"   🔄 Falling back to basic analysis...")
        
        # Fallback to basic keyword matching
        prompt_lower = prompt.lower()
        fallback_results = []
        
        # Basic keyword-based matching
        if any(word in prompt_lower for word in ["food", "eat", "cook", "recipe", "restaurant"]):
            fallback_results.append({
                'benchmark': 'mmlu',
                'explanation': 'General knowledge benchmark that may include food-related questions',
                'relevance_score': 1
            })
        
        if any(word in prompt_lower for word in ["math", "calculate", "number", "equation"]):
            fallback_results.append({
                'benchmark': 'math' if 'math' in available_benchmarks else 'mmlu',
                'explanation': 'Mathematical reasoning benchmark',
                'relevance_score': 1
            })
        
        if any(word in prompt_lower for word in ["code", "program", "python", "function"]):
            fallback_results.append({
                'benchmark': 'humaneval' if 'humaneval' in available_benchmarks else 'mmlu',
                'explanation': 'Code generation benchmark',
                'relevance_score': 1
            })
        
        # Fill with general benchmarks if needed
        general_benchmarks = ["mmlu", "truthfulqa_mc1", "hellaswag"]
        for gb in general_benchmarks:
            if gb in available_benchmarks and not any(fr['benchmark'] == gb for fr in fallback_results):
                fallback_results.append({
                    'benchmark': gb,
                    'explanation': 'General purpose benchmark for language understanding',
                    'relevance_score': len(fallback_results) + 1
                })
                if len(fallback_results) >= max_benchmarks:
                    break
        
        return fallback_results[:max_benchmarks]


def test_prompt_benchmark_matching(test_prompt: str = "I like food"):
    """Test the prompt-to-benchmark matching function."""
    print(f"🧪 Testing prompt-to-benchmark matching")
    print(f"Test prompt: '{test_prompt}'")
    print("=" * 50)
    
    results = get_relevant_benchmarks_for_prompt(test_prompt)
    
    print("\n📊 Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. **{result['benchmark']}**")
        print(f"   Explanation: {result['explanation']}")
        print(f"   Relevance Score: {result['relevance_score']}")
        print()
    
    return results

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

def test_specific_task():
    """Test the specific problematic task for error handling."""
    
    print("\n🧪 Testing specific problematic tasks...")
    
    # Test both types of problematic tasks
    test_tasks = [
        'evalita-mp_ner_fic_group',  # Dependency issue
        'flan_held_in'               # Configuration issue
    ]
    
    for task_name in test_tasks:
        print(f"\nTesting task: {task_name}")
        
        try:
            result = get_task_samples_for_analysis(task_name, num_samples=3)
            
            print(f"Result keys: {list(result.keys())}")
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                if 'skip_reason' in result:
                    print(f"📝 Skip reason: {result['skip_reason']}")
                    print("✅ Error was handled gracefully with skip reason!")
                else:
                    print("⚠️  No skip reason provided")
            else:
                print(f"✅ Success: Retrieved {len(result.get('samples', []))} samples")
                
        except Exception as e:
            print(f"💥 Exception raised: {e}")
            import traceback
            traceback.print_exc()

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
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_specific_task()
    else:
        main() 