"""
Task Manager for lm-evaluation-harness integration.

This module handles discovery, validation, and loading of tasks from the 
lm-evaluation-harness library.
"""

import json
import os
import re
import random
import yaml
import tempfile
import glob
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher


def load_available_tasks() -> List[str]:
    """Load available tasks from local tasks.json file or lm-eval registry."""
    
    # First try to load from local tasks.json file
    try:
        tasks_json_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "parameters", "tasks", "tasks.json")
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
    Load documents from the most appropriate split (validation → test → train → fewshot).
    
    Args:
        task: Task object from lm_eval
        limit: Optional limit on number of documents to load
        
    Returns:
        List of documents from the most appropriate split
    """
    docs = []
    
    # Try different doc sources in order of preference
    if task.has_validation_docs():
        docs = list(task.validation_docs())
    elif task.has_test_docs():
        docs = list(task.test_docs())
    elif task.has_training_docs():
        docs = list(task.training_docs())
    elif hasattr(task, 'has_fewshot_docs') and task.has_fewshot_docs():
        docs = list(task.fewshot_docs())
    else:
        # For tasks that use fewshot_split (like MMMLU), try to load from dataset directly
        if hasattr(task, 'dataset') and hasattr(task, 'fewshot_split'):
            try:
                from datasets import load_dataset
                dataset = load_dataset(
                    task.dataset_path if hasattr(task, 'dataset_path') else task.dataset_name,
                    task.dataset_config_name if hasattr(task, 'dataset_config_name') else None,
                    split=task.fewshot_split
                )
                docs = [dict(item) for item in dataset]
            except Exception as e:
                raise RuntimeError(f"No labelled docs available for task {task.NAME}. Error loading fewshot split: {e}")
        else:
            raise RuntimeError(f"No labelled docs available for task {task.NAME}")
    
    if limit is not None and limit > 0:
        docs = docs[:limit]
    
    return docs


def find_working_task_from_group(group_dict, max_depth=3, current_depth=0):
    """
    Recursively search through nested ConfigurableGroup structures to find a working individual task.
    
    Args:
        group_dict: Dictionary-like ConfigurableGroup object or regular dict
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth
        
    Returns:
        Tuple of (task_object, task_name) or (None, None) if no working task found
    """
    if current_depth >= max_depth:
        return None, None
    
    try:
        # Try to iterate through the group
        items = group_dict.items() if hasattr(group_dict, 'items') else []
        
        for key, value in items:
            # Skip nested ConfigurableGroup objects at first pass
            if hasattr(value, 'items') and 'ConfigurableGroup' in str(type(key)):
                continue
                
            # Check if this looks like an individual task
            if hasattr(value, 'has_validation_docs') or hasattr(value, 'has_test_docs') or hasattr(value, 'has_training_docs'):
                # Try to validate it has documents
                try:
                    has_docs = False
                    if hasattr(value, 'has_validation_docs') and value.has_validation_docs():
                        has_docs = True
                    elif hasattr(value, 'has_test_docs') and value.has_test_docs():
                        has_docs = True  
                    elif hasattr(value, 'has_training_docs') and value.has_training_docs():
                        has_docs = True
                    
                    if has_docs:
                        # Test if we can actually get documents
                        if hasattr(value, 'validation_docs') and value.has_validation_docs():
                            docs = list(value.validation_docs())
                        elif hasattr(value, 'test_docs') and value.has_test_docs():
                            docs = list(value.test_docs())
                        elif hasattr(value, 'training_docs') and value.has_training_docs():
                            docs = list(value.training_docs())
                        else:
                            docs = []
                            
                        if docs:
                            return value, str(key)
                except Exception:
                    # This task doesn't work, try next one
                    continue
        
        # If no individual tasks worked, try nested groups
        for key, value in items:
            if hasattr(value, 'items') and 'ConfigurableGroup' in str(type(key)):
                result_task, result_name = find_working_task_from_group(value, max_depth, current_depth + 1)
                if result_task is not None:
                    return result_task, result_name
                    
        return None, None
        
    except Exception as e:
        print(f"Error exploring group: {e}")
        return None, None


def handle_configurable_group_task(task_name: str):
    """
    Consolidated function to handle ConfigurableGroup tasks for both CLI and processing scripts.
    
    This function detects when a task is actually a ConfigurableGroup and finds a working
    individual task within it, handling nested groups up to 3 levels deep.
    Even handles tasks with lm-eval dependency issues by finding working alternatives.
    Also supports loading custom YAML task configurations.
    
    Args:
        task_name: Name of the potentially problematic group task
        
    Returns:
        Tuple of (working_task_object, actual_task_name) or raises ValueError if no working task found
    """
    try:
        from lm_eval.tasks import get_task_dict
    except ImportError as e:
        raise ImportError("lm-evaluation-harness is required. Install with: pip install lm-eval") from e
    
    print(f"🔍 Loading task: {task_name}")
    
    # First, try to load the task normally from the registry
    try:
        # Initialize TaskManager to ensure registry is populated
        from lm_eval.tasks import TaskManager as LMTaskManager
        task_manager = LMTaskManager()
        task_manager.initialize_tasks()
        
        task_dict = get_task_dict([task_name], task_manager=task_manager)
        if task_name in task_dict:
            task = task_dict[task_name]
            print(f"   ✅ Found {task_name} in registry")
            return task, task_name
    except Exception as e:
        print(f"   ⚠️  Registry loading failed: {e}")
        
        # Check if the task exists in the registry but has loading issues
        try:
            from lm_eval.tasks import TaskManager as LMTaskManager
            task_manager = LMTaskManager()
            task_manager.initialize_tasks()
            
            # Check in both individual tasks and groups
            all_tasks = getattr(task_manager, 'all_tasks', set())
            all_groups = getattr(task_manager, 'all_groups', set())
            
            print(f"   📊 Registry check: {len(all_tasks)} tasks, {len(all_groups)} groups available")
            print(f"   🔍 Is '{task_name}' in groups? {task_name in all_groups}")
            print(f"   🔍 Is '{task_name}' in tasks? {task_name in all_tasks}")
            
            if task_name in all_tasks or task_name in all_groups:
                print(f"   🔍 Task {task_name} exists in registry but has loading issues")
                
                # For group tasks, try to extract individual working tasks
                if task_name in all_groups:
                    print(f"   💡 Found {task_name} as a ConfigurableGroup - extracting individual tasks...")
                    result = try_extract_working_tasks_from_group(task_name, task_manager)
                    if result:
                        return result
                    else:
                        print(f"   💥 FAILED: Group {task_name} exists but no working tasks found!")
                        return None
                        
                # For individual tasks that fail loading, try aggressive search
                print(f"   💡 Found {task_name} as individual task - trying alternatives...")
                return try_find_related_working_task(task_name)
                    
            # If not found in registry at all, try aggressive search
            print(f"   🔄 Task {task_name} not found in registry, trying alternatives...")
            return try_find_related_working_task(task_name)
            
        except Exception as registry_error:
            print(f"   ⚠️  Registry check failed: {registry_error}")
            # Still try aggressive search as fallback
            return try_find_related_working_task(task_name)
    
    # If not found in registry, look for custom YAML configurations
    print(f"   🔍 Searching for custom YAML configuration for {task_name}")
    
    import os
    import glob
    
    # For specific custom tasks like flan_held_in, create the YAML files if needed
    if task_name == "flan_held_in":
        yaml_file_path = create_flan_held_in_files()
        if yaml_file_path:
            config_dir = os.path.dirname(yaml_file_path)
            print(f"   🔍 Loading flan_held_in from: {config_dir}")
            
            try:
                # Load using the proper config directory approach
                task_dict = load_task_with_config_dir(task_name, config_dir)
                
                if task_name in task_dict:
                    task = task_dict[task_name]
                    print(f"   ✅ Successfully loaded {task_name}")
                    return task, task_name
                
                # If the group task doesn't load directly, try to extract individual tasks
                print(f"   🔍 Extracting individual tasks from group...")
                individual_tasks = extract_individual_tasks_from_yaml(yaml_file_path, task_name)
                if individual_tasks:
                    print(f"   📋 Found individual tasks: {individual_tasks[:3]}...")
                    
                    for extracted_task_name in individual_tasks:
                        try:
                            individual_dict = load_task_with_config_dir(extracted_task_name, config_dir)
                            if extracted_task_name in individual_dict:
                                task = individual_dict[extracted_task_name]
                                print(f"   ✅ Successfully loaded individual task: {extracted_task_name}")
                                return task, extracted_task_name
                        except Exception as e:
                            print(f"      ❌ Failed to load {extracted_task_name}: {str(e)[:50]}")
                            continue
                            
            except Exception as e:
                print(f"   ❌ Failed to load flan_held_in: {e}")
    
    # Generic approach for other custom tasks
    # Look for existing YAML files in common directories
    yaml_candidates = []
    search_dirs = [
        "wisent_guard/parameters/tasks",
        ".",
        "tasks",
        "configs"
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            yaml_candidates.extend(glob.glob(os.path.join(search_dir, f"{task_name}.yaml")))
            yaml_candidates.extend(glob.glob(os.path.join(search_dir, f"{task_name}.yml")))
    
    # Try loading existing YAML files for the task
    for yaml_file in yaml_candidates:
        if os.path.exists(yaml_file):
            print(f"   🔍 Found YAML file: {yaml_file}")
            config_dir = os.path.dirname(yaml_file)
            
            try:
                task_dict = load_task_with_config_dir(task_name, config_dir)
                if task_name in task_dict:
                    task = task_dict[task_name]
                    print(f"   ✅ Successfully loaded {task_name}")
                    return task, task_name
                    
            except Exception as e:
                print(f"   ❌ Failed to load from {yaml_file}: {str(e)[:100]}")
    
    # If still not found, fall back to the original ConfigurableGroup handling logic
    print(f"   🔄 Falling back to ConfigurableGroup handling for {task_name}")
    
    # FIRST: Check if task exists in registry (for both individual tasks and groups)
    try:
        from lm_eval.tasks import TaskManager as LMTaskManager
        task_manager = LMTaskManager()
        task_manager.initialize_tasks()
        
        # Check in both individual tasks and groups
        all_tasks = getattr(task_manager, 'all_tasks', set())
        all_groups = getattr(task_manager, 'all_groups', set())
        
        # Convert to sets if they're lists, then merge
        if isinstance(all_tasks, list):
            all_tasks = set(all_tasks)
        if isinstance(all_groups, list):
            all_groups = set(all_groups)
        
        print(f"   📊 Registry check: {len(all_tasks)} tasks, {len(all_groups)} groups available")
        print(f"   🔍 Is '{task_name}' in groups? {task_name in all_groups}")
        print(f"   🔍 Is '{task_name}' in tasks? {task_name in all_tasks}")
        
        if task_name in all_tasks or task_name in all_groups:
            print(f"   🔍 Task {task_name} exists in registry but has loading issues")
            
            # For group tasks, try to extract individual working tasks
            if task_name in all_groups:
                print(f"   💡 Found {task_name} as a ConfigurableGroup - extracting individual tasks...")
                result = try_extract_working_tasks_from_group(task_name, task_manager)
                if result:
                    return result
                else:
                    print(f"   💥 FAILED: Group {task_name} exists but no working tasks found!")
                    return None
                    
            # For individual tasks that fail loading, try aggressive search
            print(f"   💡 Found {task_name} as individual task - trying alternatives...")
            return try_find_related_working_task(task_name)
                
        # If not found in registry at all, try aggressive search
        print(f"   🔄 Task {task_name} not found in registry, trying alternatives...")
        return try_find_related_working_task(task_name)
        
    except Exception as registry_error:
        print(f"   ⚠️  Registry check failed: {registry_error}")
        # Still try aggressive search as fallback
        return try_find_related_working_task(task_name)
    
    try:
        # Original logic for ConfigurableGroup tasks (should not reach here for known groups)
        task_dict = get_task_dict([task_name])
        if task_name not in task_dict:
            # Task doesn't exist, try aggressive search
            return try_find_related_working_task(task_name)
        
        task = task_dict[task_name]
        
        # Check if it's a ConfigurableGroup by examining the task object
        if hasattr(task, '__dict__') and isinstance(getattr(task, '__dict__', {}), dict):
            task_dict_items = getattr(task, '__dict__', {})
            
            # Look for ConfigurableGroup indicators
            if any(isinstance(v, dict) for v in task_dict_items.values()):
                print(f"   🎯 Detected ConfigurableGroup structure in {task_name}")
                
                # Try to find a working individual task within the group
                working_task = find_working_task_from_group(task_dict_items)
                if working_task:
                    return working_task
                    
        # If it's not a ConfigurableGroup or we couldn't find working tasks, 
        # try to use the task directly but handle potential dependency issues
        try:
            # Test if the task can load documents (quick validation)
            if hasattr(task, 'validation_docs'):
                docs = list(task.validation_docs())
                if docs:
                    print(f"   ✅ Task {task_name} works directly")
                    return task, task_name
            elif hasattr(task, 'test_docs'):
                docs = list(task.test_docs())
                if docs:
                    print(f"   ✅ Task {task_name} works directly")
                    return task, task_name
            elif hasattr(task, 'training_docs'):
                docs = list(task.training_docs())
                if docs:
                    print(f"   ✅ Task {task_name} works directly")
                    return task, task_name
                    
        except Exception as doc_error:
            print(f"   ⚠️  Task {task_name} has document loading issues: {doc_error}")
            
            # If there are dependency issues, try to find working alternatives
            return try_find_related_working_task(task_name)
        
        # If we get here, the task exists but has no usable documents
        print(f"   ⚠️  Task {task_name} has no usable documents")
        return try_find_related_working_task(task_name)
        
    except Exception as e:
        print(f"   ❌ Error handling {task_name}: {e}")
        # Try aggressive search for alternatives
        return try_find_related_working_task(task_name)


def extract_individual_tasks_from_yaml(yaml_file: str, group_name: str, _visited_files=None) -> List[str]:
    """
    Extract individual task names from a YAML configuration file.
    This function handles nested groups by recursively resolving group names.
    
    Args:
        yaml_file: Path to the YAML file
        group_name: Name of the group we're looking for
        _visited_files: Set of already visited files to prevent infinite recursion
        
    Returns:
        List of individual task names found in the YAML
    """
    try:
        import yaml
        import os
        
        # Initialize visited files set to prevent infinite recursion
        if _visited_files is None:
            _visited_files = set()
            
        # Check if we've already processed this file
        yaml_path_normalized = os.path.abspath(yaml_file)
        if yaml_path_normalized in _visited_files:
            print(f"   🔄 Cycle detected: {yaml_file} - skipping to prevent infinite recursion")
            return []
            
        _visited_files.add(yaml_path_normalized)
        
        with open(yaml_file, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        individual_tasks = []
        
        def extract_tasks_recursive(obj, depth=0):
            if depth > 5:  # Prevent infinite recursion
                return
                
            if isinstance(obj, dict):
                # Look for 'task' key which usually contains individual tasks
                if 'task' in obj:
                    task_value = obj['task']
                    if isinstance(task_value, str):
                        # Single task name - could be individual or group
                        individual_tasks.append(task_value)
                    elif isinstance(task_value, list):
                        # List of tasks or nested groups
                        for item in task_value:
                            extract_tasks_recursive(item, depth + 1)
                    elif isinstance(task_value, dict):
                        # Nested task definition
                        extract_tasks_recursive(task_value, depth + 1)
                
                # Also check other keys recursively
                for key, value in obj.items():
                    if key != 'task':  # Already processed above
                        extract_tasks_recursive(value, depth + 1)
                        
            elif isinstance(obj, list):
                for item in obj:
                    extract_tasks_recursive(item, depth + 1)
            elif isinstance(obj, str):
                # This is a task name (could be individual or group)
                individual_tasks.append(obj)
        
        extract_tasks_recursive(yaml_content)
        
        # Remove duplicates and filter out empty strings
        potential_tasks = list(set([task for task in individual_tasks if task and isinstance(task, str)]))
        
        print(f"   📋 Found potential tasks/groups: {potential_tasks[:5]}...")  # Limit output
        
        # Now we need to resolve any groups to their individual tasks
        resolved_tasks = []
        
        # Get the base directory for this YAML file to find related group files
        yaml_dir = os.path.dirname(yaml_file)
        
        # Limit to prevent excessive processing
        max_tasks_to_process = 5
        
        for i, task_name in enumerate(potential_tasks[:max_tasks_to_process]):
            # First check if this looks like an individual task (has specific suffixes)
            if any(suffix in task_name for suffix in ['_zeroshot_', '_fewshot_', '_cot_', '_prompt-', '_task_']):
                # This is likely an individual task
                resolved_tasks.append(task_name)
                continue
            
            # Check if this is a known group that we should resolve (limit recursion depth)
            if len(_visited_files) < 3:  # Limit recursion depth
                potential_group_file = os.path.join(yaml_dir, f"{task_name}.yaml")
                if os.path.exists(potential_group_file):
                    print(f"   🔍 Found nested group file: {os.path.basename(potential_group_file)}")
                    # Recursively extract from this group
                    nested_tasks = extract_individual_tasks_from_yaml(potential_group_file, task_name, _visited_files.copy())
                    resolved_tasks.extend(nested_tasks[:3])  # Limit results
                    continue
                    
                # Check in subdirectories (common pattern)
                for subdir in ['zeroshot', 'fewshot', 'cot']:
                    subdir_path = os.path.join(yaml_dir, task_name, subdir)
                    if os.path.isdir(subdir_path):
                        subdir_yaml = os.path.join(subdir_path, f"_{task_name}_{subdir}.yaml")
                        if os.path.exists(subdir_yaml):
                            print(f"   🔍 Found nested group in subdir: {subdir}")
                            nested_tasks = extract_individual_tasks_from_yaml(subdir_yaml, f"{task_name}_{subdir}", _visited_files.copy())
                            resolved_tasks.extend(nested_tasks[:3])  # Limit results
                            break
                else:
                    # Treat as individual task if we can't find a group file
                    resolved_tasks.append(task_name)
            else:
                # Max recursion depth reached, treat as individual task
                resolved_tasks.append(task_name)
        
        # Final cleanup - remove duplicates and limit results
        final_tasks = list(set(resolved_tasks))[:10]  # Limit to 10 tasks max
        
        print(f"   📋 Extracted individual tasks from YAML: {final_tasks}")
        return final_tasks
        
    except Exception as e:
        print(f"   ❌ Error extracting tasks from YAML {yaml_file}: {e}")
        return []


def try_find_related_working_task(task_name: str):
    """
    AGGRESSIVELY find related tasks that work when the main task has issues.
    This function will try EVERY possible variation to find a working task.
    NO TASK SHOULD BE SKIPPED!
    
    Args:
        task_name: The problematic task name
        
    Returns:
        Tuple of (task_object, task_name) or None if absolutely no alternatives found
    """
    try:
        from lm_eval.tasks import get_task_dict
        from lm_eval.tasks import TaskManager as LMTaskManager
        
        # Ensure TaskManager is properly initialized
        task_manager = LMTaskManager()
        task_manager.initialize_tasks()
        
        # Get all available tasks from the initialized manager
        all_tasks = getattr(task_manager, 'all_tasks', set())
        all_groups = getattr(task_manager, 'all_groups', set())
        
        # Convert to sets if they're lists, then merge
        if isinstance(all_tasks, list):
            all_tasks = set(all_tasks)
        if isinstance(all_groups, list):
            all_groups = set(all_groups)
        
        all_available_tasks = all_tasks | all_groups
        
        print(f"   📊 TaskManager has {len(all_tasks)} tasks, {len(all_groups)} groups")
        
        print(f"   🔄 AGGRESSIVE SEARCH for working alternatives to '{task_name}' ({len(all_available_tasks)} tasks available)...")
        
        # Strategy 1: Remove '_group' suffix
        if '_group' in task_name:
            base_name = task_name.replace('_group', '')
            print(f"   🎯 Trying base name: {base_name}")
            try:
                return handle_configurable_group_task(base_name)
            except:
                pass
        
        # Strategy 2: Try progressively shorter prefixes
        parts = task_name.split('_')
        if len(parts) > 1:
            for i in range(len(parts) - 1, 0, -1):
                parent_name = '_'.join(parts[:i])
                print(f"   🎯 Trying parent: {parent_name}")
                try:
                    return handle_configurable_group_task(parent_name)
                except:
                    continue
        
        # Strategy 3: Find ANY task with the same prefix (e.g., flan_held_in -> any flan_* task)
        prefix = parts[0] if parts else task_name
        print(f"   🎯 Searching for ANY task starting with '{prefix}_'...")
        
        matching_tasks = [t for t in all_available_tasks if t.startswith(prefix + '_') and t != task_name]
        
        # Try up to 10 matching tasks until we find one that works
        for candidate in matching_tasks[:10]:
            print(f"   🎯 Trying candidate: {candidate}")
            try:
                result = handle_configurable_group_task(candidate)
                print(f"   ✅ SUCCESS! Found working alternative: {candidate}")
                return result
            except:
                continue
        
        # Strategy 4: Try exact prefix match (e.g., flan_held_in -> flan)
        if prefix in all_available_tasks:
            print(f"   🎯 Trying exact prefix: {prefix}")
            try:
                return handle_configurable_group_task(prefix)
            except:
                pass
        
        # Strategy 5: Find tasks with similar keywords
        keywords = [part for part in parts if len(part) > 2]  # Skip short parts
        for keyword in keywords:
            print(f"   🎯 Searching for tasks containing '{keyword}'...")
            keyword_tasks = [t for t in all_available_tasks if keyword in t and t != task_name]
            
            for candidate in keyword_tasks[:5]:  # Try up to 5 per keyword
                print(f"   🎯 Trying keyword match: {candidate}")
                try:
                    result = handle_configurable_group_task(candidate)
                    print(f"   ✅ SUCCESS! Found working keyword match: {candidate}")
                    return result
                except:
                    continue
        
        # NO MORE STUPID FALLBACKS - FIX THE REAL ISSUE
        print(f"   💥 FAILED TO FIND CORRECT TASK: {task_name} - NO RANDOM FALLBACKS ALLOWED!")
        return None
        
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        return None


def try_extract_working_tasks_from_group(group_name: str, task_manager):
    """
    Try to extract and load individual working tasks from a problematic group.
    
    This handles cases like flan_held_in where the group exists in the registry
    but has loading issues (like yaml_path becoming None during include processing).
    
    Args:
        group_name: Name of the group (e.g., 'flan_held_in')
        task_manager: Initialized LM TaskManager instance
        
    Returns:
        Tuple of (task_object, task_name) or None if no working tasks found
    """
    try:
        from lm_eval.tasks import get_task_dict
        
        print(f"   🔍 Extracting working tasks from group: {group_name}")
        
        # Get the group configuration from the task manager
        if hasattr(task_manager, 'task_index') and group_name in task_manager.task_index:
            group_info = task_manager.task_index[group_name]
            yaml_path = group_info.get('yaml_path')
            
            if yaml_path and os.path.exists(yaml_path):
                print(f"   📁 Found group YAML: {yaml_path}")
                
                # Generic approach: parse the main YAML to extract task names
                
                # STEP 1: Try to parse the main group YAML for task names
                import yaml
                try:
                    with open(yaml_path, 'r') as f:
                        yaml_content = yaml.safe_load(f)
                    
                    # Extract task names from the main group YAML - more comprehensive search
                    initial_tasks = []
                    if isinstance(yaml_content, dict):
                        # Method 1: Direct 'task' field
                        if 'task' in yaml_content:
                            if isinstance(yaml_content['task'], list):
                                initial_tasks.extend(yaml_content['task'])
                            elif isinstance(yaml_content['task'], str):
                                initial_tasks.append(yaml_content['task'])
                        
                        # Method 2: Look for any list that might contain task names
                        for key, value in yaml_content.items():
                            if isinstance(value, list) and key not in ['metric_list', 'generation_kwargs', 'metadata']:
                                # Filter for task-like names (avoid metrics and config values)
                                for item in value:
                                    if isinstance(item, str) and ('_' in item or item.isalpha()):
                                        if item not in initial_tasks:
                                            initial_tasks.append(item)
                    
                    if initial_tasks:
                        print(f"   📋 Found {len(initial_tasks)} initial tasks from main YAML: {initial_tasks[:5]}...")
                        
                        # Try the initially found tasks directly
                        for task_name in initial_tasks[:15]:  # Try more tasks
                            try:
                                print(f"   🎯 Trying initial task: {task_name}")
                                result = get_task_dict([task_name], task_manager=task_manager)
                                if task_name in result:
                                    task = result[task_name]
                                    print(f"   ✅ SUCCESS: Found working initial task {task_name}")
                                    return task, task_name
                            except Exception as e:
                                print(f"      ❌ Initial task {task_name} failed: {str(e)[:50]}")
                                continue
                    else:
                        print(f"   ⚠️  No task names found in main YAML structure")
                    
                except Exception as yaml_parse_error:
                    print(f"   ⚠️  Main YAML parsing failed: {str(yaml_parse_error)[:100]}")
                
                # Fallback: try the recursive extraction method  
                try:
                    individual_tasks = extract_individual_tasks_from_yaml(yaml_path, group_name)
                    
                    if individual_tasks:
                        print(f"   📋 Found {len(individual_tasks)} individual tasks in group")
                        
                        # Try to load known working base tasks that these might be based on
                        base_tasks_to_try = []
                        
                        # Extract base task names (remove prompt suffixes)
                        for task in individual_tasks:
                            if '_prompt-' in task:
                                base_task = task.split('_prompt-')[0]
                                if base_task not in base_tasks_to_try:
                                    base_tasks_to_try.append(base_task)
                        
                        # Try the base tasks first
                        for base_task in base_tasks_to_try:
                            try:
                                print(f"   🎯 Trying base task: {base_task}")
                                result = get_task_dict([base_task], task_manager=task_manager)
                                if base_task in result:
                                    task = result[base_task]
                                    print(f"   ✅ SUCCESS: Found working base task {base_task}")
                                    return task, base_task
                            except Exception as e:
                                print(f"      ❌ Base task {base_task} failed: {str(e)[:50]}")
                                continue
                        
                        # If base tasks don't work, try some individual tasks (but skip templates/variables)
                        valid_tasks = [t for t in individual_tasks if not any(x in t for x in ['{{', '}}', '_common_yaml', 'sentence:'])]
                        for individual_task in valid_tasks[:5]:  # Try first 5 valid ones
                            try:
                                print(f"   🎯 Trying individual task: {individual_task}")
                                result = get_task_dict([individual_task], task_manager=task_manager)
                                if individual_task in result:
                                    task = result[individual_task]
                                    print(f"   ✅ SUCCESS: Found working individual task {individual_task}")
                                    return task, individual_task
                            except Exception as e:
                                print(f"      ❌ Individual task {individual_task} failed: {str(e)[:50]}")
                                continue
                                
                except Exception as yaml_error:
                    print(f"   ⚠️  YAML extraction failed (likely !function constructor): {str(yaml_error)[:100]}")
                    # Fall through to generic catch-all approach below
        
        # FINAL GENERIC CATCH-ALL: If all YAML approaches fail, search registry intelligently
        print(f"   🔍 FINAL CATCH-ALL: Searching registry for tasks matching group pattern...")
        
        # Search for tasks that contain the group name or parts of it
        all_tasks = getattr(task_manager, 'all_tasks', set())
        if isinstance(all_tasks, list):
            all_tasks = set(all_tasks)
            
        # Generate candidate task names based on the group name with smart filtering
        candidates = []
        
        # Strategy 1: Try exact group name
        if group_name in all_tasks:
            candidates.append(group_name)
            
        # Strategy 2: Try tasks that start with the group name
        group_prefix_tasks = [t for t in all_tasks if t.startswith(group_name + '_')]
        candidates.extend(group_prefix_tasks[:10])  # Limit to first 10
        
        # Strategy 3: Try tasks that contain all major parts of the group name
        group_parts = [part for part in group_name.split('_') if len(part) > 2]
        for part in group_parts:
            matching_tasks = [t for t in all_tasks if part in t and t not in candidates]
            # Prioritize exact matches and longer names
            matching_tasks.sort(key=lambda x: (part in x.split('_'), len(x)), reverse=True)
            candidates.extend(matching_tasks[:3])  # Top 3 per part
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                unique_candidates.append(candidate)
                seen.add(candidate)
                
        print(f"   📋 Found {len(unique_candidates)} candidate tasks to try...")
        
        # Try each candidate with intelligent prioritization
        for candidate in unique_candidates[:20]:  # Limit total attempts
            try:
                print(f"   🎯 Trying candidate: {candidate}")
                result = get_task_dict([candidate], task_manager=task_manager)
                if candidate in result:
                    task = result[candidate]
                    print(f"   ✅ SUCCESS: Found working candidate {candidate}")
                    return task, candidate
            except Exception as e:
                print(f"      ❌ Candidate {candidate} failed: {str(e)[:50]}")
                continue
        
        # If still no success, this group truly has no working tasks
        print(f"   💥 FAILED: Group {group_name} has no working tasks - exhausted all generic approaches")
        print(f"   ❌ No working tasks found in group {group_name}")
        return None
        
    except Exception as e:
        print(f"   ❌ Group extraction failed: {e}")
        return None


def save_custom_task_yaml(task_name: str, yaml_content: str) -> Optional[str]:
    """
    Save custom YAML task configuration to the tasks directory for future loading.
    
    Args:
        task_name: Name of the task
        yaml_content: YAML content to save
        
    Returns:
        Path to the saved file, or None if failed
    """
    try:
        # Create the tasks directory if it doesn't exist
        tasks_dir = os.path.join("wisent_guard", "parameters", "tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        
        # Save the YAML content to a file
        yaml_file_path = os.path.join(tasks_dir, f"{task_name}.yaml")
        with open(yaml_file_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"   💾 Saved custom task configuration to: {yaml_file_path}")
        return yaml_file_path
        
    except Exception as e:
        print(f"   ❌ Failed to save custom task configuration: {e}")
        return None





def create_task_yaml_from_user_content(task_name: str, user_yaml_content: str) -> Optional[str]:
    """
    Create a task YAML file from user-provided YAML content.
    This function can be called when users provide their own YAML configurations.
    
    Args:
        task_name: Name of the task
        user_yaml_content: YAML content provided by the user
        
    Returns:
        Path to the saved file, or None if failed
    """
    try:
        # Validate that the YAML is parseable
        yaml_data = yaml.safe_load(user_yaml_content)
        
        # Save the user's YAML content
        yaml_file_path = save_custom_task_yaml(f"{task_name}_user", user_yaml_content)
        
        if yaml_file_path:
            print(f"   💾 Saved user-provided YAML for {task_name}")
            return yaml_file_path
        
        return None
        
    except Exception as e:
        print(f"   ❌ Failed to process user YAML content: {e}")
        return None


def load_with_env_config(task_name: str, yaml_file: str):
    """
    Try to load a task by setting environment variables for lm_eval configuration.
    
    Args:
        task_name: Name of the task to load
        yaml_file: Path to the YAML configuration file
        
    Returns:
        Task dictionary from get_task_dict
    """
    try:
        from lm_eval.tasks import get_task_dict
        
        # Try setting various environment variables that lm_eval might use
        original_env = {}
        env_vars_to_set = [
            'LM_EVAL_CONFIG_PATH',
            'LM_EVAL_TASKS_PATH', 
            'LMEVAL_CONFIG_PATH',
            'TASK_CONFIG_PATH'
        ]
        
        # Save original environment
        for env_var in env_vars_to_set:
            original_env[env_var] = os.environ.get(env_var)
            os.environ[env_var] = yaml_file
        
        try:
            # Try to load the task with environment variables set
            return get_task_dict([task_name])
        finally:
            # Restore original environment
            for env_var in env_vars_to_set:
                if original_env[env_var] is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = original_env[env_var]
                    
    except Exception as e:
        raise Exception(f"Environment config loading failed: {e}")


def create_flan_held_in_files() -> Optional[str]:
    """
    Create the actual flan_held_in YAML files as provided by the user.
    This creates both the main file and the template file with proper include directives.
    
    Returns:
        Path to the main flan_held_in.yaml file, or None if failed
    """
    try:
        # Create the tasks directory
        tasks_dir = os.path.join("wisent_guard", "parameters", "tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        
        # Create the template file first
        template_content = """output_type: generate_until
test_split: null
doc_to_choice: null
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
generation_kwargs:
  until:
    - "</s>"
  do_sample: false
  temperature: 0.0
metadata:
  version: 1.0
"""
        
        template_path = os.path.join(tasks_dir, "_held_in_template_yaml.yaml")
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        # Create the main flan_held_in.yaml file with the exact content from the user
        main_content = """group: flan_held_in
group_alias: Flan (Held-In)
task:
  # ANLI R1
  - group: anli_r1_flan
    group_alias: ANLI R1
    aggregate_metric_list:
      - metric: acc
        weight_by_size: True
    task:
      - task: anli_r1_prompt-0
        task_alias: prompt-0
        include: _held_in_template_yaml
        doc_to_text: "{{premise}}\\n\\nChoose your answer: based on the paragraph above can we conclude that \\"{{hypothesis}}\\"?\\n\\nOPTIONS:\\n- Yes\\n- It's impossible to say\\n- No\\nI think the answer is"
        doc_to_target: "{{[\\"Yes\\", \\"It's impossible to say\\", \\"No\\"][label]}}"
      - task: anli_r1_prompt-1
        task_alias: prompt-1
        include: _held_in_template_yaml
        doc_to_text: "{{premise}}\\n\\nBased on that paragraph can we conclude that this sentence is true?\\n{{hypothesis}}\\n\\nOPTIONS:\\n- Yes\\n- It's impossible to say\\n- No"
        doc_to_target: "{{[\\"Yes\\", \\"It's impossible to say\\", \\"No\\"][label]}}"
      - task: anli_r1_prompt-2
        task_alias: prompt-2
        include: _held_in_template_yaml
        doc_to_text: "{{premise}}\\n\\nCan we draw the following conclusion?\\n{{hypothesis}}\\n\\nOPTIONS:\\n- Yes\\n- It's impossible to say\\n- No"
        doc_to_target: "{{[\\"Yes\\", \\"It's impossible to say\\", \\"No\\"][label]}}"
  # Arc Easy
  - group: arc_easy_flan
    group_alias: Arc Easy
    aggregate_metric_list:
      - metric: acc
        weight_by_size: True
    task:
      - task: arc_easy_prompt-0
        task_alias: prompt-0
        include: _held_in_template_yaml
        doc_to_text: "{{question}}\\n\\nOPTIONS:\\n- {{choices.text|join('\\n- ')}}"
        doc_to_target: "{{choices.text[choices.label.index(answerKey)]}}"
      - task: arc_easy_prompt-1
        task_alias: prompt-1
        include: _held_in_template_yaml
        doc_to_text: "Question: {{question}}\\nOPTIONS:\\n- {{choices.text|join('\\n- ')}}\\nAnswer:"
        doc_to_target: "{{choices.text[choices.label.index(answerKey)]}}"
  # BoolQ
  - group: boolq_flan
    group_alias: BoolQ
    aggregate_metric_list:
      - metric: acc
        weight_by_size: True
    task:
      - task: boolq_prompt-0
        task_alias: prompt-0
        include: _held_in_template_yaml
        doc_to_text: "{{passage}}\\n\\nCan we conclude that {{question}}?\\n\\nOPTIONS:\\n- no\\n- yes"
        doc_to_target: "{{['no', 'yes'][label]}}"
      - task: boolq_prompt-1
        task_alias: prompt-1
        include: _held_in_template_yaml
        doc_to_text: "{{passage}}\\n\\nIs it true that {{question}}?\\n\\nOPTIONS:\\n- no\\n- yes"
        doc_to_target: "{{['no', 'yes'][label]}}"
"""
        
        main_path = os.path.join(tasks_dir, "flan_held_in.yaml")
        with open(main_path, 'w') as f:
            f.write(main_content)
        
        print(f"   💾 Created flan_held_in YAML files:")
        print(f"      📄 Template: {template_path}")
        print(f"      📄 Main: {main_path}")
        
        return main_path
        
    except Exception as e:
        print(f"   ❌ Failed to create flan_held_in files: {e}")
        return None


def load_task_with_config_dir(task_name: str, config_dir: str):
    """
    Load a task by setting the lm_eval configuration directory.
    This attempts to load YAML configurations by manipulating the path and environment.
    
    Args:
        task_name: Name of the task to load
        config_dir: Directory containing YAML configuration files
        
    Returns:
        Task dictionary from get_task_dict
    """
    try:
        from lm_eval.tasks import get_task_dict
        from lm_eval.tasks import TaskManager as LMTaskManager
        import sys
        
        print(f"      🔧 Attempting to load {task_name} from config dir: {config_dir}")
        
        # Method 1: Try to use TaskManager if available
        try:
            # Check if LMTaskManager has config path functionality
            task_manager = LMTaskManager()
            if hasattr(task_manager, 'initialize_tasks') or hasattr(task_manager, 'load_config'):
                print(f"      🔧 Using TaskManager approach")
                return get_task_dict([task_name], task_manager=task_manager)
        except Exception as e:
            print(f"      ⚠️ TaskManager approach failed: {e}")
        
        # Method 2: Try adding config directory to Python path
        original_path = sys.path[:]
        try:
            if config_dir not in sys.path:
                sys.path.insert(0, config_dir)
            print(f"      🔧 Added config dir to Python path")
            return get_task_dict([task_name])
        except Exception as e:
            print(f"      ⚠️ Python path approach failed: {e}")
        finally:
            sys.path[:] = original_path
        
        # Method 3: Try setting environment variables
        original_env = {}
        env_vars = ['LM_EVAL_CONFIG_DIR', 'LMEVAL_CONFIG_PATH', 'TASK_CONFIG_PATH']
        try:
            for env_var in env_vars:
                original_env[env_var] = os.environ.get(env_var)
                os.environ[env_var] = config_dir
            print(f"      🔧 Set environment variables")
            return get_task_dict([task_name])
        except Exception as e:
            print(f"      ⚠️ Environment variable approach failed: {e}")
        finally:
            for env_var in env_vars:
                if original_env[env_var] is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = original_env[env_var]
        
        # Method 4: Fall back to basic loading
        print(f"      🔧 Falling back to basic task loading")
        return get_task_dict([task_name])
                
    except Exception as e:
        raise Exception(f"Config directory loading failed: {e}")


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
    
    def load_task(self, task_name: str, limit: Optional[int] = None):
        """
        Load a task from lm-evaluation-harness with dynamic task name resolution.
        Supports both regular tasks and ConfigurableGroup tasks.
        
        Args:
            task_name: Name of the task
            limit: Optional limit on number of documents
            
        Returns:
            Task object from lm_eval
        """
        
        # Find the actual task name dynamically
        actual_task_name = self.resolve_task_name(task_name)
        
        try:
            # First try to handle as potentially problematic ConfigurableGroup task
            task, _ = handle_configurable_group_task(actual_task_name)
            task._limit = limit
            return task
            
        except Exception as e:
            # If that fails, check if it's a task resolution issue
            if not self.is_valid_task(actual_task_name):
                raise ValueError(
                    f"Task '{task_name}' could not be resolved to a valid task. "
                    f"Use get_available_tasks() to see all available tasks."
                )
            
            # Re-raise the original error if it wasn't a resolution issue
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

    def register_custom_task_yaml(self, task_name: str, yaml_content: str) -> bool:
        """
        Register a custom YAML task configuration that can be loaded later.
        
        Args:
            task_name: Name of the task to register
            yaml_content: YAML content defining the task
            
        Returns:
            True if successfully registered, False otherwise
            
        Example:
            yaml_content = '''
            my_custom_task:
              class: custom_task
              doc_to_text: "Question: {{question}}"
              doc_to_target: "{{answer}}"
            '''
            manager.register_custom_task_yaml("my_custom_task", yaml_content)
        """
        try:
            yaml_file_path = create_task_yaml_from_user_content(task_name, yaml_content)
            if yaml_file_path:
                print(f"✅ Registered custom task configuration for '{task_name}'")
                print(f"   📁 Saved to: {yaml_file_path}")
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to register custom task '{task_name}': {e}")
            return False


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