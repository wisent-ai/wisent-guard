#!/usr/bin/env python3
"""
Systematic processing script for improving cognitive trait tags on all 6,888 tasks.
Uses Llama 3.1B Instruct to analyze benchmark samples and generate specific cognitive traits.
"""

import json
import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path

# Set environment variables to automatically trust remote code for datasets
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings

from generate_tags import generate_cognitive_tags

def load_tasks(filename: str = "tasks.json") -> Dict[str, Any]:
    """Load tasks from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return {}
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return {}

def save_tasks(tasks_data: Dict[str, Any], filename: str = "tasks.json"):
    """Save tasks to JSON file with backup."""
    # Create backup first
    backup_filename = f"{filename}.backup.{int(time.time())}"
    try:
        with open(filename, 'r') as f:
            with open(backup_filename, 'w') as backup:
                backup.write(f.read())
        print(f"✅ Backup created: {backup_filename}")
    except:
        print("⚠️  Could not create backup")
    
    # Save updated file
    try:
        with open(filename, 'w') as f:
            json.dump(tasks_data, f, indent=2)
        print(f"✅ Saved updated tasks to {filename}")
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")

def process_all_tasks(start_index: int = 0, save_every: int = 1):
    """
    Process all tasks starting from start_index.
    
    Args:
        start_index: Task index to start from (for resuming)
        save_every: Save progress every N tasks
    """
    
    print("🚀 Starting systematic tag improvement for all tasks...")
    
    # Load existing tasks
    tasks_data = load_tasks()
    if not tasks_data or 'tasks' not in tasks_data:
        print("❌ No tasks found in tasks.json")
        return
    
    tasks = tasks_data['tasks']
    task_names = list(tasks.keys())
    total_tasks = len(task_names)
    
    print(f"📊 Found {total_tasks} tasks")
    print(f"▶️  Starting from index {start_index}")
    print(f"💾 Will save progress every {save_every} tasks")
    print("=" * 60)
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    updated_count = 0
    start_time = time.time()
    
    for i in range(start_index, total_tasks):
        task_name = task_names[i]
        current_task = tasks[task_name]
        
        print(f"\n[{i+1}/{total_tasks}] Processing: {task_name}")
        print(f"   Current tags: {', '.join(current_task.get('tags', []))}")
        
        # Generate new tags using AI - FAIL HARD ON ANY ERROR
        result = generate_cognitive_tags(task_name, num_samples=3)
        
        if "error" in result:
            error_msg = result['error']
            print(f"   ❌ FATAL ERROR: {error_msg}")
            print(f"   💥 FAILING HARD - TASK MUST BE FIXED!")
            raise RuntimeError(f"Task '{task_name}' failed: {error_msg}")
        
        # Get the top 3 tags
        new_tags = result.get('generated_tags', [])
        
        if not new_tags:
            print(f"   ❌ FATAL ERROR: No tags generated!")
            print(f"   💥 FAILING HARD - TASK MUST GENERATE TAGS!")
            raise RuntimeError(f"Task '{task_name}' generated no tags")
        
        # Update the task with new tags (keep quality score unchanged)
        old_tags = current_task.get('tags', [])
        current_task['tags'] = new_tags
        
        print(f"   ✅ Updated tags: {', '.join(new_tags)}")
        
        # Show relevance scores if available
        if result.get('all_tags_with_scores'):
            top_scores = result['all_tags_with_scores'][:3]
            score_info = ", ".join([f"{tag}({score})" for tag, score in top_scores])
            print(f"   📊 Scores: {score_info}")
        
        updated_count += 1
        processed_count += 1
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.5)
        
        # Save progress periodically
        if processed_count % save_every == 0:
            elapsed = time.time() - start_time
            remaining_tasks = total_tasks - (i + 1)
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = remaining_tasks / rate if rate > 0 else 0
            
            print(f"\n💾 Saving progress... ({processed_count} tasks processed)")
            print(f"⏱️  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min, Rate: {rate*60:.1f}/min")
            save_tasks(tasks_data)
            print(f"📊 Stats: {updated_count} updated, {error_count} errors, {skipped_count} skipped")
    
    # Final save
    print(f"\n🎉 Completed processing all tasks!")
    print(f"📊 Final stats:")
    print(f"   • Tasks processed: {processed_count}")
    print(f"   • Tags updated: {updated_count}")
    print(f"   • Errors: {error_count}")
    print(f"   • Skipped: {skipped_count}")
    
    save_tasks(tasks_data)

def resume_processing(filename: str = "tasks.json"):
    """Resume processing from where we left off by checking which tasks still have generic tags."""
    
    tasks_data = load_tasks(filename)
    if not tasks_data or 'tasks' not in tasks_data:
        print("❌ No tasks found")
        return 0
    
    tasks = tasks_data['tasks']
    
    # Look for tasks with generic tags that need updating
    generic_tags = {'question-answering', 'factuality', 'safety', 'knowledge', 'reasoning'}
    
    for i, (task_name, task_data) in enumerate(tasks.items()):
        current_tags = set(task_data.get('tags', []))
        if current_tags.intersection(generic_tags):
            print(f"📍 Found first task needing update at index {i}: {task_name}")
            print(f"   Current tags: {', '.join(current_tags)}")
            return i
    
    print("✅ All tasks appear to have been processed already!")
    return len(tasks)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python process_all_tasks.py process [start_index] [save_every]")
        print("  python process_all_tasks.py resume")
        print("Examples:")
        print("  python process_all_tasks.py process 0 1     # Start from beginning, save every task")
        print("  python process_all_tasks.py process 100 5  # Start from task 100, save every 5") 
        print("  python process_all_tasks.py resume          # Resume from where we left off")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "process":
        start_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        save_every = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        process_all_tasks(start_index, save_every)
        
    elif command == "resume":
        start_index = resume_processing()
        if start_index < 6888:  # If there are tasks to process
            save_every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            process_all_tasks(start_index, save_every)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1) 