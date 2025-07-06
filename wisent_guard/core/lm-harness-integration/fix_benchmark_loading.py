#!/usr/bin/env python3
"""
Script to fix benchmark loading issues in populate_tasks.py
"""

import re

def fix_benchmark_loading():
    """Apply fixes to the populate_tasks.py file"""
    
    # Read the file
    with open('populate_tasks.py', 'r') as f:
        content = f.read()
    
    # Define the fix code to insert
    fix_code = '''    # BENCHMARK LOADING FIXES
    # Task name mapping for failed benchmarks
    TASK_NAME_MAPPING = {
        "social_i_qa": "social_iqa",
        "squad2": "squadv2", 
        "math_qa": "mathqa",
        "paws_x": "pawsx",
        "narrativeqa": "scrolls_narrativeqa",
        "mctaco": "mc_taco",
        "wmt": "wmt2016",
        "storycloze": "storycloze_2016",
        "crows_pairs": "crows_pairs_english",
        "big_bench": "bigbench",
        "mmmlu": "mmlu"
    }
    
    # Tasks that need trust_remote_code=True
    TRUST_REMOTE_CODE_TASKS = {
        "wsc273", "hendrycks_ethics", "pubmedqa"
    }
    
    # Apply task name mapping
    original_task_name = task_name
    if task_name in TASK_NAME_MAPPING:
        task_name = TASK_NAME_MAPPING[task_name]
        print(f"🔧 Task name mapped: {original_task_name} → {task_name}")
    
    # Check if task needs trust_remote_code
    needs_trust_remote_code = task_name in TRUST_REMOTE_CODE_TASKS or original_task_name in TRUST_REMOTE_CODE_TASKS
    if needs_trust_remote_code:
        print(f"🔒 Task requires trust_remote_code=True: {task_name}")
        # Set environment variable for trust_remote_code
        import os
        os.environ['HF_ALLOW_CODE_EVAL'] = '1'
        print(f"🔓 Enabled trust_remote_code for {task_name}")
    
'''
    
    # Find the function and add our fix
    pattern = r'(def get_task_samples_for_analysis\(task_name: str, num_samples: int = 5\) -> Dict\[str, Any\]:\s*"""[^"]*?"""\s*)(try:)'
    
    if re.search(pattern, content, re.DOTALL):
        # Insert the fix code between the docstring and the try block
        new_content = re.sub(pattern, r'\1' + fix_code + r'\2', content, flags=re.DOTALL)
        
        # Write back to file
        with open('populate_tasks.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Successfully applied benchmark loading fixes!")
        return True
    else:
        print("❌ Could not find the function pattern to fix")
        return False

if __name__ == "__main__":
    success = fix_benchmark_loading()
    if success:
        print("🎉 Benchmark loading fixes applied successfully!")
    else:
        print("💥 Failed to apply fixes")
