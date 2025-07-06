import re

with open('populate_tasks.py', 'r') as f:
    lines = f.readlines()

# Find the line with the function definition
for i, line in enumerate(lines):
    if 'def get_task_samples_for_analysis(task_name: str, num_samples: int = 5) -> Dict[str, Any]:' in line:
        # Find the end of the docstring
        docstring_start = i
        docstring_end = i
        in_docstring = False
        
        for j in range(i, len(lines)):
            if '"""' in lines[j]:
                if not in_docstring:
                    in_docstring = True
                else:
                    docstring_end = j
                    break
        
        # Insert the fix code after the docstring
        fix_lines = [
            '    # BENCHMARK LOADING FIXES\n',
            '    # Task name mapping for failed benchmarks\n',
            '    TASK_NAME_MAPPING = {\n',
            '        "social_i_qa": "social_iqa",\n',
            '        "squad2": "squadv2", \n',
            '        "math_qa": "mathqa",\n',
            '        "paws_x": "pawsx",\n',
            '        "narrativeqa": "scrolls_narrativeqa",\n',
            '        "mctaco": "mc_taco",\n',
            '        "wmt": "wmt2016",\n',
            '        "storycloze": "storycloze_2016",\n',
            '        "crows_pairs": "crows_pairs_english",\n',
            '        "big_bench": "bigbench",\n',
            '        "mmmlu": "mmlu"\n',
            '    }\n',
            '    \n',
            '    # Tasks that need trust_remote_code=True\n',
            '    TRUST_REMOTE_CODE_TASKS = {\n',
            '        "wsc273", "hendrycks_ethics", "pubmedqa"\n',
            '    }\n',
            '    \n',
            '    # Apply task name mapping\n',
            '    original_task_name = task_name\n',
            '    if task_name in TASK_NAME_MAPPING:\n',
            '        task_name = TASK_NAME_MAPPING[task_name]\n',
            '        print(f"🔧 Task name mapped: {original_task_name} → {task_name}")\n',
            '    \n',
            '    # Check if task needs trust_remote_code\n',
            '    needs_trust_remote_code = task_name in TRUST_REMOTE_CODE_TASKS or original_task_name in TRUST_REMOTE_CODE_TASKS\n',
            '    if needs_trust_remote_code:\n',
            '        print(f"🔒 Task requires trust_remote_code=True: {task_name}")\n',
            '        # Set environment variable for trust_remote_code\n',
            '        import os\n',
            '        os.environ[\'HF_ALLOW_CODE_EVAL\'] = \'1\'\n',
            '        print(f"🔓 Enabled trust_remote_code for {task_name}")\n',
            '    \n'
        ]
        
        # Insert the fix lines
        lines[docstring_end+1:docstring_end+1] = fix_lines
        break

# Write back to file
with open('populate_tasks.py', 'w') as f:
    f.writelines(lines)

print("✅ Applied benchmark loading fixes!")
