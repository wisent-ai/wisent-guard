#!/usr/bin/env python
"""
Simple CLI for Wisent Guard LM-Evaluation Harness integration.

Usage:
    python wisent_cli.py tasks hellaswag,mmlu layer 15 model meta-llama/Llama-3.1-8B
"""

import sys
import subprocess
import os


def main():
    if len(sys.argv) < 7:
        print("Usage: python wisent_cli.py tasks <task_list> layer <layer_num> model <model_name>")
        print("Example: python wisent_cli.py tasks hellaswag,mmlu layer 15 model meta-llama/Llama-3.1-8B")
        sys.exit(1)
    
    # Parse arguments
    if sys.argv[1] != "tasks":
        print("Error: First argument must be 'tasks'")
        sys.exit(1)
    
    tasks = sys.argv[2]
    
    if sys.argv[3] != "layer":
        print("Error: Third argument must be 'layer'")
        sys.exit(1)
    
    layer = sys.argv[4]
    
    if sys.argv[5] != "model":
        print("Error: Fifth argument must be 'model'")
        sys.exit(1)
    
    model = sys.argv[6]
    
    # Build the command to run the evaluation script
    script_path = os.path.join(os.path.dirname(__file__), "..", "evaluation", "evaluate_lm_eval_harness.py")
    
    cmd = [
        "python", script_path,
        "--tasks", tasks,
        "--layer", layer,
        "--model", model
    ]
    
    print(f"🦬 Running Wisent Guard evaluation...")
    print(f"Tasks: {tasks}")
    print(f"Layer: {layer}")
    print(f"Model: {model}")
    print("-" * 50)
    
    # Execute the evaluation script
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Evaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 