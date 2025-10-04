#!/usr/bin/env python3

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def run_script(script_path):
    """Run a Python script and return the exit code."""
    print(f"\n[{datetime.now()}] Running {script_path}...")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=False
        )
        if result.returncode == 0:
            print(f"[{datetime.now()}] {script_path} completed successfully")
        else:
            print(f"[{datetime.now()}] {script_path} failed with exit code {result.returncode}")
        return result.returncode
    except Exception as e:
        print(f"[{datetime.now()}] Error running {script_path}: {e}")
        return -1

def main():
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    print("Starting search tasks...")
    print("=" * 50)

    scripts = [
        "tests/arithmetic/arithmetic_best.py",
        "tests/prost/prost_best.py",
        "tests/untruthfulness/untruthfulness_best.py"
    ]

    for script in scripts:
        run_script(script)

    print("\n" + "=" * 50)
    print(f"[{datetime.now()}] All tasks completed")

if __name__ == "__main__":
    main()
