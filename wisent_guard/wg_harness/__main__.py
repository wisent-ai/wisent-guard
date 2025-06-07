"""
Entry point for wisent-guard lm-evaluation-harness integration CLI.

Usage:
    python -m wisent_guard.wg_harness tasks hellaswag,mmlu --layer 15 --model meta-llama/Llama-3.1-8B
"""

from .cli import main

if __name__ == "__main__":
    main() 