"""
Main entry point for wisent-guard CLI.

This module routes CLI commands to appropriate submodules.
"""

import sys
import argparse

def main():
    """Main CLI entry point with subcommand routing."""
    
    # If no arguments, show help
    if len(sys.argv) < 2:
        print("Usage: python -m wisent_guard <subcommand> [args...]")
        print("\nAvailable subcommands:")
        print("  tasks    - Run lm-evaluation benchmarks through wisent-guard pipeline")
        print("\nExamples:")
        print("  python -m wisent_guard tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B")
        print("  python -m wisent_guard tasks hellaswag,mmlu --layer 10 --shots 5")
        sys.exit(1)
    
    subcommand = sys.argv[1]
    
    if subcommand == "tasks":
        # Import and run the integrated CLI
        from .cli import main as cli_main
        # Remove the subcommand from sys.argv so the CLI parser works correctly
        sys.argv = [sys.argv[0]] + ["tasks"] + sys.argv[2:]
        cli_main()
    else:
        print(f"Unknown subcommand: {subcommand}")
        print("Available subcommands: tasks")
        sys.exit(1)

if __name__ == "__main__":
    main() 