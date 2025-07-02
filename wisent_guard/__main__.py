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
        print("  tasks          - Run lm-evaluation benchmarks through wisent-guard pipeline")
        print("  generate-pairs - Generate synthetic contrastive pairs from trait descriptions")
        print("  synthetic      - Run synthetic pair generation and evaluation pipeline")
        print("  test-nonsense  - Run test-nonsense command")
        print("  agent          - Interact with autonomous agent")
        print("\nExamples:")
        print("  python -m wisent_guard tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B-Instruct")
        print("  python -m wisent_guard generate-pairs --trait 'refuse harmful requests' --output pairs.json")
        print("  python -m wisent_guard synthetic --trait 'be helpful and honest' --steering-method KSteering")
        sys.exit(1)
    
    subcommand = sys.argv[1]
    
    if subcommand in ["tasks", "generate-pairs", "synthetic", "test-nonsense", "agent"]:
        # Import and run the integrated CLI
        from .cli import main as cli_main
        # Remove the subcommand from sys.argv so the CLI parser works correctly
        sys.argv = [sys.argv[0]] + [subcommand] + sys.argv[2:]
        cli_main()
    else:
        print(f"Unknown subcommand: {subcommand}")
        print("Available subcommands: tasks, generate-pairs, synthetic, test-nonsense, agent")
        sys.exit(1)

if __name__ == "__main__":
    main() 