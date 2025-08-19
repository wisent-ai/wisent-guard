"""
Main entry point for wisent-guard CLI.

This module routes CLI commands to appropriate submodules.
"""

import sys


def main():
    """Main CLI entry point with subcommand routing."""

    # If no arguments, show help
    if len(sys.argv) < 2:
        print("Usage: python -m wisent_guard <subcommand> [args...]")
        print("\nAvailable subcommands:")
        print("  tasks                 - Run lm-evaluation benchmarks through wisent-guard pipeline")
        print("  generate-pairs        - Generate synthetic contrastive pairs from trait descriptions")
        print("  synthetic             - Run synthetic pair generation and evaluation pipeline")
        print("  test-nonsense         - Run test-nonsense command")
        print("  agent                 - Interact with autonomous agent")
        print("  model-config          - Manage model-specific optimal parameters")
        print("  optimize-classification - Optimize classification parameters across all tasks")
        print("  optimize-steering     - Optimize steering parameters for different methods")
        print("  optimize-sample-size  - Find optimal training sample size for classifiers")
        print("  full-optimize         - Run full optimization: classification, steering, and sample size")
        print("  generate-vector       - Generate steering vectors from contrastive pairs (file or description)")
        print("  multi-steer           - Combine multiple steering vectors dynamically at inference time")
        print("  evaluate              - Evaluate single prompt with steering vector and return quality scores")
        print("\nExamples:")
        print("  python -m wisent_guard tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B-Instruct")
        print("  python -m wisent_guard generate-pairs --trait 'refuse harmful requests' --output pairs.json")
        print("  python -m wisent_guard synthetic --trait 'be helpful and honest' --steering-method KSteering")
        print("  python -m wisent_guard optimize-classification meta-llama/Llama-3.1-8B-Instruct --limit 500")
        print(
            "  python -m wisent_guard optimize-steering compare-methods meta-llama/Llama-3.1-8B-Instruct --task truthfulqa_mc1"
        )
        print("  python -m wisent_guard generate-vector --from-description 'be helpful and honest' --output vector.pt")
        print(
            "  python -m wisent_guard evaluate --vector vector.pt --prompt 'What is your view?' --model llama --trait helpful"
        )
        sys.exit(1)

    subcommand = sys.argv[1]

    if subcommand in [
        "tasks",
        "generate-pairs",
        "synthetic",
        "test-nonsense",
        "agent",
        "model-config",
        "optimize-classification",
        "optimize-steering",
        "optimize-sample-size",
        "full-optimize",
        "generate-vector",
        "multi-steer",
        "evaluate",
    ]:
        # Import and run the integrated CLI
        from .cli import main as cli_main

        # Remove the subcommand from sys.argv so the CLI parser works correctly
        sys.argv = [sys.argv[0]] + [subcommand] + sys.argv[2:]
        cli_main()
    else:
        print(f"Unknown subcommand: {subcommand}")
        print(
            "Available subcommands: tasks, generate-pairs, synthetic, test-nonsense, agent, model-config, optimize-classification, optimize-steering, optimize-sample-size, full-optimize, generate-vector, multi-steer, evaluate"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
