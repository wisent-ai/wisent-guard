"""
Wisent-Guard CLI interface for running benchmark evaluations.
This script allows running any lm-evaluation-harness benchmark with Wisent-Guard's hallucination detection.
"""

import argparse
from pathlib import Path
from wisent_guard.benchmarking.benchmark_runner import BenchmarkRunner
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function that parses arguments and runs benchmark evaluation.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Wisent-Guard Benchmark Evaluation Interface'
    )
    
    # Required arguments
    parser.add_argument(
        'benchmarks', 
        nargs='+',
        help='Benchmark names to run (e.g. truthfulqa_mc, hellaswag, mmlu)'
    )
    parser.add_argument(
        '--layer',
        type=int,
        required=True,
        help='Transformer layer number to monitor activations'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Model name (e.g. meta-llama/Llama-3.1-8B)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for computation (default: cuda)'
    )
    parser.add_argument(
        '--output-dir',
        default='benchmark_results',
        help='Directory to save evaluation results (default: benchmark_results)'
    )
    parser.add_argument(
        '--no-classifier',
        action='store_true',
        help='Skip classifier training and use threshold-based detection'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize BenchmarkRunner and run evaluation
    runner = BenchmarkRunner()
    
    results = runner.run_benchmark(
        benchmark_names=args.benchmarks,
        model_name=args.model,
        layer=args.layer,
        device=args.device,
        use_classifier=not args.no_classifier,
        output_dir=str(output_path)
    )
    
    # Save results to JSON file
    results_file = output_path / f"benchmark_results_{args.model.replace('/', '_')}_{args.layer}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display results in console
    print("\nBenchmark Results:")
    for benchmark, metrics in results.items():
        print(f"\nBenchmark: {benchmark}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

if __name__ == "__main__":
    main()
