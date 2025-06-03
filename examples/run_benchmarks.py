"""
CLI script to run multiple benchmarks with Wisent-Guard hallucination detection.
"""

import argparse
import json
import logging
from pathlib import Path
from wisent_guard.benchmark_runner import BenchmarkRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks with Wisent-Guard hallucination detection')
    
    # Required arguments
    parser.add_argument('benchmarks', nargs='+', help='Benchmark names to run')
    parser.add_argument('--layer', type=int, required=True, help='Layer number to monitor')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--device', default='mps', choices=['mps', 'cpu'], help='Device to run on (mps for M2 Macs)')
    
    # Optional arguments
    parser.add_argument('--output-dir', default='benchmark_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize and run benchmarks
    runner = BenchmarkRunner()
    results = runner.run_benchmark(
        benchmark_names=args.benchmarks,
        model_name=args.model,
        layer=args.layer,
        device=args.device
    )
    
    # Save results
    results_file = output_dir / f"results_{args.model.replace('/', '_')}_{args.layer}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    for benchmark, metrics in results.items():
        logger.info(f"\nBenchmark: {benchmark}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Detection rate: {metrics['detection_rate']:.4f}")
        logger.info(f"Hallucinations caught: {metrics['hallucinations_caught']}")
        logger.info(f"Total hallucinations: {metrics['total_hallucinations']}")

if __name__ == "__main__":
    main()
