"""
Wisent-Guard CLI interface for running benchmark evaluations.
This script allows running any lm-evaluation-harness benchmark with Wisent-Guard's hallucination detection.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmark(benchmark_names: list, layer: int, model: str, few_shot: bool = False):
    """
    Run benchmark evaluation with Wisent-Guard.
    
    Args:
        benchmark_names: List of benchmark names to run
        layer: Layer number to monitor
        model: Model name
        few_shot: Whether to use few-shot prompting
    """
    logger.info(f"Running benchmarks: {benchmark_names}")
    
    # Create output directory
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for benchmark in benchmark_names:
        logger.info(f"\nStarting benchmark: {benchmark}")
        
        # Run benchmark using subprocess
        result = subprocess.run([
            "python3",
            "run_benchmark.py",
            benchmark,
            model,
            str(layer)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error running benchmark {benchmark}: {result.stderr}")
            continue
            
        # Parse results
        try:
            result_data = json.loads(result.stdout)
            results[benchmark] = result_data
            
            # Log key metrics
            if "detection_rate" in result_data:
                logger.info(f"Detection rate: {result_data['detection_rate']:.2%}")
            if "accuracy" in result_data:
                logger.info(f"Accuracy: {result_data['accuracy']:.2%}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse results for {benchmark}")
            continue
    
    # Save combined results
    results_file = output_dir / "combined_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nEvaluation completed! Results saved to:")
    logger.info(str(results_file))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Wisent-Guard Benchmark Evaluation')
    
    # Required arguments
    parser.add_argument('benchmarks', nargs='+', help='Benchmark names to run (e.g. hellaswag, mmlu)')
    parser.add_argument('--layer', type=int, required=True, help='Layer number to monitor')
    parser.add_argument('--model', required=True, help='Model name')
    
    # Optional arguments
    parser.add_argument('--few-shot', action='store_true', help='Use few-shot prompting')
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmark(
        benchmark_names=args.benchmarks,
        layer=args.layer,
        model=args.model,
        few_shot=args.few_shot
    )

if __name__ == "__main__":
    main()
