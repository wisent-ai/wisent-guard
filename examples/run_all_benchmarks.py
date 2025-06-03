"""
Script to run Wisent-Guard evaluation on multiple benchmarks.
This script automates running benchmark evaluations across different datasets and combines the results.
"""

import subprocess
import json
from pathlib import Path
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Benchmark configuration
BENCHMARKS = [
    "truthfulqa_mc",    # TruthfulQA multiple choice
    "hellaswag",       # HellaSWAG
    "mmlu"            # Massive Multilingual Language Understanding
]

# Model configuration
MODEL = "meta-llama/Llama-3.1-8B"
LAYER = 15

def run_benchmark(benchmark: str) -> str:
    """
    Run a single benchmark evaluation.
    
    Args:
        benchmark: Name of the benchmark to run
        
    Returns:
        str: Benchmark results as JSON string
    """
    logger.info(f"Running benchmark: {benchmark}")
    
    # Run benchmark using subprocess
    result = subprocess.run([
        "python3",
        "run_benchmark.py",
        benchmark,
        MODEL,
        str(LAYER)
    ], capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        logger.error(f"Error running benchmark {benchmark}: {result.stderr}")
        return None
    
    return result.stdout

def main():
    """
    Main function that runs all benchmarks and combines results.
    """
    all_results = {}
    
    # Run each benchmark
    for benchmark in BENCHMARKS:
        result = run_benchmark(benchmark)
        if result:
            all_results[benchmark] = result
    
    # Save combined results
    output_path = Path("benchmark_results")
    output_path.mkdir(exist_ok=True)
    
    # Save results to JSON file
    with open(output_path / "all_benchmarks_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("All benchmarks completed and results saved")

if __name__ == "__main__":
    main()
