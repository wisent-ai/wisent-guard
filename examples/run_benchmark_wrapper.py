import argparse
import subprocess
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARKS = [
    "truthfulqa_mc",
    "hellaswag",
    "mmlu"
]

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
    output_path = "benchmark_results"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Save results to JSON file
    with open(os.path.join(output_path, "all_benchmarks_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("All benchmarks completed and results saved")

if __name__ == "__main__":
    main()
