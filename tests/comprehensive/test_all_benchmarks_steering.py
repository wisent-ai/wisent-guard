#!/usr/bin/env python3
"""
Test steering pipeline for all benchmarks with hybrid CPU-MPS solution.
Mimics the cli_steering_test.py approach we tested.
"""

import json
import logging
import subprocess
import time
from pathlib import Path

from wisent_guard.cli_bricks.cli_benchmarks import AVAILABLE_BENCHMARKS

# Configuration matching our MPS fix testing
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 11  # Same as cli_steering_test
LIMIT = 10
TOKEN_STRATEGY = "max_pooling"
STEERING_METHOD = "CAA"
STEERING_STRENGTH = 1.0

def test_steering_benchmark(task_name: str, verbose: bool = False):
    """Test steering on a single benchmark using cli_steering_test.py."""
    
    try:
        # Use cli_steering_test.py directly
        cmd = [
            "python", "wisent_guard/cli_bricks/cli_tests/cli_steering_test.py",
            "--model", MODEL,
            "--tasks", task_name,
            "--layer", str(LAYER),
            "--limit", str(LIMIT),
            "--training-limit", str(LIMIT),
            "--testing-limit", "5",
            "--split-ratio", "0.7",
            "--seed", "2024",
            "--token-target", TOKEN_STRATEGY,
            "--method", STEERING_METHOD,
            "--steering-strength", str(STEERING_STRENGTH),
            "--use-test-split",
            "--output-mode", "metrics"
        ]
        
        logging.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
            # No timeout
        )
        
        if result.returncode == 0:
            # Try to parse metrics from output
            output = result.stdout
            metrics = {}
            
            # Look for accuracy or other metrics in output
            for line in output.split('\n'):
                if 'accuracy' in line.lower() or 'score' in line.lower():
                    try:
                        import re
                        match = re.search(r'(accuracy|score)[:\s]+([0-9.]+)', line, re.IGNORECASE)
                        if match:
                            metrics[match.group(1).lower()] = float(match.group(2))
                    except:
                        pass
            
            return {
                "status": "success", 
                "metrics": metrics,
                "output": output[-1000:] if len(output) > 1000 else output
            }
        else:
            raise Exception(f"Command failed with return code {result.returncode}: {result.stderr}")
        
    except Exception as e:
        logging.exception(f"Error in steering test for {task_name}:")
        return {"status": "failed", "error": str(e)}

def test_steering_all_benchmarks(stop_on_error=True, max_failures=5):
    """Test steering on all benchmarks with hybrid CPU-MPS mode.
    
    Args:
        stop_on_error: If True, stop testing when an error occurs
        max_failures: Maximum consecutive failures before stopping (if stop_on_error is False)
    """
    
    # Setup logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler('steering_test.log'),
            logging.StreamHandler()
        ]
    )
    
    # Get all benchmarks
    benchmarks = list(AVAILABLE_BENCHMARKS.keys())
    logging.info(f"Testing {len(benchmarks)} benchmarks with steering")
    
    results = {}
    consecutive_failures = 0
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] Testing steering on {benchmark}...")
        logging.info(f"Starting steering test for benchmark: {benchmark}")
        start_time = time.time()
        
        result = test_steering_benchmark(benchmark, verbose=False)
        duration = time.time() - start_time
        result["duration"] = duration
        results[benchmark] = result
        
        if result["status"] == "success":
            print(f"   {benchmark} completed in {duration:.2f}s")
            logging.info(f"SUCCESS: {benchmark} completed in {duration:.2f}s")
            consecutive_failures = 0
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"   {benchmark} failed: {error_msg}")
            logging.error(f"FAILED: {benchmark} after {duration:.2f}s - Error: {error_msg}")
            
            consecutive_failures += 1
            
            # Check if we should stop
            if stop_on_error:
                logging.error(f"Stopping due to error (stop_on_error=True)")
                print(f"\n�  Stopping tests due to error in {benchmark}")
                break
            elif consecutive_failures >= max_failures:
                logging.error(f"Stopping after {consecutive_failures} consecutive failures")
                print(f"\n�  Stopping tests after {consecutive_failures} consecutive failures")
                break
    
    # Save results
    results_file = Path(__file__).parent / "steering_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    print(f"\n{'='*60}")
    print(f"Steering Testing Complete")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {results_file}")
    print(f"Logs saved to: steering_test.log")
    
    return results

if __name__ == "__main__":
    test_steering_all_benchmarks()