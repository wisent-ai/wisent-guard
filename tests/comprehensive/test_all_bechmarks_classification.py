#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test classification pipeline for all benchmarks with hybrid CPU-MPS solution.
Mimics the testing we did with the MPS bug fix.
"""

import json
import logging
import subprocess
import time
from pathlib import Path

from wisent_guard.cli.cli_benchmarks import AVAILABLE_BENCHMARKS

# Configuration matching our MPS fix testing
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 15
LIMIT = 10
TOKEN_STRATEGY = "max_pooling"

def test_classification_all_benchmarks(stop_on_error=True, max_failures=5):
    """Test classification on all benchmarks with hybrid CPU-MPS mode.
    
    Args:
        stop_on_error: If True, stop testing when an error occurs
        max_failures: Maximum consecutive failures before stopping (if stop_on_error is False)
    """
    
    # Setup logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler('classification_test.log'),
            logging.StreamHandler()
        ]
    )
    
    # Get all benchmarks
    benchmarks = list(AVAILABLE_BENCHMARKS.keys())
    logging.info(f"Testing {len(benchmarks)} benchmarks with classification")
    
    # Save benchmark list
    benchmark_file = Path(__file__).parent / "available_benchmarks.json"
    with open(benchmark_file, "w") as f:
        json.dump(benchmarks, f, indent=2)
    
    results = {}
    consecutive_failures = 0
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] Testing {benchmark}...")
        logging.info(f"Starting test for benchmark: {benchmark}")
        start_time = time.time()
        
        try:
            # Test with the same approach we used for MPS fix
            # Model will use hybrid CPU-MPS mode automatically for Llama
            cmd = [
                "python", "-m", "wisent_guard.cli", "tasks", benchmark,
                "--model", MODEL,
                "--layer", str(LAYER),
                "--limit", str(LIMIT),
                "--token-targeting-strategy", TOKEN_STRATEGY,
                "--split-ratio", "0.8",
                "--allow-small-dataset"  # Allow small datasets
            ]
            
            logging.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # No timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Parse output for metrics if available
                output = result.stdout
                metrics = {}
                if "accuracy" in output.lower():
                    # Try to extract accuracy from output
                    for line in output.split('\n'):
                        if 'accuracy' in line.lower():
                            try:
                                # Extract number from line
                                import re
                                match = re.search(r'accuracy[:\s]+([0-9.]+)', line, re.IGNORECASE)
                                if match:
                                    metrics['accuracy'] = float(match.group(1))
                            except:
                                pass
                
                results[benchmark] = {
                    "status": "success",
                    "duration": duration,
                    "metrics": metrics,
                    "output": output[-1000:] if len(output) > 1000 else output  # Keep last 1000 chars
                }
                print(f"  [PASS] {benchmark} completed in {duration:.2f}s")
                logging.info(f"SUCCESS: {benchmark} completed in {duration:.2f}s")
                consecutive_failures = 0
            else:
                raise Exception(f"Command failed with return code {result.returncode}: {result.stderr}")
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            results[benchmark] = {
                "status": "failed", 
                "error": error_msg,
                "duration": duration
            }
            print(f"  [FAIL] {benchmark} failed: {error_msg}")
            logging.error(f"FAILED: {benchmark} after {duration:.2f}s - Error: {error_msg}")
            logging.exception("Full traceback:")
            
            consecutive_failures += 1
            
            # Check if we should stop
            if stop_on_error:
                logging.error(f"Stopping due to error (stop_on_error=True)")
                print(f"\n[WARNING] Stopping tests due to error in {benchmark}")
                break
            elif consecutive_failures >= max_failures:
                logging.error(f"Stopping after {consecutive_failures} consecutive failures")
                print(f"\n[WARNING] Stopping tests after {consecutive_failures} consecutive failures")
                break
    
    # Save results
    results_file = Path(__file__).parent / "classification_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    print(f"\n{'='*60}")
    print(f"Classification Testing Complete")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {results_file}")
    print(f"Logs saved to: classification_test.log")
    
    return results

if __name__ == "__main__":
    test_classification_all_benchmarks()
