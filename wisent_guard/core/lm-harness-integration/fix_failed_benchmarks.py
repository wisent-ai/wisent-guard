#!/usr/bin/env python3
"""
Targeted script to fix only the failed benchmarks from the previous run.
Reads the existing results file, tests only the failed benchmarks with fixes,
and updates the results file with the new results.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Import the benchmark configuration and timing functions
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from only_benchmarks import CORE_BENCHMARKS, get_task_samples_for_analysis

def load_existing_results(results_file: str) -> Dict:
    """Load existing benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def get_failed_benchmarks(results_data: Dict) -> List[Dict]:
    """Extract failed benchmarks from results data."""
    failed_benchmarks = []
    
    # Get failed benchmarks from summary
    if 'summary' in results_data and 'failed_benchmarks' in results_data['summary']:
        for failed in results_data['summary']['failed_benchmarks']:
            failed_benchmarks.append(failed['name'])
    
    # Also check detailed results for failures
    if 'detailed_results' in results_data:
        for result in results_data['detailed_results']:
            if not result.get('success', False):
                benchmark_name = result['benchmark_name']
                if benchmark_name not in failed_benchmarks:
                    failed_benchmarks.append(benchmark_name)
    
    return failed_benchmarks

def test_failed_benchmark(benchmark_name: str, benchmark_config: Dict) -> Dict:
    """Test a single failed benchmark with the fixes."""
    print(f"\n🔧 Testing fixed benchmark: {benchmark_name}")
    
    task_name = benchmark_config["task"]
    trust_remote_code = benchmark_config.get("trust_remote_code", False)
    
    if trust_remote_code:
        print(f"   🔐 Using trust_remote_code=True")
    
    start_time = time.time()
    
    try:
        # Test the benchmark with enhanced loading
        result = get_task_samples_for_analysis(
            task_name=task_name,
            num_samples=5,
            trust_remote_code=trust_remote_code
        )
        
        loading_time = time.time() - start_time
        
        if result.get("samples"):
            print(f"   ✅ SUCCESS: {len(result['samples'])} samples loaded in {loading_time:.2f}s")
            
            return {
                "benchmark_name": benchmark_name,
                "task_name": task_name,
                "tags": benchmark_config["tags"],
                "num_samples": 5,
                "success": True,
                "loading_time_seconds": loading_time,
                "samples_retrieved": len(result['samples']),
                "error": None,
                "metadata": result.get("metadata", {})
            }
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"   ❌ FAILED: {error_msg}")
            
            return {
                "benchmark_name": benchmark_name,
                "task_name": task_name,
                "tags": benchmark_config["tags"],
                "num_samples": 5,
                "success": False,
                "loading_time_seconds": loading_time,
                "samples_retrieved": 0,
                "error": error_msg,
                "metadata": {}
            }
            
    except Exception as e:
        loading_time = time.time() - start_time
        error_msg = f"Exception during testing: {str(e)}"
        print(f"   ❌ EXCEPTION: {error_msg}")
        
        return {
            "benchmark_name": benchmark_name,
            "task_name": task_name,
            "tags": benchmark_config["tags"],
            "num_samples": 5,
            "success": False,
            "loading_time_seconds": loading_time,
            "samples_retrieved": 0,
            "error": error_msg,
            "metadata": {}
        }

def update_results_file(original_results: Dict, fixed_results: List[Dict], output_file: str):
    """Update the results file with fixed benchmark results."""
    
    # Create a map of benchmark names to fixed results
    fixed_map = {result['benchmark_name']: result for result in fixed_results}
    
    # Update detailed results
    updated_detailed_results = []
    for result in original_results['detailed_results']:
        benchmark_name = result['benchmark_name']
        if benchmark_name in fixed_map:
            # Replace with fixed result
            updated_detailed_results.append(fixed_map[benchmark_name])
            print(f"   📝 Updated result for {benchmark_name}")
        else:
            # Keep original result
            updated_detailed_results.append(result)
    
    # Update summary statistics
    successful_results = [r for r in updated_detailed_results if r['success']]
    failed_results = [r for r in updated_detailed_results if not r['success']]
    
    total_benchmarks = len(updated_detailed_results)
    successful_loads = len(successful_results)
    failed_loads = len(failed_results)
    success_rate = successful_loads / total_benchmarks if total_benchmarks > 0 else 0
    
    # Calculate timing stats for successful results
    if successful_results:
        times = [r['loading_time_seconds'] for r in successful_results]
        mean_time = sum(times) / len(times)
        median_time = sorted(times)[len(times) // 2]
        min_time = min(times)
        max_time = max(times)
        std_dev = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        total_time = sum(times)
    else:
        mean_time = median_time = min_time = max_time = std_dev = total_time = 0
    
    # Update summary
    updated_summary = {
        "total_benchmarks": total_benchmarks,
        "successful_loads": successful_loads,
        "failed_loads": failed_loads,
        "success_rate": success_rate,
        "timing_stats": {
            "mean_time": mean_time,
            "median_time": median_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "total_time": total_time
        },
        "failed_benchmarks": [
            {
                "name": r['benchmark_name'],
                "task": r['task_name'],
                "error": r['error'],
                "tags": r['tags']
            } for r in failed_results
        ]
    }
    
    # Keep other summary fields from original
    if 'summary' in original_results:
        for key, value in original_results['summary'].items():
            if key not in updated_summary:
                updated_summary[key] = value
    
    # Create updated results structure
    updated_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_timestamp": original_results.get("timestamp"),
        "fix_applied": True,
        "summary": updated_summary,
        "detailed_results": updated_detailed_results
    }
    
    # Write updated results
    with open(output_file, 'w') as f:
        json.dump(updated_results, f, indent=2)
    
    print(f"\n📊 Updated Results Summary:")
    print(f"   Total benchmarks: {total_benchmarks}")
    print(f"   Successful: {successful_loads} ({success_rate:.1%})")
    print(f"   Failed: {failed_loads}")
    print(f"   Improvement: {successful_loads - original_results['summary']['successful_loads']} benchmarks fixed")

def main():
    """Main function to test failed benchmarks and update results."""
    
    # Find the most recent results file
    results_dir = Path("benchmark_timing_results")
    if not results_dir.exists():
        print("❌ No benchmark_timing_results directory found")
        return
    
    # Use the specified file
    results_file = results_dir / "benchmark_timing_20250705_142530.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"📖 Loading existing results from: {results_file}")
    
    # Load existing results
    original_results = load_existing_results(results_file)
    
    # Get failed benchmarks
    failed_benchmark_names = get_failed_benchmarks(original_results)
    
    print(f"\n🔍 Found {len(failed_benchmark_names)} failed benchmarks:")
    for name in failed_benchmark_names:
        print(f"   - {name}")
    
    # Test each failed benchmark with fixes
    print(f"\n🚀 Testing {len(failed_benchmark_names)} failed benchmarks with fixes...")
    
    fixed_results = []
    for benchmark_name in failed_benchmark_names:
        if benchmark_name in CORE_BENCHMARKS:
            benchmark_config = CORE_BENCHMARKS[benchmark_name]
            result = test_failed_benchmark(benchmark_name, benchmark_config)
            fixed_results.append(result)
        else:
            print(f"   ⚠️  Skipping {benchmark_name}: not found in CORE_BENCHMARKS")
    
    # Update results file
    output_file = results_file.parent / f"benchmark_timing_20250705_142530_fixed.json"
    
    print(f"\n💾 Updating results file: {output_file}")
    update_results_file(original_results, fixed_results, output_file)
    
    print(f"\n✅ Fix completed! Updated results saved to: {output_file}")

if __name__ == "__main__":
    main() 