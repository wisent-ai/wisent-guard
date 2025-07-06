#!/usr/bin/env python3
"""
Benchmark Loading Timer Script

Measures the time to load 5 samples from each benchmark in the CORE_BENCHMARKS list.
Tests the benchmark loading performance to identify slow or problematic benchmarks.

Features:
- Times each benchmark loading individually 
- Handles errors gracefully and continues
- Reports detailed timing statistics
- Saves results to JSON file
- Shows progress and status for each benchmark
"""

import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import statistics

# Add current directory to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the sample retrieval function and benchmark list
from populate_tasks import get_task_samples_for_analysis
from only_benchmarks import CORE_BENCHMARKS

def time_benchmark_loading(benchmark_name: str, benchmark_config: dict, num_samples: int = 5) -> Dict:
    """
    Time the loading of samples from a single benchmark.
    
    Args:
        benchmark_name: Display name of the benchmark
        benchmark_config: Config dict with 'task' and 'tags' keys
        num_samples: Number of samples to retrieve (default 5)
    
    Returns:
        Dictionary with timing results and metadata
    """
    task_name = benchmark_config["task"]
    tags = benchmark_config.get("tags", [])
    
    print(f"⏱️  Timing {benchmark_name} ({task_name})...", end=" ", flush=True)
    
    result = {
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "tags": tags,
        "num_samples": num_samples,
        "success": False,
        "loading_time_seconds": None,
        "samples_retrieved": 0,
        "error": None,
        "metadata": {}
    }
    
    try:
        # Record start time
        start_time = time.time()
        
        # Load samples using the existing function
        samples_result = get_task_samples_for_analysis(task_name, num_samples=num_samples)
        
        # Record end time
        end_time = time.time()
        loading_time = end_time - start_time
        
        # Check if loading was successful
        if "error" in samples_result:
            result["error"] = samples_result["error"]
            result["loading_time_seconds"] = loading_time  # Still record time even if failed
            print(f"❌ Failed in {loading_time:.2f}s - {samples_result['error'][:100]}...")
        else:
            result["success"] = True
            result["loading_time_seconds"] = loading_time
            result["samples_retrieved"] = len(samples_result.get("samples", []))
            
            # Extract metadata if available
            if "total_docs" in samples_result:
                result["metadata"]["total_docs"] = samples_result["total_docs"]
            if "output_type" in samples_result:
                result["metadata"]["output_type"] = samples_result["output_type"]
            if "description" in samples_result:
                result["metadata"]["description"] = samples_result["description"][:200] + "..." if len(str(samples_result.get("description", ""))) > 200 else samples_result.get("description", "")
            
            print(f"✅ {result['samples_retrieved']} samples in {loading_time:.2f}s")
        
    except Exception as e:
        end_time = time.time()
        loading_time = end_time - start_time if 'start_time' in locals() else 0
        
        result["loading_time_seconds"] = loading_time
        result["error"] = f"Exception: {str(e)}"
        print(f"💥 Exception in {loading_time:.2f}s - {str(e)[:100]}...")
        
        # Add traceback for debugging
        result["traceback"] = traceback.format_exc()
    
    return result

def analyze_timing_results(results: List[Dict]) -> Dict:
    """
    Analyze timing results and generate statistics.
    
    Args:
        results: List of timing result dictionaries
    
    Returns:
        Dictionary with analysis results
    """
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    analysis = {
        "total_benchmarks": len(results),
        "successful_loads": len(successful_results),
        "failed_loads": len(failed_results),
        "success_rate": len(successful_results) / len(results) if results else 0,
        "timing_stats": {},
        "category_stats": {},
        "slowest_benchmarks": [],
        "fastest_benchmarks": [],
        "failed_benchmarks": []
    }
    
    # Timing statistics for successful loads
    if successful_results:
        times = [r["loading_time_seconds"] for r in successful_results]
        analysis["timing_stats"] = {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "total_time": sum(times)
        }
        
        # Sort by timing
        sorted_by_time = sorted(successful_results, key=lambda x: x["loading_time_seconds"])
        analysis["fastest_benchmarks"] = [
            {
                "name": r["benchmark_name"],
                "time": r["loading_time_seconds"],
                "samples": r["samples_retrieved"],
                "tags": r["tags"]
            }
            for r in sorted_by_time[:5]
        ]
        analysis["slowest_benchmarks"] = [
            {
                "name": r["benchmark_name"], 
                "time": r["loading_time_seconds"],
                "samples": r["samples_retrieved"],
                "tags": r["tags"]
            }
            for r in sorted_by_time[-5:]
        ]
    
    # Category analysis
    category_times = {}
    for result in successful_results:
        for tag in result["tags"]:
            if tag not in category_times:
                category_times[tag] = []
            category_times[tag].append(result["loading_time_seconds"])
    
    for category, times in category_times.items():
        if times:
            analysis["category_stats"][category] = {
                "count": len(times),
                "mean_time": statistics.mean(times),
                "median_time": statistics.median(times),
                "min_time": min(times),
                "max_time": max(times)
            }
    
    # Failed benchmarks
    analysis["failed_benchmarks"] = [
        {
            "name": r["benchmark_name"],
            "task": r["task_name"],
            "error": r["error"][:200] + "..." if len(r["error"]) > 200 else r["error"],
            "tags": r["tags"]
        }
        for r in failed_results
    ]
    
    return analysis

def print_timing_summary(analysis: Dict):
    """Print a formatted summary of timing results."""
    print(f"\n{'='*80}")
    print("📊 BENCHMARK LOADING TIMING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Total benchmarks tested: {analysis['total_benchmarks']}")
    print(f"  Successful loads: {analysis['successful_loads']}")
    print(f"  Failed loads: {analysis['failed_loads']}")
    print(f"  Success rate: {analysis['success_rate']:.1%}")
    
    if analysis["timing_stats"]:
        stats = analysis["timing_stats"]
        print(f"\n⏱️  Timing Statistics (successful loads only):")
        print(f"  Mean loading time: {stats['mean_time']:.2f}s")
        print(f"  Median loading time: {stats['median_time']:.2f}s")
        print(f"  Fastest load: {stats['min_time']:.2f}s")
        print(f"  Slowest load: {stats['max_time']:.2f}s")
        print(f"  Standard deviation: {stats['std_dev']:.2f}s")
        print(f"  Total time: {stats['total_time']:.2f}s")
        
        print(f"\n🚀 Fastest Benchmarks:")
        for i, benchmark in enumerate(analysis["fastest_benchmarks"], 1):
            print(f"  {i}. {benchmark['name']} - {benchmark['time']:.2f}s ({benchmark['samples']} samples)")
            print(f"     Tags: {', '.join(benchmark['tags'])}")
        
        print(f"\n🐌 Slowest Benchmarks:")
        for i, benchmark in enumerate(analysis["slowest_benchmarks"], 1):
            print(f"  {i}. {benchmark['name']} - {benchmark['time']:.2f}s ({benchmark['samples']} samples)")
            print(f"     Tags: {', '.join(benchmark['tags'])}")
    
    if analysis["category_stats"]:
        print(f"\n🏷️  Performance by Category:")
        sorted_categories = sorted(analysis["category_stats"].items(), 
                                 key=lambda x: x[1]["mean_time"])
        for category, stats in sorted_categories:
            print(f"  {category}: {stats['mean_time']:.2f}s avg ({stats['count']} benchmarks)")
    
    if analysis["failed_benchmarks"]:
        print(f"\n❌ Failed Benchmarks:")
        for benchmark in analysis["failed_benchmarks"]:
            print(f"  • {benchmark['name']} ({benchmark['task']})")
            print(f"    Error: {benchmark['error']}")
            print(f"    Tags: {', '.join(benchmark['tags'])}")

def save_detailed_results(results: List[Dict], analysis: Dict, output_file: str):
    """Save detailed results to JSON file."""
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": analysis,
        "detailed_results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

def main():
    """Main function to time all benchmark loading."""
    print("🚀 Benchmark Loading Timer")
    print("="*50)
    print(f"📋 Testing {len(CORE_BENCHMARKS)} benchmarks...")
    print("⏱️  Measuring time to load 5 samples from each benchmark")
    print()
    
    # Create output directory
    output_dir = "benchmark_timing_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Time all benchmarks
    results = []
    total_start_time = time.time()
    
    for i, (benchmark_name, benchmark_config) in enumerate(CORE_BENCHMARKS.items(), 1):
        print(f"[{i:2d}/{len(CORE_BENCHMARKS)}] ", end="")
        result = time_benchmark_loading(benchmark_name, benchmark_config)
        results.append(result)
        
        # Add a small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    total_time = time.time() - total_start_time
    
    # Analyze results
    print(f"\n🔍 Analyzing results...")
    analysis = analyze_timing_results(results)
    analysis["total_script_time"] = total_time
    
    # Print summary
    print_timing_summary(analysis)
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"benchmark_timing_{timestamp}.json")
    save_detailed_results(results, analysis, output_file)
    
    print(f"\n🏁 Total script execution time: {total_time:.2f}s")
    print("✅ Benchmark loading timing analysis complete!")

if __name__ == "__main__":
    main() 