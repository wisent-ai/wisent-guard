#!/usr/bin/env python3
"""
Quick Test Script for Benchmark Loading

Tests the benchmark loading logic on a small subset of benchmarks
to verify everything works before running the full timing script.
"""

import os
import sys
import time
import traceback

# Add current directory to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the sample retrieval function
from populate_tasks import get_task_samples_for_analysis

# Test subset of benchmarks (mix of different types)
TEST_BENCHMARKS = {
    "hellaswag": {
        "task": "hellaswag",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "truthfulqa_mc1": {
        "task": "truthfulqa_mc1",
        "tags": ["hallucination", "general knowledge", "reasoning"]
    },
    "mmlu": {
        "task": "mmlu",
        "tags": ["general knowledge", "science", "reasoning"]
    },
    "gsm8k": {
        "task": "gsm8k",
        "tags": ["mathematics", "reasoning", "science"]
    },
    "humaneval": {
        "task": "humaneval",
        "tags": ["coding", "reasoning", "mathematics"]
    }
}

def test_single_benchmark(benchmark_name: str, benchmark_config: dict) -> dict:
    """Test loading a single benchmark and return results."""
    task_name = benchmark_config["task"]
    tags = benchmark_config.get("tags", [])
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {benchmark_name}")
    print(f"📋 Task: {task_name}")
    print(f"🏷️  Tags: {', '.join(tags)}")
    print(f"{'='*60}")
    
    result = {
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "success": False,
        "loading_time": 0,
        "samples_count": 0,
        "error": None
    }
    
    try:
        print("⏱️  Starting benchmark loading...")
        start_time = time.time()
        
        # Load samples using the core function
        samples_result = get_task_samples_for_analysis(task_name, num_samples=5)
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        result["loading_time"] = loading_time
        
        if "error" in samples_result:
            result["error"] = samples_result["error"]
            print(f"❌ Failed in {loading_time:.2f}s")
            print(f"   Error: {samples_result['error']}")
        else:
            result["success"] = True
            samples = samples_result.get("samples", [])
            result["samples_count"] = len(samples)
            
            print(f"✅ Success in {loading_time:.2f}s")
            print(f"📊 Retrieved {len(samples)} samples")
            
            # Show sample details
            if samples:
                sample = samples[0]
                print(f"📝 Sample question: {sample.get('question', 'N/A')[:150]}...")
                print(f"✅ Sample answer: {sample.get('correct_answer', 'N/A')}")
                if sample.get('choices'):
                    print(f"🔤 Choices available: {len(sample['choices'])}")
                
                # Show metadata
                if "total_docs" in samples_result:
                    print(f"📈 Total docs in benchmark: {samples_result['total_docs']}")
                if "output_type" in samples_result:
                    print(f"🔧 Output type: {samples_result['output_type']}")
    
    except Exception as e:
        end_time = time.time()
        loading_time = end_time - start_time if 'start_time' in locals() else 0
        result["loading_time"] = loading_time
        result["error"] = f"Exception: {str(e)}"
        
        print(f"💥 Exception in {loading_time:.2f}s")
        print(f"   Error: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
    
    return result

def main():
    """Test the benchmark loading logic on a subset of benchmarks."""
    print("🚀 Testing Benchmark Loading Logic")
    print(f"📋 Testing {len(TEST_BENCHMARKS)} sample benchmarks...")
    print()
    
    results = []
    total_start = time.time()
    
    for benchmark_name, benchmark_config in TEST_BENCHMARKS.items():
        result = test_single_benchmark(benchmark_name, benchmark_config)
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 QUICK TEST SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    if successful:
        avg_time = sum(r["loading_time"] for r in successful) / len(successful)
        print(f"📈 Average loading time: {avg_time:.2f}s")
        
        fastest = min(successful, key=lambda x: x["loading_time"])
        slowest = max(successful, key=lambda x: x["loading_time"])
        print(f"🚀 Fastest: {fastest['benchmark_name']} ({fastest['loading_time']:.2f}s)")
        print(f"🐌 Slowest: {slowest['benchmark_name']} ({slowest['loading_time']:.2f}s)")
    
    if failed:
        print(f"\n❌ Failed benchmarks:")
        for result in failed:
            print(f"  • {result['benchmark_name']}: {result['error'][:100]}...")
    
    print(f"\n🏁 Test complete! Ready to run full timing script.")

if __name__ == "__main__":
    main() 