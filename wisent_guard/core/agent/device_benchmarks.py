"""
Device-specific performance benchmarking for wisent-guard.

This module runs quick performance tests on the current device to measure
actual execution times for different operations, then saves those estimates
for future budget calculations.
"""

import json
import time
import os
import tempfile
import subprocess
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

import torch

from wisent_guard.core.utils.device import resolve_default_device


@dataclass
class DeviceBenchmark:
    """Performance benchmark results for a specific device."""
    device_id: str
    device_type: str  # "cpu", "cuda", "mps", etc.
    model_loading_seconds: float
    benchmark_eval_seconds_per_100_examples: float
    classifier_training_seconds_per_100_samples: float  # Actually measures full classifier creation time (per 100 classifiers)
    data_generation_seconds_per_example: float
    steering_seconds_per_example: float
    benchmark_timestamp: float
    python_version: str
    platform_info: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceBenchmark':
        """Create from dictionary loaded from JSON."""
        return cls(**data)


class DeviceBenchmarker:
    """Runs performance benchmarks and manages device-specific estimates."""
    
    def __init__(self, benchmarks_file: str = "device_benchmarks.json"):
        self.benchmarks_file = benchmarks_file
        self.cached_benchmark: Optional[DeviceBenchmark] = None
        
    def get_device_id(self) -> str:
        """Generate a unique ID for the current device configuration."""
        import platform
        
        # Create device fingerprint from hardware/software info
        info_parts = [
            platform.machine(),
            platform.processor(),
            platform.system(),
            platform.release(),
            sys.version,
        ]
        
        # Add GPU info if available
        device_kind = resolve_default_device()
        if device_kind == "cuda" and torch.cuda.is_available():
            info_parts.append(f"cuda_{torch.cuda.get_device_name(torch.cuda.current_device())}")
        elif device_kind == "mps":
            info_parts.append("mps")
        
        # Create hash of the combined info
        combined = "|".join(str(part) for part in info_parts)
        device_hash = hashlib.md5(combined.encode()).hexdigest()[:12]
        return device_hash
    
    def get_device_type(self) -> str:
        """Detect the device type (cpu, cuda, mps, etc.)."""
        return resolve_default_device()
    
    def load_cached_benchmark(self) -> Optional[DeviceBenchmark]:
        """Load cached benchmark results if they exist and are recent."""
        if not os.path.exists(self.benchmarks_file):
            return None
            
        try:
            with open(self.benchmarks_file, 'r') as f:
                data = json.load(f)
            
            device_id = self.get_device_id()
            if device_id not in data:
                return None
                
            benchmark_data = data[device_id]
            benchmark = DeviceBenchmark.from_dict(benchmark_data)
            
            # Check if benchmark is recent (within 7 days)
            current_time = time.time()
            age_days = (current_time - benchmark.benchmark_timestamp) / (24 * 3600)
            
            if age_days > 7:
                print(f"   ⚠️ Cached benchmark is {age_days:.1f} days old, will re-run")
                return None
                
            return benchmark
            
        except Exception as e:
            print(f"   ⚠️ Error loading cached benchmark: {e}")
            return None
    
    def save_benchmark(self, benchmark: DeviceBenchmark) -> None:
        """Save benchmark results to JSON file."""
        try:
            # Load existing data
            existing_data = {}
            if os.path.exists(self.benchmarks_file):
                with open(self.benchmarks_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Update with new benchmark
            existing_data[benchmark.device_id] = benchmark.to_dict()
            
            # Save back to file
            with open(self.benchmarks_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            print(f"   💾 Saved benchmark results to {self.benchmarks_file}")
            
        except Exception as e:
            print(f"   ❌ Error saving benchmark: {e}")
    
    def run_model_loading_benchmark(self) -> float:
        """Benchmark actual model loading time using the real model."""
        print("   📊 Benchmarking model loading...")
        
        # Create actual model loading test script
        test_script = '''
import time
import sys
sys.path.append('.')

start_time = time.time()
try:
    from wisent_guard.core.model import Model
    # Use the actual model that will be used in production
    model = Model("meta-llama/Llama-3.1-8B-Instruct")
    end_time = time.time()
    print(f"BENCHMARK_RESULT:{end_time - start_time}")
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    raise
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            
            # Run with 2-minute timeout
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, timeout=120)
            
            # Clean up
            os.unlink(temp_script)
            
            # Parse result
            for line in result.stdout.split('\n'):
                if line.startswith('BENCHMARK_RESULT:'):
                    loading_time = float(line.split(':')[1])
                    print(f"      Model loading: {loading_time:.1f}s")
                    return loading_time
                    
        except Exception as e:
            print(f"      Error in model loading benchmark: {e}")
            raise RuntimeError(f"Model loading benchmark failed: {e}")
    
    def run_benchmark_eval_test(self) -> float:
        """Benchmark evaluation performance using real CLI functionality."""
        print("   📊 Benchmarking evaluation performance...")
        print("   🔧 DEBUG: Creating evaluation test script...")
        
        # Create evaluation test script using actual CLI
        test_script = '''
import time
import sys
sys.path.append('.')

print("BENCHMARK_DEBUG: Starting evaluation benchmark")
start_time = time.time()
try:
    print("BENCHMARK_DEBUG: Importing CLI...")
    from wisent_guard.cli import run_task_pipeline
    print("BENCHMARK_DEBUG: CLI imported successfully")
    
    print("BENCHMARK_DEBUG: Running task pipeline...")
    # Run actual evaluation with real model and minimal examples
    run_task_pipeline(
        task_name="truthfulqa_mc",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        layer="15",  # Required parameter
        limit=3,  # Minimum examples for timing
        steering_mode=False,  # No steering for baseline timing
        verbose=False,
        allow_small_dataset=True,
        output_mode="likelihoods"
    )
    print("BENCHMARK_DEBUG: Task pipeline completed")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"BENCHMARK_DEBUG: Total time: {total_time}s for 3 examples")
    # Scale to per-100-examples
    time_per_100 = (total_time / 3) * 100
    print(f"BENCHMARK_DEBUG: Scaled time per 100: {time_per_100}s")
    print(f"BENCHMARK_RESULT:{time_per_100}")
    
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    import traceback
    traceback.print_exc()
    raise
'''
        
        print("   🔧 DEBUG: Writing test script to temporary file...")
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            print(f"   🔧 DEBUG: Test script written to {temp_script}")
            
            print("   🔧 DEBUG: Running evaluation subprocess...")
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, timeout=120)  # 2-minute timeout
            
            print(f"   🔧 DEBUG: Subprocess completed with return code: {result.returncode}")
            print(f"   🔧 DEBUG: Stdout length: {len(result.stdout)} chars")
            print(f"   🔧 DEBUG: Stderr length: {len(result.stderr)} chars")
            
            if result.stderr:
                print(f"   ⚠️ DEBUG: Stderr content:\n{result.stderr}")
            
            os.unlink(temp_script)
            print("   🔧 DEBUG: Temporary script cleaned up")
            
            # Parse result
            print("   🔧 DEBUG: Parsing output for BENCHMARK_RESULT...")
            found_result = False
            for line in result.stdout.split('\n'):
                print(f"   🔍 DEBUG: Output line: {repr(line)}")
                if line.startswith('BENCHMARK_RESULT:'):
                    eval_time = float(line.split(':')[1])
                    print(f"      ✅ Evaluation: {eval_time:.1f}s per 100 examples")
                    found_result = True
                    return eval_time
            
            if not found_result:
                print("   ❌ DEBUG: No BENCHMARK_RESULT found in output!")
                print("   📜 DEBUG: Full stdout:")
                print(result.stdout)
                return None
                    
        except Exception as e:
            print(f"      ❌ Error in evaluation benchmark: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_classifier_training_test(self) -> float:
        """Benchmark ACTUAL classifier training using real synthetic classifier creation."""
        print("   📊 Benchmarking classifier training...")
        print("   🔧 DEBUG: Creating classifier training test script...")
        
        # Create test script that uses real synthetic classifier creation
        test_script = '''
import time
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Optional
try:
    print("BENCHMARK_DEBUG: Importing required modules...")
    from wisent_guard.core.model import Model
    from wisent_guard.core.agent.diagnose.synthetic_classifier_option import create_classifier_from_trait_description
    from wisent_guard.core.agent.budget import set_time_budget
    import time
    print("BENCHMARK_DEBUG: All modules imported successfully")
    
    print("BENCHMARK_DEBUG: Starting classifier benchmark")
    
    # Set a budget for the classifier creation
    print("BENCHMARK_DEBUG: Setting time budget...")
    set_time_budget(5.0)  # 5 minutes
    print("BENCHMARK_DEBUG: Set time budget to 5.0 minutes")
    
    # Load the actual model
    print("BENCHMARK_DEBUG: Loading model...")
    model_start = time.time()
    model = Model("meta-llama/Llama-3.1-8B-Instruct")
    model_time = time.time() - model_start
    print(f"BENCHMARK_DEBUG: Model loaded in {model_time}s")
    
    # Create ONE actual classifier using the real synthetic process
    print("BENCHMARK_DEBUG: Creating classifier...")
    classifier_start = time.time()
    classifier = create_classifier_from_trait_description(
        model=model,
        trait_description="accuracy and truthfulness",
        num_pairs=3  # Minimum needed for training
    )
    classifier_time = time.time() - classifier_start
    print(f"BENCHMARK_DEBUG: Classifier created in {classifier_time}s")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"BENCHMARK_DEBUG: Total benchmark time: {total_time}s")
    
    # This is time for ONE complete classifier creation
    # Scale to "per 100 classifiers" for compatibility with existing code
    time_per_100 = total_time * 100
    print(f"BENCHMARK_DEBUG: Scaled time per 100 classifiers: {time_per_100}s")
    print(f"BENCHMARK_RESULT:{time_per_100}")
    
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    import traceback
    traceback.print_exc()
    raise
'''
        
        print("   🔧 DEBUG: Writing classifier test script to temporary file...")
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            print(f"   🔧 DEBUG: Classifier test script written to {temp_script}")

            print("   🔧 DEBUG: Running classifier training subprocess (20 min timeout)...")
            result = subprocess.run([
                sys.executable,
                temp_script,
            ], capture_output=True, text=True, timeout=1200)

            print(f"   🔧 DEBUG: Classifier subprocess completed with return code: {result.returncode}")
            print(f"   🔧 DEBUG: Stdout length: {len(result.stdout)} chars")
            print(f"   🔧 DEBUG: Stderr length: {len(result.stderr)} chars")

            if result.stderr:
                print(f"   ⚠️ DEBUG: Classifier stderr content:\n{result.stderr}")

            os.unlink(temp_script)
            print("   🔧 DEBUG: Classifier temporary script cleaned up")

            # Parse result
            print("   🔧 DEBUG: Parsing classifier output for BENCHMARK_RESULT...")
            for line in result.stdout.split('\n'):
                print(f"   🔍 DEBUG: Classifier output line: {repr(line)}")
                if line.startswith('BENCHMARK_RESULT:'):
                    training_time = float(line.split(':')[1])
                    print(f"      ✅ Classifier training: {training_time:.1f}s per 100 classifiers")
                    return training_time

            print("   ❌ DEBUG: No BENCHMARK_RESULT found in classifier output!")
            print("   📜 DEBUG: Full classifier stdout:")
            print(result.stdout)
            return None

        except Exception as e:
            print(f"      ❌ Error in classifier training benchmark: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_steering_test(self) -> float:
        """Benchmark steering performance using real CLI functionality."""
        print("   📊 Benchmarking steering performance...")
        
        # Create steering test script using actual CLI
        test_script = '''
import time
import sys
sys.path.append('.')

start_time = time.time()
try:
    from wisent_guard.cli import run_task_pipeline
    
    # Run actual steering with real model and minimal examples
    run_task_pipeline(
        task_name="truthfulqa_mc",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        limit=2,  # Minimum examples for timing
        steering_mode=True,
        steering_method="CAA",
        steering_strength=1.0,
        layer="15",
        verbose=False,
        allow_small_dataset=True,
        output_mode="likelihoods"
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    # Time per example
    time_per_example = total_time / 2
    print(f"BENCHMARK_RESULT:{time_per_example}")
    
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    raise
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name

            result = subprocess.run([
                sys.executable,
                temp_script,
            ], capture_output=True, text=True, timeout=300)

            os.unlink(temp_script)

            for line in result.stdout.split('\n'):
                if line.startswith('BENCHMARK_RESULT:'):
                    steering_time = float(line.split(':')[1])
                    print(f"      Steering: {steering_time:.1f}s per example")
                    return steering_time

            print("   ❌ No BENCHMARK_RESULT found in steering output!")
            print(result.stdout)
            return None

        except Exception as e:
            print(f"      Error in steering benchmark: {e}")
            raise RuntimeError(f"Steering benchmark failed: {e}")
    
    def run_data_generation_test(self) -> float:
        """Benchmark data generation performance using real synthetic generation.""" 
        print("   📊 Benchmarking data generation...")
        
        # Create data generation test script using actual synthetic pair generation
        test_script = '''
import time
import sys
sys.path.append('.')

start_time = time.time()
try:
    from wisent_guard.core.model import Model
    from wisent_guard.core.contrastive_pairs.generate_synthetically import SyntheticContrastivePairGenerator
    
    # Load the actual model
    model = Model("meta-llama/Llama-3.1-8B-Instruct")
    
    # Create generator and generate actual synthetic pairs
    generator = SyntheticContrastivePairGenerator(model)
    
    # Generate a small set of pairs for timing
    pair_set = generator.generate_contrastive_pair_set(
        trait_description="accuracy and truthfulness",
        num_pairs=1,  # Minimum needed for estimation
        name="benchmark_test"
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate time per generated pair (each pair has 2 responses)
    num_generated_responses = len(pair_set.pairs) * 2
    if num_generated_responses == 0:
        raise RuntimeError("No pairs were generated during data generation benchmark")
    
    time_per_example = total_time / num_generated_responses
    print(f"BENCHMARK_RESULT:{time_per_example}")
    
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    raise
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, timeout=300)  # 5-minute timeout
            
            os.unlink(temp_script)
            
            # Parse result
            for line in result.stdout.split('\n'):
                if line.startswith('BENCHMARK_RESULT:'):
                    generation_time = float(line.split(':')[1])
                    print(f"      Data generation: {generation_time:.1f}s per example")
                    return generation_time
                    
        except Exception as e:
            print(f"      Error in data generation benchmark: {e}")
            raise RuntimeError(f"Data generation benchmark failed: {e}")
    
    def run_full_benchmark(self, force_rerun: bool = False) -> DeviceBenchmark:
        """Run complete device benchmark suite."""
        # Check for cached results first
        if not force_rerun:
            cached = self.load_cached_benchmark()
            if cached:
                print(f"   ✅ Using cached benchmark results (device: {cached.device_id[:8]}...)")
                self.cached_benchmark = cached
                return cached
        
        print("🚀 Running device performance benchmark...")
        print("   This will take 1-2 minutes to measure your hardware performance")
        
        import platform
        
        device_id = self.get_device_id()
        device_type = self.get_device_type()
        
        print(f"   🖥️ Device ID: {device_id[:8]}... ({device_type})")
        
        # Run all benchmarks with error handling
        try:
            model_loading = self.run_model_loading_benchmark()
            if model_loading is None:
                print(f"   ❌ Model loading benchmark returned None")
                raise RuntimeError("Model loading benchmark failed")
        except Exception as e:
            print(f"   ❌ Model loading benchmark failed: {e}")
            raise
            
        try:
            benchmark_eval = self.run_benchmark_eval_test()
            if benchmark_eval is None:
                print(f"   ⚠️ Evaluation benchmark returned None, using default value")
                benchmark_eval = 60.0  # Default 60 seconds per 100 examples
        except Exception as e:
            print(f"   ❌ Evaluation benchmark failed: {e}")
            benchmark_eval = 60.0  # Default fallback
            
        try:
            classifier_training = self.run_classifier_training_test()
            if classifier_training is None:
                print(f"   ⚠️ Classifier training benchmark returned None, using default value")
                classifier_training = 600.0  # Default 600 seconds per 100 classifiers
        except Exception as e:
            print(f"   ❌ Classifier training benchmark failed: {e}")
            classifier_training = 600.0  # Default fallback
            
        try:
            steering = self.run_steering_test()
            if steering is None:
                print(f"   ❌ Steering benchmark returned None")
                raise RuntimeError("Steering benchmark failed")
        except Exception as e:
            print(f"   ❌ Steering benchmark failed: {e}")
            raise
            
        try:
            data_generation = self.run_data_generation_test()
            if data_generation is None:
                print(f"   ❌ Data generation benchmark returned None")
                raise RuntimeError("Data generation benchmark failed")
        except Exception as e:
            print(f"   ❌ Data generation benchmark failed: {e}")
            raise
        
        # Create benchmark result
        benchmark = DeviceBenchmark(
            device_id=device_id,
            device_type=device_type,
            model_loading_seconds=model_loading,
            benchmark_eval_seconds_per_100_examples=benchmark_eval,
            classifier_training_seconds_per_100_samples=classifier_training,
            data_generation_seconds_per_example=data_generation,
            steering_seconds_per_example=steering,
            benchmark_timestamp=time.time(),
            python_version=sys.version,
            platform_info=platform.platform()
        )
        
        # Save results
        self.save_benchmark(benchmark)
        self.cached_benchmark = benchmark
        
        print("   ✅ Benchmark complete!")
        print(f"      Model loading: {model_loading:.1f}s")
        print(f"      Evaluation: {benchmark_eval:.1f}s per 100 examples")
        print(f"      Classifier creation: {classifier_training:.1f}s per 100 classifiers")
        print(f"      Steering: {steering:.1f}s per example")
        print(f"      Generation: {data_generation:.1f}s per example")
        
        return benchmark
    
    def get_current_benchmark(self, auto_run: bool = True) -> Optional[DeviceBenchmark]:
        """Get current device benchmark, optionally auto-running if needed."""
        if self.cached_benchmark:
            return self.cached_benchmark
            
        cached = self.load_cached_benchmark()
        if cached:
            self.cached_benchmark = cached
            return cached
            
        if auto_run:
            return self.run_full_benchmark()
            
        return None
    
    def estimate_task_time(self, task_type: str, quantity: int = 1) -> float:
        """
        Estimate time for a specific task type and quantity.
        
        Args:
            task_type: Type of task ("model_loading", "benchmark_eval", etc.)
            quantity: Number of items (examples, samples, etc.)
            
        Returns:
            Estimated time in seconds
        """
        benchmark = self.get_current_benchmark()
        if not benchmark:
            raise RuntimeError(f"No benchmark available for device. Run benchmark first with: python -m wisent_guard.core.agent.budget benchmark")
        else:
            # Use actual benchmark results
            if task_type == "model_loading":
                return benchmark.model_loading_seconds
            elif task_type == "benchmark_eval":
                base_time = benchmark.benchmark_eval_seconds_per_100_examples
                return (base_time / 100.0) * quantity
            elif task_type == "classifier_training":
                base_time = benchmark.classifier_training_seconds_per_100_samples  # Actually per 100 classifiers now
                return (base_time / 100.0) * quantity
            elif task_type == "steering":
                return benchmark.steering_seconds_per_example * quantity
            elif task_type == "data_generation":
                return benchmark.data_generation_seconds_per_example * quantity
            else:
                raise ValueError(f"Unknown task type: {task_type}")


# Global benchmarker instance
_device_benchmarker = DeviceBenchmarker()


def get_device_benchmarker() -> DeviceBenchmarker:
    """Get the global device benchmarker instance."""
    return _device_benchmarker


def ensure_benchmark_exists(force_rerun: bool = False) -> DeviceBenchmark:
    """Ensure device benchmark exists, running it if necessary."""
    return _device_benchmarker.run_full_benchmark(force_rerun=force_rerun)


def estimate_task_time(task_type: str, quantity: int = 1) -> float:
    """
    Convenience function to estimate task time.
    
    Args:
        task_type: Type of task ("model_loading", "benchmark_eval", etc.)
        quantity: Number of items
        
    Returns:
        Estimated time in seconds
    """
    return _device_benchmarker.estimate_task_time(task_type, quantity)


def get_current_device_info() -> Dict[str, str]:
    """Get current device information."""
    benchmarker = get_device_benchmarker()
    return {
        "device_id": benchmarker.get_device_id(),
        "device_type": benchmarker.get_device_type()
    } 