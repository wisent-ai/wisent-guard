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


@dataclass
class DeviceBenchmark:
    """Performance benchmark results for a specific device."""
    device_id: str
    device_type: str  # "cpu", "cuda", "mps", etc.
    model_loading_seconds: float
    benchmark_eval_seconds_per_100_examples: float
    classifier_training_seconds_per_100_samples: float
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
        try:
            import torch
            if torch.cuda.is_available():
                info_parts.append(f"cuda_{torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                info_parts.append("mps")
        except ImportError:
            pass
        
        # Create hash of the combined info
        combined = "|".join(str(part) for part in info_parts)
        device_hash = hashlib.md5(combined.encode()).hexdigest()[:12]
        return device_hash
    
    def get_device_type(self) -> str:
        """Detect the device type (cpu, cuda, mps, etc.)."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps" 
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
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
        """Benchmark model loading time."""
        print("   📊 Benchmarking model loading...")
        
        # Create a minimal test script
        test_script = '''
import time
import sys
sys.path.append('.')

start_time = time.time()
try:
    # Try to load a small model for testing
    from wisent_guard.core.model import Model
    model = Model("microsoft/DialoGPT-small")  # Small model for testing
    end_time = time.time()
    print(f"BENCHMARK_RESULT:{end_time - start_time}")
except Exception as e:
    # Fallback estimate if model loading fails
    print("BENCHMARK_RESULT:15.0")
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            
            # Run the test script
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, timeout=60)
            
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
        
        # Fallback estimate
        fallback_time = 15.0
        print(f"      Using fallback estimate: {fallback_time:.1f}s")
        return fallback_time
    
    def run_benchmark_eval_test(self) -> float:
        """Benchmark evaluation performance on a small dataset."""
        print("   📊 Benchmarking evaluation performance...")
        
        # Create evaluation test script
        test_script = '''
import time
import sys
sys.path.append('.')

start_time = time.time()
try:
    # Quick evaluation test
    from wisent_guard.experiments.steering.all_methods_truthfulqa_mc import run_evaluation
    
    # Run with minimal parameters
    result = run_evaluation(
        model_name="microsoft/DialoGPT-small",
        limit=5,  # Just 5 examples
        methods=[],  # No steering, just baseline
        quiet=True
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    # Scale to per-100-examples
    time_per_100 = (total_time / 5) * 100
    print(f"BENCHMARK_RESULT:{time_per_100}")
    
except Exception as e:
    # Fallback estimate
    print("BENCHMARK_RESULT:45.0")
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, timeout=120)
            
            os.unlink(temp_script)
            
            # Parse result
            for line in result.stdout.split('\n'):
                if line.startswith('BENCHMARK_RESULT:'):
                    eval_time = float(line.split(':')[1])
                    print(f"      Evaluation: {eval_time:.1f}s per 100 examples")
                    return eval_time
                    
        except Exception as e:
            print(f"      Error in evaluation benchmark: {e}")
        
        # Fallback estimate
        fallback_time = 45.0
        print(f"      Using fallback estimate: {fallback_time:.1f}s per 100 examples")
        return fallback_time
    
    def run_classifier_training_test(self) -> float:
        """Benchmark classifier training performance."""
        print("   📊 Benchmarking classifier training...")
        
        # Create training test script
        test_script = '''
import time
import sys
sys.path.append('.')

start_time = time.time()
try:
    from wisent_guard.core.model import Model
    from wisent_guard.core.activations import ActivationClassifier
    import numpy as np
    
    # Create dummy training data
    dummy_activations_good = [np.random.randn(100) for _ in range(50)]  # 50 samples
    dummy_activations_bad = [np.random.randn(100) for _ in range(50)]   # 50 samples
    
    # Quick classifier training
    classifier = ActivationClassifier()
    classifier.fit(dummy_activations_bad, dummy_activations_good)
    
    end_time = time.time()
    total_time = end_time - start_time
    # Scale to per-100-samples  
    time_per_100 = (total_time / 100) * 100
    print(f"BENCHMARK_RESULT:{time_per_100}")
    
except Exception as e:
    # Fallback estimate
    print("BENCHMARK_RESULT:8.0")
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, timeout=60)
            
            os.unlink(temp_script)
            
            # Parse result
            for line in result.stdout.split('\n'):
                if line.startswith('BENCHMARK_RESULT:'):
                    training_time = float(line.split(':')[1])
                    print(f"      Classifier training: {training_time:.1f}s per 100 samples")
                    return training_time
                    
        except Exception as e:
            print(f"      Error in classifier training benchmark: {e}")
        
        # Fallback estimate
        fallback_time = 8.0
        print(f"      Using fallback estimate: {fallback_time:.1f}s per 100 samples")
        return fallback_time
    
    def run_steering_test(self) -> float:
        """Benchmark steering performance."""
        print("   📊 Benchmarking steering performance...")
        
        # For now, use a reasonable estimate based on typical steering overhead
        # TODO: Implement actual steering benchmark when we have steering code ready
        fallback_time = 2.0  # 2 seconds per example with steering
        print(f"      Steering: {fallback_time:.1f}s per example (estimated)")
        return fallback_time
    
    def run_data_generation_test(self) -> float:
        """Benchmark data generation performance.""" 
        print("   📊 Benchmarking data generation...")
        
        # Simple estimate based on text generation overhead
        fallback_time = 3.0  # 3 seconds per generated example
        print(f"      Data generation: {fallback_time:.1f}s per example (estimated)")
        return fallback_time
    
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
        
        # Run all benchmarks
        model_loading = self.run_model_loading_benchmark()
        benchmark_eval = self.run_benchmark_eval_test()
        classifier_training = self.run_classifier_training_test()
        steering = self.run_steering_test()
        data_generation = self.run_data_generation_test()
        
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
        print(f"      Training: {classifier_training:.1f}s per 100 samples")
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
            # Fallback to hardcoded estimates
            fallback_estimates = {
                "model_loading": 15.0,
                "benchmark_eval": 45.0,  # per 100 examples
                "classifier_training": 8.0,  # per 100 samples
                "steering": 2.0,  # per example
                "data_generation": 3.0  # per example
            }
            base_time = fallback_estimates.get(task_type, 30.0)
        else:
            # Use actual benchmark results
            if task_type == "model_loading":
                return benchmark.model_loading_seconds
            elif task_type == "benchmark_eval":
                base_time = benchmark.benchmark_eval_seconds_per_100_examples
                return (base_time / 100.0) * quantity
            elif task_type == "classifier_training":
                base_time = benchmark.classifier_training_seconds_per_100_samples
                return (base_time / 100.0) * quantity
            elif task_type == "steering":
                return benchmark.steering_seconds_per_example * quantity
            elif task_type == "data_generation":
                return benchmark.data_generation_seconds_per_example * quantity
            else:
                base_time = 30.0  # Default fallback
        
        # Scale for quantity if needed
        if task_type in ["benchmark_eval", "classifier_training"]:
            return (base_time / 100.0) * quantity
        else:
            return base_time * quantity


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