#!/usr/bin/env python3
"""
Test suite for _load_single_task function in cli_prepare_dataset.py
Tests loading single benchmark tasks, similar to test_process_group_task.py
"""

import sys
from pathlib import Path
import torch
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent_guard.cli_utils.cli_prepare_dataset import (
    _load_single_task,
    PrepState,
)
from wisent_guard.core.model import Model


# Test cache directory
TEST_CACHE_DIR = "./benchmark_cache_test_load_single"


def cleanup_test_cache() -> None:
    """Remove test cache directory if it exists."""
    if Path(TEST_CACHE_DIR).exists():
        shutil.rmtree(TEST_CACHE_DIR)


def test_load_single_task_basic():
    """Test loading a single task with basic configuration"""
    
    print("="*80)
    print("Test 1: Load single task - basic configuration")
    print("="*80)
    
    cleanup_test_cache()
    
    # Check if CUDA is available and use appropriate device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device for testing")
    else:
        device = "cpu"
        print(f"CUDA not available, using CPU for testing")
    
    # Create model
    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B", device=device)
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        print("   Test cannot continue without model")
        return False
    
    # Test loading a single task
    task_name = "winogrande"
    print(f"\n2. Loading task '{task_name}'...")
    
    try:
        result = _load_single_task(
            model=model,
            task_name=task_name,
            shots=0,
            split_ratio=0.8,
            seed=42,
            limit=10,  # Small limit for fast testing
            training_limit=None,
            testing_limit=None,
            cache_benchmark=False,
            cache_dir=TEST_CACHE_DIR,
            force_download=False,
            livecodebench_version="v1",
            verbose=True
        )
        
        print("\n3. Task loaded successfully")
        
        # Check that we got a PrepState result
        assert isinstance(result, PrepState), "Should return PrepState object"
        print("   ✓ Returned PrepState object")
        
        # Check the fields
        assert result.group_processed == False, "Should not be group_processed for single task"
        assert result.group_qa_format == False, "Should not be group_qa_format for single task"
        assert result.task_data is not None, "Should have task_data"
        assert result.train_docs is not None, "Should have train_docs"
        
        print(f"   ✓ QA pairs extracted: {len(result.qa_pairs)}")
        print(f"   ✓ Test source: {len(result.test_source)}")
        print(f"   ✓ Train docs: {len(result.train_docs)}")
        
        print("\n✓ Test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_cache()


def test_load_single_task_with_limits():
    """Test loading a single task with training and testing limits"""
    
    print("\n" + "="*80)
    print("Test 2: Load single task with training and testing limits")
    print("="*80)
    
    cleanup_test_cache()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Create model
    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B", device=device)
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False
    
    # Test loading with specific limits
    task_name = "winogrande"
    training_limit = 8
    testing_limit = 2
    
    print(f"\n2. Loading task '{task_name}' with limits...")
    print(f"   - training_limit: {training_limit}")
    print(f"   - testing_limit: {testing_limit}")
    
    try:
        result = _load_single_task(
            model=model,
            task_name=task_name,
            shots=0,
            split_ratio=0.8,
            seed=42,
            limit=None,  # No overall limit
            training_limit=training_limit,
            testing_limit=testing_limit,
            cache_benchmark=False,
            cache_dir=TEST_CACHE_DIR,
            force_download=False,
            livecodebench_version="v1",
            verbose=True
        )
        
        print("\n3. Task loaded with limits")
        
        # Check limits were respected
        assert len(result.train_docs) <= training_limit, f"Training docs ({len(result.train_docs)}) should be <= {training_limit}"
        assert len(result.test_source) <= testing_limit, f"Test source ({len(result.test_source)}) should be <= {testing_limit}"
        
        print(f"   ✓ Training docs: {len(result.train_docs)} (limit: {training_limit})")
        print(f"   ✓ Test source: {len(result.test_source)} (limit: {testing_limit})")
        
        print("\n✓ Test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_cache()


def test_load_single_task_with_cache():
    """Test loading a single task with cache_benchmark enabled"""
    
    print("\n" + "="*80)
    print("Test 3: Load single task with cache_benchmark enabled")
    print("="*80)
    
    cleanup_test_cache()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Create model
    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B", device=device)
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False
    
    task_name = "winogrande"
    
    print(f"\n2. Loading task '{task_name}' with cache_benchmark=True...")
    
    try:
        result = _load_single_task(
            model=model,
            task_name=task_name,
            shots=0,
            split_ratio=0.8,
            seed=42,
            limit=10,
            training_limit=None,
            testing_limit=None,
            cache_benchmark=True,  # Enable caching
            cache_dir=TEST_CACHE_DIR,
            force_download=False,
            livecodebench_version="v1",
            verbose=True
        )
        
        print("\n3. Task loaded (cache_benchmark=True)")
        
        # Check result
        assert isinstance(result, PrepState), "Should return PrepState object"
        print("   ✓ Returned PrepState object")
        print(f"   ✓ QA pairs: {len(result.qa_pairs)}")
        print(f"   ✓ Test source: {len(result.test_source)}")
        
        # Note: Actual caching may fail if managed cache or Full Benchmark Downloader not available
        print("   Note: Cache operations depend on availability of caching infrastructure")
        
        print("\n✓ Test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_cache()


def test_load_single_task_different_benchmarks():
    """Test loading different benchmark tasks"""
    
    print("\n" + "="*80)
    print("Test 4: Load different benchmark tasks")
    print("="*80)
    
    cleanup_test_cache()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Create model
    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B", device=device)
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False
    
    # Test different benchmarks
    benchmarks = ["winogrande", "piqa", "copa"]
    
    print(f"\n2. Testing different benchmarks: {benchmarks}")
    
    for task_name in benchmarks:
        print(f"\n   Loading '{task_name}'...")
        try:
            result = _load_single_task(
                model=model,
                task_name=task_name,
                shots=0,
                split_ratio=0.8,
                seed=42,
                limit=5,  # Very small limit for speed
                training_limit=None,
                testing_limit=None,
                cache_benchmark=False,
                cache_dir=TEST_CACHE_DIR,
                force_download=False,
                livecodebench_version="v1",
                verbose=False  # Less verbose for multiple tasks
            )
            
            assert isinstance(result, PrepState), f"Failed to load {task_name}"
            print(f"   ✓ {task_name}: {len(result.qa_pairs)} QA pairs, {len(result.test_source)} test source")
            
        except Exception as e:
            print(f"   ✗ Failed to load {task_name}: {e}")
            return False
    
    print("\n✓ All benchmarks loaded successfully")
    print("✓ Test passed")
    cleanup_test_cache()
    return True


def test_load_single_task_with_shots():
    """Test loading a single task with few-shot examples"""
    
    print("\n" + "="*80)
    print("Test 5: Load single task with few-shot examples")
    print("="*80)
    
    cleanup_test_cache()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Create model
    print("\n1. Creating model...")
    try:
        model = Model(name="meta-llama/Llama-3.2-1B", device=device)
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False
    
    task_name = "winogrande"
    shots = 3
    
    print(f"\n2. Loading task '{task_name}' with {shots} shots...")
    
    try:
        result = _load_single_task(
            model=model,
            task_name=task_name,
            shots=shots,  # Few-shot examples
            split_ratio=0.8,
            seed=42,
            limit=10,
            training_limit=None,
            testing_limit=None,
            cache_benchmark=False,
            cache_dir=TEST_CACHE_DIR,
            force_download=False,
            livecodebench_version="v1",
            verbose=True
        )
        
        print(f"\n3. Task loaded with {shots} shots")
        
        assert isinstance(result, PrepState), "Should return PrepState object"
        print("   ✓ Returned PrepState object")
        print(f"   ✓ QA pairs: {len(result.qa_pairs)}")
        print(f"   ✓ Test source: {len(result.test_source)}")
        
        print("\n✓ Test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_cache()



def run_all_tests():
    """Run all tests"""
    print("\nRunning _load_single_task tests...")
    print("="*80)
    
    test_functions = [
        test_load_single_task_basic,
        test_load_single_task_with_limits,
        test_load_single_task_with_cache,
        test_load_single_task_different_benchmarks,
        test_load_single_task_with_shots,
    ]
    
    failed = []
    
    for test_func in test_functions:
        try:
            success = test_func()
            if not success:
                failed.append(test_func.__name__)
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test_func.__name__)
    
    print("\n" + "="*80)
    if failed:
        print(f"✗ {len(failed)} test(s) failed: {', '.join(failed)}")
        return False
    else:
        print("✓ All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)