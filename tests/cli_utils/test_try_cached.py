#!/usr/bin/env python3
"""
Test suite for _try_cached function and related cache functions in cli_prepare_dataset.py
Tests cache loading without using mocks.
"""

import sys
import os
import pickle
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent_guard.cli_utils.cli_prepare_dataset import (
    _try_cached,
    _is_benchmark_cached,
    _load_cached_benchmark,
    _convert_cached_data_to_qa_pairs,
)


# Default cache directory for tests
DEFAULT_CACHE_DIR = "./benchmark_cache_test"  # Use test-specific dir to avoid conflicts

# Helper function to setup test cache
def setup_test_cache(task_name: str, data: dict) -> None:
    """Create a cache file in the default cache directory."""
    cache_path = Path(DEFAULT_CACHE_DIR) / "data" / f"{task_name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def cleanup_test_cache() -> None:
    """Remove test cache directory if it exists."""
    if Path(DEFAULT_CACHE_DIR).exists():
        shutil.rmtree(DEFAULT_CACHE_DIR)

# Test _try_cached function
def test_try_cached_not_cached():
    """Test 1: No cached data available"""
    cleanup_test_cache()  # Ensure clean state
    
    used_cache, cached_pairs = _try_cached(
        task_name="nonexistent_task",
        cache_dir=DEFAULT_CACHE_DIR,
        limit=None,
        use_cached=True,
        force_download=False,
        verbose=False
    )
    
    assert used_cache == False, "Should not use cache when nothing cached"
    assert cached_pairs == [], "Should return empty list when no cache"
    
    cleanup_test_cache()


def test_try_cached_force_download():
    """Test 2: Force download bypasses cache"""
    cleanup_test_cache()
    
    # Create a fake cache file
    setup_test_cache("fake_task", {"contrastive_pairs": [{"test": "data"}]})
    
    used_cache, cached_pairs = _try_cached(
        task_name="fake_task",
        cache_dir=DEFAULT_CACHE_DIR,
        limit=None,
        use_cached=True,
        force_download=True,  # Force fresh download
        verbose=False
    )
    
    assert used_cache == False, "Should not use cache when force_download=True"
    assert cached_pairs == [], "Should return empty list"
    
    cleanup_test_cache()


def test_try_cached_disabled():
    """Test 3: Cache usage disabled"""
    cleanup_test_cache()
    
    used_cache, cached_pairs = _try_cached(
        task_name="some_task",
        cache_dir=DEFAULT_CACHE_DIR,
        limit=None,
        use_cached=False,  # Caching disabled
        force_download=False,
        verbose=False
    )
    
    assert used_cache == False, "Should not use cache when use_cached=False"
    assert cached_pairs == [], "Should return empty list"
    
    cleanup_test_cache()


def test_try_cached_with_data():
    """Test 4: Successfully load from cache"""
    cleanup_test_cache()
    
    # Create test data in cache (using the expected format)
    test_data = [
        {"context": "Q1", "good_response": "A1", "bad_response": "W1"},
        {"context": "Q2", "good_response": "A2", "bad_response": "W2"}
    ]
    setup_test_cache("test_task", test_data)
    
    used_cache, cached_pairs = _try_cached(
        task_name="test_task",
        cache_dir=DEFAULT_CACHE_DIR,
        limit=None,
        use_cached=True,
        force_download=False,
        verbose=False
    )
    
    assert used_cache == True, "Should use cache when available"
    assert len(cached_pairs) == 2, "Should return cached pairs"
    assert cached_pairs[0]["question"] == "Q1", "Should have correct data"
    assert cached_pairs[0]["correct_answer"] == "A1", "Should have correct answer"
    assert cached_pairs[0]["incorrect_answer"] == "W1", "Should have incorrect answer"
    
    cleanup_test_cache()


def test_try_cached_with_limit():
    """Test 5: Load from cache with limit"""
    cleanup_test_cache()
    
    # Create test data with more items (using the expected format)
    test_data = [
        {"context": f"Q{i}", "good_response": f"A{i}", "bad_response": f"W{i}"}
        for i in range(10)
    ]
    setup_test_cache("limited_task", test_data)
    
    used_cache, cached_pairs = _try_cached(
        task_name="limited_task",
        cache_dir=DEFAULT_CACHE_DIR,
        limit=5,  # Limit to 5 items
        use_cached=True,
        force_download=False,
        verbose=False
    )
    
    assert used_cache == True, "Should use cache"
    assert len(cached_pairs) == 5, "Should respect limit"
    # Check first item has correct data
    assert cached_pairs[0]["question"] == "Q0", "Should have correct question"
    assert cached_pairs[0]["correct_answer"] == "A0", "Should have correct answer"
    assert cached_pairs[0]["incorrect_answer"] == "W0", "Should have incorrect answer"
    # Check last item (should be Q4 since we limited to 5)
    assert cached_pairs[4]["question"] == "Q4", "Should have 5th item"
    assert cached_pairs[4]["correct_answer"] == "A4", "Should have 5th answer"
    
    cleanup_test_cache()


# Test cache helper functions
def test_is_benchmark_cached():
    """Test 6: Check if benchmark is cached"""
    cleanup_test_cache()
    
    # Initially not cached
    assert _is_benchmark_cached("test", DEFAULT_CACHE_DIR) == False
    
    # Create cache file
    setup_test_cache("test", {"contrastive_pairs": []})
    
    # Now it should be cached
    assert _is_benchmark_cached("test", DEFAULT_CACHE_DIR) == True
    
    cleanup_test_cache()


def test_load_cached_benchmark():
    """Test 7: Load benchmark from cache file"""
    cleanup_test_cache()
    
    # Save test data
    test_data = [
        {"question": "Test Q", "answer": "Test A"}
    ]
    setup_test_cache("bench", test_data)
    
    # Load it back
    loaded = _load_cached_benchmark("bench", DEFAULT_CACHE_DIR)
    assert loaded is not None, "Should load data"
    assert len(loaded) == 1, "Should have one item"
    assert loaded[0]["question"] == "Test Q", "Should preserve data"
    
    cleanup_test_cache()


def test_convert_cached_data_to_qa_pairs():
    """Test 8: Convert cached format to QA pairs"""
    cached_data = [
        {"context": "Q1", "good_response": "A1", "bad_response": "W1"},
        {"context": "Q2", "good_response": "A2", "bad_response": "W2"},
        {"context": "Q3", "good_response": "A3", "bad_response": "W3"}
    ]
    
    # Without limit
    qa_pairs = _convert_cached_data_to_qa_pairs(cached_data)
    assert len(qa_pairs) == 3, "Should convert all pairs"
    assert qa_pairs[0]["question"] == "Q1", "Should map context to question"
    assert qa_pairs[0]["correct_answer"] == "A1", "Should map good_response to correct_answer"
    assert qa_pairs[0]["incorrect_answer"] == "W1", "Should map bad_response to incorrect_answer"
    
    # With limit
    qa_pairs_limited = _convert_cached_data_to_qa_pairs(cached_data, limit=2)
    assert len(qa_pairs_limited) == 2, "Should respect limit"


def run_all_tests():
    """Run all _try_cached tests."""
    print("="*80)
    print("TESTING _try_cached() AND RELATED CACHE FUNCTIONS")
    print("="*80)
    
    # Ensure clean state before all tests
    cleanup_test_cache()
    
    test_functions = [
        test_try_cached_not_cached,
        test_try_cached_force_download,
        test_try_cached_disabled,
        test_try_cached_with_data,
        test_try_cached_with_limit,
        test_is_benchmark_cached,
        test_load_cached_benchmark,
        test_convert_cached_data_to_qa_pairs,
    ]
    
    failed = []
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except AssertionError as e:
            failed.append((test_func.__name__, str(e)))
            print(f"✗ {test_func.__name__} failed: {e}")
        except Exception as e:
            failed.append((test_func.__name__, f"Test crashed: {e}"))
            print(f"✗ {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final cleanup
    cleanup_test_cache()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(test_functions)}")
    print(f"Passed: {len(test_functions) - len(failed)} ✓")
    print(f"Failed: {len(failed)} ✗")
    
    if failed:
        print("\nFailed tests:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    
    if not failed:
        print("\n🎉 ALL TESTS PASSED! _try_cached() and cache functions verified.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())