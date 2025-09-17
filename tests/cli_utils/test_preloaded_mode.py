#!/usr/bin/env python3
"""
Test suite for _preloaded_mode() function in cli_prepare_dataset.py
Tests preloaded QA pairs mode handling.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent_guard.cli_utils.cli_prepare_dataset import _preloaded_mode, PrepState


def test_preloaded_mode_none():
    """Test 1: Preloaded mode with None"""
    result = _preloaded_mode(
        preloaded=None,
        verbose=False
    )
    
    assert result is None, "Should return None when preloaded is None"


def test_preloaded_mode_empty_list():
    """Test 2: Preloaded mode with empty list"""
    result = _preloaded_mode(
        preloaded=[],
        verbose=False
    )
    
    # Empty list is falsy in Python
    assert result is None, "Should return None when preloaded is empty list"


def test_preloaded_mode_with_qa_pairs():
    """Test 3: Preloaded mode with valid QA pairs"""
    qa_pairs = [
        {"question": "What is 2+2?", "correct": "4", "incorrect": "5"},
        {"question": "What is the capital of France?", "correct": "Paris", "incorrect": "London"},
        {"question": "What color is the sky?", "correct": "Blue", "incorrect": "Green"},
    ]
    
    result = _preloaded_mode(
        preloaded=qa_pairs,
        verbose=False
    )
    
    # Verify result
    assert result is not None, "Should return PrepState when preloaded has data"
    assert isinstance(result, PrepState), f"Expected PrepState, got {type(result)}"
    
    # Check PrepState fields based on function implementation:
    # PrepState(pairs, list(preloaded), True, True, None, [], False, False, [])
    assert result.qa_pairs == qa_pairs, "qa_pairs should match input"
    assert result.test_source == qa_pairs, "test_source should be a copy of input"
    assert result.qa_pairs is not qa_pairs, "qa_pairs should be a new list (not same reference)"
    assert result.group_processed == True, "group_processed should be True"
    assert result.group_qa_format == True, "group_qa_format should be True"
    assert result.task_data is None, "task_data should be None"
    assert result.train_docs == [], "train_docs should be empty list"
    assert result.skip_qa_display == False, "skip_qa_display should be False"
    assert result.used_cache == False, "used_cache should be False"
    assert result.all_cached_pairs == [], "all_cached_pairs should be empty list"


def test_preloaded_mode_with_verbose():
    """Test 4: Preloaded mode with verbose logging"""
    qa_pairs = [{"q": f"Question {i}"} for i in range(10)]
    
    # Note: We can't easily test the actual log output without capturing logs,
    # but we can verify the function executes correctly with verbose=True
    result = _preloaded_mode(
        preloaded=qa_pairs,
        verbose=True
    )
    
    assert result is not None, "Should return PrepState with verbose=True"
    assert isinstance(result, PrepState), "Should return PrepState object"
    assert len(result.qa_pairs) == 10, "Should have all 10 pairs"


def test_preloaded_mode_large_dataset():
    """Test 5: Preloaded mode with large dataset"""
    large_dataset = [
        {"id": i, "question": f"Q{i}", "answer": f"A{i}"}
        for i in range(10000)
    ]
    
    result = _preloaded_mode(
        preloaded=large_dataset,
        verbose=False
    )
    
    assert result is not None, "Should handle large datasets"
    assert len(result.qa_pairs) == 10000, "Should have all 10000 pairs"
    assert len(result.test_source) == 10000, "test_source should have all 10000 items"


def test_preloaded_mode_tuple_input():
    """Test 6: Preloaded mode with tuple instead of list"""
    # Function accepts Sequence type, which includes tuples
    qa_tuple = (
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    )
    
    result = _preloaded_mode(
        preloaded=qa_tuple,
        verbose=False
    )
    
    assert result is not None, "Should work with tuple input"
    assert isinstance(result.qa_pairs, list), "qa_pairs should be converted to list"
    assert isinstance(result.test_source, list), "test_source should be a list"
    assert len(result.qa_pairs) == 2, "Should have both items"



def run_all_tests():
    """Run all preloaded mode tests."""
    print("="*80)
    print("TESTING _preloaded_mode() FUNCTION")
    print("="*80)
    
    test_functions = [
        test_preloaded_mode_none,
        test_preloaded_mode_empty_list,
        test_preloaded_mode_with_qa_pairs,
        test_preloaded_mode_with_verbose,
        test_preloaded_mode_large_dataset,
        test_preloaded_mode_tuple_input,
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
        print("\n🎉 ALL TESTS PASSED! _preloaded_mode() function verified completely.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())