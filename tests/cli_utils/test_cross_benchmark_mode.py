#!/usr/bin/env python3
"""
Test suite for _cross_benchmark_mode() function in cli_prepare_dataset.py
Tests cross-benchmark evaluation mode handling.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent_guard.cli_utils.cli_prepare_dataset import _cross_benchmark_mode, PrepState
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair


def test_cross_benchmark_mode_disabled():
    """Test 1: Cross-benchmark mode when cross_mode=False"""
    result = _cross_benchmark_mode(
        cross_mode=False,
        train_pairs=None,
        eval_pairs=None,
        verbose=False
    )
    
    assert result is None, "Should return None when cross_mode=False"


def test_cross_benchmark_mode_no_train_pairs():
    """Test 2: Cross-benchmark mode enabled but no training pairs"""
    eval_pairs = ContrastivePairSet(
        name="eval_task",
        pairs=[ContrastivePair(prompt="test", positive_response=None, negative_response=None)]
    )
    
    result = _cross_benchmark_mode(
        cross_mode=True,
        train_pairs=None,
        eval_pairs=eval_pairs,
        verbose=False
    )
    
    assert result is None, "Should return None when train_pairs is None"


def test_cross_benchmark_mode_no_eval_pairs():
    """Test 3: Cross-benchmark mode enabled but no evaluation pairs"""
    train_pairs = ContrastivePairSet(
        name="train_task",
        pairs=[ContrastivePair(prompt="test", positive_response=None, negative_response=None)]
    )
    
    result = _cross_benchmark_mode(
        cross_mode=True,
        train_pairs=train_pairs,
        eval_pairs=None,
        verbose=False
    )
    
    assert result is None, "Should return None when eval_pairs is None"


def test_cross_benchmark_mode_with_both_pairs():
    """Test 4: Cross-benchmark mode with both training and evaluation pairs"""
    # Create training pairs
    train_pairs = ContrastivePairSet(
        name="math_train",
        pairs=[
            ContrastivePair(prompt="What is 2+2?", positive_response=None, negative_response=None),
            ContrastivePair(prompt="What is 3+3?", positive_response=None, negative_response=None),
            ContrastivePair(prompt="What is 4+4?", positive_response=None, negative_response=None),
        ]
    )
    
    # Create evaluation pairs
    eval_pairs = ContrastivePairSet(
        name="science_eval",
        pairs=[
            ContrastivePair(prompt="What is H2O?", positive_response=None, negative_response=None),
            ContrastivePair(prompt="What is gravity?", positive_response=None, negative_response=None),
        ]
    )
    
    result = _cross_benchmark_mode(
        cross_mode=True,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        verbose=False
    )
    
    # Verify result
    assert result is not None, "Should return PrepState when all conditions are met"
    assert isinstance(result, PrepState), f"Expected PrepState, got {type(result)}"
    
    # Check PrepState fields based on function implementation:
    # PrepState([], [], True, True, None, [], True, False, [])
    assert result.qa_pairs == [], "qa_pairs should be empty list"
    assert result.test_source == [], "test_source should be empty list"
    assert result.group_processed == True, "group_processed should be True"
    assert result.group_qa_format == True, "group_qa_format should be True"
    assert result.task_data is None, "task_data should be None"
    assert result.train_docs == [], "train_docs should be empty list"
    assert result.skip_qa_display == True, "skip_qa_display should be True (different from other modes!)"
    assert result.used_cache == False, "used_cache should be False"
    assert result.all_cached_pairs == [], "all_cached_pairs should be empty list"


def test_cross_benchmark_mode_with_verbose():
    """Test 5: Cross-benchmark mode with verbose logging"""
    # Create pairs with different sizes
    train_pairs = ContrastivePairSet(
        name="large_train_set",
        pairs=[ContrastivePair(prompt=f"Train Q{i}", positive_response=None, negative_response=None) 
               for i in range(100)]
    )
    
    eval_pairs = ContrastivePairSet(
        name="small_eval_set",
        pairs=[ContrastivePair(prompt=f"Eval Q{i}", positive_response=None, negative_response=None) 
               for i in range(20)]
    )
    
    result = _cross_benchmark_mode(
        cross_mode=True,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        verbose=True
    )
    
    assert result is not None, "Should return PrepState with verbose=True"
    assert isinstance(result, PrepState), "Should return PrepState object"
    assert result.skip_qa_display == True, "skip_qa_display should be True in cross-benchmark mode"


def test_cross_benchmark_mode_empty_pairs():
    """Test 6: Cross-benchmark mode with empty ContrastivePairSets"""
    # ContrastivePairSets with empty pairs lists
    train_pairs = ContrastivePairSet(name="empty_train", pairs=[])
    eval_pairs = ContrastivePairSet(name="empty_eval", pairs=[])
    
    # Empty ContrastivePairSets are falsy (due to __len__ method)
    result = _cross_benchmark_mode(
        cross_mode=True,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        verbose=False
    )
    
    assert result is None, "Should return None when pairs are empty (falsy)"


def test_cross_benchmark_mode_false_with_valid_pairs():
    """Test 7: cross_mode=False even with valid pairs"""
    train_pairs = ContrastivePairSet(
        name="train",
        pairs=[ContrastivePair(prompt="q1", positive_response=None, negative_response=None)]
    )
    eval_pairs = ContrastivePairSet(
        name="eval",
        pairs=[ContrastivePair(prompt="q2", positive_response=None, negative_response=None)]
    )
    
    result = _cross_benchmark_mode(
        cross_mode=False,  # Disabled
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        verbose=False
    )
    
    assert result is None, "Should return None when cross_mode=False regardless of pairs"


def test_cross_benchmark_mode_large_datasets():
    """Test 8: Cross-benchmark mode with large datasets"""
    # Create large training and evaluation sets
    train_pairs = ContrastivePairSet(
        name="massive_train",
        pairs=[ContrastivePair(prompt=f"T{i}", positive_response=None, negative_response=None) 
               for i in range(10000)]
    )
    
    eval_pairs = ContrastivePairSet(
        name="massive_eval",
        pairs=[ContrastivePair(prompt=f"E{i}", positive_response=None, negative_response=None) 
               for i in range(5000)]
    )
    
    result = _cross_benchmark_mode(
        cross_mode=True,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        verbose=False
    )
    
    assert result is not None, "Should handle large datasets"
    assert isinstance(result, PrepState), "Should return PrepState object"
    # Note: The function doesn't actually store the pairs in PrepState, just empty lists
    assert result.qa_pairs == [], "qa_pairs should still be empty regardless of input size"
    assert result.test_source == [], "test_source should still be empty"


def test_cross_benchmark_mode_different_task_types():
    """Test 9: Cross-benchmark with different task types"""
    # Training on QA task
    train_pairs = ContrastivePairSet(
        name="qa_benchmark",
        pairs=[ContrastivePair(prompt="QA question", positive_response=None, negative_response=None)],
        task_type="qa"
    )
    
    # Evaluating on classification task
    eval_pairs = ContrastivePairSet(
        name="classification_benchmark",
        pairs=[ContrastivePair(prompt="Classify this", positive_response=None, negative_response=None)],
        task_type="classification"
    )
    
    result = _cross_benchmark_mode(
        cross_mode=True,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        verbose=False
    )
    
    assert result is not None, "Should work with different task types"
    assert result.skip_qa_display == True, "skip_qa_display should be True"


def test_cross_benchmark_mode_mixed_one_empty():
    """Test 10: Cross-benchmark with one valid and one empty set"""
    # Valid training pairs
    train_pairs = ContrastivePairSet(
        name="valid_train",
        pairs=[ContrastivePair(prompt="valid", positive_response=None, negative_response=None)]
    )
    
    # Empty evaluation pairs (falsy)
    eval_pairs = ContrastivePairSet(name="empty_eval", pairs=[])
    
    result = _cross_benchmark_mode(
        cross_mode=True,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        verbose=False
    )
    
    # Should return None because eval_pairs is falsy
    assert result is None, "Should return None when one set is empty"


def run_all_tests():
    """Run all cross-benchmark mode tests."""
    print("="*80)
    print("TESTING _cross_benchmark_mode() FUNCTION")
    print("="*80)
    
    test_functions = [
        test_cross_benchmark_mode_disabled,
        test_cross_benchmark_mode_no_train_pairs,
        test_cross_benchmark_mode_no_eval_pairs,
        test_cross_benchmark_mode_with_both_pairs,
        test_cross_benchmark_mode_with_verbose,
        test_cross_benchmark_mode_empty_pairs,
        test_cross_benchmark_mode_false_with_valid_pairs,
        test_cross_benchmark_mode_large_datasets,
        test_cross_benchmark_mode_different_task_types,
        test_cross_benchmark_mode_mixed_one_empty,
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
        print("\n🎉 ALL TESTS PASSED! _cross_benchmark_mode() function verified completely.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())