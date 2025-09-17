#!/usr/bin/env python3
"""
Test suite for _synthetic_mode() function in cli_prepare_dataset.py
Tests synthetic pair mode handling using real ContrastivePairSet objects.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent_guard.cli_utils.cli_prepare_dataset import _synthetic_mode, PrepState
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair


def test_synthetic_mode_disabled():
    """Test 1: Synthetic mode when from_synth=False"""
    result = _synthetic_mode(
        from_synth=False,
        synth_pairs=None,
        verbose=False
    )
    
    assert result is None, "Should return None when from_synth=False"


def test_synthetic_mode_no_pairs():
    """Test 2: Synthetic mode enabled but no pairs provided"""
    result = _synthetic_mode(
        from_synth=True,
        synth_pairs=None,
        verbose=False
    )
    
    assert result is None, "Should return None when synth_pairs is None"


def test_synthetic_mode_enabled_with_pairs():
    """Test 3: Synthetic mode with valid ContrastivePairSet"""
    # Create real ContrastivePairSet with some pairs
    pairs = [
        ContrastivePair(
            prompt="What is 2+2?",
            positive_response=None,  # These can be None for this test
            negative_response=None
        ),
        ContrastivePair(
            prompt="What is the capital of France?",
            positive_response=None,
            negative_response=None
        ),
        ContrastivePair(
            prompt="Explain quantum mechanics",
            positive_response=None,
            negative_response=None
        ),
    ]
    
    pair_set = ContrastivePairSet(
        name="test_synthetic_task",
        pairs=pairs,
        task_type="qa"
    )
    
    result = _synthetic_mode(
        from_synth=True,
        synth_pairs=pair_set,
        verbose=False
    )
    
    # Verify result
    assert result is not None, "Should return PrepState when both conditions are met"
    assert isinstance(result, PrepState), f"Expected PrepState, got {type(result)}"
    
    # Check PrepState fields based on function implementation:
    # PrepState([], [], True, True, None, [], False, False, [])
    assert result.qa_pairs == [], "qa_pairs should be empty list"
    assert result.test_source == [], "test_source should be empty list"
    assert result.group_processed == True, "group_processed should be True"
    assert result.group_qa_format == True, "group_qa_format should be True"
    assert result.task_data is None, "task_data should be None"
    assert result.train_docs == [], "train_docs should be empty list"
    assert result.skip_qa_display == False, "skip_qa_display should be False"
    assert result.used_cache == False, "used_cache should be False"
    assert result.all_cached_pairs == [], "all_cached_pairs should be empty list"


def test_synthetic_mode_with_verbose():
    """Test 4: Synthetic mode with verbose logging"""
    # Create ContrastivePairSet with 10 pairs
    pairs = [
        ContrastivePair(
            prompt=f"Question {i}",
            positive_response=None,
            negative_response=None
        )
        for i in range(10)
    ]
    
    pair_set = ContrastivePairSet(
        name="verbose_test_task",
        pairs=pairs
    )
    
    # Note: We can't easily test the actual log output without mocking the logger,
    # but we can verify the function executes correctly with verbose=True
    result = _synthetic_mode(
        from_synth=True,
        synth_pairs=pair_set,
        verbose=True
    )
    
    assert result is not None, "Should return PrepState with verbose=True"
    assert isinstance(result, PrepState), "Should return PrepState object"


def test_synthetic_mode_empty_pairs_list():
    """Test 5: Synthetic mode with empty pairs list"""
    # ContrastivePairSet with empty pairs list
    pair_set = ContrastivePairSet(
        name="empty_task",
        pairs=[]  # Empty list
    )
    
    # ContrastivePairSet with empty pairs evaluates to False (due to __len__ method returning 0)
    # So the function should return None
    result = _synthetic_mode(
        from_synth=True,
        synth_pairs=pair_set,
        verbose=False
    )
    
    assert result is None, "Should return None when ContrastivePairSet has no pairs (evaluates to False)"


def test_synthetic_mode_large_dataset():
    """Test 6: Synthetic mode with large dataset"""
    # Create large dataset
    pairs = [
        ContrastivePair(
            prompt=f"Question {i}",
            positive_response=None,
            negative_response=None
        )
        for i in range(10000)  # 10k pairs
    ]
    
    pair_set = ContrastivePairSet(
        name="large_synthetic_dataset",
        pairs=pairs
    )
    
    result = _synthetic_mode(
        from_synth=True,
        synth_pairs=pair_set,
        verbose=False
    )
    
    assert result is not None, "Should handle large datasets"
    assert isinstance(result, PrepState), "Should return PrepState object"
    # The function doesn't actually store the pairs in PrepState, just empty lists
    assert result.qa_pairs == [], "qa_pairs should still be empty regardless of input size"


def test_synthetic_mode_with_special_chars_in_name():
    """Test 7: Synthetic mode with special characters in task name"""
    pair_set = ContrastivePairSet(
        name="task-with_special.chars/2024@v1",
        pairs=[ContrastivePair(prompt="test", positive_response=None, negative_response=None)]
    )
    
    result = _synthetic_mode(
        from_synth=True,
        synth_pairs=pair_set,
        verbose=False
    )
    
    assert result is not None, "Should handle special characters in task name"
    assert isinstance(result, PrepState), "Should return PrepState object"


def test_synthetic_mode_false_with_valid_pairs():
    """Test 8: from_synth=False even with valid pairs"""
    pair_set = ContrastivePairSet(
        name="ignored_task",
        pairs=[ContrastivePair(prompt="test", positive_response=None, negative_response=None)]
    )
    
    result = _synthetic_mode(
        from_synth=False,  # Disabled
        synth_pairs=pair_set,  # But pairs provided
        verbose=False
    )
    
    assert result is None, "Should return None when from_synth=False regardless of pairs"




def run_all_tests():
    """Run all synthetic mode tests."""
    print("="*80)
    print("TESTING _synthetic_mode() FUNCTION")
    print("="*80)
    
    test_functions = [
        test_synthetic_mode_disabled,
        test_synthetic_mode_no_pairs,
        test_synthetic_mode_enabled_with_pairs,
        test_synthetic_mode_with_verbose,
        test_synthetic_mode_empty_pairs_list,
        test_synthetic_mode_large_dataset,
        test_synthetic_mode_with_special_chars_in_name,
        test_synthetic_mode_false_with_valid_pairs,
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
        print("\n🎉 ALL TESTS PASSED! _synthetic_mode() function verified completely.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())