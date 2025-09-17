#!/usr/bin/env python3
"""
Test suite for _is_group_task function in cli_prepare_dataset.py
Tests group task detection without using mocks.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent_guard.cli_utils.cli_prepare_dataset import _is_group_task, SLOW_GROUP_TASKS


def test_is_group_task_slow_tasks():
    """Test 1: Skip group detection for known slow tasks"""
    # These tasks are in SLOW_GROUP_TASKS and should be skipped
    for task in SLOW_GROUP_TASKS:
        is_group, expanded = _is_group_task(task, verbose=False)
        assert is_group == False, f"{task} should skip group detection"
        assert expanded == [], f"{task} should have empty expansion"


def test_is_group_task_slow_tasks_verbose():
    """Test 2: Skip group detection for known slow tasks with verbose logging"""
    # Test with verbose=True to ensure logging works
    task = "livecodebench"  # Known to be in SLOW_GROUP_TASKS
    is_group, expanded = _is_group_task(task, verbose=True)
    
    assert is_group == False, "Should skip group detection for livecodebench"
    assert expanded == [], "Should have empty expansion for skipped task"


def test_is_group_task_unknown():
    """Test 3: Handle unknown/non-existent tasks"""
    # Test with a task that doesn't exist
    is_group, expanded = _is_group_task("definitely_not_a_real_task_xyz123", verbose=False)
    
    # Should return valid results even for unknown tasks
    assert isinstance(is_group, bool), "Should return boolean"
    assert isinstance(expanded, list), "Should return list"
    assert is_group == False, "Unknown task should not be a group"
    assert expanded == [], "Unknown task should have empty expansion"


def test_is_group_task_return_types():
    """Test 4: Verify return types are always correct"""
    # Test various task names to ensure consistent return types
    test_cases = [
        "livecodebench",  # Known slow task
        "unknown_task_abc",  # Unknown task
        "",  # Empty string
        "test_task_123",  # Another unknown
    ]
    
    for task_name in test_cases:
        is_group, expanded = _is_group_task(task_name, verbose=False)
        
        assert isinstance(is_group, bool), f"is_group should be bool for {task_name}"
        assert isinstance(expanded, list), f"expanded should be list for {task_name}"
        
        # If not a group, expanded should be empty
        if not is_group:
            assert expanded == [], f"Non-group task {task_name} should have empty expansion"


def test_is_group_task_with_timeout_simulation():
    """Test 5: Handle potential timeout scenarios"""
    # Test with a very long task name that might cause processing delays
    # This should still return within reasonable time due to timeout
    long_task_name = "a" * 1000 + "_task_that_does_not_exist"
    
    is_group, expanded = _is_group_task(long_task_name, verbose=False)
    
    assert is_group == False, "Long non-existent task should not be a group"
    assert expanded == [], "Long non-existent task should have empty expansion"


def test_is_group_task_special_characters():
    """Test 6: Handle task names with special characters"""
    # Test task names with various special characters
    special_tasks = [
        "task-with-hyphens",
        "task_with_underscores",
        "task.with.dots",
        "task/with/slashes",
        "task@with@symbols",
    ]
    
    for task_name in special_tasks:
        is_group, expanded = _is_group_task(task_name, verbose=False)
        
        # Should handle gracefully without crashing
        assert isinstance(is_group, bool), f"Should handle {task_name} gracefully"
        assert isinstance(expanded, list), f"Should return list for {task_name}"


def test_is_group_task_case_sensitivity():
    """Test 7: Test case sensitivity in task names"""
    # Test same task with different cases
    test_cases = [
        ("livecodebench", True),  # Exact match with SLOW_GROUP_TASKS
        ("LIVECODEBENCH", False),  # Uppercase variant
        ("LiveCodeBench", False),  # Mixed case variant
    ]
    
    for task_name, should_skip in test_cases:
        is_group, expanded = _is_group_task(task_name, verbose=False)
        
        # Only exact lowercase match should skip
        if should_skip and task_name in SLOW_GROUP_TASKS:
            assert is_group == False, f"{task_name} should skip detection"
            assert expanded == [], f"{task_name} should have empty expansion"
        else:
            # Others will try to load and fail (returning False, [])
            assert is_group == False, f"{task_name} should not be detected as group"
            assert expanded == [], f"{task_name} should have empty expansion"


def test_is_group_task_verbose_mode():
    """Test 8: Test verbose mode doesn't affect results"""
    # Test same task with verbose on and off
    task_name = "test_task_for_verbose"
    
    # Without verbose
    is_group_quiet, expanded_quiet = _is_group_task(task_name, verbose=False)
    
    # With verbose
    is_group_verbose, expanded_verbose = _is_group_task(task_name, verbose=True)
    
    # Results should be identical
    assert is_group_quiet == is_group_verbose, "Verbose mode should not affect is_group result"
    assert expanded_quiet == expanded_verbose, "Verbose mode should not affect expanded result"


def run_all_tests():
    """Run all _is_group_task tests."""
    print("="*80)
    print("TESTING _is_group_task() FUNCTION")
    print("="*80)
    
    test_functions = [
        test_is_group_task_slow_tasks,
        test_is_group_task_slow_tasks_verbose,
        test_is_group_task_unknown,
        test_is_group_task_return_types,
        test_is_group_task_with_timeout_simulation,
        test_is_group_task_special_characters,
        test_is_group_task_case_sensitivity,
        test_is_group_task_verbose_mode,
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
        print("\n🎉 ALL TESTS PASSED! _is_group_task() function verified.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())