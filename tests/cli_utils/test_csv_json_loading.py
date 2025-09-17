#!/usr/bin/env python3
"""
Comprehensive test for _load_from_csv_json() function in cli_prepare_dataset.py
Tests CSV and JSON loading with various conditions and edge cases.
"""

import json
import csv
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent_guard.cli_utils.cli_prepare_dataset import (
    _load_from_csv_json, 
    PrepState, 
    Caps
)
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet


def create_test_csv(path: Path, data: List[Dict[str, str]]):
    """Create a test CSV file."""
    if not data:
        path.write_text("")
        return
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def create_test_json(path: Path, data: List[Dict[str, Any]]):
    """Create a test JSON file."""
    path.write_text(json.dumps(data, indent=2))




def test_csv_loading_disabled():
    """Test 1: CSV loading when from_csv=False"""
    result = _load_from_csv_json(
        from_csv=False,
        from_json=False,
        task_name="dummy.csv",
        question_col="question",
        correct_col="correct",
        incorrect_col="incorrect",
        limit=None,
        caps=Caps(train=100, test=50),
        split_ratio=0.8,
        seed=42,
        verbose=False
    )
    
    assert result is None, f"Expected None when from_csv=False, got {type(result)}"


def test_csv_loading_basic():
    """Test 2: Basic CSV loading with valid data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        
        # Create test data
        test_data = [
            {"question": "What is 2+2?", "correct_answer": "4", "incorrect_answer": "5"},
            {"question": "What is 3+3?", "correct_answer": "6", "incorrect_answer": "7"},
            {"question": "What is 4+4?", "correct_answer": "8", "incorrect_answer": "9"},
            {"question": "What is 5+5?", "correct_answer": "10", "incorrect_answer": "11"},
            {"question": "What is 6+6?", "correct_answer": "12", "incorrect_answer": "13"},
        ]
        create_test_csv(csv_path, test_data)
        
        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=42,
            verbose=True
        )
        
        # Verify result
        assert result is not None, "CSV loading returned None"
        assert isinstance(result, PrepState), f"Expected PrepState, got {type(result)}"
        
        # Check data split (80/20)
        total_items = len(result.qa_pairs) + len(result.test_source)
        assert total_items == 5, f"Expected 5 items, got {total_items}"
        
        # Check split ratio (approximately 80/20)
        expected_train = 4  # 80% of 5
        expected_test = 1   # 20% of 5
        assert len(result.qa_pairs) == expected_train, f"Expected {expected_train} train, got {len(result.qa_pairs)}"
        assert len(result.test_source) == expected_test, f"Expected {expected_test} test, got {len(result.test_source)}"
        
        # Check flags
        assert result.group_processed == True, f"group_processed should be True, got {result.group_processed}"
        assert result.group_qa_format == True, f"group_qa_format should be True, got {result.group_qa_format}"


def test_json_loading_basic():
    """Test 3: Basic JSON loading with valid data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test.json"
        
        # Create test data in the expected JSON format
        test_data = [
            {"question": f"Question {i}", "correct_answer": f"Answer {i}", "incorrect_answer": f"Wrong {i}"}
            for i in range(10)
        ]
        create_test_json(json_path, test_data)
        
        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=False,
            from_json=True,
            task_name=str(json_path),
            question_col="question",
            correct_col="correct_answer",  # Fixed column name
            incorrect_col="incorrect_answer",  # Fixed column name
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=0.7,  # 70/30 split
            seed=42,
            verbose=False
        )
        
        # Verify result
        assert result is not None, "JSON loading returned None"
        assert isinstance(result, PrepState), f"Expected PrepState, got {type(result)}"
        
        # Check data split
        total_items = len(result.qa_pairs) + len(result.test_source)
        assert total_items == 10, f"Expected 10 items, got {total_items}"
        
        # Check 70/30 split
        expected_train = 7  # 70% of 10
        expected_test = 3   # 30% of 10
        assert len(result.qa_pairs) == expected_train, f"Expected {expected_train} train, got {len(result.qa_pairs)}"
        assert len(result.test_source) == expected_test, f"Expected {expected_test} test, got {len(result.test_source)}"
        
        # Check flags
        assert result.group_processed == True, f"group_processed should be True, got {result.group_processed}"
        assert result.group_qa_format == True, f"group_qa_format should be True, got {result.group_qa_format}"


def test_with_limits():
    """Test 4: CSV loading with limit parameter"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        
        # Create large dataset
        test_data = [
            {"question": f"Q{i}", "correct_answer": f"A{i}", "incorrect_answer": f"W{i}"}
            for i in range(100)
        ]
        create_test_csv(csv_path, test_data)
        
        # Only load first 20 items
        limit = 20
        
        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=limit,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=42,
            verbose=False
        )
        
        # Check that limit was applied
        total = len(result.qa_pairs) + len(result.test_source)
        assert total == limit, f"Expected {limit} items, got {total}"


def test_with_caps():
    """Test 5: Loading with training and testing caps"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        
        # Create dataset
        test_data = [
            {"question": f"Q{i}", "correct_answer": f"A{i}", "incorrect_answer": f"W{i}"}
            for i in range(50)
        ]
        create_test_csv(csv_path, test_data)
        
        # Set small caps
        train_cap = 5
        test_cap = 2
        
        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=train_cap, test=test_cap),
            split_ratio=0.8,
            seed=42,
            verbose=True
        )
        
        # Check caps were applied
        assert len(result.qa_pairs) <= train_cap, f"Train cap exceeded: {len(result.qa_pairs)} > {train_cap}"
        assert len(result.test_source) <= test_cap, f"Test cap exceeded: {len(result.test_source)} > {test_cap}"


def test_empty_file():
    """Test 6: Loading empty CSV file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "empty.csv"
        # Create empty CSV with headers only
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "correct_answer", "incorrect_answer"])
            writer.writeheader()
        
        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=42,
            verbose=False
        )
        
        assert result is not None, "Result is None for empty file"
        assert len(result.qa_pairs) == 0, f"Expected 0 train items for empty file, got {len(result.qa_pairs)}"
        assert len(result.test_source) == 0, f"Expected 0 test items for empty file, got {len(result.test_source)}"


def test_deterministic_split():
    """Test 7: Verify deterministic splitting with seed"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        
        test_data = [
            {"question": f"Q{i}", "correct_answer": f"A{i}", "incorrect_answer": f"W{i}"}
            for i in range(20)
        ]
        create_test_csv(csv_path, test_data)
        
        # Run twice with same seed - using actual ContrastivePairSet
        result1 = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=42,
            verbose=False
        )
        
        result2 = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=42,
            verbose=False
        )
        
        # Check if splits are identical
        assert result1.qa_pairs == result2.qa_pairs, "Same seed should produce identical train splits"
        assert result1.test_source == result2.test_source, "Same seed should produce identical test splits"
        
        # Run with different seed
        result3 = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="question",
            correct_col="correct_answer",
            incorrect_col="incorrect_answer",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=99,
            verbose=False
        )
        
        # Should be different
        assert result1.qa_pairs != result3.qa_pairs, "Different seeds should produce different splits"


def test_different_column_names():
    """Test 8: CSV with custom column names"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "custom.csv"
        
        # Use different column names
        test_data = [
            {"query": "What is 2+2?", "right": "4", "wrong": "5"},
            {"query": "What is 3+3?", "right": "6", "wrong": "7"},
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["query", "right", "wrong"])
            writer.writeheader()
            writer.writerows(test_data)
        
        # Use actual ContrastivePairSet - no mocking
        result = _load_from_csv_json(
            from_csv=True,
            from_json=False,
            task_name=str(csv_path),
            question_col="query",
            correct_col="right",
            incorrect_col="wrong",
            limit=None,
            caps=Caps(train=100, test=50),
            split_ratio=0.8,
            seed=42,
            verbose=False
        )
        
        # Just verify that custom columns work and we get results
        assert result is not None, "Result is None with custom columns"
        assert len(result.qa_pairs) + len(result.test_source) == 2, "Custom columns should load all data"


def test_split_ratios():
    """Test 9: Various split ratios"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        
        test_data = [
            {"question": f"Q{i}", "correct_answer": f"A{i}", "incorrect_answer": f"W{i}"}
            for i in range(100)
        ]
        create_test_csv(csv_path, test_data)
        
        test_ratios = [0.5, 0.7, 0.9, 0.99]
        
        for ratio in test_ratios:
            # Use actual ContrastivePairSet - no mocking
            result = _load_from_csv_json(
                from_csv=True,
                from_json=False,
                task_name=str(csv_path),
                question_col="question",
                correct_col="correct_answer",
                incorrect_col="incorrect_answer",
                limit=None,
                caps=Caps(train=1000, test=1000),  # High caps to test ratio
                split_ratio=ratio,
                seed=42,
                verbose=False
            )
            
            expected_train = int(100 * ratio)
            expected_test = 100 - expected_train
            
            # Allow ±1 for rounding
            assert abs(len(result.qa_pairs) - expected_train) <= 1, f"Ratio {ratio}: Expected ~{expected_train} train, got {len(result.qa_pairs)}"




def run_all_tests():
    """Run all CSV/JSON loading tests."""
    print("="*80)
    print("COMPREHENSIVE TESTING OF _load_from_csv_json() FUNCTION")
    print("="*80)
    
    test_functions = [
        test_csv_loading_disabled,
        test_csv_loading_basic,
        test_json_loading_basic,
        test_with_limits,
        test_with_caps,
        test_empty_file,
        test_deterministic_split,
        test_different_column_names,
        test_split_ratios,
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
        print("\n🎉 ALL TESTS PASSED! _load_from_csv_json() function verified completely.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())