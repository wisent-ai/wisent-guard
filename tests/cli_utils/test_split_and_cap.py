#!/usr/bin/env python3
"""
Test suite for _split_and_cap function in cli_prepare_dataset.py
Tests data splitting and capping functionality.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent_guard.cli_utils.cli_prepare_dataset import _split_and_cap

# Import Caps dataclass
@dataclass
class Caps:
    train: int
    test: int


def test_split_and_cap_basic():
    """Test basic splitting with 80/20 ratio"""
    
    print("="*80)
    print("Test 1: Basic split with 80/20 ratio")
    print("="*80)
    
    # Create test data
    items = [
        {"id": i, "text": f"Item {i}"} 
        for i in range(100)
    ]
    
    # Split with 80/20 ratio, no caps
    train, test = _split_and_cap(
        items=items,
        split_ratio=0.8,
        caps=Caps(train=1000, test=1000),  # High caps, won't limit
        seed=42,
        verbose=True
    )
    
    print(f"\n   Input: {len(items)} items")
    print(f"   Split ratio: 0.8")
    print(f"   Results:")
    print(f"   - Training: {len(train)} items")
    print(f"   - Testing: {len(test)} items")
    print(f"   - Total: {len(train) + len(test)} items")
    
    # Verify split
    assert len(train) == 80, f"Expected 80 training items, got {len(train)}"
    assert len(test) == 20, f"Expected 20 test items, got {len(test)}"
    assert len(train) + len(test) == 100, "Total should equal input"
    
    print("\n   ✓ Split correctly: 80 train, 20 test")
    print("   ✓ Test passed")


def test_split_and_cap_with_train_cap():
    """Test splitting with training cap applied"""
    
    print("\n" + "="*80)
    print("Test 2: Split with training cap")
    print("="*80)
    
    # Create test data
    items = [
        {"id": i, "text": f"Item {i}"} 
        for i in range(100)
    ]
    
    # Split with cap on training data
    train, test = _split_and_cap(
        items=items,
        split_ratio=0.8,
        caps=Caps(train=50, test=1000),  # Cap training at 50
        seed=42,
        verbose=True
    )
    
    print(f"\n   Input: {len(items)} items")
    print(f"   Split ratio: 0.8 (would be 80 train)")
    print(f"   Caps: train=50, test=1000")
    print(f"   Results:")
    print(f"   - Training: {len(train)} items (capped)")
    print(f"   - Testing: {len(test)} items")
    
    # Verify cap was applied
    assert len(train) == 50, f"Expected 50 training items (capped), got {len(train)}"
    assert len(test) == 20, f"Expected 20 test items, got {len(test)}"
    
    print("\n   ✓ Training capped at 50")
    print("   ✓ Test unchanged at 20")
    print("   ✓ Test passed")


def test_split_and_cap_with_test_cap():
    """Test splitting with test cap applied"""
    
    print("\n" + "="*80)
    print("Test 3: Split with test cap")
    print("="*80)
    
    # Create test data
    items = [
        {"id": i, "text": f"Item {i}"} 
        for i in range(100)
    ]
    
    # Split with cap on test data
    train, test = _split_and_cap(
        items=items,
        split_ratio=0.8,
        caps=Caps(train=1000, test=10),  # Cap test at 10
        seed=42,
        verbose=True
    )
    
    print(f"\n   Input: {len(items)} items")
    print(f"   Split ratio: 0.8 (would be 20 test)")
    print(f"   Caps: train=1000, test=10")
    print(f"   Results:")
    print(f"   - Training: {len(train)} items")
    print(f"   - Testing: {len(test)} items (capped)")
    
    # Verify cap was applied
    assert len(train) == 80, f"Expected 80 training items, got {len(train)}"
    assert len(test) == 10, f"Expected 10 test items (capped), got {len(test)}"
    
    print("\n   ✓ Training unchanged at 80")
    print("   ✓ Test capped at 10")
    print("   ✓ Test passed")


def test_split_and_cap_both_caps():
    """Test splitting with both caps applied"""
    
    print("\n" + "="*80)
    print("Test 4: Split with both caps")
    print("="*80)
    
    # Create test data
    items = [
        {"id": i, "text": f"Item {i}"} 
        for i in range(100)
    ]
    
    # Split with both caps
    train, test = _split_and_cap(
        items=items,
        split_ratio=0.8,
        caps=Caps(train=30, test=5),  # Cap both
        seed=42,
        verbose=True
    )
    
    print(f"\n   Input: {len(items)} items")
    print(f"   Split ratio: 0.8")
    print(f"   Caps: train=30, test=5")
    print(f"   Results:")
    print(f"   - Training: {len(train)} items (capped)")
    print(f"   - Testing: {len(test)} items (capped)")
    
    # Verify both caps were applied
    assert len(train) == 30, f"Expected 30 training items (capped), got {len(train)}"
    assert len(test) == 5, f"Expected 5 test items (capped), got {len(test)}"
    
    print("\n   ✓ Training capped at 30")
    print("   ✓ Test capped at 5")
    print("   ✓ Test passed")


def test_split_and_cap_deterministic():
    """Test that splitting is deterministic with same seed"""
    
    print("\n" + "="*80)
    print("Test 5: Deterministic splitting with seed")
    print("="*80)
    
    # Create test data
    items = [
        {"id": i, "text": f"Item {i}"} 
        for i in range(50)
    ]
    
    # First split
    train1, test1 = _split_and_cap(
        items=items,
        split_ratio=0.7,
        caps=Caps(train=1000, test=1000),
        seed=123,
        verbose=False
    )
    
    # Second split with same seed
    train2, test2 = _split_and_cap(
        items=items,
        split_ratio=0.7,
        caps=Caps(train=1000, test=1000),
        seed=123,
        verbose=False
    )
    
    # Third split with different seed
    train3, test3 = _split_and_cap(
        items=items,
        split_ratio=0.7,
        caps=Caps(train=1000, test=1000),
        seed=456,
        verbose=False
    )
    
    print(f"\n   Input: {len(items)} items")
    print(f"   Split ratio: 0.7")
    print(f"\n   First split (seed=123):")
    print(f"   - First 3 train IDs: {[t['id'] for t in train1[:3]]}")
    print(f"\n   Second split (seed=123):")
    print(f"   - First 3 train IDs: {[t['id'] for t in train2[:3]]}")
    print(f"\n   Third split (seed=456):")
    print(f"   - First 3 train IDs: {[t['id'] for t in train3[:3]]}")
    
    # Verify same seed produces same results
    assert train1 == train2, "Same seed should produce identical training sets"
    assert test1 == test2, "Same seed should produce identical test sets"
    
    # Verify different seed produces different results
    assert train1 != train3, "Different seeds should produce different training sets"
    
    print("\n   ✓ Same seed (123) produces identical splits")
    print("   ✓ Different seed (456) produces different split")
    print("   ✓ Test passed")


def test_split_and_cap_edge_cases():
    """Test edge cases: empty list, single item, extreme ratios"""
    
    print("\n" + "="*80)
    print("Test 6: Edge cases")
    print("="*80)
    
    # Test 1: Empty list
    print("\n   Test 6a: Empty list")
    train, test = _split_and_cap(
        items=[],
        split_ratio=0.8,
        caps=Caps(train=100, test=100),
        seed=42,
        verbose=False
    )
    assert len(train) == 0 and len(test) == 0, "Empty input should produce empty outputs"
    print("   ✓ Empty list handled correctly")
    
    # Test 2: Single item with 0.5 split
    print("\n   Test 6b: Single item")
    train, test = _split_and_cap(
        items=[{"id": 1}],
        split_ratio=0.5,
        caps=Caps(train=100, test=100),
        seed=42,
        verbose=False
    )
    assert len(train) + len(test) == 1, "Single item should be in either train or test"
    print(f"   - Single item went to: {'train' if len(train)==1 else 'test'}")
    print("   ✓ Single item handled correctly")
    
    # Test 3: Extreme split ratio (0.0)
    print("\n   Test 6c: Split ratio 0.0 (all test)")
    items = [{"id": i} for i in range(10)]
    train, test = _split_and_cap(
        items=items,
        split_ratio=0.0,
        caps=Caps(train=100, test=100),
        seed=42,
        verbose=False
    )
    assert len(train) == 0, "Ratio 0.0 should put all in test"
    assert len(test) == 10, "Ratio 0.0 should put all in test"
    print("   ✓ Split ratio 0.0 handled correctly (0 train, 10 test)")
    
    # Test 4: Extreme split ratio (1.0)
    print("\n   Test 6d: Split ratio 1.0 (all train)")
    train, test = _split_and_cap(
        items=items,
        split_ratio=1.0,
        caps=Caps(train=100, test=100),
        seed=42,
        verbose=False
    )
    assert len(train) == 10, "Ratio 1.0 should put all in train"
    assert len(test) == 0, "Ratio 1.0 should put all in train"
    print("   ✓ Split ratio 1.0 handled correctly (10 train, 0 test)")
    
    print("\n   ✓ All edge cases passed")


def test_split_and_cap_different_ratios():
    """Test various split ratios"""
    
    print("\n" + "="*80)
    print("Test 7: Various split ratios")
    print("="*80)
    
    items = [{"id": i} for i in range(100)]
    
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n   Input: {len(items)} items")
    print("\n   Testing different ratios:")
    
    for ratio in ratios:
        train, test = _split_and_cap(
            items=items,
            split_ratio=ratio,
            caps=Caps(train=1000, test=1000),
            seed=42,
            verbose=False
        )
        
        expected_train = int(100 * ratio)
        expected_test = 100 - expected_train
        
        print(f"   - Ratio {ratio}: {len(train)} train, {len(test)} test " +
              f"(expected {expected_train}/{expected_test})")
        
        assert len(train) == expected_train, f"Incorrect train size for ratio {ratio}"
        assert len(test) == expected_test, f"Incorrect test size for ratio {ratio}"
    
    print("\n   ✓ All ratios split correctly")
    print("   ✓ Test passed")


def test_split_and_cap_preserves_data():
    """Test that no data is lost or duplicated during split"""
    
    print("\n" + "="*80)
    print("Test 8: Data preservation (no loss or duplication)")
    print("="*80)
    
    # Create test data with unique IDs
    items = [{"id": i, "value": f"val_{i}"} for i in range(73)]  # Odd number
    
    train, test = _split_and_cap(
        items=items,
        split_ratio=0.7,
        caps=Caps(train=1000, test=1000),
        seed=999,
        verbose=False
    )
    
    print(f"\n   Input: {len(items)} items with unique IDs")
    print(f"   Split ratio: 0.7")
    print(f"   Results: {len(train)} train, {len(test)} test")
    
    # Check no data loss
    total_after = len(train) + len(test)
    assert total_after == len(items), f"Data lost! Had {len(items)}, now have {total_after}"
    
    # Check no duplication
    all_ids = [item['id'] for item in train + test]
    unique_ids = set(all_ids)
    assert len(all_ids) == len(unique_ids), "Duplicate items found!"
    
    # Check all original IDs are present
    original_ids = set(item['id'] for item in items)
    split_ids = set(item['id'] for item in train + test)
    assert original_ids == split_ids, "Some IDs missing or changed!"
    
    print("\n   ✓ No data lost")
    print("   ✓ No duplicates created")
    print("   ✓ All original items preserved")
    print("   ✓ Test passed")


def run_all_tests():
    """Run all tests"""
    print("\nRunning _split_and_cap tests...")
    print("="*80)
    
    test_functions = [
        test_split_and_cap_basic,
        test_split_and_cap_with_train_cap,
        test_split_and_cap_with_test_cap,
        test_split_and_cap_both_caps,
        test_split_and_cap_deterministic,
        test_split_and_cap_edge_cases,
        test_split_and_cap_different_ratios,
        test_split_and_cap_preserves_data,
    ]
    
    failed = []
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} FAILED: {e}")
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