#!/usr/bin/env python3
"""
Simple test to verify that wg_harness imports work correctly.
"""

def test_imports():
    """Test that all wg_harness modules can be imported."""
    
    print("Testing wisent_guard.wg_harness imports...")
    
    try:
        # Test basic package import
        import wisent_guard.wg_harness
        print("✓ wisent_guard.wg_harness imported successfully")
        
        # Test individual module imports
        from wisent_guard.wg_harness import data
        print("✓ data module imported successfully")
        
        from wisent_guard.wg_harness import generate
        print("✓ generate module imported successfully")
        
        from wisent_guard.wg_harness import labeler
        print("✓ labeler module imported successfully")
        
        from wisent_guard.wg_harness import train_guard
        print("✓ train_guard module imported successfully")
        
        from wisent_guard.wg_harness import evaluate
        print("✓ evaluate module imported successfully")
        
        from wisent_guard.wg_harness import cli
        print("✓ cli module imported successfully")
        
        # Test key class imports
        from wisent_guard.wg_harness.train_guard import GuardPipeline
        print("✓ GuardPipeline class imported successfully")
        
        from wisent_guard.wg_harness.data import load_task, split_docs
        print("✓ data functions imported successfully")
        
        from wisent_guard.wg_harness.generate import generate_responses
        print("✓ generate_responses function imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_optional_dependencies():
    """Test optional dependencies with graceful fallbacks."""
    
    print("\nTesting optional dependencies...")
    
    # Test lm-eval
    try:
        import lm_eval
        print("✓ lm-eval is available")
    except ImportError:
        print("⚠ lm-eval not available (install with: pip install lm-eval)")
    
    # Test sklearn
    try:
        import sklearn
        print("✓ scikit-learn is available")
    except ImportError:
        print("⚠ scikit-learn not available (install with: pip install scikit-learn)")
    
    # Test pandas
    try:
        import pandas
        print("✓ pandas is available")
    except ImportError:
        print("⚠ pandas not available (install with: pip install pandas)")

if __name__ == "__main__":
    success = test_imports()
    test_optional_dependencies()
    
    if success:
        print("\n✅ All tests passed! wg_harness is ready to use.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        exit(1) 