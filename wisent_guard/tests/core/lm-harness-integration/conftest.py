"""
Pytest configuration for lm-harness integration tests.
"""

import pytest
import os


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for all tests."""
    # Set environment variable for code evaluation
    os.environ['HF_ALLOW_CODE_EVAL'] = '1'
    
    # Set other test-specific environment variables if needed
    original_env = dict(os.environ)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)