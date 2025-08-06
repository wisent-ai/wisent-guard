"""
Pytest configuration for steering validation tests.
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def validation_data_dir():
    """Path to validation test data directory."""
    return Path(__file__).parent / "reference_data"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "vector_generation: Tests for steering vector generation")
    config.addinivalue_line("markers", "steering_application: Tests for steering application logic")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their names."""
    for item in items:
        if "vector_generation" in item.nodeid:
            item.add_marker(pytest.mark.vector_generation)
        elif "steering_application" in item.nodeid:
            item.add_marker(pytest.mark.steering_application)

        # All tests now use real models, so mark them as slow
        item.add_marker(pytest.mark.slow)
