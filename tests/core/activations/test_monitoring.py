"""
Minimal happy-path tests for ActivationMonitor and TestActivationCache.

Focuses on basic functionality validation with synthetic data.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from wisent_guard.core.activations import (
    ActivationAggregationStrategy,
    ActivationMonitor,
    Activations,
    TestActivationCache,
)
from wisent_guard.core.layer import Layer


class TestActivationMonitor:
    """Test ActivationMonitor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = ActivationMonitor()

        # Create test activations
        self.layer1 = Layer(index=1, type="transformer")
        self.layer2 = Layer(index=2, type="transformer")

        self.activations1 = Activations(
            tensor=torch.randn(1, 5, 256),
            layer=self.layer1,
            aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
        )

        self.activations2 = Activations(
            tensor=torch.randn(1, 5, 256),
            layer=self.layer2,
            aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
        )

    def test_store_activations(self):
        """Test storing activations."""
        activations_dict = {1: self.activations1, 2: self.activations2}

        self.monitor.store_activations(activations_dict, text="Test text")

        assert len(self.monitor.activation_history) == 1
        assert self.monitor.activation_history[0]["text"] == "Test text"
        assert 1 in self.monitor.current_activations
        assert 2 in self.monitor.current_activations

    def test_analyze_activations(self):
        """Test analyzing stored activations."""
        activations_dict = {1: self.activations1, 2: self.activations2}

        self.monitor.store_activations(activations_dict)
        analysis = self.monitor.analyze_activations()

        assert 1 in analysis
        assert 2 in analysis
        assert "statistics" in analysis[1]
        assert "feature_vector" in analysis[1]
        assert "tensor_shape" in analysis[1]
        assert "device" in analysis[1]

    def test_compare_with_baseline(self):
        """Test comparing with baseline activations."""
        current_dict = {1: self.activations1, 2: self.activations2}

        # Create baseline with slightly different values
        baseline_dict = {
            1: Activations(tensor=torch.randn(1, 5, 256), layer=self.layer1),
            2: Activations(tensor=torch.randn(1, 5, 256), layer=self.layer2),
        }

        self.monitor.store_activations(current_dict)
        comparisons = self.monitor.compare_with_baseline(baseline_dict)

        assert 1 in comparisons
        assert 2 in comparisons
        assert "cosine_similarity" in comparisons[1]
        assert "dot_product" in comparisons[1]
        assert "euclidean_distance" in comparisons[1]

    def test_detect_anomalies_no_history(self):
        """Test anomaly detection with no history."""
        activations_dict = {1: self.activations1, 2: self.activations2}

        self.monitor.store_activations(activations_dict)
        anomalies = self.monitor.detect_anomalies(threshold=0.8)

        # Should return False for all since not enough history
        assert anomalies[1] is False
        assert anomalies[2] is False

    def test_detect_anomalies_with_history(self):
        """Test anomaly detection with history."""
        # Add some history
        for _ in range(3):
            activations_dict = {
                1: Activations(tensor=torch.randn(1, 5, 256), layer=self.layer1),
                2: Activations(tensor=torch.randn(1, 5, 256), layer=self.layer2),
            }
            self.monitor.store_activations(activations_dict)

        # Now check for anomalies
        anomalies = self.monitor.detect_anomalies(threshold=0.8)

        assert isinstance(anomalies[1], bool)
        assert isinstance(anomalies[2], bool)

    def test_clear_history(self):
        """Test clearing activation history."""
        activations_dict = {1: self.activations1, 2: self.activations2}

        self.monitor.store_activations(activations_dict)
        self.monitor.store_activations(activations_dict)
        assert len(self.monitor.activation_history) == 2

        self.monitor.clear_history()
        assert len(self.monitor.activation_history) == 0

    def test_save_and_load_activations(self):
        """Test saving and loading activations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_activations.pt")

            activations_dict = {1: self.activations1, 2: self.activations2}

            self.monitor.store_activations(activations_dict)
            self.monitor.save_activations(filepath)

            assert os.path.exists(filepath)

            # Create new monitor and load
            new_monitor = ActivationMonitor()
            loaded = new_monitor.load_activations(filepath)

            assert 1 in loaded
            assert 2 in loaded
            assert loaded[1].tensor.shape == self.activations1.tensor.shape


class TestTestActivationCache:
    """Test TestActivationCache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = TestActivationCache()

        # Create test data
        self.layer = Layer(index=3, type="transformer")
        self.activation = Activations(
            tensor=torch.randn(1, 10, 512),
            layer=self.layer,
            aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
        )

    def test_add_activation(self):
        """Test adding activation to cache."""
        self.cache.add_activation(question="What is 2+2?", response="4", activations=self.activation, layer=3)

        assert len(self.cache) == 1
        assert self.cache.activations[0]["question"] == "What is 2+2?"
        assert self.cache.activations[0]["response"] == "4"
        assert self.cache.activations[0]["layer"] == 3

    def test_get_activations_for_layer(self):
        """Test retrieving activations for specific layer."""
        # Add activations for different layers
        self.cache.add_activation("Q1", "R1", self.activation, layer=3)
        self.cache.add_activation("Q2", "R2", self.activation, layer=3)
        self.cache.add_activation("Q3", "R3", self.activation, layer=5)

        layer3_activations = self.cache.get_activations_for_layer(3)
        assert len(layer3_activations) == 2

        layer5_activations = self.cache.get_activations_for_layer(5)
        assert len(layer5_activations) == 1

    def test_save_and_load_cache(self):
        """Test saving and loading cache to/from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_cache.pt")

            # Add some activations
            self.cache.add_activation("Q1", "R1", self.activation, layer=3)
            self.cache.add_activation("Q2", "R2", self.activation, layer=3)

            # Save
            self.cache.save_to_file(filepath)
            assert os.path.exists(filepath)

            # Load into new cache
            loaded_cache = TestActivationCache.load_from_file(filepath)

            assert len(loaded_cache) == 2
            assert loaded_cache.activations[0]["question"] == "Q1"
            assert loaded_cache.activations[0]["response"] == "R1"
            assert loaded_cache.activations[0]["layer"] == 3

    def test_cache_repr(self):
        """Test cache string representation."""
        self.cache.add_activation("Q1", "R1", self.activation, layer=3)
        self.cache.add_activation("Q2", "R2", self.activation, layer=5)

        repr_str = repr(self.cache)
        assert "2 samples" in repr_str
        assert "layers: " in repr_str

    def test_empty_cache_save_error(self):
        """Test that saving empty cache raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty_cache.pt")

            with pytest.raises(ValueError, match="No activations to save"):
                self.cache.save_to_file(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
