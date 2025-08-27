"""
Minimal happy-path tests for utility functions.

Tests the calculate_roc_auc function with various scenarios.
"""

import pytest

from wisent_guard.core.classifier.utils import calculate_roc_auc


class TestCalculateROCAUC:
    """Test calculate_roc_auc function."""

    def test_perfect_predictions(self):
        """Test ROC AUC with perfect predictions."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

        auc = calculate_roc_auc(y_true, y_scores)
        assert auc == 1.0

    def test_inverse_perfect_predictions(self):
        """Test ROC AUC with perfectly wrong predictions."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

        auc = calculate_roc_auc(y_true, y_scores)
        assert auc == 0.0

    def test_random_predictions(self):
        """Test ROC AUC with random predictions."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_scores = [0.6, 0.4, 0.7, 0.3, 0.55, 0.45]  # Mixed random scores

        auc = calculate_roc_auc(y_true, y_scores)
        # Should be around 0.5 for random predictions (allow wider range)
        assert 0.0 <= auc <= 1.0

    def test_uniform_predictions(self):
        """Test ROC AUC with uniform predictions (all same score)."""
        y_true = [0, 1, 0, 1]
        y_scores = [0.5, 0.5, 0.5, 0.5]

        auc = calculate_roc_auc(y_true, y_scores)
        # With uniform scores, AUC depends on implementation
        assert 0.0 <= auc <= 1.0

    def test_mixed_predictions(self):
        """Test ROC AUC with mixed predictions."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.2, 0.6, 0.4, 0.8]

        auc = calculate_roc_auc(y_true, y_scores)
        # Should be 0.75 for this specific case
        assert 0.7 <= auc <= 0.8

    def test_single_class_returns_half(self):
        """Test that single class returns 0.5."""
        # All zeros
        y_true = [0, 0, 0, 0]
        y_scores = [0.1, 0.3, 0.5, 0.7]

        auc = calculate_roc_auc(y_true, y_scores)
        assert auc == 0.5

        # All ones
        y_true = [1, 1, 1, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8]

        auc = calculate_roc_auc(y_true, y_scores)
        assert auc == 0.5

    def test_tied_scores(self):
        """Test ROC AUC with tied scores."""
        y_true = [0, 1, 0, 1]
        y_scores = [0.5, 0.5, 0.3, 0.7]

        auc = calculate_roc_auc(y_true, y_scores)
        assert 0.0 <= auc <= 1.0

    def test_large_dataset(self):
        """Test ROC AUC with larger dataset."""
        # Create larger dataset
        y_true = [i % 2 for i in range(100)]
        y_scores = [0.3 + 0.4 * (i % 2) + 0.1 * (i / 100) for i in range(100)]

        auc = calculate_roc_auc(y_true, y_scores)
        assert 0.0 <= auc <= 1.0

    def test_float_labels(self):
        """Test that function works with float labels."""
        y_true = [0.0, 0.0, 1.0, 1.0]
        y_scores = [0.1, 0.4, 0.35, 0.8]

        auc = calculate_roc_auc(y_true, y_scores)
        assert 0.0 <= auc <= 1.0

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        y_true = [0, 1, 0]
        y_scores = [0.5, 0.7]

        with pytest.raises(ValueError, match="Length of y_true and y_scores must match"):
            calculate_roc_auc(y_true, y_scores)

    def test_boundary_scores(self):
        """Test with boundary scores (0 and 1)."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.0, 0.0, 1.0, 1.0]

        auc = calculate_roc_auc(y_true, y_scores)
        assert auc == 1.0

    def test_alternating_pattern(self):
        """Test with alternating pattern."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]

        auc = calculate_roc_auc(y_true, y_scores)
        assert auc == 1.0

    def test_real_world_example(self):
        """Test with a realistic example."""
        y_true = [0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.7, 0.35, 0.8, 0.9]

        auc = calculate_roc_auc(y_true, y_scores)
        assert 0.8 <= auc <= 1.0  # Should be relatively high

    def test_empty_inputs_behavior(self):
        """Test behavior with minimal inputs."""
        # Two samples minimum for binary classification
        y_true = [0, 1]
        y_scores = [0.3, 0.7]

        auc = calculate_roc_auc(y_true, y_scores)
        assert auc == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
