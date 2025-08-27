"""
Minimal happy-path tests for ActivationClassifier class.

Tests the integration between Activations and Classifier.
"""

import os
import tempfile

import pytest
import torch

from wisent_guard.core.activations import ActivationAggregationStrategy, Activations
from wisent_guard.core.classifier import ActivationClassifier
from wisent_guard.core.layer import Layer


class TestActivationClassifier:
    """Test ActivationClassifier functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create synthetic activations
        self.layer = Layer(index=5, type="transformer")
        self.hidden_dim = 256

        # Create harmful and harmless activation samples
        torch.manual_seed(42)
        self.harmful_activations = []
        self.harmless_activations = []

        for _ in range(20):
            # Harmful activations (slightly higher values)
            harmful_tensor = torch.randn(1, 10, self.hidden_dim) + 0.5
            self.harmful_activations.append(
                Activations(
                    tensor=harmful_tensor,
                    layer=self.layer,
                    aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                )
            )

            # Harmless activations (slightly lower values)
            harmless_tensor = torch.randn(1, 10, self.hidden_dim) - 0.5
            self.harmless_activations.append(
                Activations(
                    tensor=harmless_tensor,
                    layer=self.layer,
                    aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                )
            )

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        classifier = ActivationClassifier(model_type="mlp", threshold=0.7, device="cpu")

        assert classifier.classifier.model_type == "mlp"
        assert classifier.classifier.threshold == 0.7
        assert classifier.classifier.device == "cpu"

    def test_train_on_activations(self):
        """Test training on activation data."""
        classifier = ActivationClassifier(device="cpu")

        results = classifier.train_on_activations(
            harmful_activations=self.harmful_activations[:10], harmless_activations=self.harmless_activations[:10]
        )

        assert classifier.is_trained is True
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results

    def test_predict_from_activations(self):
        """Test prediction from activations."""
        classifier = ActivationClassifier(device="cpu")

        # Train first
        classifier.train_on_activations(
            harmful_activations=self.harmful_activations[:10], harmless_activations=self.harmless_activations[:10]
        )

        # Test prediction on harmful activation
        result = classifier.predict_from_activations(self.harmful_activations[15])

        assert "is_harmful" in result
        assert "probability" in result
        assert "confidence" in result
        assert isinstance(result["is_harmful"], bool)
        assert 0.0 <= result["probability"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_before_training_raises_error(self):
        """Test that prediction before training raises error."""
        classifier = ActivationClassifier()

        with pytest.raises(ValueError, match="Classifier must be trained before prediction"):
            classifier.predict_from_activations(self.harmful_activations[0])

    def test_multiple_predictions(self):
        """Test batch predictions on multiple activations."""
        classifier = ActivationClassifier(device="cpu")

        # Train
        classifier.train_on_activations(
            harmful_activations=self.harmful_activations[:10], harmless_activations=self.harmless_activations[:10]
        )

        # Predict on multiple samples individually
        results = []
        for activation in self.harmful_activations[10:15]:
            result = classifier.predict_from_activations(activation)
            results.append(result)

        assert len(results) == 5
        for result in results:
            assert "is_harmful" in result
            assert "probability" in result

    def test_save_and_load_classifier(self):
        """Test saving and loading trained classifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "activation_classifier.pt")

            # Train and save
            classifier1 = ActivationClassifier(device="cpu")
            classifier1.train_on_activations(
                harmful_activations=self.harmful_activations[:10], harmless_activations=self.harmless_activations[:10]
            )

            # Get prediction before saving
            pred1 = classifier1.predict_from_activations(self.harmful_activations[15])

            # Save
            classifier1.save(model_path)
            assert os.path.exists(model_path)

            # Load in new classifier
            classifier2 = ActivationClassifier(device="cpu")
            classifier2.load(model_path)
            assert classifier2.is_trained is True

            # Get prediction after loading
            pred2 = classifier2.predict_from_activations(self.harmful_activations[15])

            # Predictions should be very similar
            assert pred1["is_harmful"] == pred2["is_harmful"]
            assert abs(pred1["probability"] - pred2["probability"]) < 0.01

    def test_mlp_classifier(self):
        """Test ActivationClassifier with MLP model."""
        classifier = ActivationClassifier(model_type="mlp", device="cpu")

        results = classifier.train_on_activations(
            harmful_activations=self.harmful_activations[:10], harmless_activations=self.harmless_activations[:10]
        )

        assert classifier.is_trained is True
        assert "accuracy" in results

        # Should be able to predict
        result = classifier.predict_from_activations(self.harmless_activations[15])
        assert "is_harmful" in result

    def test_different_thresholds(self):
        """Test classification with different thresholds."""
        # Train with default threshold
        classifier_low = ActivationClassifier(threshold=0.3, device="cpu")
        classifier_high = ActivationClassifier(threshold=0.7, device="cpu")

        # Train both
        for classifier in [classifier_low, classifier_high]:
            classifier.train_on_activations(
                harmful_activations=self.harmful_activations[:10], harmless_activations=self.harmless_activations[:10]
            )

        # Test on borderline activation
        test_activation = Activations(tensor=torch.randn(1, 10, self.hidden_dim), layer=self.layer)

        result_low = classifier_low.predict_from_activations(test_activation)
        result_high = classifier_high.predict_from_activations(test_activation)

        # Same probability but potentially different classification
        assert abs(result_low["probability"] - result_high["probability"]) < 0.01

    def test_confidence_calculation(self):
        """Test confidence calculation in predictions."""
        classifier = ActivationClassifier(device="cpu")

        classifier.train_on_activations(
            harmful_activations=self.harmful_activations[:10], harmless_activations=self.harmless_activations[:10]
        )

        # Test confidence for different probabilities
        for activation in self.harmful_activations[10:13]:
            result = classifier.predict_from_activations(activation)

            # Confidence should be 0 at probability 0.5 and increase towards 1
            expected_confidence = abs(result["probability"] - 0.5) * 2
            assert abs(result["confidence"] - expected_confidence) < 0.01

    def test_empty_training_data_handling(self):
        """Test handling of edge case with minimal training data."""
        classifier = ActivationClassifier(device="cpu")

        # Train with minimal data
        results = classifier.train_on_activations(
            harmful_activations=self.harmful_activations[:1], harmless_activations=self.harmless_activations[:1]
        )

        assert classifier.is_trained is True
        # Should still return metrics even with small dataset
        assert "accuracy" in results

    def test_activation_aggregation_strategies(self):
        """Test with different aggregation strategies."""
        # Create activations with different strategies
        strategies = [
            ActivationAggregationStrategy.LAST_TOKEN,
            ActivationAggregationStrategy.MEAN_POOLING,
            ActivationAggregationStrategy.MAX_POOLING,
        ]

        for strategy in strategies:
            # Create activations with this strategy
            harmful = [
                Activations(
                    tensor=torch.randn(1, 10, self.hidden_dim) + 0.5, layer=self.layer, aggregation_strategy=strategy
                )
                for _ in range(5)
            ]

            harmless = [
                Activations(
                    tensor=torch.randn(1, 10, self.hidden_dim) - 0.5, layer=self.layer, aggregation_strategy=strategy
                )
                for _ in range(5)
            ]

            classifier = ActivationClassifier(device="cpu")
            results = classifier.train_on_activations(harmful_activations=harmful, harmless_activations=harmless)

            assert "accuracy" in results

            # Test prediction
            test_activation = Activations(
                tensor=torch.randn(1, 10, self.hidden_dim), layer=self.layer, aggregation_strategy=strategy
            )
            result = classifier.predict_from_activations(test_activation)
            assert "is_harmful" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
