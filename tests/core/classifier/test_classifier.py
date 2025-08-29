"""
Minimal happy-path tests for Classifier class.

Focuses on basic functionality validation with synthetic data.
"""

import os
import tempfile

import pytest
import torch

from wisent_guard.core.classifier import Classifier


class TestClassifier:
    """Test Classifier class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create synthetic data
        self.input_dim = 128
        self.num_samples = 50

        torch.manual_seed(42)
        self.X_list = [torch.randn(self.input_dim) for _ in range(self.num_samples)]
        self.y_list = [float(i % 2) for i in range(self.num_samples)]  # Alternating 0s and 1s

        # Tensor versions
        self.X_tensor = torch.stack(self.X_list)
        self.y_tensor = torch.tensor(self.y_list, dtype=torch.float32)

    def test_initialization_default(self):
        """Test default initialization."""
        classifier = Classifier()

        assert classifier.model_type == "logistic"
        assert classifier.threshold == 0.5
        assert classifier.device in ["cpu", "cuda", "mps"]
        assert classifier.model is None

    def test_initialization_with_params(self):
        """Test initialization with specific parameters."""
        classifier = Classifier(model_type="mlp", device="cpu", threshold=0.7)

        assert classifier.model_type == "mlp"
        assert classifier.device == "cpu"
        assert classifier.threshold == 0.7

    def test_feature_extraction(self):
        """Test feature extraction from tensors."""
        classifier = Classifier(device="cpu")

        # Test with single tensor
        tensor = torch.randn(10, 5)
        features = classifier._extract_features(tensor)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (50,)  # Flattened
        assert features.device.type == "cpu"

    def test_fit_logistic(self):
        """Test training logistic classifier."""
        classifier = Classifier(model_type="logistic", device="cpu")

        results = classifier.fit(self.X_list, self.y_list, num_epochs=10, batch_size=16, test_size=0.2)

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "auc" in results
        assert 0.0 <= results["accuracy"] <= 1.0
        assert classifier.model is not None

    def test_fit_mlp(self):
        """Test training MLP classifier."""
        classifier = Classifier(model_type="mlp", device="cpu")

        results = classifier.fit(
            self.X_list,  # Use list input to avoid dimension issues
            self.y_list,  # Use list labels
            num_epochs=10,
            batch_size=16,
            test_size=0.2,
        )

        assert "accuracy" in results
        assert "epochs" in results
        assert results["epochs"] <= 10
        assert classifier.model is not None

    def test_predict(self):
        """Test prediction after training."""
        classifier = Classifier(model_type="logistic", device="cpu")

        # Train first
        classifier.fit(self.X_list, self.y_list, num_epochs=5)

        # Test single prediction
        prediction = classifier.predict([self.X_list[0]])
        assert prediction in [0, 1]

        # Test batch prediction
        predictions = classifier.predict(self.X_list[:5])
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self):
        """Test probability prediction."""
        classifier = Classifier(model_type="logistic", device="cpu")

        # Train first
        classifier.fit(self.X_list, self.y_list, num_epochs=5)

        # Test single probability
        prob = classifier.predict_proba([self.X_list[0]])
        assert 0.0 <= prob <= 1.0

        # Test batch probabilities
        probs = classifier.predict_proba(self.X_list[:5])
        assert len(probs) == 5
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_evaluate(self):
        """Test evaluation method."""
        classifier = Classifier(model_type="logistic", device="cpu")

        # Train first
        classifier.fit(self.X_list, self.y_list, num_epochs=5)

        # Evaluate on subset
        metrics = classifier.evaluate(self.X_list[:20], self.y_list[:20])

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_and_load_model(self):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pt")

            # Train and save
            classifier1 = Classifier(model_type="logistic", device="cpu")
            classifier1.fit(self.X_list, self.y_list, num_epochs=5)
            original_pred = classifier1.predict(self.X_list[:5])
            classifier1.save_model(model_path)

            assert os.path.exists(model_path)

            # Load in new classifier
            classifier2 = Classifier(model_type="logistic", device="cpu")
            classifier2.load_model(model_path)
            loaded_pred = classifier2.predict(self.X_list[:5])

            # Predictions should be the same
            assert list(loaded_pred) == list(original_pred)

    def test_initialization_with_model_path(self):
        """Test loading model during initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pt")

            # Train and save first
            classifier1 = Classifier(model_type="logistic", device="cpu")
            classifier1.fit(self.X_list, self.y_list, num_epochs=5)
            classifier1.save_model(model_path)

            # Create new classifier with model path
            classifier2 = Classifier(model_type="logistic", device="cpu", model_path=model_path)
            assert classifier2.model is not None

            # Should be able to predict immediately
            pred = classifier2.predict([self.X_list[0]])
            assert pred in [0, 1]

    def test_device_handling(self):
        """Test that tensors are moved to correct device."""
        classifier = Classifier(device="cpu")

        # Create tensor on different device (cpu in this case)
        tensor = torch.randn(10)
        features = classifier._extract_features(tensor)

        assert features.device.type == "cpu"

    def test_mixed_input_types(self):
        """Test classifier handles both list and tensor inputs."""
        classifier = Classifier(model_type="logistic", device="cpu")

        # Mix of input types in training
        results = classifier.fit(
            self.X_list,  # List input
            self.y_tensor,  # Tensor labels
            num_epochs=5,
        )

        assert "accuracy" in results

        # Can predict with list input after training with mixed types
        pred = classifier.predict(self.X_list[:5])
        assert len(pred) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
