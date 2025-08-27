"""
Minimal happy-path tests for Activations and ActivationAggregationStrategy.

Uses tiny-random-gpt2 model for real model output testing.
Focuses on basic functionality validation rather than edge cases.
"""

import os

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from wisent_guard.core.activations import ActivationAggregationStrategy, Activations
from wisent_guard.core.layer import Layer


class TestActivations:
    """Test Activations class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Create test tensors
        self.test_tensor = torch.randn(1, 10, 768)  # [batch, seq_len, hidden_dim]
        self.layer = Layer(index=5, type="transformer")

    def teardown_method(self):
        """Clean up after tests."""
        for env_var in ["HF_HUB_DISABLE_PROGRESS_BARS", "TOKENIZERS_PARALLELISM"]:
            if env_var in os.environ:
                del os.environ[env_var]

    def test_basic_creation(self):
        """Test basic Activations object creation."""
        activations = Activations(
            tensor=self.test_tensor, layer=self.layer, aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN
        )

        assert activations.tensor.shape == (1, 10, 768)
        assert activations.layer.index == 5
        assert activations.aggregation_method == ActivationAggregationStrategy.LAST_TOKEN

    def test_aggregation_last_token(self):
        """Test last token aggregation."""
        activations = Activations(
            tensor=self.test_tensor, layer=self.layer, aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN
        )

        aggregated = activations.get_aggregated()
        assert aggregated.shape == (768,)
        # Should be last token of first batch
        assert torch.allclose(aggregated, self.test_tensor[0, -1, :])

    def test_aggregation_mean_pooling(self):
        """Test mean pooling aggregation."""
        # Note: MEAN_POOLING maps to MEAN internally in get_aggregated
        activations = Activations(
            tensor=self.test_tensor, layer=self.layer, aggregation_strategy=ActivationAggregationStrategy.MEAN_POOLING
        )

        aggregated = activations.get_aggregated()
        assert aggregated.shape == (768,)

    def test_aggregation_max_pooling(self):
        """Test max pooling aggregation."""
        # Note: MAX_POOLING maps to MAX internally in get_aggregated
        activations = Activations(
            tensor=self.test_tensor, layer=self.layer, aggregation_strategy=ActivationAggregationStrategy.MAX_POOLING
        )

        aggregated = activations.get_aggregated()
        assert aggregated.shape == (768,)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        activations = Activations(self.test_tensor, self.layer)
        other_tensor = torch.randn(768)

        similarity = activations.calculate_similarity(other_tensor, method="cosine")
        assert 0.0 <= similarity <= 1.0  # Normalized cosine similarity

    def test_dot_product_similarity(self):
        """Test dot product similarity calculation."""
        activations = Activations(self.test_tensor, self.layer)
        other_tensor = torch.randn(768)

        similarity = activations.calculate_similarity(other_tensor, method="dot")
        assert isinstance(similarity, float)

    def test_euclidean_distance(self):
        """Test euclidean distance calculation."""
        activations = Activations(self.test_tensor, self.layer)
        other_tensor = torch.randn(768)

        distance = activations.calculate_similarity(other_tensor, method="euclidean")
        assert distance <= 0  # Negative distance (higher = more similar)

    def test_normalization(self):
        """Test activation normalization."""
        activations = Activations(self.test_tensor, self.layer)
        normalized = activations.normalize()

        aggregated = normalized.get_aggregated()
        norm = torch.norm(aggregated, p=2)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)

    def test_statistics(self):
        """Test getting statistics."""
        activations = Activations(self.test_tensor, self.layer)
        stats = activations.get_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "norm" in stats
        assert "shape" in stats
        assert stats["shape"] == [768]  # Aggregated shape

    def test_to_device(self):
        """Test moving activations to different device."""
        activations = Activations(self.test_tensor, self.layer)
        cpu_activations = activations.to_device("cpu")

        assert str(cpu_activations.tensor.device) == "cpu"

    def test_extract_features_for_classifier(self):
        """Test feature extraction for classifier."""
        activations = Activations(self.test_tensor, self.layer)
        features = activations.extract_features_for_classifier()

        assert isinstance(features, torch.Tensor)
        assert features.shape == (768,)  # Flattened

    @pytest.mark.slow
    def test_from_model_output(self):
        """Test creation from real model output using tiny-random-gpt2."""
        # Load tiny model
        model_name = "hf-internal-testing/tiny-random-gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        # Set pad_token_id if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Generate some output
        inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # Create Activations from model output
        layer = Layer(index=2, type="transformer")
        activations = Activations.from_model_output(
            outputs, layer, aggregation_method=ActivationAggregationStrategy.LAST_TOKEN
        )

        assert activations.layer.index == 2
        assert activations.tensor.shape[0] == 1  # Batch size
        assert activations.aggregation_method == ActivationAggregationStrategy.LAST_TOKEN

    def test_compare_with_vectors(self):
        """Test comparing with multiple contrastive vectors."""
        activations = Activations(self.test_tensor, self.layer)

        vector_dict = {"harmful": torch.randn(768), "safe": torch.randn(768)}

        results = activations.compare_with_vectors(vector_dict, threshold=0.7)

        assert "harmful" in results
        assert "safe" in results
        assert "similarity" in results["harmful"]
        assert "is_harmful" in results["harmful"]
        assert "threshold" in results["harmful"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
