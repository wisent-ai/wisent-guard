"""
Tests for the ContrastiveVectors class
"""

import os
import torch
import pytest
import tempfile
import shutil
from wisent_guard.vectors import ContrastiveVectors


@pytest.mark.slow
@pytest.mark.unit
class TestContrastiveVectors:
    """Test suite for ContrastiveVectors class."""
    
    @pytest.fixture(autouse=True)
    def setup_vectors(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        yield
        
        # Tear down test fixtures
        shutil.rmtree(self.test_dir)
    
    def test_vectors_initialization(self):
        """Test that ContrastiveVectors initializes correctly."""
        # Use a small model for testing
        vectors = ContrastiveVectors(
            model_name="distilbert/distilgpt2",
            layers=[0, 1],
            save_dir=self.test_dir
        )
        
        # Check that initialization worked
        assert vectors.model is not None
        assert len(vectors.layers) == 2
        assert vectors.save_dir.exists()
        
        # Check that layers are correctly set
        layer_indices = [layer.index for layer in vectors.layers]
        assert layer_indices == [0, 1]
    
    def test_create_pair_set(self):
        """Test creating a contrastive pair set."""
        vectors = ContrastiveVectors(
            model_name="distilbert/distilgpt2",
            layers=[0],
            save_dir=self.test_dir
        )
        
        # Create a simple pair set
        harmful_texts = ["This is harmful text", "Another harmful example"]
        harmless_texts = ["This is harmless text", "Another harmless example"]
        
        pair_set = vectors.create_pair_set(
            name="test_pairs",
            harmful_texts=harmful_texts,
            harmless_texts=harmless_texts
        )
        
        # Check that pair set was created
        assert pair_set is not None
        assert "test_pairs" in vectors.pair_sets
        assert len(pair_set.phrase_pairs) == 2
    
    def test_compute_vectors_basic(self):
        """Test basic vector computation."""
        vectors = ContrastiveVectors(
            model_name="distilbert/distilgpt2",
            layers=[0],
            save_dir=self.test_dir
        )
        
        # Create a simple pair set
        harmful_texts = ["Bad content"]
        harmless_texts = ["Good content"]
        
        vectors.create_pair_set(
            name="test_pairs",
            harmful_texts=harmful_texts,
            harmless_texts=harmless_texts
        )
        
        # Compute vectors
        computed_vectors = vectors.compute_vectors("test_pairs", layer_idx=0)
        
        # Check that vectors were computed
        assert isinstance(computed_vectors, dict)
        assert 0 in computed_vectors  # Layer 0 should have a vector
        
        # Check that the vector is stored
        assert "test_pairs" in vectors.vectors
        assert 0 in vectors.vectors["test_pairs"]
    
    def test_get_similarity_basic(self):
        """Test basic similarity computation."""
        vectors = ContrastiveVectors(
            model_name="distilbert/distilgpt2",
            layers=[0],
            save_dir=self.test_dir
        )
        
        # Create and compute vectors
        harmful_texts = ["Bad content"]
        harmless_texts = ["Good content"]
        
        vectors.create_pair_set(
            name="test_pairs",
            harmful_texts=harmful_texts,
            harmless_texts=harmless_texts
        )
        
        vectors.compute_vectors("test_pairs", layer_idx=0)
        
        # Test similarity
        similarity = vectors.get_similarity(
            text="Some test text",
            category="test_pairs",
            layer_idx=0
        )
        
        # Check that similarity is a number
        assert isinstance(similarity, (int, float))
    
    def test_vectors_storage_structure(self):
        """Test that vectors are stored in correct structure."""
        vectors = ContrastiveVectors(
            model_name="distilbert/distilgpt2",
            layers=[0, 1],
            save_dir=self.test_dir
        )
        
        # Create pair set
        vectors.create_pair_set(
            name="test_pairs",
            harmful_texts=["Bad"],
            harmless_texts=["Good"]
        )
        
        # Compute for all layers
        vectors.compute_vectors("test_pairs")
        
        # Check storage structure
        assert "test_pairs" in vectors.vectors
        assert isinstance(vectors.vectors["test_pairs"], dict)
        
        # Should have computed vectors for both layers
        available_layers = list(vectors.vectors["test_pairs"].keys())
        assert len(available_layers) > 0  # At least one layer should work
    
    def test_layer_configuration(self):
        """Test different layer configurations."""
        # Test with single layer
        vectors_single = ContrastiveVectors(
            model_name="distilbert/distilgpt2",
            layers=[0],
            save_dir=self.test_dir
        )
        assert len(vectors_single.layers) == 1
        
        # Test with multiple layers
        vectors_multi = ContrastiveVectors(
            model_name="distilbert/distilgpt2",
            layers=[0, 1, 2],
            save_dir=self.test_dir
        )
        assert len(vectors_multi.layers) == 3
        
        # Check layer indices
        layer_indices = [layer.index for layer in vectors_multi.layers]
        assert layer_indices == [0, 1, 2]
    
    def test_save_directory_creation(self):
        """Test that save directory is created correctly."""
        # Use a nested path
        nested_path = os.path.join(self.test_dir, "nested", "vectors")
        
        vectors = ContrastiveVectors(
            model_name="distilbert/distilgpt2",
            layers=[0],
            save_dir=nested_path
        )
        
        # Check that directory was created
        assert os.path.exists(nested_path)
        assert vectors.save_dir.exists()
        assert vectors.save_dir.is_dir()