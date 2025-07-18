"""
Tests for model persistence with steering vectors.
"""

import pytest
import torch
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from transformers import GPT2Config

from wisent_guard.core.models import SteeringCompatibleModel, create_steering_compatible_model


class TestSteeringCompatibleModel:
    """Test suite for SteeringCompatibleModel."""
    
    def test_init(self):
        """Test model initialization."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        assert model.steering_active is True
        assert len(model.steering_vectors) == 0
        assert len(model.steering_metadata) == 0
        assert model._steering_layer_indices.numel() == 0
    
    def test_add_steering_vector(self):
        """Test adding steering vectors."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Add steering vector
        steering_vector = torch.randn(64)
        metadata = {"method": "CAA", "timestamp": "2025-01-18"}
        
        model.add_steering_vector(0, steering_vector, metadata)
        
        assert 0 in model.steering_vectors
        assert torch.equal(model.steering_vectors[0], steering_vector)
        assert model.steering_metadata[0] == metadata
        assert 0 in model._steering_layer_indices
        
        # Check buffer was registered
        assert hasattr(model, "_steering_vector_0")
    
    def test_add_multiple_steering_vectors(self):
        """Test adding multiple steering vectors."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=3, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Add multiple steering vectors
        for layer_idx in [0, 2]:
            steering_vector = torch.randn(64)
            model.add_steering_vector(layer_idx, steering_vector)
        
        assert len(model.steering_vectors) == 2
        assert 0 in model.steering_vectors
        assert 2 in model.steering_vectors
        assert torch.equal(model._steering_layer_indices, torch.tensor([0, 2]))
    
    def test_add_steering_vector_invalid_layer(self):
        """Test adding steering vector to invalid layer."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        steering_vector = torch.randn(64)
        
        with pytest.raises(ValueError, match="Layer index 5 out of range"):
            model.add_steering_vector(5, steering_vector)
    
    def test_remove_steering_vector(self):
        """Test removing steering vectors."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Add steering vector
        steering_vector = torch.randn(64)
        model.add_steering_vector(0, steering_vector)
        
        # Remove steering vector
        model.remove_steering_vector(0)
        
        assert 0 not in model.steering_vectors
        assert 0 not in model.steering_metadata
        assert 0 not in model._steering_layer_indices
        assert not hasattr(model, "_steering_vector_0")
    
    def test_enable_disable_steering(self):
        """Test enabling and disabling steering."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Initially enabled
        assert model.steering_active is True
        
        # Disable
        model.disable_steering()
        assert model.steering_active is False
        
        # Enable
        model.enable_steering()
        assert model.steering_active is True
    
    def test_forward_without_steering(self):
        """Test forward pass without steering vectors."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Create test input
        input_ids = torch.randint(0, 100, (1, 10))
        
        # Forward pass should work normally
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.logits.shape == (1, 10, 100)
    
    def test_forward_with_steering_disabled(self):
        """Test forward pass with steering vectors but steering disabled."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Add steering vector but disable steering
        steering_vector = torch.randn(64)
        model.add_steering_vector(0, steering_vector)
        model.disable_steering()
        
        # Create test input
        input_ids = torch.randint(0, 100, (1, 10))
        
        # Forward pass should work normally (no steering applied)
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.logits.shape == (1, 10, 100)
    
    def test_forward_with_steering_enabled(self):
        """Test forward pass with steering vectors enabled."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Add steering vector
        steering_vector = torch.randn(64)
        model.add_steering_vector(0, steering_vector)
        
        # Create test input
        input_ids = torch.randint(0, 100, (1, 10))
        
        # Forward pass should work with steering applied
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.logits.shape == (1, 10, 100)
    
    def test_get_steering_info(self):
        """Test getting steering information."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Initially no steering
        info = model.get_steering_info()
        assert info["active"] is True
        assert info["num_vectors"] == 0
        assert info["layers"] == []
        
        # Add steering vector
        steering_vector = torch.randn(64)
        metadata = {"method": "CAA"}
        model.add_steering_vector(0, steering_vector, metadata)
        
        info = model.get_steering_info()
        assert info["active"] is True
        assert info["num_vectors"] == 1
        assert info["layers"] == [0]
        assert info["metadata"][0] == metadata
    
    def test_save_and_load_steering_vectors(self):
        """Test saving and loading steering vectors."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Add steering vectors
        steering_vector_0 = torch.randn(64)
        steering_vector_1 = torch.randn(64)
        metadata_0 = {"method": "CAA", "layer": 0}
        metadata_1 = {"method": "CAA", "layer": 1}
        
        model.add_steering_vector(0, steering_vector_0, metadata_0)
        model.add_steering_vector(1, steering_vector_1, metadata_1)
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_steering_vectors(temp_dir)
            
            # Check files were created
            files = os.listdir(temp_dir)
            assert len(files) == 4  # 2 .pt files + 2 .json files
            
            # Create new model and load steering vectors
            new_model = SteeringCompatibleModel(config)
            new_model.load_steering_vectors(temp_dir)
            
            # Check vectors were loaded
            assert len(new_model.steering_vectors) == 2
            assert torch.allclose(new_model.steering_vectors[0], steering_vector_0)
            assert torch.allclose(new_model.steering_vectors[1], steering_vector_1)
    
    def test_restore_steering_vectors_after_save_load(self):
        """Test that steering vectors are restored after saving/loading full model."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Add steering vector
        steering_vector = torch.randn(64)
        metadata = {"method": "CAA"}
        model.add_steering_vector(0, steering_vector, metadata)
        
        # Save and load model state dict
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            torch.save(model.state_dict(), model_path)
            
            # Create new model and load state dict with strict=False
            new_model = SteeringCompatibleModel(config)
            new_model.load_state_dict(torch.load(model_path), strict=False)
            
            # Check steering vector was restored
            assert len(new_model.steering_vectors) == 1
            assert torch.allclose(new_model.steering_vectors[0], steering_vector)
    
    @patch('wisent_guard.core.models.steering_compatible_model.GPT2LMHeadModel.from_pretrained')
    @patch('wisent_guard.core.models.steering_compatible_model.AutoConfig.from_pretrained')
    def test_from_pretrained_with_steering(self, mock_config, mock_model):
        """Test loading from pretrained with steering vectors."""
        # Setup mocks
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        mock_config.return_value = config
        
        mock_base_model = MagicMock()
        mock_base_model.state_dict.return_value = {}
        mock_model.return_value = mock_base_model
        
        # Test loading without steering
        model = SteeringCompatibleModel.from_pretrained_with_steering("distilgpt2")
        assert isinstance(model, SteeringCompatibleModel)
        
        # Test loading with steering directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy steering files - correct format: steering_vector_<model>_<layer>_<date>.pt
            torch.save(torch.randn(64), os.path.join(temp_dir, "steering_vector_distilgpt2_0_20250118.pt"))
            
            model = SteeringCompatibleModel.from_pretrained_with_steering(
                "distilgpt2",
                steering_directory=temp_dir
            )
            assert isinstance(model, SteeringCompatibleModel)
    
    def test_create_steering_compatible_model(self):
        """Test the convenience function for creating models."""
        with patch('wisent_guard.core.models.steering_compatible_model.SteeringCompatibleModel.from_pretrained_with_steering') as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model
            
            model = create_steering_compatible_model("distilgpt2")
            
            mock_from_pretrained.assert_called_once_with("distilgpt2", steering_directory=None)
            assert model == mock_model
    
    def test_load_steering_vectors_nonexistent_directory(self):
        """Test loading steering vectors from nonexistent directory."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        with pytest.raises(ValueError, match="Directory .* does not exist"):
            model.load_steering_vectors("/nonexistent/directory")
    
    def test_load_steering_vectors_invalid_filename(self):
        """Test loading steering vectors with invalid filenames."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with invalid name
            torch.save(torch.randn(64), os.path.join(temp_dir, "invalid_filename.pt"))
            
            # Should not raise error but should log warning
            model.load_steering_vectors(temp_dir)
            
            # No steering vectors should be loaded
            assert len(model.steering_vectors) == 0


@pytest.mark.integration
class TestModelPersistenceIntegration:
    """Integration tests for model persistence."""
    
    def test_end_to_end_persistence(self):
        """Test complete end-to-end model persistence."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        
        # Create model with steering vectors
        model = SteeringCompatibleModel(config)
        steering_vector = torch.randn(64)
        metadata = {
            "method": "CAA",
            "timestamp": "2025-01-18T10:00:00",
            "experiment": "test_experiment"
        }
        model.add_steering_vector(0, steering_vector, metadata)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            torch.save(model.state_dict(), model_path)
            
            # Create new model and load
            new_model = SteeringCompatibleModel(config)
            new_model.load_state_dict(torch.load(model_path), strict=False)
            
            # Test that steering vectors were restored
            assert len(new_model.steering_vectors) == 1
            assert torch.allclose(new_model.steering_vectors[0], steering_vector)
            
            # Test that both models have the same base weights (should be identical)
            # Compare a few key parameters to verify the models are the same
            assert torch.allclose(model.transformer.wte.weight, new_model.transformer.wte.weight)
            assert torch.allclose(model.lm_head.weight, new_model.lm_head.weight)
            
            # Test that steering can be enabled/disabled
            new_model.disable_steering()
            assert not new_model.steering_active
            
            new_model.enable_steering()
            assert new_model.steering_active
            
            # Test steering info
            info = new_model.get_steering_info()
            assert info["num_vectors"] == 1
            assert info["layers"] == [0]
    
    def test_persistence_with_multiple_vectors(self):
        """Test persistence with multiple steering vectors."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=3, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Add multiple steering vectors
        vectors = {}
        for layer_idx in [0, 2]:
            vector = torch.randn(64)
            vectors[layer_idx] = vector
            model.add_steering_vector(layer_idx, vector)
        
        # Save and load
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            torch.save(model.state_dict(), model_path)
            
            new_model = SteeringCompatibleModel(config)
            new_model.load_state_dict(torch.load(model_path), strict=False)
            
            # Check all vectors were restored
            assert len(new_model.steering_vectors) == 2
            for layer_idx, vector in vectors.items():
                assert torch.allclose(new_model.steering_vectors[layer_idx], vector)
    
    def test_separate_steering_vector_persistence(self):
        """Test separate steering vector file persistence."""
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        model = SteeringCompatibleModel(config)
        
        # Add steering vector with metadata
        steering_vector = torch.randn(64)
        metadata = {"method": "CAA", "training_data": "livecodebench_v1"}
        model.add_steering_vector(0, steering_vector, metadata)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save steering vectors separately
            model.save_steering_vectors(temp_dir)
            
            # Check files exist
            files = os.listdir(temp_dir)
            pt_files = [f for f in files if f.endswith('.pt')]
            json_files = [f for f in files if f.endswith('.json')]
            
            assert len(pt_files) == 1
            assert len(json_files) == 1
            
            # Load into new model
            new_model = SteeringCompatibleModel(config)
            new_model.load_steering_vectors(temp_dir)
            
            # Check vector and metadata
            assert torch.allclose(new_model.steering_vectors[0], steering_vector)
            # Note: metadata is loaded from JSON file, not from model state