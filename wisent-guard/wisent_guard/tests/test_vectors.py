"""
Tests for the ContrastiveVectors class
"""

import os
import torch
import unittest
import tempfile
import shutil
from wisent_guard.vectors import ContrastiveVectors

class TestContrastiveVectors(unittest.TestCase):
    """Test suite for ContrastiveVectors class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.vectors = ContrastiveVectors(save_dir=self.test_dir)
        
        # Create sample vectors
        self.harmful_vector = torch.randn(768)
        self.harmless_vector = torch.randn(768)
    
    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_add_vector_pair(self):
        """Test adding a vector pair."""
        category = "test_category"
        layer = 0
        
        # Add a vector pair
        self.vectors.add_vector_pair(
            category=category,
            layer=layer,
            harmful_vector=self.harmful_vector,
            harmless_vector=self.harmless_vector
        )
        
        # Check that the vectors were added correctly
        self.assertIn(category, self.vectors.harmful_vectors)
        self.assertIn(category, self.vectors.harmless_vectors)
        self.assertIn(layer, self.vectors.harmful_vectors[category])
        self.assertIn(layer, self.vectors.harmless_vectors[category])
        
        # Check that metadata was updated
        self.assertIn(category, self.vectors.metadata["categories"])
        self.assertEqual(self.vectors.metadata["num_pairs"][category], 1)
        self.assertIn(str(layer), self.vectors.metadata["layers"])
    
    def test_compute_contrastive_vectors(self):
        """Test computing contrastive vectors."""
        category = "test_category"
        layer = 0
        
        # Add a vector pair
        self.vectors.add_vector_pair(
            category=category,
            layer=layer,
            harmful_vector=self.harmful_vector,
            harmless_vector=self.harmless_vector
        )
        
        # Compute contrastive vectors
        contrastive_vectors = self.vectors.compute_contrastive_vectors()
        
        # Check that contrastive vectors were computed correctly
        self.assertIn(category, contrastive_vectors)
        self.assertIn(layer, contrastive_vectors[category])
        
        # Check that the contrastive vector is normalized
        contrastive_vector = contrastive_vectors[category][layer]
        norm = torch.norm(contrastive_vector, p=2).item()
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_save_and_load_vectors(self):
        """Test saving and loading vectors."""
        category = "test_category"
        layer = 0
        
        # Add a vector pair
        self.vectors.add_vector_pair(
            category=category,
            layer=layer,
            harmful_vector=self.harmful_vector,
            harmless_vector=self.harmless_vector
        )
        
        # Compute and save vectors
        self.vectors.compute_contrastive_vectors()
        self.vectors.save_vectors()
        
        # Check that files were created
        category_dir = os.path.join(self.test_dir, "vectors", category)
        vector_path = os.path.join(category_dir, f"contrastive_layer_{layer}.pt")
        harmful_path = os.path.join(category_dir, "harmful_vectors.pkl")
        harmless_path = os.path.join(category_dir, "harmless_vectors.pkl")
        
        self.assertTrue(os.path.exists(vector_path))
        self.assertTrue(os.path.exists(harmful_path))
        self.assertTrue(os.path.exists(harmless_path))
        
        # Create a new vectors object and load the saved vectors
        new_vectors = ContrastiveVectors(save_dir=self.test_dir)
        success = new_vectors.load_vectors()
        
        # Check that vectors were loaded successfully
        self.assertTrue(success)
        self.assertIn(category, new_vectors.contrastive_vectors)
        self.assertIn(layer, new_vectors.contrastive_vectors[category])
        
        # Check that metadata was loaded correctly
        self.assertIn(category, new_vectors.metadata["categories"])
        self.assertEqual(new_vectors.metadata["num_pairs"][category], 1)
    
    def test_get_contrastive_vector(self):
        """Test getting a contrastive vector."""
        category = "test_category"
        layer = 0
        
        # Add a vector pair and compute contrastive vectors
        self.vectors.add_vector_pair(
            category=category,
            layer=layer,
            harmful_vector=self.harmful_vector,
            harmless_vector=self.harmless_vector
        )
        self.vectors.compute_contrastive_vectors()
        
        # Get the contrastive vector
        contrastive_vector = self.vectors.get_contrastive_vector(category, layer)
        
        # Check that the contrastive vector is correct
        self.assertIsNotNone(contrastive_vector)
        self.assertEqual(contrastive_vector.shape, self.harmful_vector.shape)
        
        # Check that getting a non-existent vector returns None
        self.assertIsNone(self.vectors.get_contrastive_vector("nonexistent", layer))
        self.assertIsNone(self.vectors.get_contrastive_vector(category, 999))
    
    def test_get_available_categories_and_layers(self):
        """Test getting available categories and layers."""
        categories = ["category1", "category2"]
        layers = [0, 1, 2]
        
        # Add vector pairs for different categories and layers
        for category in categories:
            for layer in layers:
                self.vectors.add_vector_pair(
                    category=category,
                    layer=layer,
                    harmful_vector=self.harmful_vector,
                    harmless_vector=self.harmless_vector
                )
        
        # Check that available categories and layers are correct
        self.assertEqual(set(self.vectors.get_available_categories()), set(categories))
        self.assertEqual(set(self.vectors.get_available_layers()), set(layers))

if __name__ == "__main__":
    unittest.main() 