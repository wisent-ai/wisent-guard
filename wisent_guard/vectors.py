"""
ContrastiveVectors module for creating and managing activation vectors
"""

import os
import torch
import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import pickle
from .utils.helpers import ensure_dir, normalize_vector, calculate_average_vector

class ContrastiveVectors:
    """
    Class for creating, storing, and managing contrastive activation vectors.
    
    This class handles the creation of vector representations that distinguish
    harmful content from harmless content by analyzing model activations.
    """
    def __init__(self, save_dir: str = "./wisent_guard_data"):
        """
        Initialize the ContrastiveVectors class.
        
        Args:
            save_dir: Directory for saving vectors and metadata
        """
        self.save_dir = save_dir
        self.vectors_dir = os.path.join(save_dir, "vectors")
        self.metadata_path = os.path.join(save_dir, "metadata.json")
        ensure_dir(self.vectors_dir)
        
        self.harmful_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        self.harmless_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        self.contrastive_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        self.metadata: Dict[str, Any] = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from disk if available.
        
        Returns:
            Dictionary containing metadata
        """
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {
            "categories": [],
            "layers": {},
            "version": "0.1.0",
            "num_pairs": {},
        }
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_vector_pair(
        self, 
        category: str, 
        layer: int, 
        harmful_vector: torch.Tensor, 
        harmless_vector: torch.Tensor
    ) -> None:
        """
        Add a pair of harmful and harmless activation vectors.
        
        Args:
            category: Category label for the vector pair (e.g., "hate_speech")
            layer: Model layer the vectors are from
            harmful_vector: Activation vector from harmful content
            harmless_vector: Activation vector from harmless content
        """
        # Initialize dictionaries if needed
        if category not in self.harmful_vectors:
            self.harmful_vectors[category] = {}
            self.harmless_vectors[category] = {}
            
            # Update metadata
            if category not in self.metadata["categories"]:
                self.metadata["categories"].append(category)
                self.metadata["num_pairs"][category] = 0
            
        if layer not in self.harmful_vectors[category]:
            self.harmful_vectors[category][layer] = []
            self.harmless_vectors[category][layer] = []
            
            # Update metadata
            if str(layer) not in self.metadata["layers"]:
                self.metadata["layers"][str(layer)] = 0
        
        # Normalize vectors
        harmful_norm = normalize_vector(harmful_vector)
        harmless_norm = normalize_vector(harmless_vector)
        
        # Store vectors
        self.harmful_vectors[category][layer].append(harmful_norm)
        self.harmless_vectors[category][layer].append(harmless_norm)
        
        # Update metadata counts
        self.metadata["num_pairs"][category] = self.metadata["num_pairs"].get(category, 0) + 1
        self.metadata["layers"][str(layer)] = self.metadata["layers"].get(str(layer), 0) + 1
        
        # Save metadata
        self._save_metadata()
    
    def compute_contrastive_vectors(self) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Compute contrastive vectors for all categories and layers.
        
        Returns:
            Dictionary mapping categories to dictionaries mapping layers to contrastive vectors
        """
        for category in self.harmful_vectors:
            if category not in self.contrastive_vectors:
                self.contrastive_vectors[category] = {}
                
            for layer in self.harmful_vectors[category]:
                if len(self.harmful_vectors[category][layer]) == 0 or len(self.harmless_vectors[category][layer]) == 0:
                    continue
                    
                # Compute average vectors
                harmful_avg = calculate_average_vector(self.harmful_vectors[category][layer])
                harmless_avg = calculate_average_vector(self.harmless_vectors[category][layer])
                
                # Compute contrastive vector (harmful - harmless)
                contrastive = harmful_avg - harmless_avg
                
                # Normalize
                contrastive_norm = normalize_vector(contrastive)
                
                self.contrastive_vectors[category][layer] = contrastive_norm
        
        return self.contrastive_vectors
    
    def save_vectors(self) -> None:
        """Save all vectors to disk."""
        # First compute contrastive vectors if not already done
        if not self.contrastive_vectors:
            self.compute_contrastive_vectors()
            
        # Save each category's vectors
        for category in self.contrastive_vectors:
            category_dir = os.path.join(self.vectors_dir, category)
            ensure_dir(category_dir)
            
            # Save contrastive vectors
            for layer, vector in self.contrastive_vectors[category].items():
                vector_path = os.path.join(category_dir, f"contrastive_layer_{layer}.pt")
                torch.save(vector, vector_path)
            
            # Also save raw vectors for potential re-calculation
            harmful_path = os.path.join(category_dir, "harmful_vectors.pkl")
            harmless_path = os.path.join(category_dir, "harmless_vectors.pkl")
            
            with open(harmful_path, 'wb') as f:
                pickle.dump(self.harmful_vectors[category], f)
            
            with open(harmless_path, 'wb') as f:
                pickle.dump(self.harmless_vectors[category], f)
    
    def load_vectors(self, categories: Optional[List[str]] = None) -> bool:
        """
        Load vectors from disk.
        
        Args:
            categories: List of categories to load. If None, loads all available categories.
            
        Returns:
            True if vectors were loaded successfully, False otherwise
        """
        # Load metadata first
        self.metadata = self._load_metadata()
        
        if categories is None:
            categories = self.metadata["categories"]
        
        success = True
        for category in categories:
            category_dir = os.path.join(self.vectors_dir, category)
            
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory {category} does not exist")
                success = False
                continue
            
            # Initialize dictionaries
            self.contrastive_vectors[category] = {}
            
            # Load contrastive vectors
            layers = [int(layer) for layer in self.metadata["layers"].keys()]
            for layer in layers:
                vector_path = os.path.join(category_dir, f"contrastive_layer_{layer}.pt")
                if os.path.exists(vector_path):
                    self.contrastive_vectors[category][layer] = torch.load(vector_path)
                else:
                    print(f"Warning: Vector file {vector_path} does not exist")
                    success = False
            
            # Load raw vectors if available
            harmful_path = os.path.join(category_dir, "harmful_vectors.pkl")
            harmless_path = os.path.join(category_dir, "harmless_vectors.pkl")
            
            if os.path.exists(harmful_path) and os.path.exists(harmless_path):
                with open(harmful_path, 'rb') as f:
                    self.harmful_vectors[category] = pickle.load(f)
                
                with open(harmless_path, 'rb') as f:
                    self.harmless_vectors[category] = pickle.load(f)
        
        return success
    
    def get_contrastive_vector(self, category: str, layer: int) -> Optional[torch.Tensor]:
        """
        Get the contrastive vector for a specific category and layer.
        
        Args:
            category: Category label
            layer: Model layer
            
        Returns:
            Contrastive vector tensor if available, None otherwise
        """
        if category in self.contrastive_vectors and layer in self.contrastive_vectors[category]:
            return self.contrastive_vectors[category][layer]
        return None
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available categories.
        
        Returns:
            List of category names
        """
        return self.metadata["categories"]
    
    def get_available_layers(self) -> List[int]:
        """
        Get list of available layers.
        
        Returns:
            List of layer indices
        """
        return [int(layer) for layer in self.metadata["layers"].keys()] 