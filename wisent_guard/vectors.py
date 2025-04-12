"""
ContrastiveVectors module for creating and managing activation vectors
"""

import os
import torch
import json
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import pickle
from .utils.helpers import ensure_dir, normalize_vector, calculate_average_vector
from .utils.logger import get_logger

class ContrastiveVectors:
    """
    Class for creating, storing, and managing contrastive activation vectors.
    
    This class handles the creation of vector representations that distinguish
    harmful content from harmless content by analyzing model activations.
    """
    def __init__(self, save_dir: str = "./wisent_guard_data", log_level: str = "info"):
        """
        Initialize the ContrastiveVectors class.
        
        Args:
            save_dir: Directory for saving vectors and metadata
            log_level: Logging level ('debug', 'info', 'warning', 'error')
        """
        self.logger = get_logger(name="wisent_guard.vectors", level=log_level)
        self.logger.info("Initializing ContrastiveVectors")
        
        self.save_dir = save_dir
        self.vectors_dir = os.path.join(save_dir, "vectors")
        self.metadata_path = os.path.join(save_dir, "metadata.json")
        ensure_dir(self.vectors_dir)
        
        self.logger.debug(f"Vectors directory: {self.vectors_dir}")
        self.logger.debug(f"Metadata path: {self.metadata_path}")
        
        self.harmful_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        self.harmless_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        self.contrastive_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        self.metadata: Dict[str, Any] = self._load_metadata()
        
        self.logger.info(f"Found {len(self.metadata['categories'])} existing categories")
        if self.metadata['categories']:
            self.logger.debug(f"Categories: {', '.join(self.metadata['categories'])}")
        
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from disk if available.
        
        Returns:
            Dictionary containing metadata
        """
        if os.path.exists(self.metadata_path):
            self.logger.debug(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.logger.debug(f"Loaded metadata with {len(metadata.get('categories', []))} categories")
                return metadata
        
        self.logger.debug("No existing metadata found, initializing new metadata")
        return {
            "categories": [],
            "layers": {},
            "version": "0.1.0",
            "num_pairs": {},
        }
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        self.logger.debug(f"Saving metadata to {self.metadata_path}")
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        self.logger.debug("Metadata saved successfully")
    
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
        self.logger.debug(f"Adding vector pair for category '{category}', layer {layer}")
        
        # Initialize dictionaries if needed
        if category not in self.harmful_vectors:
            self.logger.debug(f"Initializing new category: {category}")
            self.harmful_vectors[category] = {}
            self.harmless_vectors[category] = {}
            
            # Update metadata
            if category not in self.metadata["categories"]:
                self.metadata["categories"].append(category)
                self.metadata["num_pairs"][category] = 0
                self.logger.info(f"Added new category to metadata: {category}")
            
        if layer not in self.harmful_vectors[category]:
            self.logger.debug(f"Initializing layer {layer} for category '{category}'")
            self.harmful_vectors[category][layer] = []
            self.harmless_vectors[category][layer] = []
            
            # Update metadata
            if str(layer) not in self.metadata["layers"]:
                self.metadata["layers"][str(layer)] = 0
                self.logger.debug(f"Added new layer to metadata: {layer}")
        
        # Log vector shapes and statistics
        self.logger.debug(f"Harmful vector shape: {harmful_vector.shape}, Harmless vector shape: {harmless_vector.shape}")
        
        # Normalize vectors
        self.logger.debug("Normalizing vectors")
        harmful_norm = normalize_vector(harmful_vector)
        harmless_norm = normalize_vector(harmless_vector)
        
        # Store vectors
        self.harmful_vectors[category][layer].append(harmful_norm)
        self.harmless_vectors[category][layer].append(harmless_norm)
        
        # Update metadata counts
        self.metadata["num_pairs"][category] = self.metadata["num_pairs"].get(category, 0) + 1
        self.metadata["layers"][str(layer)] = self.metadata["layers"].get(str(layer), 0) + 1
        
        self.logger.debug(f"Total pairs for category '{category}': {self.metadata['num_pairs'][category]}")
        
        # Save metadata
        self._save_metadata()
        self.logger.debug("Vector pair added successfully")
    
    def compute_contrastive_vectors(self) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Compute contrastive vectors for all categories and layers.
        
        Returns:
            Dictionary mapping categories to dictionaries mapping layers to contrastive vectors
        """
        self.logger.info("Computing contrastive vectors")
        
        for category in self.harmful_vectors:
            self.logger.debug(f"Processing category: {category}")
            
            if category not in self.contrastive_vectors:
                self.contrastive_vectors[category] = {}
                
            for layer in self.harmful_vectors[category]:
                self.logger.debug(f"Computing contrastive vector for layer {layer}")
                
                if len(self.harmful_vectors[category][layer]) == 0 or len(self.harmless_vectors[category][layer]) == 0:
                    self.logger.warning(f"No vectors found for category '{category}', layer {layer}")
                    continue
                
                # Log number of vectors being averaged
                self.logger.debug(f"Averaging {len(self.harmful_vectors[category][layer])} harmful vectors")
                self.logger.debug(f"Averaging {len(self.harmless_vectors[category][layer])} harmless vectors")
                    
                # Compute average vectors
                harmful_avg = calculate_average_vector(self.harmful_vectors[category][layer])
                harmless_avg = calculate_average_vector(self.harmless_vectors[category][layer])
                
                self.logger.debug(f"Average harmful vector shape: {harmful_avg.shape}")
                self.logger.debug(f"Average harmless vector shape: {harmless_avg.shape}")
                
                # Compute contrastive vector (harmful - harmless)
                self.logger.debug("Computing difference vector (harmful - harmless)")
                contrastive = harmful_avg - harmless_avg
                
                # Normalize
                self.logger.debug("Normalizing contrastive vector")
                contrastive_norm = normalize_vector(contrastive)
                
                self.contrastive_vectors[category][layer] = contrastive_norm
                self.logger.debug(f"Contrastive vector for layer {layer} computed successfully")
        
        self.logger.info(f"Computed contrastive vectors for {len(self.contrastive_vectors)} categories")
        return self.contrastive_vectors
    
    def save_vectors(self) -> bool:
        """
        Save all vectors to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # First compute contrastive vectors if not already done
            if not self.contrastive_vectors:
                self.logger.info("No contrastive vectors found, computing now")
                self.compute_contrastive_vectors()
            
            # Ensure we have contrastive vectors to save
            if not self.contrastive_vectors:
                self.logger.warning("No contrastive vectors to save")
                return False
            
            # Save each category's vectors
            self.logger.info("Saving vectors to disk")
            saved_count = 0
            
            for category in self.contrastive_vectors:
                self.logger.debug(f"Saving vectors for category: {category}")
                category_dir = os.path.join(self.vectors_dir, category)
                ensure_dir(category_dir)
                
                # Save contrastive vectors
                for layer, vector in self.contrastive_vectors[category].items():
                    vector_path = os.path.join(category_dir, f"contrastive_layer_{layer}.pt")
                    torch.save(vector, vector_path)
                    saved_count += 1
                    self.logger.debug(f"Saved contrastive vector to {vector_path}")
                
                # Also save raw vectors for potential re-calculation if available
                if category in self.harmful_vectors:
                    harmful_path = os.path.join(category_dir, "harmful_vectors.pkl")
                    with open(harmful_path, 'wb') as f:
                        pickle.dump(self.harmful_vectors[category], f)
                        self.logger.debug(f"Saved harmful vectors to {harmful_path}")
                
                if category in self.harmless_vectors:
                    harmless_path = os.path.join(category_dir, "harmless_vectors.pkl")
                    with open(harmless_path, 'wb') as f:
                        pickle.dump(self.harmless_vectors[category], f)
                        self.logger.debug(f"Saved harmless vectors to {harmless_path}")
            
            self.logger.info(f"Successfully saved {saved_count} contrastive vectors")
            return True
        except Exception as e:
            self.logger.error(f"Error saving vectors: {e}")
            return False
    
    def load_vectors(self, categories: Optional[List[str]] = None, layers: Optional[List[int]] = None) -> bool:
        """
        Load vectors from disk.
        
        Args:
            categories: List of categories to load. If None, loads all available categories.
            layers: List of layers to load. If None, loads all available layers.
            
        Returns:
            True if vectors were loaded successfully, False otherwise
        """
        # Load metadata first
        self.metadata = self._load_metadata()
        self.logger.info("Loading vectors from disk")
        
        if categories is None:
            categories = self.metadata["categories"]
            self.logger.debug(f"Loading all {len(categories)} available categories")
        else:
            self.logger.debug(f"Loading specific categories: {categories}")
        
        # Convert layers to integers if provided
        if layers is not None:
            layers = [int(layer) for layer in layers]
            self.logger.debug(f"Loading specific layers: {layers}")
        
        success = True
        loaded_count = 0
        
        for category in categories:
            category_dir = os.path.join(self.vectors_dir, category)
            self.logger.debug(f"Looking for vectors in {category_dir}")
            
            if not os.path.exists(category_dir):
                self.logger.warning(f"Category directory {category} does not exist")
                success = False
                continue
            
            # Initialize dictionaries
            self.contrastive_vectors[category] = {}
            
            # Load contrastive vectors
            metadata_layers = [int(layer) for layer in self.metadata["layers"].keys()]
            # Filter layers if specified
            target_layers = layers if layers is not None else metadata_layers
            
            for layer in target_layers:
                # Skip layers not in metadata
                if layer not in metadata_layers:
                    self.logger.debug(f"Layer {layer} not in metadata, skipping")
                    continue
                    
                vector_path = os.path.join(category_dir, f"contrastive_layer_{layer}.pt")
                self.logger.debug(f"Attempting to load vector from {vector_path}")
                
                if os.path.exists(vector_path):
                    try:
                        vector_tensor = torch.load(vector_path, map_location=torch.device('cpu'))
                        self.contrastive_vectors[category][layer] = vector_tensor
                        loaded_count += 1
                        self.logger.debug(f"Successfully loaded vector from {vector_path}")
                    except Exception as e:
                        self.logger.warning(f"Error loading vector file {vector_path}: {e}")
                        success = False
                else:
                    self.logger.warning(f"Vector file {vector_path} does not exist")
                    success = False
        
        if loaded_count > 0:
            self.logger.info(f"Successfully loaded {loaded_count} contrastive vectors")
            return True
        else:
            if not success:
                self.logger.warning(f"Loaded {loaded_count} vectors with some errors")
            else:
                self.logger.info("No vectors found to load")
            return loaded_count > 0
    
    def clear_vectors(self, category: Optional[str] = None) -> None:
        """
        Clear vectors for a specific category or all categories.
        
        Args:
            category: Category to clear. If None, clears all categories.
        """
        if category is not None:
            # Clear specific category
            if category in self.harmful_vectors:
                del self.harmful_vectors[category]
            if category in self.harmless_vectors:
                del self.harmless_vectors[category]
            if category in self.contrastive_vectors:
                del self.contrastive_vectors[category]
            
            # Remove from metadata
            if category in self.metadata["categories"]:
                self.metadata["categories"].remove(category)
            if category in self.metadata["num_pairs"]:
                del self.metadata["num_pairs"][category]
            
            # Remove files from disk
            category_dir = os.path.join(self.vectors_dir, category)
            if os.path.exists(category_dir):
                # Remove all files in the directory
                for filename in os.listdir(category_dir):
                    file_path = os.path.join(category_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                
                # Remove the directory
                os.rmdir(category_dir)
        else:
            # Clear all categories
            self.harmful_vectors = {}
            self.harmless_vectors = {}
            self.contrastive_vectors = {}
            
            # Reset metadata
            self.metadata["categories"] = []
            self.metadata["num_pairs"] = {}
            # Keep layers info in case it's useful
            
            # Remove files from disk
            if os.path.exists(self.vectors_dir):
                # Remove all category directories
                for category_name in os.listdir(self.vectors_dir):
                    category_dir = os.path.join(self.vectors_dir, category_name)
                    if os.path.isdir(category_dir):
                        # Remove all files in the directory
                        for filename in os.listdir(category_dir):
                            file_path = os.path.join(category_dir, filename)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        # Remove the directory
                        os.rmdir(category_dir)
        
        # Save updated metadata
        self._save_metadata()
    
    def get_contrastive_vector(self, category: str, layer: int) -> Optional[torch.Tensor]:
        """
        Get a contrastive vector for a specific category and layer.
        
        Args:
            category: Category to get vector for
            layer: Layer to get vector for
            
        Returns:
            Contrastive vector tensor or None if not found
        """
        if category not in self.contrastive_vectors:
            return None
            
        if layer not in self.contrastive_vectors[category]:
            return None
            
        # Ensure the tensor is properly allocated
        vector = self.contrastive_vectors[category][layer]
        
        # If the vector is on CPU but should be on MPS/CUDA, we handle that at the point of use
        # rather than here, to avoid device assumptions in this class
        
        return vector
    
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
    
    def get_contrastive_vectors(self) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Get all contrastive vectors.
        
        Returns:
            Dictionary mapping categories to dictionaries mapping layers to contrastive vectors
        """
        return self.contrastive_vectors 

    def get_similarity(self, 
                    activation_data: Dict[str, Any],
                    layer: int, 
                    category: str,
                    token_strategy: str = "last") -> float:
        """
        Calculate similarity between an activation and a contrastive vector.
        
        Args:
            activation_data: Dictionary of activation data
            layer: Layer to compare
            category: Category of contrastive vector to compare against
            token_strategy: Kept for backward compatibility, but only "last" is supported
            
        Returns:
            Cosine similarity score (0.0-1.0)
        """
        if token_strategy != "last":
            self.logger.warning(f"Ignoring provided token_strategy='{token_strategy}'. Always using 'last' token strategy.")
        
        # Get the contrastive vector for this category and layer
        contrastive_vector = self.get_contrastive_vector(category, layer)
        if contrastive_vector is None:
            return 0.0
            
        # Extract activation tensor - always using the "last" method
        if isinstance(activation_data, dict):
            # For raw activation data (e.g., from hooks)
            if torch.is_tensor(activation_data):
                activation = activation_data
            elif hasattr(activation_data, 'get') and torch.is_tensor(activation_data.get('activations')):
                activation = activation_data['activations']
            else:
                self.logger.warning(f"Unknown activation data format: {type(activation_data)}")
                return 0.0
        elif torch.is_tensor(activation_data):
            # Direct tensor input
            activation = activation_data
        else:
            self.logger.warning(f"Unknown activation data type: {type(activation_data)}")
            return 0.0
            
        # Calculate similarity
        try:
            # Handle tensor device and shape differences
            if activation.device != contrastive_vector.device:
                self.logger.debug(f"Moving activation to {contrastive_vector.device} for comparison")
                activation = activation.to(contrastive_vector.device)
                
            # Handle dimension differences
            if activation.shape != contrastive_vector.shape:
                self.logger.debug(f"Adjusting dimensions: activation {activation.shape}, vector {contrastive_vector.shape}")
                
                # Flatten if needed
                if len(activation.shape) > 1:
                    activation = activation.flatten()
                if len(contrastive_vector.shape) > 1:
                    contrastive_vector = contrastive_vector.flatten()
                
                # If still different, truncate to smaller size
                if activation.shape != contrastive_vector.shape:
                    min_size = min(activation.shape[0], contrastive_vector.shape[0])
                    activation = activation[:min_size]
                    contrastive_vector = contrastive_vector[:min_size]
            
            similarity = cosine_sim(activation, contrastive_vector)
            return similarity.item() if torch.is_tensor(similarity) else similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0 

    def has_any_vectors(self) -> bool:
        """
        Check if any vectors are loaded.
        
        Returns:
            True if any vectors are loaded, False otherwise
        """
        return len(self.contrastive_vectors) > 0 

    def add_contrastive_vector(self, harmful_activations: List[torch.Tensor], harmless_activations: List[torch.Tensor], category: str, layer: int):
        """
        Add contrastive vector directly from lists of activations.
        
        Args:
            harmful_activations: List of activation tensors for harmful examples
            harmless_activations: List of activation tensors for harmless examples
            category: Category name for the contrastive vector
            layer: Layer number
        """
        if not harmful_activations or not harmless_activations:
            self.logger.warning(f"Empty activation lists for category '{category}' layer {layer}")
            return
        
        # Stack harmful activations
        try:
            harmful_stack = torch.stack(harmful_activations)
        except Exception as e:
            self.logger.error(f"Error stacking harmful activations: {e}")
            return
        
        # Stack harmless activations
        try:
            harmless_stack = torch.stack(harmless_activations)
        except Exception as e:
            self.logger.error(f"Error stacking harmless activations: {e}")
            return
        
        # Calculate means
        harmful_mean = torch.mean(harmful_stack, dim=0)
        harmless_mean = torch.mean(harmless_stack, dim=0)
        
        # Calculate contrastive vector
        contrastive_vector = harmful_mean - harmless_mean
        
        # Normalize
        contrastive_vector = contrastive_vector / torch.norm(contrastive_vector)
        
        # Initialize category if needed
        if category not in self.contrastive_vectors:
            self.contrastive_vectors[category] = {}
        
        # Store the vector
        self.contrastive_vectors[category][layer] = contrastive_vector.cpu()
        self.logger.info(f"Added contrastive vector for category '{category}' layer {layer}")
        
        # Update metadata
        if category not in self.metadata["categories"]:
            self.metadata["categories"].append(category)
            self.metadata["num_pairs"][category] = len(harmful_activations)
        else:
            self.metadata["num_pairs"][category] = self.metadata["num_pairs"].get(category, 0) + len(harmful_activations)
        
        if str(layer) not in self.metadata["layers"]:
            self.metadata["layers"][str(layer)] = len(harmful_activations)
        else:
            self.metadata["layers"][str(layer)] = self.metadata["layers"].get(str(layer), 0) + len(harmful_activations)
        
        # Save metadata
        self._save_metadata() 