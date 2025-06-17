"""
Model persistence utilities for saving and loading trained classifiers and steering vectors.
"""

import os
import pickle
import json
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from pathlib import Path


class ModelPersistence:
    """Utilities for saving and loading trained models."""
    
    @staticmethod
    def save_classifier(classifier, layer: int, save_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Save a trained classifier to disk.
        
        Args:
            classifier: Trained classifier object
            layer: Layer index this classifier was trained for
            save_path: Base path for saving (will add layer suffix)
            metadata: Additional metadata to save with the classifier
            
        Returns:
            Actual path where the classifier was saved
        """
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Split path and sanitize only the filename part
        directory = os.path.dirname(save_path)
        filename = os.path.basename(save_path)
        # Sanitize filename to handle periods in model names
        safe_filename = filename.replace('.', '_')
        safe_path = os.path.join(directory, safe_filename)
        
        # Add layer suffix to filename
        base, ext = os.path.splitext(safe_path)
        classifier_path = f"{base}_layer_{layer}{ext or '.pkl'}"
        
        # Prepare data to save
        save_data = {
            'classifier': classifier,
            'layer': layer,
            'metadata': metadata or {}
        }
        
        # Save classifier
        with open(classifier_path, 'wb') as f:
            pickle.dump(save_data, f)
            
        return classifier_path
    
    @staticmethod
    def load_classifier(load_path: str, layer: int) -> tuple:
        """
        Load a trained classifier from disk.
        
        Args:
            load_path: Base path for loading (will add layer suffix)
            layer: Layer index to load classifier for
            
        Returns:
            Tuple of (classifier, metadata)
        """
        # Split path and sanitize only the filename part to match save format
        directory = os.path.dirname(load_path)
        filename = os.path.basename(load_path)
        safe_filename = filename.replace('.', '_')
        safe_path = os.path.join(directory, safe_filename)
        
        # Add layer suffix to filename
        base, ext = os.path.splitext(safe_path)
        classifier_path = f"{base}_layer_{layer}{ext or '.pkl'}"
        
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
        
        # Load classifier
        with open(classifier_path, 'rb') as f:
            save_data = pickle.load(f)
        
        return save_data['classifier'], save_data.get('metadata', {})
    
    @staticmethod
    def save_multi_layer_classifiers(classifiers: Dict[int, Any], save_path: str, metadata: Dict[str, Any] = None) -> List[str]:
        """
        Save multiple classifiers for different layers.
        
        Args:
            classifiers: Dictionary mapping layer indices to trained classifiers
            save_path: Base path for saving
            metadata: Additional metadata to save with classifiers
            
        Returns:
            List of actual paths where classifiers were saved
        """
        saved_paths = []
        for layer, classifier in classifiers.items():
            path = ModelPersistence.save_classifier(classifier, layer, save_path, metadata)
            saved_paths.append(path)
        return saved_paths
    
    @staticmethod
    def load_multi_layer_classifiers(load_path: str, layers: List[int]) -> Dict[int, tuple]:
        """
        Load multiple classifiers for different layers.
        
        Args:
            load_path: Base path for loading
            layers: List of layer indices to load classifiers for
            
        Returns:
            Dictionary mapping layer indices to (classifier, metadata) tuples
        """
        classifiers = {}
        for layer in layers:
            try:
                classifier, metadata = ModelPersistence.load_classifier(load_path, layer)
                classifiers[layer] = (classifier, metadata)
            except FileNotFoundError:
                print(f"⚠️  Warning: Classifier for layer {layer} not found at {load_path}")
                continue
        return classifiers
    
    @staticmethod
    def save_steering_vector(vector: torch.Tensor, layer: int, save_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Save a steering vector to disk.
        
        Args:
            vector: Steering vector tensor
            layer: Layer index this vector was computed for
            save_path: Base path for saving (will add layer suffix)
            metadata: Additional metadata to save with the vector
            
        Returns:
            Actual path where the vector was saved
        """
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Add layer suffix to filename
        base, ext = os.path.splitext(save_path)
        vector_path = f"{base}_layer_{layer}{ext or '.pt'}"
        
        # Prepare data to save
        save_data = {
            'vector': vector.cpu() if isinstance(vector, torch.Tensor) else vector,
            'layer': layer,
            'metadata': metadata or {}
        }
        
        # Save vector
        torch.save(save_data, vector_path)
            
        return vector_path
    
    @staticmethod
    def load_steering_vector(load_path: str, layer: int, device: str = None) -> tuple:
        """
        Load a steering vector from disk.
        
        Args:
            load_path: Base path for loading (will add layer suffix)
            layer: Layer index to load vector for
            device: Device to load tensor to
            
        Returns:
            Tuple of (vector, metadata)
        """
        # Add layer suffix to filename
        base, ext = os.path.splitext(load_path)
        vector_path = f"{base}_layer_{layer}{ext or '.pt'}"
        
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Steering vector file not found: {vector_path}")
        
        # Load vector
        save_data = torch.load(vector_path, map_location=device)
        
        return save_data['vector'], save_data.get('metadata', {})
    
    @staticmethod
    def list_available_models(model_dir: str, model_type: str = "classifier") -> Dict[str, List[int]]:
        """
        List available saved models in a directory.
        
        Args:
            model_dir: Directory to search
            model_type: Type of model ("classifier" or "steering_vector")
            
        Returns:
            Dictionary mapping base model names to lists of available layers
        """
        if not os.path.exists(model_dir):
            return {}
        
        extension = ".pkl" if model_type == "classifier" else ".pt"
        models = {}
        
        for filename in os.listdir(model_dir):
            if filename.endswith(extension) and f"_layer_" in filename:
                # Extract base name and layer
                parts = filename.replace(extension, "").split("_layer_")
                if len(parts) == 2:
                    base_name = parts[0]
                    try:
                        layer = int(parts[1])
                        if base_name not in models:
                            models[base_name] = []
                        models[base_name].append(layer)
                    except ValueError:
                        continue
        
        # Sort layers for each model
        for base_name in models:
            models[base_name].sort()
        
        return models


def create_classifier_metadata(
    model_name: str,
    task_name: str,
    layer: int,
    classifier_type: str,
    training_accuracy: float,
    training_samples: int,
    token_aggregation: str,
    detection_threshold: float,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized metadata for a trained classifier.
    
    Args:
        model_name: Name of the language model
        task_name: Name of the training task
        layer: Layer index
        classifier_type: Type of classifier (logistic, mlp, etc.)
        training_accuracy: Accuracy achieved during training
        training_samples: Number of training samples used
        token_aggregation: Token aggregation method used
        detection_threshold: Classification threshold used
        **kwargs: Additional metadata fields
        
    Returns:
        Metadata dictionary
    """
    import datetime
    
    metadata = {
        'model_name': model_name,
        'task_name': task_name,
        'layer': layer,
        'classifier_type': classifier_type,
        'training_accuracy': training_accuracy,
        'training_samples': training_samples,
        'token_aggregation': token_aggregation,
        'detection_threshold': detection_threshold,
        'created_at': datetime.datetime.now().isoformat(),
        'wisent_guard_version': '1.0.0'  # Could be dynamically determined
    }
    
    # Add any additional metadata
    metadata.update(kwargs)
    
    return metadata


def create_steering_vector_metadata(
    model_name: str,
    task_name: str,
    layer: int,
    vector_strength: float,
    training_samples: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized metadata for a steering vector.
    
    Args:
        model_name: Name of the language model
        task_name: Name of the training task
        layer: Layer index
        vector_strength: Strength/magnitude of the steering vector
        training_samples: Number of training samples used
        **kwargs: Additional metadata fields
        
    Returns:
        Metadata dictionary
    """
    import datetime
    
    metadata = {
        'model_name': model_name,
        'task_name': task_name,
        'layer': layer,
        'vector_strength': vector_strength,
        'training_samples': training_samples,
        'created_at': datetime.datetime.now().isoformat(),
        'wisent_guard_version': '1.0.0'
    }
    
    # Add any additional metadata
    metadata.update(kwargs)
    
    return metadata 