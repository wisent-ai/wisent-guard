"""
Helper functions for Wisent-Guard
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity

def ensure_dir(directory: str) -> None:
    """
    Make sure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def cosine_sim(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity value
    """
    if isinstance(v1, torch.Tensor):
        v1 = v1.cpu().detach().numpy()
    if isinstance(v2, torch.Tensor):
        v2 = v2.cpu().detach().numpy()
        
    # Reshape if needed
    if len(v1.shape) > 1:
        v1 = v1.reshape(1, -1)
    if len(v2.shape) > 1:
        v2 = v2.reshape(1, -1)
        
    return float(cosine_similarity(v1, v2)[0][0])

def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = torch.norm(vector, p=2)
    if norm > 0:
        return vector / norm
    return vector

def calculate_average_vector(vectors: List[torch.Tensor]) -> torch.Tensor:
    """
    Calculate the average of a list of vectors.
    
    Args:
        vectors: List of vectors to average
        
    Returns:
        Average vector
    """
    if not vectors:
        raise ValueError("Empty list of vectors provided")
    
    # Stack tensors and calculate mean
    stacked = torch.stack(vectors)
    return torch.mean(stacked, dim=0)

def get_layer_name(model_type: str, layer_idx: int) -> str:
    """
    Get the appropriate layer name based on model type and layer index.
    
    Args:
        model_type: Type of model ('opt', 'llama', 'gpt2', etc.)
        layer_idx: Layer index
        
    Returns:
        Layer name for the specific model architecture
    """
    layer_name_map = {
        "opt": f"model.decoder.layers.{layer_idx}",
        "llama": f"model.layers.{layer_idx}",
        "gpt2": f"transformer.h.{layer_idx}",
        "gpt_neox": f"gpt_neox.layers.{layer_idx}",
        "gptj": f"transformer.h.{layer_idx}",
        "t5": f"encoder.block.{layer_idx}",
        "bart": f"model.encoder.layers.{layer_idx}",
    }
    
    return layer_name_map.get(model_type.lower(), f"layers.{layer_idx}") 