"""
A simplified standalone example that demonstrates how activation monitoring works
"""

import torch
import tempfile
import os
import numpy as np
from typing import Dict, List, Any

def normalize_vector(vector):
    """Normalize a vector to unit length."""
    norm = torch.norm(vector)
    if norm > 0:
        return vector / norm
    return vector

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    # Make sure both vectors are 1D
    if len(v1.shape) > 1:
        v1 = v1.flatten()
    if len(v2.shape) > 1:
        v2 = v2.flatten()
        
    # Compute cosine similarity
    dot_product = torch.sum(v1 * v2)
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return (dot_product / (norm_v1 * norm_v2)).item()

class SimpleContrastiveVectors:
    """A simplified version of ContrastiveVectors for demonstration."""
    
    def __init__(self):
        self.harmful_vectors = {}
        self.harmless_vectors = {}
        self.contrastive_vectors = {}
        
    def add_vector_pair(self, category, layer, harmful_vector, harmless_vector):
        """Add a pair of harmful and harmless vectors."""
        # Initialize dictionaries if needed
        if category not in self.harmful_vectors:
            self.harmful_vectors[category] = {}
            self.harmless_vectors[category] = {}
            
        if layer not in self.harmful_vectors[category]:
            self.harmful_vectors[category][layer] = []
            self.harmless_vectors[category][layer] = []
        
        # Normalize vectors
        harmful_norm = normalize_vector(harmful_vector)
        harmless_norm = normalize_vector(harmless_vector)
        
        # Store vectors
        self.harmful_vectors[category][layer].append(harmful_norm)
        self.harmless_vectors[category][layer].append(harmless_norm)
    
    def compute_contrastive_vectors(self):
        """Compute contrastive vectors for all categories and layers."""
        for category in self.harmful_vectors:
            if category not in self.contrastive_vectors:
                self.contrastive_vectors[category] = {}
                
            for layer in self.harmful_vectors[category]:
                if len(self.harmful_vectors[category][layer]) == 0 or len(self.harmless_vectors[category][layer]) == 0:
                    continue
                    
                # Compute average vectors
                harmful_avg = torch.stack(self.harmful_vectors[category][layer]).mean(dim=0)
                harmless_avg = torch.stack(self.harmless_vectors[category][layer]).mean(dim=0)
                
                # Compute contrastive vector (harmful - harmless)
                contrastive = harmful_avg - harmless_avg
                
                # Normalize
                contrastive_norm = normalize_vector(contrastive)
                
                self.contrastive_vectors[category][layer] = contrastive_norm
        
        return self.contrastive_vectors
    
    def get_contrastive_vector(self, category, layer):
        """Get a contrastive vector for a specific category and layer."""
        if category in self.contrastive_vectors and layer in self.contrastive_vectors[category]:
            return self.contrastive_vectors[category][layer]
        return None
    
    def get_available_categories(self):
        """Get list of available categories."""
        return list(self.harmful_vectors.keys())
    
    def get_available_layers(self):
        """Get list of available layers across all categories."""
        layers = set()
        for category in self.harmful_vectors:
            layers.update(self.harmful_vectors[category].keys())
        return sorted(list(layers))

class SimpleActivationMonitor:
    """A simplified version of ActivationMonitor for demonstration."""
    
    def __init__(self, vectors, threshold=0.7):
        self.vectors = vectors
        self.threshold = threshold
        self.activations = {}
        self.max_similarities = {}
        self.triggered_layers = {}
        self.is_triggered = False
        
    def reset(self):
        """Reset the monitor state."""
        self.activations = {}
        self.max_similarities = {}
        self.triggered_layers = {}
        self.is_triggered = False
    
    def set_activations(self, layer, activation):
        """Set activation for a specific layer."""
        self.activations[layer] = activation
    
    def check_activations(self, categories=None):
        """Check if current activations are similar to harmful patterns."""
        # Get all available categories if not specified
        if categories is None:
            categories = self.vectors.get_available_categories()
        
        # Initialize result dictionary
        results = {}
        self.is_triggered = False
        
        # Check each category
        for category in categories:
            # Initialize category results
            if category not in self.max_similarities:
                self.max_similarities[category] = {}
            
            if category not in self.triggered_layers:
                self.triggered_layers[category] = set()
            
            category_results = {
                "is_harmful": False,
                "max_similarity": 0.0,
                "triggered_layers": [],
                "layer_similarities": {}
            }
            
            # Check each layer
            for layer in self.activations:
                # Get the activation and contrastive vector
                activation = self.activations[layer]
                contrastive_vector = self.vectors.get_contrastive_vector(category, layer)
                
                if contrastive_vector is None:
                    continue
                
                # Calculate similarity
                similarity = cosine_similarity(activation, contrastive_vector)
                
                # Update max similarity for this layer
                self.max_similarities[category][layer] = max(
                    similarity,
                    self.max_similarities[category].get(layer, -1.0)
                )
                
                # Check if threshold is exceeded
                if similarity >= self.threshold:
                    self.triggered_layers[category].add(layer)
                    category_results["is_harmful"] = True
                    self.is_triggered = True
                
                # Store layer similarity in results
                category_results["layer_similarities"][str(layer)] = float(similarity)
            
            # Update category results
            if self.max_similarities[category]:
                category_results["max_similarity"] = max(self.max_similarities[category].values())
            
            if self.triggered_layers[category]:
                category_results["triggered_layers"] = list(self.triggered_layers[category])
            
            results[category] = category_results
        
        return results
    
    def is_harmful(self, categories=None):
        """Check if current activations indicate harmful content."""
        results = self.check_activations(categories)
        
        # Check if any category is flagged as harmful
        for category_result in results.values():
            if category_result["is_harmful"]:
                return True
        
        return False
    
    def get_most_harmful_category(self):
        """Get the category with the highest similarity to harmful patterns."""
        max_similarity = -1.0
        max_category = None
        
        for category, layer_similarities in self.max_similarities.items():
            if not layer_similarities:
                continue
                
            category_max = max(layer_similarities.values())
            if category_max > max_similarity:
                max_similarity = category_max
                max_category = category
        
        if max_category is not None:
            return (max_category, max_similarity)
        
        return None

def print_monitor_results(results: Dict[str, Dict[str, Any]]):
    """Helper function to print monitor results in a readable format."""
    for category, category_results in results.items():
        print(f"Category: {category}")
        print(f"  Is harmful: {category_results['is_harmful']}")
        print(f"  Max similarity: {category_results['max_similarity']:.4f}")
        if category_results['triggered_layers']:
            print(f"  Triggered layers: {category_results['triggered_layers']}")
        
        # Print layer similarities
        print("  Layer similarities:")
        for layer, similarity in category_results['layer_similarities'].items():
            print(f"    Layer {layer}: {similarity:.4f}")
        print()

def main():
    print("Testing activation monitoring with mock data...")
    
    # Create sample vectors
    print("\nCreating sample vectors...")
    hidden_size = 768
    harmful_vector = torch.randn(hidden_size)
    harmless_vector = torch.randn(hidden_size)
    
    # Define categories and layers
    categories = ["illegal_content", "violence", "self_harm"]
    layers = [0, 1, 2]
    
    # Initialize vectors
    vectors = SimpleContrastiveVectors()
    
    # Add vector pairs and compute contrastive vectors
    print("Adding vector pairs...")
    for category in categories:
        for layer in layers:
            # Create slightly different vectors for each category and layer
            h_vector = harmful_vector + torch.randn(hidden_size) * 0.1
            hl_vector = harmless_vector + torch.randn(hidden_size) * 0.1
            
            vectors.add_vector_pair(
                category=category,
                layer=layer,
                harmful_vector=h_vector,
                harmless_vector=hl_vector
            )
            print(f"Added vector pair for category '{category}', layer {layer}")
    
    # Compute contrastive vectors
    print("\nComputing contrastive vectors...")
    contrastive_vectors = vectors.compute_contrastive_vectors()
    
    # Print some information about the contrastive vectors
    for category in categories:
        for layer in layers:
            vector = contrastive_vectors[category][layer]
            norm = torch.norm(vector).item()
            print(f"Category: {category}, Layer: {layer}, Vector norm: {norm:.6f}")
    
    # Initialize the monitor
    print("\nInitializing SimpleActivationMonitor...")
    monitor = SimpleActivationMonitor(vectors=vectors, threshold=0.7)
    
    # Test with harmless activation (random)
    print("\nTesting with random activation (should be harmless)...")
    monitor.reset()
    
    # Set random activations
    for layer in layers:
        random_activation = torch.randn(hidden_size)
        monitor.set_activations(layer, random_activation)
    
    # Check activations
    results = monitor.check_activations()
    print_monitor_results(results)
    print(f"Is harmful: {monitor.is_harmful()}")
    
    # Test with harmful activation (similar to contrastive vector)
    print("\nTesting with harmful activation (similar to contrastive vector)...")
    monitor.reset()
    
    # Choose one category to simulate harmful content
    harmful_category = "violence"
    
    # Set activations similar to contrastive vectors
    for layer in layers:
        contrastive = vectors.get_contrastive_vector(harmful_category, layer)
        
        # Create an activation vector that's VERY similar to the contrastive vector
        # Now using 98% contrastive vector + 2% random noise to ensure it crosses threshold
        similar_activation = contrastive * 0.98 + torch.randn_like(contrastive) * 0.02
        
        # For layer 1, make it even more similar (99.5%) to demonstrate layer-specific triggering
        if layer == 1:
            similar_activation = contrastive * 0.995 + torch.randn_like(contrastive) * 0.005
        
        monitor.set_activations(layer, similar_activation)
    
    # Check activations
    results = monitor.check_activations()
    print_monitor_results(results)
    print(f"Is harmful: {monitor.is_harmful()}")
    
    # Get the most harmful category
    most_harmful = monitor.get_most_harmful_category()
    if most_harmful:
        category, similarity = most_harmful
        print(f"Most harmful category: {category}, similarity: {similarity:.4f}")
    
    # Test with category-specific check
    print("\nTesting with category-specific check...")
    # Just check against 'violence' category
    is_harmful_violence = monitor.is_harmful(categories=["violence"])
    print(f"Is harmful (violence only): {is_harmful_violence}")
    
    # Just check against 'illegal_content' category
    is_harmful_illegal = monitor.is_harmful(categories=["illegal_content"])
    print(f"Is harmful (illegal_content only): {is_harmful_illegal}")
    
    # Test with threshold adjustment
    print("\nTesting with adjusted threshold...")
    original_threshold = monitor.threshold
    
    # Try with higher threshold (should be less sensitive)
    monitor.threshold = 0.99
    print(f"With threshold = {monitor.threshold}:")
    print(f"Is harmful: {monitor.is_harmful()}")
    
    # Try with lower threshold (should be more sensitive)
    monitor.threshold = 0.5
    print(f"With threshold = {monitor.threshold}:")
    print(f"Is harmful: {monitor.is_harmful()}")
    
    # Reset threshold
    monitor.threshold = original_threshold

if __name__ == "__main__":
    main() 