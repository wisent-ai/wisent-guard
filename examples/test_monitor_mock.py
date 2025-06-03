"""
Test example for the ActivationMonitor class using mock data
"""

import torch
import tempfile
import os
from typing import Dict, List, Any
from wisent_guard.vectors import ContrastiveVectors
from wisent_guard.monitor import ActivationMonitor

# Create a mock model class that can be used with ActivationMonitor
class MockModel(torch.nn.Module):
    """A mock model that returns pre-defined activations when called."""
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cpu")
        
        # Create some mock layers to hook into
        self.model = torch.nn.ModuleDict()
        self.model.decoder = torch.nn.ModuleDict()
        self.model.decoder.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        
        # Set proper module names to ensure hooks can find our layers
        for i, layer in enumerate(self.model.decoder.layers):
            layer.name = f"layer_{i}"
    
    def forward(self, input_ids):
        """Mock forward pass that returns dummy output."""
        # This is called by the hooks system
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Return a dummy tensor as output
        return torch.randn(batch_size, seq_len, self.hidden_size)

# Create a simplified utility function to calculate cosine similarity without sklearn
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

def main():
    print("Testing ActivationMonitor with mock data...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Initialize ContrastiveVectors
        vectors = ContrastiveVectors(save_dir=temp_dir)
        
        # Create sample vectors
        print("Creating sample vectors...")
        hidden_size = 768
        harmful_vector = torch.randn(hidden_size)
        harmless_vector = torch.randn(hidden_size)
        
        # Define categories and layers
        categories = ["illegal_content", "violence", "self_harm"]
        layers = [0, 1, 2]
        
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
        
        # Compute and save contrastive vectors
        contrastive_vectors = vectors.compute_contrastive_vectors()
        vectors.save_vectors()
        
        # Create a mock model
        model = MockModel(hidden_size=hidden_size, num_layers=max(layers) + 1)
        
        # Override the cosine_sim function in the ActivationMonitor's helpers module
        # to use our simplified version that handles dimension issues better
        import wisent_guard.utils.helpers
        original_cosine_sim = wisent_guard.utils.helpers.cosine_sim
        wisent_guard.utils.helpers.cosine_sim = cosine_similarity
        
        try:
            # Initialize the monitor
            print("\nInitializing ActivationMonitor...")
            monitor = ActivationMonitor(
                model=model,
                vectors=vectors,
                layers=layers,
                model_type="opt",  # We're mocking an OPT model
                threshold=0.7
            )
            
            # Test with harmless activation (random)
            print("\nTesting with random activation (should be harmless)...")
            
            # Clear previous activations
            monitor.hooks.clear_activations()
            
            # Manually inject random activations into the activation hooks
            for layer in layers:
                # Use random activations that are different from the contrastive vectors
                random_activation = torch.randn(hidden_size)
                monitor.hooks.layer_activations[layer] = random_activation
            
            # Check activations and print results
            results = monitor.check_activations()
            print_monitor_results(results)
            print(f"Is harmful: {monitor.is_harmful()}")
            
            # Test with harmful activation (similar to contrastive vector)
            print("\nTesting with harmful activation (similar to contrastive vector)...")
            
            # Clear previous activations
            monitor.hooks.clear_activations()
            
            # Choose one category to simulate harmful content
            harmful_category = "violence"
            
            # Manually inject activations that are similar to contrastive vectors
            for layer in layers:
                # Get the contrastive vector for this category and layer
                contrastive = vectors.get_contrastive_vector(harmful_category, layer)
                
                # Ensure contrastive vector is 1D
                if len(contrastive.shape) > 1:
                    contrastive = contrastive.flatten()
                
                # Create an activation vector that's similar to the contrastive vector
                # Add some noise but maintain high similarity by making it 90% similar
                similar_activation = contrastive * 0.9 + torch.randn_like(contrastive) * 0.1
                
                # Store in monitor's activation hooks
                monitor.hooks.layer_activations[layer] = similar_activation
            
            # Check activations and print results
            results = monitor.check_activations()
            print_monitor_results(results)
            print(f"Is harmful: {monitor.is_harmful()}")
            
            # Get the most harmful category
            most_harmful = monitor.get_most_harmful_category()
            if most_harmful:
                category, similarity = most_harmful
                print(f"Most harmful category: {category}, similarity: {similarity:.4f}")
            
            # Test with threshold adjustment
            print("\nTesting with adjusted threshold...")
            original_threshold = monitor.threshold
            
            # Try with higher threshold (should be less sensitive)
            monitor.threshold = 0.9
            print(f"With threshold = {monitor.threshold}:")
            print(f"Is harmful: {monitor.is_harmful()}")
            
            # Try with lower threshold (should be more sensitive)
            monitor.threshold = 0.5
            print(f"With threshold = {monitor.threshold}:")
            print(f"Is harmful: {monitor.is_harmful()}")
            
            # Reset threshold
            monitor.threshold = original_threshold
            
        finally:
            # Restore the original cosine_sim function
            wisent_guard.utils.helpers.cosine_sim = original_cosine_sim

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

if __name__ == "__main__":
    main() 