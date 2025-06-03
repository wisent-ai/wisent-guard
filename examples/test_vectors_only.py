"""
Simple test of the ContrastiveVectors class without downloading a model
"""

import os
import torch
import tempfile
from wisent_guard.vectors import ContrastiveVectors

def main():
    print("Testing ContrastiveVectors functionality...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Initialize ContrastiveVectors
        vectors = ContrastiveVectors(save_dir=temp_dir)
        
        # Create sample vectors
        print("Creating sample vectors...")
        harmful_vector = torch.randn(768)
        harmless_vector = torch.randn(768)
        
        # Define categories and layers
        categories = ["illegal_content", "violence", "self_harm"]
        layers = [0, 1, 2]
        
        # Add vector pairs
        print("Adding vector pairs...")
        for category in categories:
            for layer in layers:
                # Create slightly different vectors for each category and layer
                h_vector = harmful_vector + torch.randn(768) * 0.1
                hl_vector = harmless_vector + torch.randn(768) * 0.1
                
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
        
        # Save vectors
        print("\nSaving vectors...")
        vectors.save_vectors()
        
        # Check that files were created
        print("\nVerifying saved files...")
        for category in categories:
            category_dir = os.path.join(temp_dir, "vectors", category)
            print(f"Category directory: {category_dir}")
            
            # List files in the category directory
            files = os.listdir(category_dir)
            print(f"Files: {files}")
        
        # Load vectors
        print("\nLoading vectors...")
        new_vectors = ContrastiveVectors(save_dir=temp_dir)
        success = new_vectors.load_vectors()
        
        if success:
            print("Successfully loaded vectors!")
        else:
            print("Failed to load vectors.")
        
        # Verify loaded vectors
        print("\nVerifying loaded vectors...")
        loaded_categories = new_vectors.get_available_categories()
        loaded_layers = new_vectors.get_available_layers()
        
        print(f"Loaded categories: {loaded_categories}")
        print(f"Loaded layers: {loaded_layers}")
        
        # Compare original and loaded vectors
        print("\nComparing original and loaded vectors...")
        for category in categories:
            for layer in layers:
                original = contrastive_vectors[category][layer]
                loaded = new_vectors.get_contrastive_vector(category, layer)
                
                # Calculate cosine similarity
                cos_sim = torch.sum(original * loaded) / (torch.norm(original) * torch.norm(loaded))
                
                print(f"Category: {category}, Layer: {layer}, Cosine similarity: {cos_sim.item():.6f}")

if __name__ == "__main__":
    main() 