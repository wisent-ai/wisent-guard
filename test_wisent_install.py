from wisent_guard.vectors import ContrastiveVectors
import torch
import tempfile

# Create a temporary test
with tempfile.TemporaryDirectory() as temp_dir:
    print(f'Created temp directory: {temp_dir}')
    
    # Initialize ContrastiveVectors
    vectors = ContrastiveVectors(save_dir=temp_dir)
    
    # Create sample vectors
    harmful_vector = torch.randn(768)
    harmless_vector = torch.randn(768)
    
    # Add a vector pair
    vectors.add_vector_pair(
        category='test',
        layer=0,
        harmful_vector=harmful_vector,
        harmless_vector=harmless_vector
    )
    
    print('Successfully added vector pair')
    
    # Compute and save
    contrastive = vectors.compute_contrastive_vectors()
    vectors.save_vectors()
    
    print(f'Computed contrastive vector with shape: {contrastive["test"][0].shape}')
    print('Test completed successfully!') 