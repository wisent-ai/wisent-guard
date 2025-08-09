#!/usr/bin/env python3
"""
Synthetic Steering Demo using MLX with Qwen3-4B-MLX-4bit model
Using wisent_guard's SyntheticContrastivePairGenerator as in the notebook.
"""

import os
import sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate

# Disable parallel processing to avoid Metal command buffer conflicts
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Disable multiprocessing in Python
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Add project root to path
project_root = Path.cwd().parent if Path.cwd().name == "examples" else Path.cwd()
sys.path.insert(0, str(project_root))

MODEL_NAME = "Qwen/Qwen3-4B-MLX-4bit"
LAYER_INDEX = 15
STEERING_STRENGTH = 1.0
MAX_LENGTH = 15
NUM_PAIRS = 5

TEST_PROMPTS = [
    "Tell me about cats",
    "How do I learn programming?", 
    "What's the weather like?"
]

TRAITS = {
    "sarcastic": "sarcastic and witty responses with subtle mockery and irony",
    "helpful": "extremely helpful, supportive, and eager to assist responses"
}

import torch
import numpy as np

class MLXModelWrapper:
    """Wrapper to make MLX model compatible with wisent_guard's Model class."""
    
    def __init__(self, mlx_model, tokenizer):
        self.mlx_model = mlx_model
        self.tokenizer = tokenizer
        self.config = type('Config', (), {
            'hidden_size': 2048,  # Typical for 4B model
            'num_hidden_layers': 32
        })()
        self.device = 'cpu'  # Pretend to be CPU to avoid CUDA ops
    
    def generate(self, **kwargs):
        """Generate text using MLX model with synchronization."""
        try:
            # Extract input
            input_ids = kwargs.get('input_ids')
            if hasattr(input_ids, 'tolist'):
                input_ids = input_ids.tolist()[0]
            
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            
            # Add synchronization to avoid Metal command buffer conflicts
            mx.eval(mx.array([0]))  # Force sync
            
            # Generate with MLX
            response = generate(
                self.mlx_model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=kwargs.get('max_new_tokens', MAX_LENGTH),
                verbose=False
            )
            
            # Force another sync
            mx.eval(mx.array([0]))
            
            # Convert response back to token ids as PyTorch tensor
            response_ids = self.tokenizer.encode(response, return_tensors='pt')
            return response_ids
            
        except Exception as e:
            print(f"Generation error: {e}")
            # Return a dummy response
            return torch.tensor([[self.tokenizer.eos_token_id]])
    
    def forward(self, **kwargs):
        """Forward pass - needed for Model wrapper."""
        # Return a dummy output that looks like transformer output
        batch_size = 1
        seq_len = 1
        vocab_size = len(self.tokenizer)
        
        return type('Output', (), {
            'logits': torch.zeros((batch_size, seq_len, vocab_size))
        })()
    
    def to(self, device):
        """MLX models are already on device."""
        return self
    
    def eval(self):
        """Set to eval mode."""
        return self
    
    def parameters(self):
        """Return empty parameters list for compatibility."""
        return []

def main():
    print("=" * 80)
    print("SYNTHETIC STEERING DEMO WITH WISENT_GUARD + MLX")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: Apple Silicon (MLX)")
    print()
    
    # Load MLX model and tokenizer
    print("Loading MLX model and tokenizer...")
    print("(This will download ~2.5GB on first run)")
    mlx_model, tokenizer = load(MODEL_NAME)
    print("✓ MLX Model loaded successfully")
    
    # NOW patch ThreadPoolExecutor after model is loaded
    import concurrent.futures
    
    class DummyExecutor:
        """Dummy executor that runs everything sequentially."""
        def submit(self, fn, *args, **kwargs):
            """Execute function immediately and return a completed future."""
            future = concurrent.futures.Future()
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future
        
        def shutdown(self, wait=True):
            """No-op shutdown."""
            pass
    
    # Replace ThreadPoolExecutor with our dummy version
    concurrent.futures.ThreadPoolExecutor = lambda max_workers=1: DummyExecutor()
    
    # Import after patching
    from wisent_guard.core.contrastive_pairs.generate_synthetically import SyntheticContrastivePairGenerator
    from wisent_guard.core.model import Model
    
    # Create wrapper to make MLX model compatible with wisent_guard
    print("\nCreating model wrapper for wisent_guard compatibility...")
    hf_compatible_model = MLXModelWrapper(mlx_model, tokenizer)
    
    # Create Model wrapper for synthetic generator (as in notebook)
    model = Model(name=MODEL_NAME, hf_model=hf_compatible_model)
    print("✓ Model wrapper created successfully")
    print()
    
    # Initialize synthetic generator using the same approach as notebook
    print("Initializing synthetic generator...")
    generator = SyntheticContrastivePairGenerator(model)
    
    # Define trait descriptions (just like in notebook)
    print(f"Will generate {NUM_PAIRS} synthetic pairs for each trait:")
    for name, description in TRAITS.items():
        print(f"- {name}: {description}")
    
    # Generate synthetic contrastive pairs for each trait
    pair_sets = {}
    
    for trait_name, trait_description in TRAITS.items():
        print(f"\nGenerating {trait_name} behavior pairs...")
        try:
            pair_set = generator.generate_contrastive_pair_set(
                trait_description=trait_description,
                num_pairs=NUM_PAIRS
            )
            pair_set.name = trait_name  # Set name after creation
            pair_sets[trait_name] = pair_set
            print(f"✓ Generated {len(pair_set.pairs)} {trait_name} pairs")
        except Exception as e:
            print(f"Error generating pairs: {e}")
            print("Note: Full integration requires adapting wisent_guard for MLX tensors")
    
    print("\n✅ Synthetic pairs generation attempted")
    
    # Show examples of generated pairs (if any)
    if pair_sets:
        for trait_name, pair_set in pair_sets.items():
            print(f"\n=== Example {trait_name.upper()} pairs ===")
            for i, pair in enumerate(pair_set.pairs[:2]):
                print(f"\nPair {i+1}:")
                print(f"Prompt: {pair.prompt}")
                print(f"Positive: {pair.positive_response.text}")
                print(f"Negative: {pair.negative_response.text}")
    
    print("\n" + "=" * 80)
    print("Note: Full wisent_guard integration with MLX requires tensor conversion")
    print("between MLX arrays and PyTorch tensors. The core generation logic is shown.")
    print("=" * 80)

if __name__ == "__main__":
    main()