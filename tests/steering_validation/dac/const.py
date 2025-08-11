"""
Constants and configuration for DAC steering validation tests.
"""

from pathlib import Path

# Test configuration
MAX_EXAMPLES = 20  # Use 20 examples (original working setting)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Default model from DAC implementation
DATASET_A_NAME = "ITA"  # Italian responses dataset
DATASET_B_NAME = "ENG"  # English responses dataset
MAX_NEW_TOKENS = 30  # Maximum new tokens to generate (original working setting)
ICL_EXAMPLES = 0  # Number of in-context learning examples

# Paths
TEST_DIR = Path(__file__).parent
REFERENCE_DATA_PATH = TEST_DIR / "reference_data"

# Dataset paths
DATASET_A_PATH = REFERENCE_DATA_PATH / f"{DATASET_A_NAME.lower()}_train.json"
DATASET_B_PATH = REFERENCE_DATA_PATH / f"{DATASET_B_NAME.lower()}_train.json"
DATASET_A_INSTR_PATH = REFERENCE_DATA_PATH / f"{DATASET_A_NAME.lower()}_instr.txt"
DATASET_B_INSTR_PATH = REFERENCE_DATA_PATH / f"{DATASET_B_NAME.lower()}_instr.txt"

# Vector storage paths (DAC works across all layers)
ACTIVATIONS_A_PATH = REFERENCE_DATA_PATH / f"mean_activations_{DATASET_A_NAME.lower()}.pt"
ACTIVATIONS_B_PATH = REFERENCE_DATA_PATH / f"mean_activations_{DATASET_B_NAME.lower()}.pt"
DIFF_ACTIVATIONS_PATH = REFERENCE_DATA_PATH / f"diff_activations_{DATASET_A_NAME.lower()}_{DATASET_B_NAME.lower()}.pt"

# Expected vector shapes
EXPECTED_VECTOR_SHAPE = (MAX_NEW_TOKENS, 32, 32, 128)  # [steps, n_layers, n_heads, d_head] for Mistral-7B

# Model configuration (Mistral-7B-Instruct-v0.2)
MODEL_CONFIG = {
    "n_layers": 32,
    "n_heads": 32,
    "d_model": 4096,
    "d_head": 128,  # d_model / n_heads
    "attn_hook_names": [f"model.layers.{i}" for i in range(32)],
}

# Model inference settings
import torch

TORCH_DTYPE = torch.float16  # Use half precision for efficiency
DEVICE = "auto"  # Let transformers choose the best device

# Tolerance for cosine similarity comparison
COSINE_SIMILARITY_THRESHOLD = 0.99
