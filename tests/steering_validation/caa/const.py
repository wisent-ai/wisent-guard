from pathlib import Path

import torch

# Model configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_SIZE = "7b"
MODEL_HIDDEN_DIM = 4096  # Hidden dimension for Llama-2-7b
LAYER_INDEX = 14
MAX_EXAMPLES = 20  # Use only first 20 examples for fast testing (full dataset has 1000)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16
NORMALIZATION_METHOD = "none"  # Normalization method for CAA steering vectors

# Generation parameters
MAX_NEW_TOKENS = 50
TOP_K = 1  # Greedy decoding
RANDOM_SEED = 42
MAX_TEXT_EXAMPLES = 5  # For full text completion generation

# Behavior and steering
BEHAVIOR = "hallucination"
STEERING_STRENGTH = 1.0

# Path constants
WISENT_PATH = Path(__file__).parent.parent.parent.parent
REFERENCE_DATA_PATH = Path(__file__).parent / "reference_data"
HALLUCINATION_DATASET_PATH = REFERENCE_DATA_PATH / "hallucination.json"
TEXT_COMPLETIONS_UNSTEERED_PATH = REFERENCE_DATA_PATH / "text_completions_unsteered.json"
TEXT_COMPLETIONS_STEERED_PATH = REFERENCE_DATA_PATH / "text_completions_steered.json"

# Steering vector paths
HALLUCINATION_VECTOR_PATH = REFERENCE_DATA_PATH / f"hallucination_layer{LAYER_INDEX}.pt"
