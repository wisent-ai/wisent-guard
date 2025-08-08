from pathlib import Path
import torch

# Model configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_HIDDEN_DIM = 4096  # Hidden dimension for Llama-2-7b
LAYER_INDEX = 14
MAX_EXAMPLES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16
NORMALIZATION_METHOD = "none"  # Normalization method for CAA steering vectors

# Path constants
REFERENCE_DATA_PATH = Path(__file__).parent / "reference_data"
HALLUCINATION_DATASET_PATH = REFERENCE_DATA_PATH / "hallucination.json"
TEXT_COMPLETIONS_UNSTEERED_PATH = REFERENCE_DATA_PATH / "text_completions_unsteered.json"
TEXT_COMPLETIONS_STEERED_PATH = REFERENCE_DATA_PATH / "text_completions_steered.json"
