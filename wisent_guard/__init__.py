"""
Wisent-Guard: Monitor and guard against harmful content in language models

Wisent-Guard provides tools to detect and prevent language models from generating harmful content
by monitoring model activations and comparing them to known harmful patterns.

Features:
- Create contrastive vectors from harmful/harmless phrase pairs
- Monitor model activations during inference
- Block responses that show activation patterns similar to harmful content
- Convert examples to multiple-choice format for consistent activation collection
- Token-by-token analysis of all response tokens (default behavior)

Device Support:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon GPUs)
- CPU
"""

from .guard import ActivationGuard
from .monitor import ActivationMonitor
from .vectors import ContrastiveVectors
from .inference import SafeInference

# Updated with Torch only support 
__version__ = "0.4.1" 