"""
Wisent-Guard: Monitor and guard against harmful content in language models
"""

from .guard import ActivationGuard
from .monitor import ActivationMonitor
from .vectors import ContrastiveVectors
from .inference import SafeInference

__version__ = "0.1.0" 