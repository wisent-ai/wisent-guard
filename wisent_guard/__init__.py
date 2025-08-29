"""
Wisent Guard: Activation-based safety guardrails for language models.
Clean implementation using enhanced core primitives.
"""

# Import from core primitives
from .core import (
    ActivationClassifier,
    ActivationHooks,
    Activations,
    Classifier,
    ContrastivePairSet,
    Layer,
    Model,
    ModelParameterOptimizer,
    PromptFormat,
    SteeringMethod,
    TokenScore,
)
from .guard import WisentGuard
from .inference import SafeInference
from .vectors import ContrastiveVectors

__version__ = "0.4.46"

__all__ = [
    # Main classes
    "WisentGuard",
    "SafeInference",
    "ContrastiveVectors",
    # Core primitives
    "Model",
    "PromptFormat",
    "TokenScore",
    "ModelParameterOptimizer",
    "ActivationHooks",
    "Activations",
    "Layer",
    "ContrastivePairSet",
    "Classifier",
    "ActivationClassifier",
    "SteeringMethod",
]
