"""
Wisent Guard: Activation-based safety guardrails for language models.
Clean implementation using enhanced core primitives.
"""

from .guard import WisentGuard
from .inference import SafeInference
from .vectors import ContrastiveVectors

# Import from core primitives
from .core import (
    Model,
    PromptFormat,
    TokenScore,
    ModelParameterOptimizer,
    ActivationHooks,
    Activations,
    ActivationAggregationMethod,
    ActivationMonitor,
    Layer,
    ContrastivePairSet,
    Classifier,
    ActivationClassifier,
    SteeringMethod
)

__version__ = "0.4.2"

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
    "ActivationAggregationMethod",
    "ActivationMonitor",
    "Layer",
    "ContrastivePairSet",
    "Classifier",
    "ActivationClassifier",
    "SteeringMethod"
] 