"""
Steering methods package for wisent-guard.

This package contains implementations of various steering methods:
- CAA: Contrastive Activation Addition
- HPR: Householder Pseudo-Rotation  
- DAC: Dynamic Activation Composition
- BiPO: Bi-directional Preference Optimization
- K-Steering: Multi-directional steering using gradient-based optimization
"""

from .base import SteeringMethod
from .caa import CAA
from .hpr import HPR
from .dac import DAC
from .bipo import BiPO
from .k_steering import KSteering

__all__ = [
    'SteeringMethod',
    'CAA',
    'HPR', 
    'DAC',
    'BiPO',
    'KSteering'
] 