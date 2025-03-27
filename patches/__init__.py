"""
wisent-guard patches

This package contains patches and compatibility fixes for the wisent-guard library.
"""

from .mps_compatibility import apply_mps_patches

__all__ = ["apply_mps_patches"] 