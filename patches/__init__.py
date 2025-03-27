"""
wisent-guard patches

This package contains patches and compatibility fixes for the wisent-guard library.
"""

from .mps_compatibility import apply_mps_patches
from .mps_comprehensive_fix import apply_comprehensive_mps_fixes

__all__ = ["apply_mps_patches", "apply_comprehensive_mps_fixes"] 