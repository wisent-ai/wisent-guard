# MPS Compatibility Patches for wisent-guard

## Overview

This directory contains comprehensive MPS (Metal Performance Shaders) compatibility patches for the wisent-guard package, enabling it to run efficiently on Apple Silicon GPUs (M1/M2/M3 chips).

## Key Accomplishments

1. **Full MPS Compatibility**: Implemented patches to ensure all core functionality of the wisent-guard library works correctly on Apple Silicon GPUs.

2. **Fixed Device Allocation Issues**: Addressed critical tensor allocation issues that previously caused errors when running on MPS.

3. **Improved Multiple-Choice Approach**: Enhanced the multiple-choice format for more reliable detection of harmful content and hallucinations.

4. **Added Missing Helper Methods**: Implemented necessary helper methods for consistent behavior across different devices.

5. **Error Handling**: Added robust error handling to gracefully continue operation even when specific checks fail.

6. **Centralized Patch System**: Created a single, easy-to-import patch file that can be applied with a single function call.

## Patched Methods

The following methods have been patched for MPS compatibility:

- `ActivationHooks._activation_hook`: Fixed tensor allocation and device handling
- `ActivationHooks.has_activations` and `get_activations`: Added new helper methods
- `SafeInference._check_prompt_safety`: Improved device handling during safety checks
- `SafeInference.generate`: Enhanced generation process with proper error handling
- `ActivationGuard.train_on_phrase_pairs`: Updated to ensure consistent device allocation
- `ActivationGuard._train_on_formatted_pairs`: Fixed tensor movement between devices
- `ActivationGuard.generate_safe_response`: Improved output format consistency
- `ActivationGuard.generate_multiple_choice_response`: Enhanced multiple-choice handling
- `ActivationGuard.train_on_multiple_choice_pairs`: Fixed multi-choice training
- `ActivationMonitor.check_activations`: Added dimension mismatch handling
- `ContrastiveVectors`: Added helper methods for vector management

## Usage

Apply these patches in your code with:

```python
import torch
from wisent_guard import ActivationGuard

# Apply MPS patches if on Apple Silicon
if torch.backends.mps.is_available():
    from patches import apply_mps_patches
    apply_mps_patches()
```

## Examples

See `examples/mps_patch_example.py` for a complete demonstration of using wisent-guard with MPS support.

## Key Benefits

- **Performance**: Run wisent-guard on Apple Silicon GPUs for significantly faster processing compared to CPU
- **Compatibility**: Use all features of the library seamlessly on Mac devices with M1/M2/M3 chips
- **Reliability**: Process larger models and datasets with proper GPU memory management
- **Consistency**: Get the same results whether running on CPU, CUDA, or MPS devices 