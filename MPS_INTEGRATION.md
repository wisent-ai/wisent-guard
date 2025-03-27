# MPS Integration for Wisent Guard

This document describes the changes made to integrate Apple Silicon MPS (Metal Performance Shaders) support directly into the Wisent Guard codebase.

## Overview

Wisent Guard now includes native support for Apple Silicon GPUs through the MPS backend in PyTorch. This integration eliminates the need for separate patch files by incorporating proper device handling directly into the core files.

## Key Changes

### 1. Device Detection and Selection

Added native MPS device detection in the `ActivationGuard` initialization:

```python
if device is None:
    if torch.cuda.is_available():
        self.device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        self.device = "mps"
        print("Using Apple Silicon GPU via MPS")
    else:
        self.device = "cpu"
else:
    self.device = device
```

### 2. Tensor Operations on MPS

Updated tensor operations throughout the codebase to properly handle MPS-specific requirements:

- Added `.to(device)` calls to ensure tensors are allocated on the correct device
- Implemented proper error handling for MPS-specific failures
- Added dimension mismatch handling for MPS tensors

### 3. Activation Hook Improvements

Modified the `ActivationHooks` class to properly handle MPS tensors:

- Ensured tensors are properly allocated on the device
- Added proper error handling for MPS tensor operations
- Added `has_activations()` helper method for consistency

### 4. Vector Processing for MPS

Improved the vector processing operations:

- Updated `cosine_sim` function to safely handle MPS tensors
- Enhanced `normalize_vector` to handle edge cases on MPS
- Modified `calculate_average_vector` for better MPS compatibility

### 5. Training Process Updates

Updated the training process to handle MPS devices:

- Added CPU transfer for vector storage to avoid MPS allocation issues
- Enhanced error handling during training

### 6. Inference Process Improvements

Made the inference process more robust on MPS:

- Enhanced error handling in `_check_prompt_safety`
- Improved tensor device management in generation

## Important Note: Patches No Longer Needed

**The external patches directory has been completely removed.** All MPS-specific fixes and enhancements have been fully integrated into the main codebase. Users no longer need to apply any patches manually.

This means:
1. No need to import any patches module
2. No need to call any `apply_mps_patches()` function
3. MPS support works out-of-the-box

## Benefits

1. **Simplified Setup**: Native support for MPS without any additional steps
2. **Better Maintainability**: Device-specific handling is integrated into the core code
3. **Improved Error Handling**: Robust error handling for all device types
4. **Consistent API**: Same API works across all supported devices

## Usage

With these changes, Wisent Guard can be used on Apple Silicon without any special configuration:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent_guard import ActivationGuard

# Initialize model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# MPS will be automatically detected and used
guard = ActivationGuard(
    model=model,
    tokenizer=tokenizer,
    layers=[15]
)

# Train and use as normal
guard.train_on_phrase_pairs(phrase_pairs)
```

## Future Work

While the MPS integration is now working, there are potential areas for further optimization:

1. Performance tuning for specific MPS operations
2. Implementing device-specific optimizations for handling large models
3. Adding telemetry to measure performance across different devices 