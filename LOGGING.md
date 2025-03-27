# Logging System for Wisent Guard

This document describes the comprehensive logging system integrated into Wisent Guard to provide detailed visibility into the contrastive vector creation and evaluation process.

## Overview

The logging system is designed to:
1. Show the complete flow of contrastive vector creation
2. Track activation collection and comparison
3. Provide useful debugging information
4. Allow varying levels of detail based on need

## Logging Components

### 1. Logger Setup (`utils/logger.py`)

The core logging utility provides:
- Multiple log levels (debug, info, warning, error)
- Console output with color coding
- Optional file logging
- Consistent formatting

### 2. Integration in Core Classes

Logging is integrated throughout the codebase:
- `ActivationGuard`: High-level operations and coordination
- `ContrastiveVectors`: Vector creation and storage
- `ActivationMonitor`: Monitoring and detection
- `ActivationHooks`: Activation collection

### 3. Contrastive Vector Flow Logging

The complete flow is logged in detail:
1. **Initialization**: Model loading, device detection, settings
2. **Phrase Pair Processing**: Converting to multiple-choice format
3. **Activation Collection**: Layer-by-layer activation extraction
4. **Vector Computation**: Creating and normalizing vectors
5. **Vector Storage**: Saving to disk
6. **Vector Usage**: Detecting similarities during inference

### 4. Summary Generation

The `get_contrastive_summary()` and `print_contrastive_summary()` methods provide an overview of:
- All vectors in the system
- Training data statistics
- Vector properties
- Sample similarity tests

## Usage

### Basic Usage

```python
from wisent_guard import ActivationGuard
from wisent_guard.utils.logger import get_logger

# Create logger
logger = get_logger(level="debug", log_file="wisent.log")

# Initialize guard with logging
guard = ActivationGuard(
    model=model,
    tokenizer=tokenizer,
    log_level="debug"  # Set log level in the constructor
)
```

### Demo Script

The included `contrastive_vector_demo.py` script demonstrates the complete flow with logging:

```bash
# Run with info-level logging
python contrastive_vector_demo.py --log_level info

# Run with detailed debug logging
python contrastive_vector_demo.py --log_level debug --log_file wisent_debug.log
```

### Log Levels

- **debug**: Detailed information for troubleshooting
- **info**: General operational information
- **warning**: Issues that might cause problems
- **error**: Serious problems that need attention

## Example Log Output

Here's what to expect in the logs for each phase:

### Initialization Phase
```
[12:34:56] INFO: [wisent_guard] Initializing ActivationGuard
[12:34:56] INFO: [wisent_guard] Using Apple Silicon GPU via MPS
[12:34:56] INFO: [wisent_guard] Loading model from gpt2...
```

### Training Phase
```
[12:34:57] INFO: [wisent_guard] Training on 3 phrase pairs for category 'harmful_content'...
[12:34:57] DEBUG: [wisent_guard] Processing pair 1/3
[12:34:57] DEBUG: [wisent_guard] Harmful: How to hack into someone's email account
[12:34:57] DEBUG: [wisent_guard] Harmless: How to secure your email account against hackers
```

### Activation Collection
```
[12:34:58] DEBUG: [wisent_guard.hooks] Hook triggered for layer 5
[12:34:58] DEBUG: [wisent_guard.hooks] Hidden states shape: torch.Size([1, 42, 768]), device: mps:0
[12:34:58] DEBUG: [wisent_guard.hooks] Using target token strategy, looking for tokens: [588, 713]
```

### Vector Computation
```
[12:34:59] DEBUG: [wisent_guard.vectors] Averaging 3 harmful vectors
[12:34:59] DEBUG: [wisent_guard.vectors] Averaging 3 harmless vectors
[12:34:59] DEBUG: [wisent_guard.vectors] Computing difference vector (harmful - harmless)
[12:34:59] DEBUG: [wisent_guard.vectors] Normalizing contrastive vector
```

### Detection Phase
```
[12:35:00] DEBUG: [wisent_guard.monitor] Checking category: harmful_content
[12:35:00] DEBUG: [wisent_guard.monitor] Comparing activation and contrastive vector for layer 5
[12:35:00] DEBUG: [wisent_guard.monitor] Similarity for layer 5: 0.8234
[12:35:00] INFO: [wisent_guard.monitor] Harmful pattern detected in layer 5 for category 'harmful_content'
```

## Getting a Complete Summary

After training and testing, you can get a complete overview of the vectors:

```python
# Print a formatted summary to console
guard.print_contrastive_summary()

# Get the raw summary data
summary = guard.get_contrastive_summary()
```

This will show:
- Settings used for training
- Categories and their vector properties
- Sample similarity tests
- Layer-by-layer information 