# DAC Steering Validation

This directory contains validation tests for the DAC (Dynamic Activation Composition) steering method. The setup generates reference steering vectors using the original DAC implementation and compares them with our wisent-guard implementation.

## Overview

DAC is a language steering method that works by:
1. Computing mean activations for dataset A (Italian responses) 
2. Computing mean activations for dataset B (English responses)
3. Creating a steering vector as the difference: `steering_vector = mean_A - mean_B`
4. Using this vector to steer the model toward Italian-style responses

## Files

- **`const.py`** - Configuration constants (model, datasets, paths, etc.)
- **`model_utils.py`** - Utilities for working with the original DAC implementation
- **`generate_data_with_original_dac_implementation.py`** - Main script to generate reference vectors
- **`test_dac_setup.py`** - Test script to validate the setup
- **`reference_data/`** - Directory containing reference datasets and generated vectors

## Datasets

We use limited versions of the original DAC datasets for fast testing:

- **ITA dataset** (`ita_train.json`): 20 examples with Italian responses
  - Instruction: "Rispondi alle seguenti domande" (Answer the following questions)
  - Examples: Questions with Italian answers
  
- **ENG dataset** (`eng_train.json`): 20 examples with English responses  
  - Instruction: "Answer the following questions"
  - Examples: Same questions with English answers

## Configuration

### Model
- **Model**: `mistralai/Mistral-7B-Instruct-v0.2` (default from DAC implementation)
- **Layers**: ALL layers (DAC operates across all 32 layers simultaneously)
- **Max new tokens**: 30
- **ICL examples**: 4 (in-context learning examples)

### Expected Vector Shapes
- **Mean activations**: `[30, 32, 32, 128]` (steps, layers, heads, d_head)
- **Difference vector**: Same shape as mean activations

## Usage

### 1. Test the Setup
```bash
cd /workspace/wisent-guard/tests/steering_validation/dac
python test_dac_setup.py
```

### 2. Generate Reference Vectors
```bash
python generate_data_with_original_dac_implementation.py
```

This will create:
- `mean_activations_ita.pt` - Mean activations for Italian responses (all layers)
- `mean_activations_eng.pt` - Mean activations for English responses (all layers)
- `diff_activations_ita_eng.pt` - Steering vector (ITA - ENG) for all layers

### 3. Run Validation Tests

**Fast tests (default):**
```bash
cd /workspace/wisent-guard/tests/steering_validation/dac
python -m pytest test_dac_vector_generation.py -v
```

**Heavy GPU tests (requires GPU and model loading):**
```bash
python -m pytest test_dac_vector_generation.py -m "heavy" -v
```

**All tests (including heavy):**
```bash
python -m pytest test_dac_vector_generation.py -m "heavy or not heavy" -v
```

## Dependencies

The setup requires:
- Access to the original DAC implementation in `Dynamic-Activation-Composition/`
- PyTorch for tensor operations
- Transformers library for model loading
- CUDA-capable GPU for model inference (recommended)

## Technical Details

### Dataset Processing
1. Load ITA and ENG datasets (20 examples each)
2. Apply in-context learning formatting with 4 examples per prompt
3. Tokenize using the model's tokenizer
4. Extract activations during generation

### Activation Extraction
1. Run the model on tokenized prompts
2. Extract residual stream activations from ALL layers (0-31)
3. Reshape activations to [steps, layers, heads, d_head] format
4. Compute mean across all examples for each dataset
5. Create difference vector for steering across all layers

### Memory Optimization
- Use 8-bit model loading to reduce memory usage
- Process datasets in batches
- Limit to 20 examples per dataset for fast testing
- Set max_new_tokens=30 to limit generation length

## Troubleshooting

### Import Errors
If you get import errors from DAC modules, ensure:
- The `Dynamic-Activation-Composition` directory exists
- All required Python packages are installed
- The path to DAC modules is correctly set

### Memory Issues
If you run out of GPU memory:
- Reduce `MAX_EXAMPLES` in `const.py`
- Reduce `MAX_NEW_TOKENS` in `const.py`  
- Enable 8-bit loading in model setup
- Use a smaller model for testing

### Shape Mismatches
If vector shapes don't match expectations:
- Check the model configuration in `const.py`
- Verify the model architecture matches Mistral-7B-Instruct-v0.2
- Ensure `LAYER_INDEX` is valid for the model