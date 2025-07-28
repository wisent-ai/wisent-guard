# GSM8K Optimization Pipeline

A comprehensive pipeline for optimizing steering methods on GSM8K using Optuna hyperparameter search with support for deterministic and stochastic generation.

## Overview

This pipeline implements a scientifically rigorous workflow for evaluating activation steering methods:

1. **Train Split**: Train probes and steering vectors
2. **Validation Split**: Optuna-based hyperparameter search 
3. **Test Split**: Single evaluation with best configuration

## Key Features

### 🧪 **Scientific Rigor** 
- ✅ **Deterministic Generation**: `temperature=0.0` for reproducible debugging
- ✅ **Zero-Steering Validation**: Ensures α=0 equals baseline (critical for steering validity)
- ✅ **Proper Train/Val/Test Splits**: No data leakage between optimization and evaluation

### ⚡ **Performance & Efficiency**
- ✅ **Activation Caching**: Smart caching system avoids recomputing activations
- ✅ **Optuna Integration**: TPE sampler + MedianPruner for efficient optimization
- ✅ **Multi-GPU Support**: Auto-detection (MPS/CUDA/CPU) with memory management

### 🎛️ **Flexibility**
- ✅ **Temperature Control**: Configurable generation randomness (0.0-2.0)
- ✅ **Multiple Steering Methods**: DAC and CAA implementations
- ✅ **Model Agnostic**: Works with any HuggingFace model

### 📊 **Reproducibility**
- ✅ **Complete Artifact Saving**: Configs, results, random seeds, git hashes
- ✅ **Steering Re-forwarding**: Proper model re-forwarding for steering evaluation

## Quick Start

### 🚀 Minimal Example (Recommended)

Test fine-tuned model behavior with small steering ranges:

```bash
python optuna_minimal_example.py
```

**Expected behavior**: Fine-tuned models (like `realtreetune/rho-1b-sft-GSM8K`) should show:
- High baseline accuracy (~27%)
- Optimal steering α ≈ 0.0 
- Performance degradation as α increases

### 🐛 Debug Mode

Quick validation with small datasets:

```bash
python run_gsm8k_optimization.py --mode debug --model realtreetune/rho-1b-sft-GSM8K
```

### 🎯 Full Optimization

Complete hyperparameter search:

```bash
python run_gsm8k_optimization.py --mode full --model realtreetune/rho-1b-sft-GSM8K --n-trials 50
```

## Configuration Options

### Generation Control
```bash
# Deterministic (for debugging/comparison)
--temperature 0.0

# Stochastic (for natural generation) 
--temperature 0.7 --do-sample

# Conservative sampling
--temperature 0.1 --do-sample
```

### Model & Data
```bash
--model realtreetune/rho-1b-sft-GSM8K  # Fine-tuned GSM8K model
--model microsoft/DialoGPT-medium       # Base model for comparison
```

### Optimization
```bash
--n-trials 50          # Number of Optuna trials
--output-dir my_exp     # Custom output directory
```

## Recommended Models

### 🎯 **Fine-tuned Models** (High baseline performance)
- `realtreetune/rho-1b-sft-GSM8K` - Fine-tuned on GSM8K (~27% accuracy)
- Expect: Optimal steering α ≈ 0.0, steering hurts performance

### 🔄 **Base Models** (Low baseline performance)  
- `microsoft/DialoGPT-medium` - General conversation model
- `EleutherAI/gpt-neo-1.3B` - General language model
- Expect: Higher optimal steering α, steering may help performance

## Pipeline Components

### Core Files
- `gsm8k_optimization_pipeline.py` - Main pipeline implementation
- `run_gsm8k_optimization.py` - Command-line runner with full configuration
- `optuna_minimal_example.py` - Focused example for fine-tuned model testing

### Debug & Analysis
- `debug_zero_steering.py` - Validates zero-steering behavior (α=0 should equal baseline)
- `test_steering_validation.py` - Comprehensive steering sensitivity analysis
- `test_model_directly.py` - Direct model testing utilities
- `analyze_results.py` - Results visualization and analysis

## Output Structure

```
outputs/my_experiment/
├── best_configuration_*.json    # Best hyperparameters + metadata
├── config_*.json               # Full pipeline configuration  
├── final_results_*.json        # Complete test results
├── study_trials_*.csv          # All Optuna trial results
└── optuna_study_*.db          # Optuna study database

cache/my_experiment/
└── activations_*.pkl          # Cached activation matrices
```

## Scientific Validation

### ✅ Zero-Steering Test
The pipeline includes validation that α=0 steering equals baseline generation:

```bash
python debug_zero_steering.py
```

This ensures steering implementation is mathematically correct.

### 📊 Steering Sensitivity
Analyze how performance changes across steering strengths:

```bash
python test_steering_validation.py
```

## Key Implementation Details

### Deterministic vs Stochastic Generation

**Deterministic** (`temperature=0.0, do_sample=False`):
- Same input always produces same output
- Required for: debugging, zero-steering validation, reproducible comparison
- Use for: scientific evaluation, pipeline validation

**Stochastic** (`temperature>0.0, do_sample=True`):
- Introduces randomness for more natural generation
- Use for: exploring model behavior, practical applications

### Activation Caching System

Caches activations based on:
- Split name (train/val/test)
- Layer ID  
- Tokenization configuration
- Prompt variant

Cache ensures efficiency while maintaining correctness.

### Steering Implementation

**Critical**: Steering evaluation re-runs model forward passes because steering modifies activations during generation. The pipeline uses PyTorch hooks to apply steering vectors at specific layers.

## Requirements

### Dependencies
```bash
pip install optuna scikit-learn torch transformers
```

### Hardware Recommendations
- **Debug mode**: 4GB GPU memory
- **Full mode**: 8GB+ GPU memory
- **CPU fallback**: Available but slower

## Troubleshooting

### Common Issues

1. **"Zero steering doesn't equal baseline"**
   - Solution: Use `temperature=0.0` for deterministic generation
   - Check: Run `debug_zero_steering.py` to validate

2. **All trials fail**
   - Check: Model name is correct and accessible
   - Check: Sufficient GPU memory
   - Try: Reduce batch_size or sample limits

3. **No steering improvement**
   - Expected for fine-tuned models (steering should hurt performance)
   - Try: Base models instead of fine-tuned models

### Debug Workflow

1. Start with `optuna_minimal_example.py` 
2. Validate with `debug_zero_steering.py`
3. Scale up with debug mode
4. Run full optimization

## Examples

### Testing Fine-tuned Model
```python
from gsm8k_optimization_pipeline import GSM8KOptimizationPipeline, OptimizationConfig

config = OptimizationConfig(
    model_name="realtreetune/rho-1b-sft-GSM8K",
    train_limit=20,
    val_limit=20, 
    test_limit=20,
    n_trials=8,
    temperature=0.0,  # Deterministic
    do_sample=False
)

pipeline = GSM8KOptimizationPipeline(config)
results = pipeline.run_optimization()

print(f"Optimal steering α: {results['best_trial_params']['steering_alpha']:.4f}")
print(f"Improvement: {results['accuracy_improvement']:+.4f}")
```

## Expected Results

### Fine-tuned Models
- **Baseline accuracy**: 20-30% (model already optimized)
- **Optimal steering α**: 0.0-0.1 (minimal steering)
- **Improvement**: Negative (steering hurts fine-tuned performance)

### Base Models  
- **Baseline accuracy**: 0-10% (model not optimized for math)
- **Optimal steering α**: 0.2-1.0 (more steering needed)
- **Improvement**: Positive or neutral (steering may help)

## Recent Updates

### v2.1 (January 2025)
- ✅ **Temperature Control**: Added configurable generation temperature
- ✅ **Zero-Steering Fix**: Fixed bug where α=0 didn't equal baseline
- ✅ **Minimal Example**: Added focused example for fine-tuned models
- ✅ **Deterministic Mode**: Default to reproducible generation

### v2.0 (January 2025)
- ✅ **Optuna Integration**: Complete hyperparameter optimization
- ✅ **Activation Caching**: Efficient activation reuse
- ✅ **Multiple Steering Methods**: DAC and CAA support