---
language: en
license: apache-2.0
tags:
- mathematics
- math-reasoning
- causal-lm
- steering
- contrastive-activation-addition
- caa
- qwen
- wisent
library_name: transformers
datasets:
- hendrycks_math
metrics:
- accuracy
base_model: Qwen/Qwen2.5-Math-7B-Instruct
model-index:
- name: wisent-ai/qwen2.5-math-7b-wisent-caa
  results:
  - task:
      type: mathematical-reasoning
      name: Mathematical Reasoning
    dataset:
      type: hendrycks_math
      name: MATH Dataset
    metrics:
    - type: accuracy
      value: 0.774
      name: Accuracy
---

# Wisent-Qwen2.5-Math-7B-Instruct with CAA Steering

## Model Description

This is an enhanced version of Qwen2.5-Math-7B-Instruct that integrates **Contrastive Activation Addition (CAA)** steering directly into the model architecture. The steering parameters have been optimized using Optuna to improve mathematical reasoning performance on the MATH (Hendrycks) dataset.

### Key Features

- 🚀 **Automatic CAA Steering**: No manual hook management required
- 🎯 **Optimized Parameters**: Layer 23, α=0.45
- 🗂️ **Trait-Based Organization**: Steering vectors organized by traits
- 🔧 **Runtime Configurable**: Adjust or disable steering on the fly
- 🤗 **HuggingFace Compatible**: Works with standard transformers API

## Installation

```bash
pip install transformers torch safetensors
# Or install from requirements.txt
pip install -r requirements.txt
```

## Hardware Requirements

### Minimum Requirements:
- **GPU Memory**: 16GB VRAM (for inference with bfloat16)
- **System RAM**: 32GB recommended
- **Storage**: 15MB (model configuration + steering vectors)

### Recommended Setup:
- **GPU**: NVIDIA RTX 4090, A100, or similar
- **CUDA**: 11.8 or newer
- **Python**: 3.8-3.11

### Performance Notes:
- Model automatically loads base Qwen2.5-Math weights (7B parameters)
- CAA steering adds minimal computational overhead (~1-2% inference time)
- Supports CPU inference but GPU recommended for practical use
- Memory usage: ~14GB GPU memory for bfloat16 inference

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model - CAA steering is automatically applied!
model = AutoModelForCausalLM.from_pretrained("./huggingface_qwen25-7b-math-caa", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./huggingface_qwen25-7b-math-caa")

# Solve mathematical problems
prompt = "Find the derivative of f(x) = 3x^2 + 2x - 1"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Advanced Usage

### Adjusting Steering Strength

```python
# Increase steering strength for enhanced mathematical reasoning
model.set_caa_alpha(0.6)

# Decrease for more flexible problem solving
model.set_caa_alpha(0.3)
```

### Disabling CAA Steering

```python
# Disable CAA to get baseline model behavior
model.set_caa_enabled(False)

# Re-enable CAA
model.set_caa_enabled(True)
```

### Accessing Steering Configuration

```python
print(f"CAA Layer: {model.caa_layer_id}")
print(f"CAA Alpha: {model.caa_alpha}")
print(f"Steering Method: {model.steering_method}")
```

### Trait-Based Vector Organization

The model uses a trait-based organization for steering vectors:

```
vectors/
├── hendrycks_math/ # Current: Optimized for MATH dataset
├── algebra/        # Future: Algebra-specific reasoning
├── geometry/       # Future: Geometric problem solving
├── calculus/       # Future: Calculus and analysis
└── number_theory/  # Future: Number theory problems
```

To switch traits, simply update the configuration:

```json
{
  "steering_vector_path": "./vectors/hendrycks_math/steering_vector.safetensors"
}
```

## Technical Details

### CAA Steering Parameters

- **Steering Method**: Contrastive Activation Addition (CAA)
- **Optimal Layer**: 23 (out of 28 transformer layers)
- **Steering Strength (α)**: 0.45
- **Vector Format**: Safetensors format for efficient loading and HuggingFace compatibility
- **Vector Dimension**: 3584 (pre-normalized during training)
- **Storage Path**: `./vectors/hendrycks_math/steering_vector.safetensors`

### How It Works

1. **Trait-based Organization**: Steering vectors are organized by behavioral traits (`vectors/{trait}/`)
2. **Dynamic Loading**: The model loads the specified steering vector from the configured path
3. **Layer Application**: Steering is applied to hidden states at layer 23 during forward pass
4. **Generation Integration**: Steering affects the last token position during generation
5. **Configurable Strength**: The α parameter (default: 0.45) controls steering intensity
6. **Pre-optimized Vectors**: Steering vectors are pre-normalized and ready for immediate use

### Optimization Process

The CAA parameters were optimized using:
- **Framework**: Optuna with TPE sampler
- **Search Space**: Layers 18-27, α ∈ [0.05, 1.5]
- **Objective**: Maximize accuracy on MATH dataset validation set
- **Best Performance**: 77.4% accuracy on MATH dataset (competition mathematics)

## Model Architecture

```
WisentQwen2ForCausalLM
├── Base: Qwen2.5-Math-7B-Instruct
├── CAA Integration: Layer 23
├── Steering Vector: ./vectors/hendrycks_math/steering_vector.safetensors
└── Auto-applied during generation
```

## File Structure

```
huggingface_qwen25-7b-math-caa/
├── config.json                    # Model configuration with CAA params
├── modeling_wisent_qwen.py        # Custom model class
├── tokenizer files               # Standard Qwen tokenizer
├── wisent_config.json            # Optimization results
└── vectors/                       # Trait-based steering vectors
    └── hendrycks_math/
        └── steering_vector.safetensors  # MATH dataset optimized steering vector
```

## Evaluation

### MATH Dataset Benchmark

The model has been optimized using Optuna on the MATH (Hendrycks) dataset. For reliable performance metrics, evaluation should be conducted on the complete MATH dataset using competition mathematics problems with step-by-step reasoning.

### Running Evaluation

```python
# Use with lm-eval-harness for mathematical evaluation
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "./huggingface_qwen25-7b-math-caa",
    trust_remote_code=True
)

# CAA steering is automatically applied during evaluation!
# Optimized for mathematical reasoning with deterministic generation
```

## Citation

If you use this model, please cite:

```bibtex
@software{wisent_qwen_math_caa_2025,
  title={Wisent-Qwen2.5-Math with CAA Steering},
  author={Wisent AI},
  year={2025},
  url={https://github.com/wisent-ai/wisent-guard}
}
```

## License

This model inherits the license from the base Qwen2.5-Math-7B-Instruct model. Please refer to the original model's license for usage terms.

## Acknowledgments

- Base model: Qwen2.5-Math-7B-Instruct by Alibaba
- CAA method: Contrastive Activation Addition
- Optimization: Optuna framework
- Implementation: Wisent Guard framework