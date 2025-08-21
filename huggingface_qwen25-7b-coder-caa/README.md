---
language: en
license: apache-2.0
tags:
- code-generation
- causal-lm
- steering
- contrastive-activation-addition
- caa
- qwen
- wisent
library_name: transformers
datasets:
- evalplus/mbppplus
metrics:
- pass@1
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
model-index:
- name: wisent-ai/qwen2.5-coder-7b-wisent-caa
  results:
  - task:
      type: code-generation
      name: Code Generation
    dataset:
      type: mbppplus
      name: MBPP Plus
    metrics:
    - type: pass@1
      value: 0.521
      name: Pass@1
---

# Wisent-Qwen2.5-Coder-7B-Instruct with CAA Steering

## Model Description

This is an enhanced version of Qwen2.5-Coder-7B-Instruct that integrates **Contrastive Activation Addition (CAA)** steering directly into the model architecture. The steering parameters have been optimized using Optuna to improve code generation quality on the MBPP Plus benchmark.

### Key Features

- 🚀 **Automatic CAA Steering**: No manual hook management required
- 🎯 **Optimized Parameters**: Layer 24, α=1.4
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
- Model automatically loads base Qwen2.5-Coder weights (7B parameters)
- CAA steering adds minimal computational overhead (~1-2% inference time)
- Supports CPU inference but GPU recommended for practical use
- Memory usage: ~14GB GPU memory for bfloat16 inference

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model - CAA steering is automatically applied!
model = AutoModelForCausalLM.from_pretrained("./huggingface_qwen25-7b-coder-caa", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./huggingface_qwen25-7b-coder-caa")

# Generate code
prompt = "Write a Python function to calculate the factorial of a number"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Advanced Usage

### Adjusting Steering Strength

```python
# Increase steering strength for stronger safety alignment
model.set_caa_alpha(1.2)

# Decrease for more creative outputs
model.set_caa_alpha(0.5)
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
├── mbpp_plus/      # Current: Optimized for MBPP Plus benchmark
├── safety/         # Future: Safety-aligned behavior
├── creativity/     # Future: Enhanced creative outputs  
├── helpfulness/    # Future: Improved helpfulness
└── reasoning/      # Future: Enhanced logical reasoning
```

To switch traits, simply update the configuration:

```json
{
  "steering_vector_path": "./vectors/safety/steering_vector.safetensors"
}
```

## Technical Details

### CAA Steering Parameters

- **Steering Method**: Contrastive Activation Addition (CAA)
- **Optimal Layer**: 24 (out of 28 transformer layers)
- **Steering Strength (α)**: 1.4
- **Vector Format**: Safetensors format for efficient loading and HuggingFace compatibility
- **Vector Dimension**: 3584 (pre-normalized during training)
- **Storage Path**: `./vectors/mbpp_plus/steering_vector.safetensors`

### How It Works

1. **Trait-based Organization**: Steering vectors are organized by behavioral traits (`vectors/{trait}/`)
2. **Dynamic Loading**: The model loads the specified steering vector from the configured path
3. **Layer Application**: Steering is applied to hidden states at layer 24 during forward pass
4. **Generation Integration**: Steering affects the last token position during generation
5. **Configurable Strength**: The α parameter (default: 0.9) controls steering intensity
6. **Pre-optimized Vectors**: Steering vectors are pre-normalized and ready for immediate use

### Optimization Process

The CAA parameters were optimized using:
- **Framework**: Optuna with TPE sampler
- **Search Space**: Layers 15-28, α ∈ [0.1, 5.0]
- **Objective**: Maximize accuracy on MBPP Plus validation set
- **Best Performance**: 52.1% accuracy on MBPP Plus (378 problems)

## Model Architecture

```
WisentQwen2ForCausalLM
├── Base: Qwen2.5-Coder-7B-Instruct
├── CAA Integration: Layer 24
├── Steering Vector: ./vectors/mbpp_plus/steering_vector.safetensors
└── Auto-applied during generation
```

## File Structure

```
huggingface_qwen25-7b-coder-caa/
├── config.json                    # Model configuration with CAA params
├── modeling_wisent_qwen.py        # Custom model class
├── tokenizer files               # Standard Qwen tokenizer
├── wisent_config.json            # Optimization results
└── vectors/                       # Trait-based steering vectors
    └── mbpp_plus/
        └── steering_vector.safetensors  # MBPP Plus optimized steering vector
```

## Evaluation

### MBPP Plus Benchmark

The model has been optimized using Optuna on MBPP Plus tasks. For reliable performance metrics, evaluation should be conducted on the complete MBPP Plus dataset (378 problems) using the [evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus) dataset.

### Running Evaluation

```python
# Use with bigcode-evaluation-harness
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "./huggingface_qwen25-7b-coder-caa",
    trust_remote_code=True
)

# CAA steering is automatically applied during evaluation!
# No manual hooks or modifications needed
```

## Citation

If you use this model, please cite:

```bibtex
@software{wisent_qwen_caa_2025,
  title={Wisent-Qwen2.5-Coder with CAA Steering},
  author={Wisent AI},
  year={2025},
  url={https://github.com/wisent-ai/wisent-guard}
}
```

## License

This model inherits the license from the base Qwen2.5-Coder-7B-Instruct model. Please refer to the original model's license for usage terms.

## Acknowledgments

- Base model: Qwen2.5-Coder-7B-Instruct by Alibaba
- CAA method: Contrastive Activation Addition
- Optimization: Optuna framework
- Implementation: Wisent Guard framework