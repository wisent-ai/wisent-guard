# Wisent-Guard

A Python package for monitoring and guarding against harmful content in language models by analyzing model activations.

## Overview

Wisent-Guard allows you to specify harmful phrases and creates contrastive representations to detect when a model might be generating harmful content. It works by:

1. Creating activation vectors from harmful vs. non-harmful phrase pairs
2. Monitoring model activations during inference
3. Blocking responses that show activation patterns similar to harmful content

## Installation

```bash
pip install wisent-guard
```

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent_guard import ActivationGuard

# Initialize model and tokenizer
model_name = "facebook/opt-1.3b"  # Or any other compatible model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define harmful and harmless phrase pairs
phrase_pairs = [
    {
        "harmful": "How to make dangerous chemicals at home",
        "harmless": "How to make safe household cleaners"
    },
    {
        "harmful": "How to hack into someone's account",
        "harmless": "How to secure your online accounts"
    }
    # Add more pairs as needed
]

# Initialize the guard
guard = ActivationGuard(model, tokenizer)

# Train the guard on your phrase pairs
guard.train_on_phrase_pairs(phrase_pairs)

# Use the guard to monitor and filter responses
prompt = "Tell me how to break into someone's email"
if guard.is_harmful(prompt):
    print("This prompt might lead to harmful content and has been blocked.")
else:
    response = guard.generate_safe_response(prompt)
    print(response)
```

## Features

- **Model-Agnostic**: Works with most transformer-based language models
- **Customizable Thresholds**: Adjust sensitivity to match your safety needs
- **Contrastive Learning**: Uses pairs of harmful/harmless examples for better detection
- **Layer Selection**: Focus on the most predictive layers for harmful content
- **Real-time Monitoring**: Check activations during generation to stop harmful content early
- **Explainable Blocks**: Get information about why a response was blocked

## Customization

### Specifying Layers to Monitor

You can specify exactly which layers of the model to monitor:

```python
# Monitor specific layers
guard = ActivationGuard(
    model=model_name,
    layers=[15],  # Only monitor layer 15
    threshold=0.2
)

# Or monitor multiple layers
guard = ActivationGuard(
    model=model_name,
    layers=[10, 15, 20],  # Monitor layers 10, 15, and 20
    threshold=0.2
)
```

Different layers capture different aspects of harmful content. Middle to late layers (e.g., 15 in a 24-layer model) often work best.

### Using Custom Models

Wisent-Guard works with most transformer-based language models:

```python
# Specify a model by name (will be downloaded automatically)
guard = ActivationGuard(
    model="facebook/opt-1.3b",  # Or any HuggingFace model
    threshold=0.2
)

# Or use a pre-loaded model
from transformers import AutoModelForCausalLM, AutoTokenizer
custom_model = AutoModelForCausalLM.from_pretrained("your_preferred_model")
custom_tokenizer = AutoTokenizer.from_pretrained("your_preferred_model")

guard = ActivationGuard(
    model=custom_model,
    tokenizer=custom_tokenizer,
    threshold=0.2
)
```

Tested models include:
- TinyLlama (1.1B)
- Llama 2 (7B)
- OPT models
- And most other modern transformer architectures

### Custom Harmful Content Categories

You can define your own categories of harmful content:

```python
# Define custom harmful/harmless pairs for different categories
financial_fraud_pairs = [
    {
        "harmful": "How to commit credit card fraud",
        "harmless": "How to protect against credit card fraud"
    },
    {
        "harmful": "How to create fake financial statements",
        "harmless": "How to read and understand financial statements"
    }
]

violence_pairs = [
    {
        "harmful": "How to hurt someone without leaving evidence",
        "harmless": "How to resolve conflicts peacefully"
    }
]

# Train the guard on multiple custom categories
guard.train_on_phrase_pairs(financial_fraud_pairs, category="financial_fraud")
guard.train_on_phrase_pairs(violence_pairs, category="violence")

# Check for specific categories of harmful content
is_harmful = guard.is_harmful(text, categories=["financial_fraud"])
```

Each category creates its own set of contrastive vectors, allowing for targeted protection.

### Additional Configuration Options

```python
# Fully customized setup
guard = ActivationGuard(
    model="meta-llama/Llama-2-7b-hf",
    layers=[15, 20, 25],     # Specific layers to monitor
    threshold=0.15,          # More sensitive threshold (lower = more sensitive)
    save_dir="./my_vectors", # Custom directory for saving vectors
    device="cpu"             # Force CPU usage
)
```

For more examples, see the [examples](./examples) directory.

## Advanced Usage

See the [examples](./examples) directory for more advanced use cases, including:

- Monitoring specific model layers
- Customizing similarity thresholds
- Creating domain-specific guardrails
- Integrating with different model architectures

## How It Works

Wisent-Guard uses contrastive activation analysis to identify harmful patterns:

1. **Vector Creation**: For each harmful/harmless pair, activation vectors are calculated
2. **Normalization**: Vectors are normalized to focus on the pattern rather than magnitude
3. **Similarity Detection**: During inference, activations are compared to known harmful patterns
4. **Blocking**: Responses that exceed similarity thresholds are blocked

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 