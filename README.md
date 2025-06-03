# Wisent-Guard

<p align="center">
  <a href="https://github.com/wisent-ai/wisent-guard/stargazers">
    <img src="https://img.shields.io/github/stars/wisent-ai/wisent-guard" alt="stars" />
  </a>
  <a href="https://pypi.org/project/wisent-guard">
    <img src="https://static.pepy.tech/badge/wisent-guard" alt="PyPI - Downloads" />
  </a>
  <br />
</p>

<p align="center">
  <img src="wisent-guard-logo.png" alt="Wisent Guard" width="200">
</p>

A Python package for latent space monitoring and guardrails. Delivered to you by the [Wisent](https://wisent.ai) team led by [Lukasz Bartoszcze](https://lukaszbartoszcze.com).

## Overview

Wisent-Guard allows you to control your AI by identifying brain patterns corresponding to responses you don't like, like hallucinations or harmful outputs. We use contrastive pairs of representations to detect when a model might be generating harmful content or hallucinating. Learn more at https://www.wisent.ai/wisent-guard.  
Contributions are welcome! Please feel free to submit a Pull Request or open an issue so that we can fix all bugs potentially troubling you.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
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