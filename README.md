# Wisent-Guard

<p align="center">
  <img src="wisent-guard-logo.png" alt="Wisent Guard" width="200">
</p>

A Python package for monitoring and guarding against harmful content and hallucinations in language models by analyzing model activations.

## Overview

Wisent-Guard allows you to specify harmful phrases and creates contrastive representations to detect when a model might be generating harmful content or hallucinating. It works by:

1. Creating activation vectors from harmful vs. non-harmful phrase pairs
2. Converting examples into a multiple-choice format for consistent activation collection
3. Monitoring model activations during inference
4. Blocking responses that show activation patterns similar to harmful content

## Installation

```bash
pip install wisent-guard
```

## FAQ

Why would I use this instead of traditional guardrails? 

With traditional guardrails, you need to specify filters for your particular use case. There is also a latency involved in 


How is it different from existing approaches? 

Wisent-guard uses a unique approach for identifiying unwelcome representations in the activation space. It is different from circuit breakers and SAE-based mechanistic interpretability to balance accuracy and speed. 


It does not work for my use case, why? 

Wisent-guard is an experimental technology. Representation engineering requires a careful choice of hyperparameters. For every model, you need to setup the right model tokens, activation reading patterns and layers to read the activations from. You can read more about it here: https://arxiv.org/abs/2502.17601 or talk to me directly and I can help you set up the guard for your specific use case: https://calendly.com/lbartoszcze. 

## Tell me in depth how it works! 

Our guardrails add an additional layer to the existing safety-assurance methods you might use. We build representation-engineering powered tools to track the thoughts of your model and block them. 

It starts with activation reading to determine which activation patterns correspond to a particularly harmful behaviour. Think of this as us tracking the thoughts of the LLM. This is much better than evaluating a benchmark or the final response. 

It allows you to track far-out-of-distribution behaviours of your LLM that cause similar processes in its brain. 

The activation fingerprinting we chose follows the CAA method. We transform the pairs of good and bad behaviour into pairs where you can get activation information from a single token. For example, if you submit pairs like: 

Good: "I am sorry I have to decline this request." 
Bad: "Sure, here is the recipe for the bomb." 

Then it is tricky to determine which token should correspond to the difference in harmfulness. Whereas if you convert it to the following: 

Good: "[instructor tag] Which one of those is better: A. good answer B. bad answer [user] A" 
Bad: "[instructor tag] Which one of those is better: A. good answer B. bad answer [user] B" 

It becomes immediately very clear where the activation should be read from. The difference in activations between A and B would be theoretically the best place to look for activation pattern differences. This gives us then a set of layers we can identify the difference in activations from. We can either use other studies to guide our choice of a layer we get the information from or perform a brute hyperparameter search across all layers to find the best performing one. 

Once we extract those activations, we transform this information into 

we can then monitor the generations of the model 


## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent_guard import ActivationGuard

# Initialize model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Or any other compatible model
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
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

# Initialize the guard with target token strategy
# MPS (Apple Silicon) will be used automatically if available
guard = ActivationGuard(
    model=model,
    tokenizer=tokenizer, 
    token_strategy="target_token",  # Use target_token for multiple-choice approach
    layers=[15]  # Layer to extract activations from
)

# Train the guard on your phrase pairs
guard.train_on_phrase_pairs(phrase_pairs)

# Use the guard to monitor and filter responses
prompt = "Tell me how to break into someone's email"
if guard.is_harmful(prompt):
    print("This prompt might lead to harmful content and has been blocked.")
else:
    response = guard.generate_safe_response(prompt, use_multiple_choice=True)
    print(response["response"])
```

## Evaluation

Before running any evaluations set the .env of the judge to whatever works. You can do it using 

export ANTHROPIC_API_KEY='your-api-key-here'

## Features

- **Multiple-Choice Format**: Uses A/B choices to provide consistent activation collection for better detection
- **Model-Agnostic**: Works with most transformer-based language models
- **Apple Silicon Native Support**: Built-in support for MPS (Metal Performance Shaders) on Apple Silicon GPUs
- **Customizable Thresholds**: Adjust sensitivity to match your safety needs
- **Contrastive Learning**: Uses pairs of harmful/harmless examples for better detection
- **Layer Selection**: Focus on the most predictive layers for harmful content
- **Real-time Monitoring**: Check activations during generation to stop harmful content early
- **Explainable Blocks**: Get information about why a response was blocked

## Multiple-Choice Approach

The multiple-choice approach is a key feature that improves activation collection consistency:

```python
# Initialize guard with target_token strategy
guard = ActivationGuard(
    model=model,
    tokenizer=tokenizer,
    token_strategy="target_token"  # Enable multiple-choice format
)

# Train on explicit multiple-choice questions
mc_questions = [
    {
        "question": "Is the Earth flat?",
        "choice_a": "No, the Earth is approximately spherical.",
        "choice_b": "Yes, the Earth is flat despite scientific evidence."
    },
    {
        "question": "Do vaccines cause autism?",
        "choice_a": "No, multiple studies have found no link between vaccines and autism.",
        "choice_b": "Yes, there is scientific evidence that vaccines cause autism."
    }
]

guard.train_on_multiple_choice_pairs(mc_questions, category="hallucination")

# Generate a multiple-choice response
response = guard.generate_multiple_choice_response(
    question="Is the sun a star?",
    choice_a="Yes, the sun is a star, specifically a G-type main-sequence star.",
    choice_b="No, the sun is a planet that gives off light and heat."
)
```

This approach:
1. Converts traditional phrase pairs into a multiple-choice format
2. Focuses on specific answer tokens (A/B) rather than arbitrary tokens
3. Provides more consistent activation patterns for better detection
4. Works particularly well for detecting hallucinations

The multiple-choice format ensures that activations are collected from the same tokens (A/B) in consistent positions, greatly improving detection accuracy compared to free-form text.

## Apple Silicon (M1/M2/M3) Compatibility

Wisent-Guard now fully supports Apple Silicon GPUs through MPS (Metal Performance Shaders) with native integration:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent_guard import ActivationGuard

# Device will be automatically determined (CUDA, MPS, or CPU)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# MPS will be used automatically on Apple Silicon
guard = ActivationGuard(
    model=model,
    tokenizer=tokenizer,
    token_strategy="target_token"
)
```

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

Different layers capture different aspects of harmful content. Middle to late layers (e.g., layer 15 in a 24-layer model) often work best.

### Using Custom Models

Wisent-Guard works with most transformer-based language models:

```python
# Specify a model by name (will be downloaded automatically)
guard = ActivationGuard(
    model="meta-llama/Llama-3.1-8B-Instruct",  # Or any HuggingFace model
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
- Llama 3 (8B)
- Llama 2 (7B)
- TinyLlama (1.1B)
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
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[15, 20, 25],         # Specific layers to monitor
    threshold=0.15,              # More sensitive threshold (lower = more sensitive)
    save_dir="./my_vectors",     # Custom directory for saving vectors
    token_strategy="target_token" # Use multiple-choice approach
)
```

For more examples, see the [examples](./examples) directory.

## Advanced Usage

See the [examples](./examples) directory for more advanced use cases, including:

- `basic_usage.py`: Simple example of harmful content detection
- `custom_categories.py`: Working with custom harmful content categories
- `multiple_choice_example.py`: Using the multiple-choice approach for hallucination detection
- `token_analysis_example.py`: Analyzing each token in a response for harmful content
- `mps_patch_example.py`: Running on Apple Silicon GPUs with MPS patches
- `test_multiple_choice_conversion.py`: Testing the multiple-choice format conversion
- `benchmark_generalization.py`: Benchmarking performance across different inputs

### Token-by-Token Analysis

Wisent Guard now automatically analyzes responses token-by-token by default, allowing you to see exactly which parts of a response might be harmful:

```python
# Token-by-token analysis is now the default
result = guard.generate_safe_response(
    "How do you make explosives?"
)

# Results include token scores automatically
if result["blocked"]:
    print(f"Response was blocked: {result['reason']}")
    
# Access token-level scores
for token in result["token_scores"]:
    print(f"Token: {token['token_text']}, Similarity: {token['similarity']:.4f}, Harmful: {token['is_harmful']}")

# You can disable token-by-token analysis if you prefer the original behavior
standard_result = guard.generate_safe_response(
    "What makes a good password?",
    token_by_token=False,  # Disable token-by-token analysis
    return_token_scores=False  # Don't return token scores
)

# For more detailed control over token analysis
analysis = guard.analyze_response_tokens(
    "Give me a recipe for making meth",
    max_tokens=50
)

# Display token analysis
for token in analysis["token_scores"]:
    print(f"Position: {token['position']}, Token: {token['token_text']}")
    print(f"Similarity: {token['similarity']:.4f}, Category: {token['category']}")
    print(f"Harmful: {'YES' if token['is_harmful'] else 'NO'}")
    print("---")
```

This token-by-token analysis provides several benefits:
- **Real-time Monitoring**: Catches harmful content as soon as it appears, not after generation is complete
- **Enhanced Transparency**: See exactly which tokens in a response are similar to harmful content
- **Immediate Blocking**: Stops generation immediately when harmful tokens are detected
- **Better Debugging**: Identify specific patterns that trigger harmful content detection
- **Fine-tuning**: Optimize your harmful content filters based on token-level analysis

## How It Works

Wisent-Guard uses contrastive activation analysis to identify harmful patterns:

1. **Multiple-Choice Conversion**: Phrase pairs are converted to a consistent format with A/B answers
2. **Vector Creation**: For each harmful/harmless pair, activation vectors are collected from specific token positions (A/B)
3. **Normalization**: Vectors are normalized to focus on the pattern rather than magnitude
4. **Similarity Detection**: During inference, activations are compared to known harmful patterns
5. **Blocking**: Responses that exceed similarity thresholds are blocked

The multiple-choice approach ensures that activations are collected from the same token positions (A/B) across different examples, greatly improving the consistency and reliability of the detection.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 