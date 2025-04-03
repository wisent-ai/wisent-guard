# Wisent-Guard

<p align="center">
  <img src="wisent-guard-logo.png" alt="Wisent Guard" width="200">
</p>

A Python package for monitoring and guarding against harmful content and hallucinations in language models by analyzing model activations. Delivered to you by the [Wisent](https://wisent.ai) team of 1 in the form of [Lukasz Bartoszcze](https://lukaszbartoszcze.com).

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

With traditional guardrails, you need to specify filters for your particular use case. You essentially hope that the model is doing what you want. You cannot track what is happening in the brain of your AI. Just because it says "I will reject the request." right now does give you the information how far it was from actually saying "Sure, here is the harmful stuff you requested." Also, the safety traditional safeguards provide is really brittle. If you are using regexes, it is hard to detect out of distribution stuff. How do you encode the concept of a hallucination in a traditional safeguard? Or harmfulness across languages? If you'd rather not beg your LLM to do what you want using prompt engineering, consider interventions directly on the AI brain. 


Show me the results! 

This approach outperforms every possible alternative, resulting in a 43 percent hallucination rate reduction on Llama 3.1 8B and TruthfulQA. But it works even better for harmfulness evaluation. You can test it yourself using the code in the evaluation folder. 


How is it different from existing approaches? 

Wisent-guard uses a unique approach for identifiying unwelcome representations in the activation space. It is different from circuit breakers and SAE-based mechanistic interpretability to balance accuracy and speed. 

It does not work for my use case, why? 

Wisent-guard is an experimental technology. Representation engineering requires a careful choice of hyperparameters. For every model, you need to setup the right model tokens, activation reading patterns and layers to read the activations from. You can read more about it here: https://arxiv.org/abs/2502.17601 or talk to me directly and I can help you set up the guard for your specific use case: https://calendly.com/lbartoszcze. If you are struggling with latency 

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

Once we extract those activations, we make the first big choice of what to do with this data. One of the options is to compute a vector from the activation means, normalize it and at inference time compare the cosine similarity to the activations from the tokens. The other we recommend is to train a classifier directly on the activation data and monitor activations. Depending on whether you want to prioritize speed or have a particular use case where one method performs better than the other. You can also run optimization algos to see whether everything works as you intend it to. 

We can then monitor the generations of the model for their similarity with the harmful examples you provide us with to better fit your needs. 


## Quick Start

```
```

## Evaluation

Before running any evaluations set the .env of the judge to whatever works. You can do it using Claude or a human evaluator. Claude is really bad at most complex evaluations like evaluating TruthfulQA questions. 

export ANTHROPIC_API_KEY='your-api-key-here'. 

## Features

- **Generalized Framework**: Provides a flexible approach for blocking activations corresponding to a wide set of properties, including the ones you can customize on your own.
- **Model-Agnostic**: Works with most transformer-based language models
- **Apple Silicon Native Support**: Built-in support for MPS (Metal Performance Shaders) on Apple Silicon GPUs
- **Customizable Thresholds**: Adjust sensitivity to match your safety needs
- **Contrastive Learning**: Uses pairs of harmful/harmless examples for better detection
- **Layer Selection**: Focus on the most predictive layers for harmful content
- **Real-time Monitoring**: Check activations during generation to stop harmful content early
- **Deep Insights**: Understand whether your model is having evil thoughts even when its outputs do not suggest so
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 