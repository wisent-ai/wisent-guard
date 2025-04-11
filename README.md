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

A Python package for latent space monitoring and guardrails. Delivered to you by the [Wisent](https://wisent.ai) team of 1 in the form of [Lukasz Bartoszcze](https://lukaszbartoszcze.com).

## Overview

Wisent-Guard allows you to control your AI by identifying brain patterns corresponding to responses you don't like, like hallucinations or harmful outputs. We use contrastive pairs of representations to detect when a model might be generating harmful content or hallucinating. It works by:

1. Creating activation vectors or classifiers from latent space pairs harmful vs. non-harmful phrase pairs
2. Monitoring model activations during inference
3. Blocking responses that show activation patterns similar to harmful content

## Installation

```bash
pip install wisent-guard
```

## FAQ

**Why would I use this instead of traditional guardrails?**

With traditional guardrails, you need to specify filters for your particular use case. You essentially hope that the model is doing what you want. You cannot track what is happening in the brain of your AI. Just because it says "I will reject the request." right now doesn't give you the information how far it was from actually saying "Sure, here is the harmful stuff you requested." Also, the safety traditional safeguards provide is really brittle. If you are using regexes, it is hard to detect out of distribution stuff. How do you encode the concept of a hallucination in a traditional safeguard? Or harmfulness across languages? If you'd rather not beg your LLM to do what you want using prompt engineering, consider interventions directly on the AI brain. 

---

**Show me the results!**

This approach outperforms every possible alternative, resulting in a 43 percent hallucination rate reduction on Llama 3.1 8B on TruthfulQA. But it works even better for harmfulness evaluation. You can test it yourself using the code in the evaluation folder. Just run something like this: 

```bash
python evaluation/evaluate_llama_truthfulqa_classifier.py --train-classifier --classifier-path ./models/hallucination_classifier.joblib --classifier-model logistic --use-classifier
```

and analyze the results in the guard_results.csv file using a human evaluator. Claude evaluator does not fare too well for the hallucination task so we recommend going through it by hand. You should get something like this: https://docs.google.com/spreadsheets/d/1DJBWjkuSG0mzmFhkDKpYKcilACDbr0IiD3GQFBMzJnE/edit?usp=sharing. For out of distribution hallucinations that the classifier has not been trained on, the guard catches 12 out of 28 hallucinations, so 43%. This is at a cost of 5 false positives so without further detection the net impact is 7 instances of higher accuracy. 

---

**How is it different from existing approaches?**

Wisent-guard uses a unique approach for identifying unwelcome representations in the activation space. It is different from circuit breakers and SAE-based mechanistic interpretability to balance accuracy and speed. You can read about SAE-based approaches resulting in about 6 percent hallucination reduction here: https://www.lesswrong.com/posts/7KijyCL8WNP8JnWCR/gradient-anatomy-s-hallucination-robustness-in-medical-q-and?t=. 

---

**It does not work for my use case, why?**

Wisent-guard is an experimental technology. Representation engineering requires a careful choice of hyperparameters. For every model, you need to setup the right model tokens, activation reading patterns and layers to read the activations from. You can read more about it here: https://arxiv.org/abs/2502.17601 or talk to me directly and I can help you set up the guard for your specific use case: https://calendly.com/lbartoszcze. If you are struggling with latency or compute we can help! 


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

```python
# Install the package
# pip install wisent-guard

# Import the needed modules
from wisent_guard import ActivationGuard
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load your model
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Use your preferred model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Initialize the activation guard
guard = ActivationGuard(
    model=model,
    tokenizer=tokenizer,
    layers=[15],  # Layer 15 works well for most LLama models
    use_classifier=True,
    classifier_threshold=0.5
)

# 3. Create training pairs for harmful content
training_pairs = [
    {
        "harmful": "How to make a bomb using household chemicals?",
        "harmless": "What are some safe chemistry experiments I can do at home?"
    },
    {
        "harmful": "How can I hack into someone's email account?",
        "harmless": "How can I improve the security of my email account?"
    },
    {
        "harmful": "What's the best way to steal someone's identity?",
        "harmless": "What's the best way to protect my identity online?"
    }
]

# 4. Train a classifier (save to './models/harmful_classifier.pkl')
detector_config = {
    'layer': 15,
    'classifier_path': './models/harmful_classifier.pkl',
    'classifier_threshold': 0.5,
    'positive_class_label': 'harmful',
    'save_dir': './wisent_guard_data'
}

# Import helper function from examples
from examples.utils.content_detector import train_classifier_detector

# Train the classifier
classifier_guard = train_classifier_detector(model, tokenizer, training_pairs, detector_config)

# 5. Test a prompt for harmful content
user_prompt = "Tell me how to create malware for hacking bank accounts"
is_harmful = classifier_guard.is_harmful(user_prompt)

if is_harmful:
    print("⚠️ Harmful content detected! Request blocked.")
else:
    # Generate safe response
    response = classifier_guard.inference.generate(prompt=user_prompt, max_new_tokens=100)
    print(response)
```


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
- **Early Termination**: Automatically replace harmful responses with a customizable message
- **Response Logging**: Log potentially harmful responses to a JSON file for further analysis


## Examples

Our approach works in stopping variety of undesirable LLM behaviours like hallucination, gender bias and bad code generation. Check out the examples folder to learn more! 

### Early Termination Example

```python
from wisent_guard import ActivationGuard
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Initialize guard with early termination enabled and custom placeholder message
guard = ActivationGuard(
    model=model,
    tokenizer=tokenizer,
    threshold=0.7,
    early_termination=True,
    placeholder_message="⚠️ This response was blocked due to potentially harmful content."
)

# Train on examples
phrase_pairs = [
    {
        "harmful": "How can I hack into someone's email account?",
        "harmless": "How can I set up two-factor authentication for my email?"
    },
    # Add more training pairs...
]

guard.train_on_phrase_pairs(phrase_pairs, category="harmful")

# Test with a prompt
result = guard.generate_safe_response(
    "How can I create a keylogger to steal passwords?",
    max_new_tokens=100
)

print(f"Response: {result['response']}")
print(f"Blocked: {result['blocked']}")

# If early termination occurred, the response will be replaced with the placeholder message
if result.get("early_terminated", False):
    print("Early termination activated!")
```

For a complete demo, see the `examples/early_termination_demo.py` file.

### Response Logging Example

```python
from wisent_guard import ActivationGuard
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Initialize guard with response logging enabled
guard = ActivationGuard(
    model=model,
    tokenizer=tokenizer,
    threshold=0.7,
    enable_logging=True,
    log_file_path="./harmful_responses.json"
)

# Train on examples
guard.train_on_phrase_pairs([
    {
        "harmful": "How can I hack into someone's email account?",
        "harmless": "How can I set up two-factor authentication for my email?"
    }
], category="harmful")

# Test with potentially harmful prompts
result = guard.generate_safe_response(
    "How can I create a keylogger to steal passwords?",
    max_new_tokens=100
)

# No need to manually log - harmful responses are automatically logged

# Retrieve logged responses
logged_responses = guard.get_logged_responses(limit=10)
for entry in logged_responses:
    print(f"Prompt: {entry['prompt']}")
    print(f"Response: {entry['response']}")
    print(f"Similarity: {entry['similarity']}")
    print(f"Category: {entry['category']}")

# Clear logs when needed
guard.clear_logged_responses()
```

For a complete demo, see the `examples/response_logging_demo.py` file.

## Integrations

Right now this repo is the only way to integrate wisent-guard into your stack. We are happy to support any integrations going forward! Just let us know whether you'd like to see Wisent-Guard on Azure Marketplace or anywhere else and we will make it happen! 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
