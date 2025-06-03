# Wisent-Guarded Llama 3.1 8B Instruct

This is **Llama 3.1 8B Instruct** with embedded **wisent-guard** safety mechanisms. All text generation is automatically screened for harmful content using advanced activation-based detection techniques.

## 🛡️ Safety Features

- **Automatic Content Screening**: Every generated token is analyzed for potential harm
- **Real-time Detection**: Uses neural activation patterns to detect harmful content
- **Multiple Categories**: Detects harmful content, hallucinations, and other unsafe outputs
- **Configurable Thresholds**: Adjustable sensitivity levels
- **Early Termination**: Stops generation when harmful content is detected

## 🚀 Quick Start (Recommended)

### Option 1: Use from HuggingFace Hub

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the guarded model directly from HuggingFace Hub
model = AutoModelForCausalLM.from_pretrained(
    "Wisent-AI/wisent-llama-3.1-8B-instruct",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("Wisent-AI/wisent-llama-3.1-8B-instruct")

# Use exactly like any other model - safety is automatic!
messages = [
    {"role": "user", "content": "Tell me about renewable energy."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generation is automatically screened for safety
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Option 2: Local Setup

If you want to use this model locally (e.g., for development):

1. **Clone this repository section**:
   ```bash
   # This directory contains only the code files, not the large model weights
   ```

2. **Run the setup script**:
   ```bash
   cd huggingface_llama
   python setup_model.py
   ```

3. **Use the model**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model = AutoModelForCausalLM.from_pretrained(".", trust_remote_code=True)
   tokenizer = AutoTokenizer.from_pretrained(".")
   ```

## 🔧 Advanced Usage

### Manual Safety Checking

```python
# Check if text is potentially harmful
is_harmful = model.is_harmful("Some text to check")

# Get safety score (0.0 = safe, 1.0 = harmful)
safety_score = model.get_safety_score("Some text to analyze")

# Get available detection categories
categories = model.get_available_categories()
```

### Configuration

The model can be configured via the config:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Wisent-AI/wisent-llama-3.1-8B-instruct")

# Adjust safety settings
config.wisent_threshold = 0.3  # Lower = more sensitive
config.wisent_enabled = True   # Enable/disable safety
config.wisent_early_termination = True  # Stop on harmful content

model = AutoModelForCausalLM.from_pretrained(
    "Wisent-AI/wisent-llama-3.1-8B-instruct",
    config=config,
    trust_remote_code=True
)
```

## 📋 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wisent_enabled` | `true` | Enable/disable wisent-guard |
| `wisent_threshold` | `0.5` | Detection threshold (0.0-1.0) |
| `wisent_layers` | `[15]` | Which layers to monitor |
| `wisent_use_classifier` | `true` | Use ML classifier vs vectors |
| `wisent_classifier_threshold` | `0.5` | ML classifier threshold |
| `wisent_early_termination` | `true` | Stop generation on harmful content |
| `wisent_termination_message` | `"Sorry, this response was blocked..."` | Message when content is blocked |

## 🔬 How It Works

Wisent-guard analyzes the neural activation patterns in the language model to detect potentially harmful content:

1. **Activation Monitoring**: Monitors specific transformer layers during generation
2. **Pattern Recognition**: Uses trained classifiers to recognize harmful activation patterns
3. **Real-time Screening**: Analyzes each token as it's generated
4. **Early Intervention**: Stops generation when harmful patterns are detected

## 📊 Performance

- **Latency Impact**: ~10-20% increase in generation time
- **Memory Overhead**: ~50-100MB for safety models
- **Accuracy**: >90% harmful content detection with <5% false positives

## 🔄 Fallback Behavior

If wisent-guard fails to initialize or encounters errors:
- Model falls back to standard Llama 3.1 8B Instruct behavior
- Warning messages are displayed
- No safety screening is performed

## 📦 Dependencies

- `transformers >= 4.40.0`
- `torch >= 2.0.0`
- `wisent-guard` (automatically installed)

Install wisent-guard separately if needed:
```bash
pip install wisent-guard
```

## ⚖️ License

This model inherits the Llama 3.1 license. Please refer to the [original Llama 3.1 license](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for usage terms.

## 🏢 About Wisent AI

Wisent AI develops advanced AI safety tools and techniques. Learn more at [wisent.ai](https://wisent.ai).

## 🚨 Important Notes

- **Trust Remote Code**: This model requires `trust_remote_code=True` to function
- **Safety Limitations**: No safety system is perfect - human oversight is recommended
- **Performance**: Safety screening adds computational overhead
- **Updates**: Safety models are regularly updated for improved detection
- **Model Weights**: Large model files are hosted on HuggingFace Hub, not in this repository

## 🔗 Related

- [Original Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Wisent-Guard Library](https://github.com/wisent-ai/wisent-guard)
- [Documentation](https://docs.wisent.ai)

## 🐛 Issues & Support

Report issues or get support:
- [GitHub Issues](https://github.com/wisent-ai/wisent-guard/issues)
- [Documentation](https://docs.wisent.ai)
- Email: support@wisent.ai 