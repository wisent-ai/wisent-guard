# Examples

This directory contains practical examples for using Wisent-Guard with different benchmarks.

## Available Examples

### 1. LiveCodeBench Evaluation Example

**File**: `livecodebench_evaluation_example.py`

Complete workflow from training to evaluation:

```bash
python examples/livecodebench_evaluation_example.py
```

This example:
- Trains steering vectors on LiveCodeBench v1 (5 problems for quick testing)
- Saves the model in HuggingFace format
- Creates a ready-to-run evaluation script for LiveCodeBench v2
- Provides step-by-step instructions for evaluation

**Output**:
- `./example_models/distilgpt2_with_steering/` - Trained model
- `./run_livecodebench_evaluation.sh` - Evaluation script
- `./example_experiments/` - Training results and metadata

### 2. Quick Test

For a very quick test without training:

```bash
python -c "
from wisent_guard.core.models import create_steering_compatible_model
import torch

# Create a model with dummy steering vectors
model = create_steering_compatible_model('distilgpt2')
model.add_steering_vector(5, torch.randn(768))  # Add dummy steering vector
model.save_pretrained('./quick_test_model')

print('Quick test model saved to ./quick_test_model')
print('You can now use this with LiveCodeBench or other benchmarks')
"
```

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- Wisent-Guard (installed in development mode)

## Running Examples

1. **Install Wisent-Guard in development mode**:
   ```bash
   pip install -e .
   ```

2. **Run an example**:
   ```bash
   python examples/livecodebench_evaluation_example.py
   ```

3. **Follow the generated instructions** for evaluation with external benchmarks.

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure Wisent-Guard is installed: `pip install -e .`
2. **CUDA errors**: Set `device='cpu'` in training config or use `export CUDA_VISIBLE_DEVICES=""`
3. **Memory issues**: Reduce `batch_size` or `data_limit` in training config
4. **Network issues**: Some examples require internet access to download datasets

### Getting Help

- Check the main [EVALUATION_GUIDE.md](../EVALUATION_GUIDE.md) for detailed instructions
- Review the training logs in `./example_experiments/`
- Check model info with `model.get_steering_info()`