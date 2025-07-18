# Evaluation Guide: Training and Evaluating Steering Vectors

This guide shows how to train steering vectors on LiveCodeBench and evaluate them for contamination-free assessment.

## Overview

The workflow is simple:
1. **Train**: Create steering vectors using LiveCodeBench v1 data
2. **Save**: Export the model with steering vectors
3. **Evaluate**: Use the model with LiveCodeBench v2 for evaluation

## Step 1: Train Steering Vectors

### Quick Start (5 problems for testing)
```bash
python -c "
from wisent_guard.core.pipelines import ExperimentRunner

# Create quick experiment
runner = ExperimentRunner.create_quick_experiment(data_limit=5)

# Run the experiment
results = runner.run_experiment()

print(f'Training completed!')
print(f'Steering vectors saved to: {results.training_results.config.output_directory}')
print(f'Model available at: {results.steering_model}')
"
```

### Full Training (recommended)
```bash
python -c "
from wisent_guard.core.pipelines import ExperimentRunner
from wisent_guard.core.pipelines.steering_trainer import TrainingConfig
from wisent_guard.core.pipelines.experiment_runner import ExperimentConfig

# Configure training
training_config = TrainingConfig(
    model_name='distilgpt2',
    target_layers=[5],  # DistilGPT2 has 6 layers (0-5), use layer 5
    steering_method='CAA',
    batch_size=4,
    device='auto'
)

# Configure experiment
experiment_config = ExperimentConfig(
    train_version='release_v1',      # Train on v1
    eval_version='release_v2',       # Evaluate on v2  
    version_split_type='new_only',   # Contamination-free
    data_limit=100,                  # Use 100 problems for training
    pairs_per_problem=1,
    training_config=training_config,
    experiment_name='livecodebench_steering_full'
)

# Run experiment
runner = ExperimentRunner(experiment_config)
results = runner.run_experiment()

print(f'Training completed!')
print(f'Model saved to: ./steering_experiments/')
"
```

## Step 2: Save Model for Evaluation

The trained model is automatically saved in HuggingFace format. You can also manually save it:

```bash
python -c "
from wisent_guard.core.models import SteeringCompatibleModel

# Load your trained model (replace with your actual path)
model = SteeringCompatibleModel.from_pretrained_with_steering(
    'distilgpt2',
    steering_directory='./steering_experiments/livecodebench_steering_full_20250118_123456'
)

# Save in HuggingFace format for evaluation
model.save_pretrained('./models/distilgpt2_with_steering')

print('Model saved to ./models/distilgpt2_with_steering')
"
```

## Step 3: Evaluate with LiveCodeBench

### Install LiveCodeBench
```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
pip install -e .
```

### Run Evaluation

#### Option A: Direct Evaluation
```bash
python -m lcb_runner.runner.main \
    --model ./models/distilgpt2_with_steering \
    --scenario code_generation \
    --release_version v2 \
    --temperature 0.2 \
    --max_length_generation 512 \
    --n_samples 1 \
    --evaluate
```

#### Option B: Custom Evaluation Script
```bash
python -c "
import os
import subprocess
import json

# Configuration
model_path = './models/distilgpt2_with_steering'
output_dir = './evaluation_results'
os.makedirs(output_dir, exist_ok=True)

# Run LiveCodeBench evaluation
cmd = [
    'python', '-m', 'lcb_runner.runner.main',
    '--model', model_path,
    '--scenario', 'code_generation',
    '--release_version', 'v2',
    '--temperature', '0.2',
    '--max_length_generation', '512',
    '--n_samples', '1',
    '--output_dir', output_dir,
    '--use_cache',
    '--continue_existing',
    '--evaluate'
]

print('Running LiveCodeBench evaluation...')
result = subprocess.run(cmd, capture_output=True, text=True)

print('STDOUT:')
print(result.stdout)
print('STDERR:')
print(result.stderr)

# Results are saved in output_dir
print(f'Results saved to: {output_dir}')
"
```

## Step 4: Compare Results

### Baseline Comparison
To compare against the baseline model without steering:

```bash
# Evaluate baseline model
python -m lcb_runner.runner.main \
    --model distilgpt2 \
    --scenario code_generation \
    --release_version v2 \
    --temperature 0.2 \
    --max_length_generation 512 \
    --n_samples 1 \
    --evaluate

# Compare results
python -c "
import json
import os

# Load results (adjust paths as needed)
with open('./evaluation_results/results.json', 'r') as f:
    steering_results = json.load(f)

with open('./baseline_results/results.json', 'r') as f:
    baseline_results = json.load(f)

print('Results Comparison:')
print(f'Baseline pass@1: {baseline_results.get(\"pass@1\", \"N/A\")}')
print(f'Steering pass@1: {steering_results.get(\"pass@1\", \"N/A\")}')
"
```

## Common Issues and Solutions

### 1. Model Not Found
If you get a model not found error:
```bash
# Check if model exists
ls -la ./models/distilgpt2_with_steering

# If not, re-run the save step or check your paths
```

### 2. CUDA/GPU Issues
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Or specify GPU
export CUDA_VISIBLE_DEVICES="0"
```

### 3. Memory Issues
```bash
# Reduce batch size in training
# Use smaller data_limit for training
# Use gradient checkpointing
```

### 4. LiveCodeBench Installation Issues
```bash
# Make sure you have the right Python version
python --version  # Should be 3.11+

# Install with uv (recommended)
pip install uv
uv pip install -e .
```

## Example Complete Workflow

Here's a complete example from start to finish:

```bash
# 1. Train steering vectors (quick test)
python -c "
from wisent_guard.core.pipelines import ExperimentRunner
runner = ExperimentRunner.create_quick_experiment(data_limit=5)
results = runner.run_experiment()
print('Training done!')
"

# 2. Save model
python -c "
from wisent_guard.core.models import SteeringCompatibleModel
model = SteeringCompatibleModel.from_pretrained_with_steering(
    'distilgpt2',
    steering_directory='./steering_experiments/quick_test_20250118_123456'  # Adjust path
)
model.save_pretrained('./models/distilgpt2_steering_test')
print('Model saved!')
"

# 3. Install LiveCodeBench (if not already installed)
# git clone https://github.com/LiveCodeBench/LiveCodeBench.git
# cd LiveCodeBench && pip install -e .

# 4. Run evaluation
python -m lcb_runner.runner.main \
    --model ./models/distilgpt2_steering_test \
    --scenario code_generation \
    --release_version v2 \
    --temperature 0.2 \
    --max_length_generation 512 \
    --n_samples 1 \
    --limit 10 \
    --evaluate

echo "Evaluation complete!"
```

## Tips for Production Use

1. **Use larger models**: Replace `distilgpt2` with `gpt2` or `microsoft/CodeGPT-small-py` for better performance
2. **Increase training data**: Use `data_limit=None` for full dataset training
3. **Multiple steering vectors**: Train on multiple layers by setting `target_layers=[3, 4, 5]`
4. **Contamination-free evaluation**: Always use `version_split_type='new_only'` when training on v1 and evaluating on v2
5. **Save experiment metadata**: Keep track of your experiments in `./steering_experiments/`

## Next Steps

- Try different steering methods (currently supports CAA)
- Experiment with different target layers
- Use different base models
- Compare performance on different LiveCodeBench versions
- Evaluate on other benchmarks (HumanEval, MBPP) using similar patterns