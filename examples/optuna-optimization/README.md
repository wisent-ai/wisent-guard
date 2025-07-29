# Optimization Examples

This directory contains comprehensive examples for optimizing steering vectors with different models and configurations.

## Files

### `llama_optuna.py` - Production Llama 3.1 8B Optimization
Comprehensive optimization script for Llama 3.1 8B Instruct with multiple steering methods, SQLite persistence, and WandB integration.

**Recommended Hardware:**
- 2x 24GB GPUs (RTX 4090, A6000, etc.)
- 64GB+ RAM for dataset loading and caching
- Fast SSD for database and caching

**Recommended Batch Sizes:**
```bash
# Conservative (recommended for stability)
python llama_optuna.py --batch-size 2

# Aggressive (if you have ample VRAM)
python llama_optuna.py --batch-size 4

# Memory-constrained systems
python llama_optuna.py --batch-size 1
```

**Memory Usage Guidelines:**
- Batch size 1: ~18GB VRAM
- Batch size 2: ~22GB VRAM (recommended)
- Batch size 4: ~28GB VRAM (requires 32GB+ cards)

### `optuna_minimal_example.py` - Testing and Validation
Minimal example for testing the optimization pipeline with small models and datasets.

## Quick Start

1. **Quick Test:**
   ```bash
   # Test with small model and dataset
   python optuna_minimal_example.py
   
   # Quick test of Llama setup
   python llama_optuna.py --quick-test
   ```

2. **Full Optimization:**
   ```bash
   # Basic run
   python llama_optuna.py
   
   # With WandB tracking
   python llama_optuna.py --use-wandb --wandb-project my-project
   
   # Resume existing study
   python llama_optuna.py --study-name existing_study
   ```

## Steering Methods Compared

The `llama_optuna.py` script investigates 4 different steering methods:

### 1. CAA (Contrastive Activation Addition)
- **Description:** Classic vector steering by adding/subtracting activation differences
- **Parameters:** `steering_alpha` (0.05-2.0)
- **Best for:** Simple, interpretable steering with good baseline performance
- **Typical optimal α:** 0.1-0.5 for most tasks

### 2. DAC (Dynamic Activation Control)
- **Description:** Adaptive steering with entropy-based control
- **Parameters:** 
  - `steering_alpha` (0.1-3.0)
  - `entropy_threshold` (0.5-2.5)
  - `ptop` (0.1-0.9)
  - `max_alpha` (1.0-5.0)
- **Best for:** Tasks requiring dynamic adaptation to generation confidence
- **Typical optimal α:** 0.2-1.0, adapts automatically

### 3. BiPO (Bidirectional Preference Optimization)
- **Description:** Preference-based steering considering both positive and negative directions
- **Parameters:**
  - `steering_alpha` (0.05-1.5)
  - `preference_strength` (0.1-2.0)
  - `bidirectional_weight` (0.1-0.9)
- **Best for:** Fine-grained preference learning and alignment
- **Typical optimal α:** 0.1-0.3

### 4. HPR (Harmful Probe Regression)
- **Description:** Regression-based approach using probe confidence for steering strength
- **Parameters:**
  - `steering_alpha` (0.1-2.0)
  - `regression_strength` (0.1-1.0)
  - `threshold_percentile` (0.7-0.9)
- **Best for:** Safety-critical applications with conservative steering
- **Typical optimal α:** 0.2-0.8

## Expected Results

### Llama 3.1 8B Instruct on GSM8K
- **Baseline:** ~65-75% accuracy
- **CAA:** Usually maintains or slightly improves baseline
- **DAC:** Can handle more aggressive steering due to adaptive control
- **BiPO:** Often provides fine-grained improvements with lower variance
- **HPR:** Conservative improvements with high safety margins

### Performance vs Batch Size
| Batch Size | VRAM Usage | Speed | Stability |
|------------|------------|-------|-----------|
| 1          | ~18GB      | Slow  | High      |
| 2          | ~22GB      | Good  | High      |
| 4          | ~28GB      | Fast  | Medium    |
| 8+         | >32GB      | Fastest | Low     |

**Recommendation:** Use batch_size=2 for most setups. Only increase if you have >32GB VRAM and have verified stability.

## Monitoring and Debugging

### SQLite Database
```bash
# View study progress
sqlite3 optuna_studies.db
SELECT study_name, trial_id, value, state FROM trials ORDER BY trial_id DESC LIMIT 10;
```

### WandB Integration
- **Setup:** Run `wandb login` in terminal before using `--use-wandb`
- **Metrics tracked:** validation_accuracy, probe_score, method, layer
- **Hyperparameters:** All steering method parameters
- **System metrics:** GPU usage, memory, training time
- **Visualizations:** Hyperparameter importance, optimization history

### Common Issues

1. **CUDA OOM:** Reduce batch_size or limit dataset sizes
2. **Study not resuming:** Check study_name matches exactly
3. **WandB authentication:** Run `wandb login` first
4. **Slow convergence:** Try different sampler or increase n_trials

## Advanced Usage

### Multi-GPU Setup
```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python llama_optuna.py --batch-size 4

# GPU memory fraction control
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Custom Hyperparameter Ranges
Modify the `_objective_function` in `LlamaSteeringPipeline` to adjust search spaces:

```python
# More conservative CAA range
steering_alpha = trial.suggest_float("steering_alpha", 0.01, 0.5)

# Wider DAC exploration
steering_alpha = trial.suggest_float("steering_alpha", 0.05, 5.0)
```

### Integration with Other Models
The pipeline can be adapted for other models by:
1. Updating `model_name` and `layer_search_range`
2. Adjusting `batch_size` and `max_new_tokens` for the model size
3. Modifying probe types if needed

## Citation

If you use these optimization examples in your research, please cite:

```bibtex
@software{wisent_guard_optimization,
  title={WisentGuard Steering Vector Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/wisent-guard}
}
```