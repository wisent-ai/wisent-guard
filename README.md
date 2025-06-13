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

## LM-Harness Integration

Wisent-Guard now supports standardized benchmarking through [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Train guardrails on established benchmarks and get objective performance metrics.

The integrated CLI implements a complete pipeline that:

1. **Loads benchmark tasks** from lm-evaluation-harness
2. **Splits data** into training and testing sets (80/20 by default)
3. **Generates responses** and extracts hidden states from specified layers
4. **Labels responses** as good/bad based on task-specific criteria
5. **Trains guard classifiers** to detect harmful behavior patterns
6. **Evaluates effectiveness** of the trained guards
7. **Saves comprehensive results** in JSON, CSV, and Markdown formats

## Installation

```bash
# Install with benchmark support
pip install wisent-guard[harness]

# Or with required dependencies
pip install lm-eval torch transformers scikit-learn pandas tqdm
```

## Quick Start

### Basic Usage

```bash
# Test on TruthfulQA (hallucination detection)
python -m wisent_guard tasks truthfulqa --model gpt2 --layer 6 --limit 50

# Multiple benchmarks
python -m wisent_guard tasks hellaswag,truthfulqa --model gpt2 --layer 6 --limit 50
```

### Advanced Usage

```bash
# Basic Command
python -m wisent_guard tasks truthfulqa --model meta-llama/Llama-3.1-8B --layer 15 --limit 3

# Multiple Tasks
python -m wisent_guard tasks hellaswag,mmlu,truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B

# With Custom Parameters
python -m wisent_guard tasks arc_easy \
    --layer 10 \
    --model gpt2 \
    --shots 5 \
    --limit 100 \
    --output ./my_results \
    --classifier-type mlp \
    --max-new-tokens 30
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Model name or path |
| `--layer` | `15` | Layer to extract activations from |
| `--shots` | `0` | Number of few-shot examples |
| `--split-ratio` | `0.8` | Train/test split ratio |
| `--limit` | `None` | Limit documents per task |
| `--output` | `./results` | Output directory |
| `--classifier-type` | `logistic` | Classifier type (`logistic` or `mlp`) |
| `--max-new-tokens` | `50` | Max tokens for generation |
| `--batch-size` | `8` | Generation batch size |
| `--cache-dir` | `./cache` | Cache directory |
| `--device` | `auto` | Device (`cuda`, `mps`, `cpu`) |
| `--seed` | `42` | Random seed |
| `--verbose` | `False` | Enable verbose logging |
| `--no-cache` | `False` | Disable caching |

## Supported Benchmarks

**Supported benchmarks:**
- `truthfulqa` - Detect hallucinations vs truthful responses (90% detection rate)
- `hellaswag` - Common sense reasoning errors (100% detection rate)  
- `arc_easy/challenge` - Science reasoning mistakes
- `winogrande` - Language understanding errors
- `piqa` - Physical reasoning mistakes
- `mmlu` - Massive multitask language understanding

The pipeline supports any lm-evaluation-harness task, with specialized handling for:

- **TruthfulQA**: Detects hallucinations vs. truthful responses
- **HellaSwag**: Uses multiple-choice correct vs. incorrect options
- **MMLU**: Uses multiple-choice correct vs. incorrect options
- **ARC**: Uses multiple-choice correct vs. incorrect options
- **Generic tasks**: Falls back to evaluator-based labeling

## Output Files

The pipeline generates several output files:

- `{task_name}_results.json`: Detailed results per task
- `guard_evaluation_summary.csv`: Aggregated metrics across tasks
- `guard_evaluation_report_{timestamp}.md`: Human-readable report
- Cached generation results in `cache/`

## Example Output

```
Guard evaluation complete:
  Detection Rate: 75.0% (12/16)
  False Positive Rate: 10.0% (2/20)
  F1 Score: 0.8000
  Accuracy: 0.8333
```

Results are saved as JSON, CSV, and Markdown reports with detailed metrics.

## Pipeline Steps

### 1. Data Loading and Splitting
- Load task from lm-evaluation-harness
- Split documents into train/test sets

### 2. Response Generation
- Generate responses for training prompts
- Extract hidden states from specified layer
- Cache results for efficiency

### 3. Response Labeling
- **TruthfulQA**: Good = truthful, Bad = hallucinated
- **Multiple Choice**: Good = correct option, Bad = random incorrect
- **Generic**: Use task evaluator to determine correctness

### 4. Guard Training
- Extract activations from good/bad response pairs
- Train binary classifier (logistic regression or MLP)
- Validate on held-out data

### 5. Testing and Evaluation
- Generate responses for test set
- Classify using trained guard
- Compare against ground truth labels
- Calculate detection rates and false positives

### 6. Results Saving
- Individual task results (JSON)
- Aggregated summary (CSV)
- Human-readable report (Markdown)

## Task-Specific Behavior

### TruthfulQA
- **Good responses**: Reference truthful answers
- **Bad responses**: Model hallucinations or fabricated facts
- **Detection target**: Factual inaccuracies and false claims

### Multiple Choice Tasks (HellaSwag, MMLU, ARC)
- **Good responses**: Correct answer choices
- **Bad responses**: Randomly selected incorrect choices
- **Detection target**: Wrong reasoning patterns

### Custom Tasks
- **Fallback**: Generate multiple responses with different temperatures
- **Filter**: Use task evaluator to identify incorrect responses
- **Adapt**: Flexible labeling based on task structure

## Performance Considerations

- **Memory**: Large models require significant GPU memory
- **Caching**: Responses are cached to avoid re-generation
- **Batching**: Adjust `--batch-size` based on available memory
- **Limiting**: Use `--limit` for quick testing

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch-size` or use smaller model
2. **Task not found**: Check task name with `lm_eval --list_tasks`
3. **Import errors**: Ensure `lm_eval` is installed correctly
4. **Cache issues**: Use `--no-cache` to disable caching
5. **MPS mixed precision errors** (Apple Silicon): If you encounter MPS tensor type errors, force CPU usage with `--device cpu`

### Debug Mode

Enable verbose logging for detailed output:

```bash
python -m wisent_guard tasks truthfulqa --verbose

# For MPS issues on Apple Silicon:
python -m wisent_guard tasks truthfulqa --device cpu --verbose
```

## Contributing

To add support for new tasks:

1. Modify `labeler.py` to add task-specific labeling logic
2. Update `evaluate.py` if custom evaluation is needed
3. Test with small dataset using `--limit` parameter

Contributions are welcome! Please feel free to submit a Pull Request or open an issue so that we can fix all bugs potentially troubling you.

## Architecture

```
results/
├── __init__.py          # Package exports
├── cli.py              # Command-line interface
├── data.py             # Task loading and data splitting
├── generate.py         # Response generation with hidden states
├── labeler.py          # Good/bad response labeling
├── train_guard.py      # Guard classifier training
├── evaluate.py         # Evaluation and metrics
└── README.md           # Module documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
