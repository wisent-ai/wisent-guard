# Wisent-Guard LM-Evaluation-Harness Integration

This module provides integration between wisent-guard and lm-evaluation-harness, enabling automated evaluation of language models with activation-based guards.

## Overview

The `wg_harness` package implements a complete pipeline that:

1. **Loads benchmark tasks** from lm-evaluation-harness
2. **Splits data** into training and testing sets (80/20 by default)
3. **Generates responses** and extracts hidden states from specified layers
4. **Labels responses** as good/bad based on task-specific criteria
5. **Trains guard classifiers** to detect harmful behavior patterns
6. **Evaluates effectiveness** of the trained guards
7. **Saves comprehensive results** in JSON, CSV, and Markdown formats

## Installation

Ensure you have the required dependencies:

```bash
pip install lm-eval torch transformers scikit-learn pandas tqdm
```

## Usage

### Basic Command

```bash
python -m wisent tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B
```

### Multiple Tasks

```bash
python -m wisent tasks hellaswag,mmlu,truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B
```

### With Custom Parameters

```bash
python -m wisent tasks arc_easy \
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
| `--output` | `./wg_harness_results` | Output directory |
| `--classifier-type` | `logistic` | Classifier type (`logistic` or `mlp`) |
| `--max-new-tokens` | `50` | Max tokens for generation |
| `--batch-size` | `8` | Generation batch size |
| `--cache-dir` | `./wg_harness_cache` | Cache directory |
| `--device` | `auto` | Device (`cuda`, `mps`, `cpu`) |
| `--seed` | `42` | Random seed |
| `--verbose` | `False` | Enable verbose logging |
| `--no-cache` | `False` | Disable caching |

## Supported Tasks

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
- Cached generation results in `wg_harness_cache/`

## Example Output

```
Guard evaluation complete:
  Detection Rate: 75.0% (12/16)
  False Positive Rate: 10.0% (2/20)
  F1 Score: 0.8000
  Accuracy: 0.8333
```

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

### Debug Mode

Enable verbose logging for detailed output:

```bash
python -m wisent tasks truthfulqa --verbose
```

## Contributing

To add support for new tasks:

1. Modify `labeler.py` to add task-specific labeling logic
2. Update `evaluate.py` if custom evaluation is needed
3. Test with small dataset using `--limit` parameter

## Architecture

```
wg_harness/
├── __init__.py          # Package exports
├── cli.py              # Command-line interface
├── data.py             # Task loading and data splitting
├── generate.py         # Response generation with hidden states
├── labeler.py          # Good/bad response labeling
├── train_guard.py      # Guard classifier training
├── evaluate.py         # Evaluation and metrics
└── README.md           # This file
``` 