# WisentGuard Classifier Evaluation

A simplified evaluation pipeline for training and evaluating WisentGuard classifiers on real-world code data.

## Overview

This evaluation system focuses on classifier performance rather than steering vectors. It uses ground truth data from LiveCodeBench submissions to train and evaluate classifiers for code quality detection.

## Key Features

- **Ground Truth Data**: Uses real pass/fail data from LiveCodeBench submissions
- **Cross-Version Evaluation**: Trains on one version, evaluates on another
- **Simple Features**: Uses code complexity metrics instead of neural activations
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Fallback Handling**: Synthetic data when real data is insufficient

## Architecture

The system implements Option 1 approach:
1. Load LiveCodeBench submissions with pass@1 scores
2. Extract positive (high pass rate) and negative (low pass rate) code examples
3. Train ActivationClassifier using simple code features
4. Evaluate cross-version generalization (v1 → v2)
5. Report comprehensive performance metrics

## Usage

### Basic Usage

```bash
python classifier_evaluation/train_and_evaluate_classifier.py classifier_evaluation/configs/toy_example_gpt2.yaml
```

### With Verbose Logging

```bash
python classifier_evaluation/train_and_evaluate_classifier.py classifier_evaluation/configs/toy_example_gpt2.yaml --verbose
```

### With Overrides

```bash
python classifier_evaluation/train_and_evaluate_classifier.py classifier_evaluation/configs/toy_example_gpt2.yaml --override train_data.limit=50 --override model.device=cpu
```

## Configuration

### Example Configuration

```yaml
model:
  name: "gpt2"
  layers: [6]
  device: "auto"

train_data:
  version: "release_v1"
  difficulty: ["easy"]
  limit: 20
  min_pass_rate: 0.5  # Minimum pass rate for "correct" code
  max_pass_rate: 0.4  # Maximum pass rate for "incorrect" code

test_data:
  version: "release_v2"
  difficulty: ["easy"]
  limit: 10
  min_pass_rate: 0.5
  max_pass_rate: 0.4

classifier:
  classifier_type: "logistic"
  classifier_threshold: 0.5
  max_pairs: 5

output:
  results_dir: "demo_results"
  classifier_path: "models/gpt2_toy_classifier.pkl"

logging:
  level: "INFO"
  save_log: true
  log_file: "logs/toy_example_gpt2.log"
```

## Features

### Code Feature Extraction

The system extracts simple features from code text:
- Text length
- Number of lines
- Number of function definitions
- Number of if statements
- Number of loops (for, while)
- Number of imports
- Number of returns

### Cross-Version Evaluation

- **Training**: Uses `release_v1` data
- **Testing**: Uses `release_v2` data
- **Generalization**: Measures how well classifiers trained on one version perform on another

### Fallback Mechanisms

- **Synthetic Data**: When real data is insufficient
- **Error Handling**: Graceful degradation when training fails
- **Default Values**: Sensible defaults for all parameters

## Results

The system provides comprehensive evaluation results:

```
EVALUATION RESULTS SUMMARY
============================================================
Training Version: release_v1
Test Version: release_v2
Layer: 6
Training Examples: 8
Test Examples: 8

Performance Metrics:
  Accuracy:  0.5000
  Precision: 0.5000
  Recall:    1.0000
  F1 Score:  0.6667

Confusion Matrix:
  True Negatives:  0
  False Positives: 4
  False Negatives: 0
  True Positives:  4

Performance Interpretation:
  🔴 Poor generalization, significant version-specific problems
============================================================
```

## Limitations

- **Simple Features**: Uses basic code metrics instead of neural activations
- **Data Availability**: LiveCodeBench may not have enough failing submissions
- **Synthetic Fallback**: May not represent real-world code complexity
- **Training Pipeline**: Uses direct ActivationClassifier instead of full WisentGuard pipeline

## Future Enhancements

- **Neural Activations**: Replace simple features with actual model activations
- **Better Data**: Find datasets with more diverse pass/fail ratios
- **Advanced Features**: Add AST-based features, code metrics, etc.
- **Full Integration**: Connect to complete WisentGuard training pipeline

## Troubleshooting

### Common Issues

1. **No negative examples found**: Lower `max_pass_rate` or use synthetic fallback
2. **Import errors**: Ensure wisent-guard is installed and accessible
3. **Device issues**: Use `--override model.device=cpu` for CPU-only execution
4. **Memory issues**: Reduce `train_data.limit` and `test_data.limit`

### Data Issues

LiveCodeBench submissions may have high pass rates, making it difficult to find negative examples. The system automatically falls back to synthetic data when needed.

## Architecture Notes

This implementation bypasses the incomplete WisentGuard training pipeline and uses ActivationClassifier directly. The key insight is that `WisentGuard.train_from_pairs()` expects a classifier but `steering_method.classifier` is None until training is called, creating a circular dependency.

The solution uses ActivationClassifier directly with simple code features as a proof-of-concept for the evaluation pipeline.