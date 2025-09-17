# Nosense Testing for Wisent-Guard

Simple framework for testing evaluation robustness by comparing performance on real vs nonsensical benchmark data.

## Overview

This framework tests if your evaluation pipeline can distinguish between real and nonsensical data. If nonsense data gets high accuracy, it indicates problems with the evaluation system.

## Files

- **`base_nosense.py`**: Base class for creating nonsense generators
- **`math500_nosense.py`**: Math500-specific nonsense generator
- **`test_robustness.py`**: Main test script that compares real vs nonsense performance
- **`README.md`**: This file

## How It Works

1. **Load real data** using existing wisent-guard task loaders (e.g., `Math500Task`)
2. **Generate nonsense data** by randomly changing words but keeping numerical answers
3. **Run evaluation** on both real and nonsense data using standard wisent-guard CLI
4. **Compare results** to detect evaluation issues

## Quick Start

### Test Math500 Robustness

```bash
cd tests/nosense

# Basic test
python test_robustness.py

# With custom model and more samples
python test_robustness.py --model "meta-llama/Llama-3.1-8B-Instruct" --limit 20

# Verbose output
python test_robustness.py --verbose
```

### Test Nonsense Generation Only

```bash
# Test the nonsense generator
python math500_nosense.py
```

## Expected Results

A robust system should show:
- **Nonsense accuracy**: < 10% (ideally < 5%)
- **Real accuracy**: > 30% (depends on task difficulty)
- **Difference**: > 30% gap between real and nonsense

### Red Flags 🚨

- Nonsense accuracy > 20% → Evaluation is likely broken
- Real vs nonsense difference < 30% → System not learning meaningful patterns
- Both accuracies very low → Model or pipeline issues

## Example Output

```
🧪 MATH500 ROBUSTNESS TEST
==================================================
Model: EleutherAI/gpt-neo-1.3B
Sample limit: 10
==================================================

📊 Testing with REAL data...
Real data accuracy: 45.0%

🎭 Generating NONSENSE data...
Generated 10 nonsense items

🎭 Testing with NONSENSE data...
Nonsense data accuracy: 5.0%

📈 COMPARISON RESULTS
==================================================
Real data accuracy:     45.0%
Nonsense data accuracy: 5.0%
Difference:             40.0%

🔍 ROBUSTNESS ANALYSIS
------------------------------
✅ GOOD: Nonsense data has low accuracy (<10%)
✅ GOOD: Large difference between real and nonsense (≥30%)

🎉 ROBUSTNESS TEST: PASSED
   The evaluation system appears robust
```

## How Nonsense is Generated

For Math500, the generator:

1. **Loads original data** using `Math500Task.load_data()`
2. **Randomizes words** in problem text: `"solve this equation"` → `"blfmk wxyz qrstuvw"`
3. **Changes numerical answer**: Original answer `42` → Random answer `187`
4. **Ensures answer is extractable**: Adds `"The answer is 187"` to problem text
5. **Keeps data format**: All fields preserved, only content changed

## Adding New Task Types

To add support for other benchmark types:

1. **Create new generator** (e.g., `aime_nosense.py`):
```python
from base_nosense import BaseNosenseGenerator
from wisent_guard.core.tasks.aime_task import AIMETask

class AIMENosenseGenerator(BaseNosenseGenerator):
    def __init__(self):
        original_task = AIMETask()
        super().__init__(original_task)

    def make_nonsense(self, data):
        # Implement AIME-specific nonsense logic
        pass
```

2. **Update test script** to support the new task type
3. **Add specific handling** for the task's data format

## Architecture

```
BaseNosenseGenerator (abstract)
├── generate_random_words()     # Word randomization
├── extract_number()            # Number extraction
├── ensure_number_in_text()     # Answer preservation
└── make_nonsense()            # Abstract method

Math500NosenseGenerator
└── make_nonsense()            # Math500-specific implementation
```

## Integration with CI/CD

Add to your testing pipeline:

```bash
# Run robustness test
cd tests/nosense
python test_robustness.py --limit 20

# Check exit code (0 = passed, 1 = issues detected)
if [ $? -ne 0 ]; then
  echo "Robustness test failed - evaluation issues detected"
  exit 1
fi
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the right directory and Python path is correct
2. **Model not found**: Check model name and availability
3. **Low accuracy on real data**: May indicate model or task issues, not necessarily robustness problems

### Debug Mode

Use `--verbose` to see detailed output including:
- Full wisent-guard command output
- Sample nonsense data
- Detailed accuracy extraction

## Future Extensions

The framework is designed to easily add:
- **More task types**: AIME, LM-eval tasks, etc.
- **Different nonsense strategies**: Pure random, adversarial, etc.
- **Additional metrics**: Beyond just accuracy
- **Batch testing**: Test multiple benchmarks at once

## Philosophy

This framework follows the principle: **"If your evaluation system can't tell the difference between a real math problem and random gibberish, it's not really evaluating mathematical reasoning."**

The goal is to catch evaluation bugs before they lead to overconfident conclusions about model capabilities.