# Control Vector Generation from Seed Pairs

This directory contains a complete pipeline for generating control vectors from JSON seed pairs and serializing them as plain activations.

## Directory Structure

```
tests/controls_from_seed_pairs/
├── seed_pairs/                    # 663 JSON files with contrastive pairs
├── control_vectors/               # Generated control vector JSON files (output)
├── json_loader.py                 # Loads JSON files into ContrastivePairSet
├── control_vector_generator.py    # Generates control vectors and serializes to JSON
├── generate_all_control_vectors.py # Main script to process all files
└── README.md                      # This file
```

## Pipeline Overview

1. **JSON Input**: Each JSON file in `seed_pairs/` contains contrastive pairs like:
   ```json
   {
     "pairs": [
       {
         "question": "How do you respond when someone is struggling?",
         "positive": "Let me help you with that.",
         "negative": "Figure it out yourself."
       }
     ]
   }
   ```
   Change it to:
   ```json
   {
     "pairs": [
       {
         "question": "How do you respond when someone is struggling?",
         "choice_a": "Let me help you with that.",
         "choice_b": "Figure it out yourself."
       }
     ]
   }
   This format is expected by ContrastivePairSet.create_multiple_choice_questions().

2. **Control Vector Generation**:
   - Load JSON → Format data → ContrastivePairSet
   - Extract activations using model
   - Train CAA steering method
   - Generate control vector

3. **JSON Serialization**: Each control vector is saved as JSON with plain activations:
   ```json
   {
     "trait_name": "helpful",
     "steering_vector": [0.123, -0.456, ...],  // Plain float array
     "vector_shape": [2560],
     "vector_norm": 8.45,
     "method": "CAA",
     "model_name": "unsloth/Qwen3-4B-bnb-4bit",
     "layer_index": 17
   }
   ```

## Usage

### Process All Files
```bash
cd /home/bc/Desktop/Documents/wisent-guard/tests/controls_from_seed_pairs

# Process all 663 JSON files (will take time!)
python generate_all_control_vectors.py

# Process with specific model and layer
python generate_all_control_vectors.py --model microsoft/DialoGPT-medium --layer 8

# Test with limited files
python generate_all_control_vectors.py --limit 10

# Resume from specific trait (if interrupted)
python generate_all_control_vectors.py --resume-from helpful
```

### Process Single File
```bash
# Test the pipeline with one file
python control_vector_generator.py
```

### Load and Test JSON
```bash
# Test JSON loading
python json_loader.py
```

## Expected Output

After running `generate_all_control_vectors.py`, you will have:

- `control_vectors/` directory with 663 JSON files
- Each JSON file named like `{trait}_control_vector.json`
- `generation_summary.json` with processing statistics

## Configuration Options

- `--model`: HuggingFace model name (default: unsloth/Qwen3-4B-bnb-4bit)
- `--layer`: Layer index for activation extraction (default: 17)
- `--device`: Device to use (auto/cuda/cpu, default: auto)
- `--limit`: Limit number of files for testing
- `--resume-from`: Resume processing from specific trait name
- `--verbose`: Enable detailed logging

## Requirements

The pipeline uses the existing wisent-guard infrastructure:
- `wisent_guard.core.Model` for activation extraction
- `wisent_guard.core.steering_methods.CAA` for control vector generation
- `wisent_guard.core.contrastive_pairs` for data handling

## Performance Notes

- Processing all 663 files will take considerable time
- Each file requires model inference for activation extraction
- Consider using `--limit` for testing before full run
- Progress is logged and can be resumed if interrupted