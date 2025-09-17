# LiveCodeBench Benchmark Runner

This Docker-based application loads the LiveCodeBench dataset using Wisent functions and generates answers using the deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B model from HuggingFace.

## Features

- Uses Wisent's `LiveCodeBenchLoader` for dataset loading
- Uses Wisent's `Model` class for HuggingFace model loading and inference  
- GPU support with NVIDIA runtime
- Configurable parameters for release version, problem limit, and generation settings
- Results saved in JSON format with comprehensive metadata

## Requirements

- Docker with NVIDIA Container Toolkit for GPU support
- NVIDIA GPU with CUDA support (optional, will fallback to CPU)

## Usage

### Build the Image

```bash
# Build from the wisent-guard root directory
cd ../../
docker build -f tests/benchmark_runner/Dockerfile -t benchmark-runner .
```

### Run the Benchmark

```bash
# Run the benchmark (without --rm to keep container after exit)
docker run --gpus all --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm benchmark-runner --limit 20 --max-tokens 1024

# After it finishes, find the container ID
docker ps -a

# Copy results from the stopped container (replace CONTAINER_ID with actual ID)
docker cp CONTAINER_ID:/app/results ./results

# Clean up - remove the stopped container
docker rm CONTAINER_ID
```

## Configuration Options

- `--model-name`: HuggingFace model identifier (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- `--release-version`: LiveCodeBench release version (default: release_v1)
- `--limit`: Number of problems to process (default: 10)
- `--max-tokens`: Maximum tokens to generate per problem (default: 512)
- `--output-dir`: Output directory for results (default: results)
- `--device`: Device to use - cuda/cpu/auto (default: auto)

## Available LiveCodeBench Versions

- `release_v1`: 400 problems (May 2023 - Mar 2024)
- `release_v2`: 511 problems (May 2023 - May 2024)
- `release_v3`: 612 problems (May 2023 - Jul 2024)
- `release_v4`: 713 problems (May 2023 - Sep 2024)
- `release_v5`: 880 problems (May 2023 - Jan 2025)
- `release_v6`: 1055 problems (May 2023 - Apr 2025)

## Output

Results are saved in JSON format with the following structure:

```json
{
  "metadata": {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "release_version": "release_v1",
    "total_problems": 10,
    "successful_generations": 8,
    "failed_generations": 2,
    "timestamp": "20250910_143022",
    "device": "cuda"
  },
  "results": [
    {
      "task_id": "lcb_release_v1_12345",
      "prompt": "Please solve the following coding problem:\n\nTitle: Example Problem\n\nProblem Description:\nWrite a function that...\n\nStarter Code:\ndef solution():\n    pass\n\nPlease provide a complete solution:",
      "generated_code": "def solution():\n    # Generated code here\n    pass",
      "generation_time": 2.34,
      "success": true,
      "error": null
    }
  ]
}
```

## Wisent Integration

This runner leverages several Wisent components:

1. **LiveCodeBenchLoader**: Loads real LiveCodeBench data from HuggingFace datasets
2. **Model**: Handles HuggingFace model loading with proper device management and prompt formatting
3. **Automatic prompt formatting**: Uses model-specific chat templates when available

The integration ensures that the benchmark runner follows Wisent's established patterns for data loading and model inference.