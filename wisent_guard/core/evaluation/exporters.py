"""
Model exporters for preparing models for evaluation in different benchmark formats.
"""

import os
import shutil
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import torch

from ..models import SteeringCompatibleModel, create_steering_compatible_model
from ..pipelines import ExperimentRunner

logger = logging.getLogger(__name__)


class ModelExporter:
    """Exports models for evaluation in different benchmark formats."""
    
    def __init__(self, output_dir: str = "./exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelExporter with output directory: {self.output_dir}")
    
    def export_huggingface_model(
        self,
        model_path: str,
        export_name: str,
        base_model_name: str = "distilgpt2",
        include_steering: bool = True
    ) -> str:
        """
        Export a model in HuggingFace format for evaluation.
        
        Args:
            model_path: Path to the trained model or steering vectors
            export_name: Name for the exported model
            base_model_name: Base model name (e.g., "distilgpt2")
            include_steering: Whether to include steering vectors
            
        Returns:
            Path to the exported model directory
        """
        export_path = self.output_dir / export_name
        export_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting model to HuggingFace format: {export_path}")
        
        try:
            # Load model with steering vectors
            if include_steering and (model_path.endswith('.pt') or model_path.endswith('.pth')):
                # Load steering vectors from file
                model = create_steering_compatible_model(base_model_name)
                
                # Load steering vectors if it's a directory
                if os.path.isdir(model_path):
                    model.load_steering_vectors(model_path)
                else:
                    # It's a single steering vector file
                    steering_data = torch.load(model_path, map_location='cpu')
                    if isinstance(steering_data, dict) and 'steering_vectors' in steering_data:
                        # Handle experiment results format
                        for layer_idx, vector in steering_data['steering_vectors'].items():
                            model.add_steering_vector(layer_idx, vector)
                    else:
                        # Handle single vector format
                        # Extract layer index from filename
                        filename = os.path.basename(model_path)
                        parts = filename.split('_')
                        if len(parts) >= 4:
                            layer_idx = int(parts[3])
                            model.add_steering_vector(layer_idx, steering_data)
            
            elif os.path.isdir(model_path):
                # Load from directory (experiment results or HuggingFace format)
                model = create_steering_compatible_model(base_model_name)
                if include_steering:
                    model.load_steering_vectors(model_path)
            
            else:
                # Use base model without steering
                model = create_steering_compatible_model(base_model_name)
            
            # Save model in HuggingFace format
            model.save_pretrained(export_path)
            
            # Save additional metadata
            metadata = {
                "export_info": {
                    "export_name": export_name,
                    "base_model": base_model_name,
                    "source_path": str(model_path),
                    "export_timestamp": datetime.now().isoformat(),
                    "include_steering": include_steering,
                    "steering_info": model.get_steering_info() if include_steering else None
                }
            }
            
            with open(export_path / "export_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully exported model to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise
    
    def export_vllm_model(
        self,
        model_path: str,
        export_name: str,
        base_model_name: str = "distilgpt2"
    ) -> str:
        """
        Export a model in vLLM format for evaluation.
        
        Args:
            model_path: Path to the trained model
            export_name: Name for the exported model
            base_model_name: Base model name
            
        Returns:
            Path to the exported model directory
        """
        # For vLLM, we essentially use the same format as HuggingFace
        # but with specific metadata for vLLM compatibility
        export_path = self.export_huggingface_model(model_path, export_name, base_model_name)
        
        # Add vLLM-specific metadata
        vllm_metadata = {
            "vllm_config": {
                "tensor_parallel_size": 1,
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.9,
                "dtype": "float16"
            }
        }
        
        with open(Path(export_path) / "vllm_metadata.json", "w") as f:
            json.dump(vllm_metadata, f, indent=2)
        
        logger.info(f"Added vLLM metadata to {export_path}")
        return export_path
    
    def create_benchmark_script(
        self,
        benchmark_type: str,
        model_path: str,
        output_dir: str,
        **kwargs
    ) -> str:
        """
        Create a benchmark execution script.
        
        Args:
            benchmark_type: Type of benchmark (livecodebench, bigcode, etc.)
            model_path: Path to the exported model
            output_dir: Output directory for results
            **kwargs: Additional benchmark-specific parameters
            
        Returns:
            Path to the generated script
        """
        script_path = self.output_dir / f"run_{benchmark_type}.sh"
        
        if benchmark_type == "livecodebench":
            script_content = self._create_livecodebench_script(model_path, output_dir, **kwargs)
        elif benchmark_type == "bigcode":
            script_content = self._create_bigcode_script(model_path, output_dir, **kwargs)
        elif benchmark_type == "humaneval":
            script_content = self._create_humaneval_script(model_path, output_dir, **kwargs)
        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")
        
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created benchmark script: {script_path}")
        return str(script_path)
    
    def _create_livecodebench_script(
        self,
        model_path: str,
        output_dir: str,
        **kwargs
    ) -> str:
        """Create LiveCodeBench evaluation script."""
        version = kwargs.get("dataset_version", "v2")
        temperature = kwargs.get("temperature", 0.2)
        max_length = kwargs.get("max_length", 512)
        n_samples = kwargs.get("n_samples", 1)
        problem_limit = kwargs.get("problem_limit", "")
        
        limit_flag = f"--limit {problem_limit}" if problem_limit else ""
        
        return f"""#!/bin/bash
# LiveCodeBench Evaluation Script
# Generated by Wisent-Guard ModelExporter

set -e

echo "Starting LiveCodeBench evaluation..."
echo "Model: {model_path}"
echo "Output: {output_dir}"
echo "Version: {version}"

# Check if LiveCodeBench is installed
if ! command -v python -c "import lcb_runner" &> /dev/null; then
    echo "LiveCodeBench not found. Please install it first:"
    echo "git clone https://github.com/LiveCodeBench/LiveCodeBench.git"
    echo "cd LiveCodeBench && pip install -e ."
    exit 1
fi

# Create output directory
mkdir -p {output_dir}

# Run evaluation
python -m lcb_runner.runner.main \\
    --model {model_path} \\
    --scenario code_generation \\
    --release_version {version} \\
    --temperature {temperature} \\
    --max_length_generation {max_length} \\
    --n_samples {n_samples} \\
    --output_dir {output_dir} \\
    --use_cache \\
    --continue_existing \\
    {limit_flag} \\
    --evaluate

echo "LiveCodeBench evaluation completed!"
echo "Results saved to: {output_dir}"
"""
    
    def _create_bigcode_script(
        self,
        model_path: str,
        output_dir: str,
        **kwargs
    ) -> str:
        """Create BigCode evaluation script."""
        tasks = kwargs.get("tasks", "humaneval")
        temperature = kwargs.get("temperature", 0.2)
        max_length = kwargs.get("max_length", 512)
        n_samples = kwargs.get("n_samples", 100)
        batch_size = kwargs.get("batch_size", 10)
        problem_limit = kwargs.get("problem_limit", "")
        
        limit_flag = f"--limit {problem_limit}" if problem_limit else ""
        
        return f"""#!/bin/bash
# BigCode Evaluation Script
# Generated by Wisent-Guard ModelExporter

set -e

echo "Starting BigCode evaluation..."
echo "Model: {model_path}"
echo "Output: {output_dir}"
echo "Tasks: {tasks}"

# Check if bigcode-evaluation-harness is installed
if ! command -v python -c "import bigcode_eval" &> /dev/null; then
    echo "BigCode evaluation harness not found. Please install it first:"
    echo "git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git"
    echo "cd bigcode-evaluation-harness && pip install -e ."
    exit 1
fi

# Create output directory
mkdir -p {output_dir}

# Run evaluation
accelerate launch main.py \\
    --model {model_path} \\
    --tasks {tasks} \\
    --max_length_generation {max_length} \\
    --temperature {temperature} \\
    --do_sample True \\
    --n_samples {n_samples} \\
    --batch_size {batch_size} \\
    --allow_code_execution \\
    --save_generations \\
    --save_generations_path {output_dir}/generations.json \\
    {limit_flag}

echo "BigCode evaluation completed!"
echo "Results saved to: {output_dir}"
"""
    
    def _create_humaneval_script(
        self,
        model_path: str,
        output_dir: str,
        **kwargs
    ) -> str:
        """Create HumanEval evaluation script."""
        temperature = kwargs.get("temperature", 0.2)
        max_length = kwargs.get("max_length", 512)
        n_samples = kwargs.get("n_samples", 1)
        
        return f"""#!/bin/bash
# HumanEval Evaluation Script
# Generated by Wisent-Guard ModelExporter

set -e

echo "Starting HumanEval evaluation..."
echo "Model: {model_path}"
echo "Output: {output_dir}"

# Check if human-eval is installed
if ! command -v python -c "import human_eval" &> /dev/null; then
    echo "HumanEval not found. Please install it first:"
    echo "pip install human-eval"
    exit 1
fi

# Create output directory
mkdir -p {output_dir}

# Run evaluation (using the common evaluation pattern)
python -c "
import json
from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained('{model_path}')
model = AutoModelForCausalLM.from_pretrained('{model_path}')

# Load problems
problems = read_problems()

# Generate completions
completions = []
for task_id, problem in problems.items():
    prompt = problem['prompt']
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + {max_length},
            temperature={temperature},
            do_sample=True,
            num_return_sequences={n_samples},
            pad_token_id=tokenizer.eos_token_id
        )
    
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = completion[len(prompt):]
    
    completions.append({{'task_id': task_id, 'completion': completion}})

# Save completions
write_jsonl('{output_dir}/completions.jsonl', completions)

# Run evaluation
import subprocess
result = subprocess.run([
    'evaluate_functional_correctness',
    '{output_dir}/completions.jsonl'
], capture_output=True, text=True)

print(result.stdout)
with open('{output_dir}/evaluation_results.txt', 'w') as f:
    f.write(result.stdout)
"

echo "HumanEval evaluation completed!"
echo "Results saved to: {output_dir}"
"""
    
    def get_export_info(self, export_path: str) -> Dict[str, Any]:
        """Get information about an exported model."""
        export_path = Path(export_path)
        
        info = {
            "export_path": str(export_path),
            "exists": export_path.exists(),
            "files": []
        }
        
        if export_path.exists():
            info["files"] = [f.name for f in export_path.iterdir()]
            
            # Load metadata if available
            metadata_path = export_path / "export_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    info["metadata"] = json.load(f)
        
        return info
    
    def list_exports(self) -> List[Dict[str, Any]]:
        """List all exported models."""
        exports = []
        
        if not self.output_dir.exists():
            return exports
        
        for export_dir in self.output_dir.iterdir():
            if export_dir.is_dir():
                exports.append(self.get_export_info(str(export_dir)))
        
        return exports