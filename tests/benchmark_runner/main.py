#!/usr/bin/env python3
"""
LiveCodeBench benchmark runner using Wisent functions.
Loads LiveCodeBench dataset and Llama-3.2-1B model to generate answers.
"""

import json
import time
from typing import Dict, Any
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

# Import Wisent functions
from wisent_guard.core.data_loaders.livecodebench_loader import LiveCodeBenchLoader
from wisent_guard.core.model import Model

# Check and print datasets version
import datasets
print(f"🔍 Using datasets version: {datasets.__version__}")


def setup_output_directory(output_dir: str = "results") -> Path:
    """Create and setup output directory for results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return output_path


def generate_code_completion(model: Model, problem: Dict[str, Any], max_tokens: int = 512) -> Dict[str, Any]:
    """
    Generate code completion for a given problem using the model.
    
    Args:
        model: Wisent Model instance
        problem: LiveCodeBench problem dictionary
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with generated code and metadata
    """
    # Format the prompt for code generation
    prompt = f"""Please solve the following coding problem:

Title: {problem['question_title']}

Problem Description:
{problem['question_content']}

Starter Code:
{problem['starter_code']}

Please provide a complete solution:"""

    try:
        # Generate response using Wisent model
        start_time = time.time()
        response = model.generate(
            prompt=prompt,
            layer_index=15,  # Use middle layer
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        generation_time = time.time() - start_time
        
        # Extract generated text
        if isinstance(response, tuple):
            generated_text, _ = response
        else:
            generated_text = response
            
        return {
            "task_id": problem["task_id"],
            "prompt": prompt,
            "generated_code": generated_text.strip(),
            "generation_time": generation_time,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "prompt": prompt,
            "generated_code": "",
            "generation_time": 0,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="LiveCodeBench benchmark runner with DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                       help="HuggingFace model name")
    parser.add_argument("--release-version", default="release_v1", 
                       help="LiveCodeBench release version")
    parser.add_argument("--limit", type=int, default=10, 
                       help="Limit number of problems to process")
    parser.add_argument("--max-tokens", type=int, default=512, 
                       help="Maximum tokens to generate per problem")
    parser.add_argument("--output-dir", default="results", 
                       help="Output directory for results")
    parser.add_argument("--device", default=None, 
                       help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    print("🚀 Starting LiveCodeBench benchmark runner")
    print(f"Model: {args.model_name}")
    print(f"Release Version: {args.release_version}")
    print(f"Limit: {args.limit}")
    print(f"Max Tokens: {args.max_tokens}")
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available. GPU: {torch.cuda.get_device_name()}")
        device = args.device or "cuda"
    else:
        print("⚠️ CUDA not available, using CPU")
        device = args.device or "cpu"
    
    print(f"Using device: {device}")
    
    try:
        # Load LiveCodeBench dataset using Wisent loader
        print("\n📊 Loading LiveCodeBench dataset...")
        loader = LiveCodeBenchLoader()
        problems = loader.load_problems(
            release_version=args.release_version,
            limit=args.limit
        )
        print(f"✅ Loaded {len(problems)} problems from {args.release_version}")
        
        # Load model using Wisent Model class
        print(f"\n🤖 Loading model: {args.model_name}")
        model = Model(name=args.model_name, device=device)
        print("✅ Model loaded successfully")
        
        # Generate answers for each problem
        print(f"\n🔄 Generating answers for {len(problems)} problems...")
        results = []
        
        for problem in tqdm(problems, desc="Generating solutions"):
            problem_dict = problem.to_dict()
            result = generate_code_completion(model, problem_dict, args.max_tokens)
            results.append(result)
            
            # Print progress
            if result["success"]:
                print(f"✅ {result['task_id']}: Generated in {result['generation_time']:.2f}s")
            else:
                print(f"❌ {result['task_id']}: Failed - {result['error']}")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"livecodebench_{args.release_version}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "metadata": {
                    "model_name": args.model_name,
                    "release_version": args.release_version,
                    "total_problems": len(problems),
                    "successful_generations": sum(1 for r in results if r["success"]),
                    "failed_generations": sum(1 for r in results if not r["success"]),
                    "timestamp": timestamp,
                    "device": device
                },
                "results": results
            }, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        total_time = sum(r["generation_time"] for r in results if r["success"])
        
        print(f"\n📊 Summary:")
        print(f"Total problems: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        if successful > 0:
            print(f"Average generation time: {total_time/successful:.2f}s")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())