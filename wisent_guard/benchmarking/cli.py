import argparse
import logging
import os
import sys
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent_guard.benchmarking.benchmark_runner import BenchmarkRunner
from wisent_guard.utils.content_utils import load_model, get_device

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hallucination detection benchmark", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ) 
    
    # Model and device
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., meta-llama/Llama-3.1-8B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu/mps)"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark name(s) (comma-separated, e.g., 'truthfulqa,hellaswag')"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Transformer layer to monitor activations"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="benchmark_results"
    )

    parser.add_argument(
        "--shots", 
        type=int, 
        default=0
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    if args.layer < 0:
        raise ValueError("Layer index must be >= 0")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting benchmark with model: {args.model}")
    logger.info(f"Using device: {args.device}")
    logger.info(f"Monitoring layer: {args.layer}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(
        random_seed=42,
        log_level="info"
    )
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        truncation_side="left"
    )
    
    # Split benchmarks string into list
    benchmark_names = [b.strip() for b in args.benchmark.split(',')]
    
    # Set model and tokenizer before running benchmarks
    runner.set_model(model, tokenizer)

    results = {}

    for benchmark_name in benchmark_names:
        logger.info(f"Running benchmark: {benchmark_name}")
    
        result = runner.run_benchmark(
            benchmark_names=[benchmark_name],
            model_name=args.model,
            model=model,
            tokenizer=tokenizer,
            layer=args.layer,
            device=args.device,
            output_dir=args.output_dir
        )

        # Add result to full dict
        results.update(result)

        # Print per-benchmark result
        benchmark_result = result[benchmark_name]
        logger.info(f"\nResults for {benchmark_name}:")
        logger.info(f"Model: {benchmark_result['model']}")
        logger.info(f"Layer: {benchmark_result['layer']}")
        logger.info(f"Accuracy: {benchmark_result['metrics']['accuracy']:.4f}")
        logger.info(f"Precision: {benchmark_result['metrics']['precision']:.4f}")
        logger.info(f"Recall: {benchmark_result['metrics']['recall']:.4f}")
        logger.info(f"F1 Score: {benchmark_result['metrics']['f1']:.4f}")
        logger.info(f"Total samples: {benchmark_result['num_samples']}")

    logger.info("All benchmarks completed!")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
