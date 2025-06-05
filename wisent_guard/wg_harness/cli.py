"""
Command-line interface for wisent-guard lm-evaluation-harness integration.
"""

import argparse
import logging
import sys
import os
from typing import List, Dict, Any
from tqdm import tqdm

from .data import load_task, split_data, prepare_prompts_from_docs, get_reference_answers
from .generate import generate_responses
from .labeler import label_responses, validate_labels
from .train_guard import GuardPipeline
from .evaluate import (
    evaluate_guard, evaluate_lm_harness_task, save_results_json, 
    save_results_csv, create_evaluation_report, calculate_aggregate_metrics
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation benchmarks through wisent-guard pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m wisent_guard tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks hellaswag,mmlu --layer 10 --model meta-llama/Llama-3.1-8B --shots 5
  python -m wisent_guard tasks arc_easy --layer 20 --model gpt2 --limit 100 --output ./results
        """
    )
    
    parser.add_argument(
        "command",
        choices=["tasks"],
        help="Command to run (currently only 'tasks' is supported)"
    )
    
    parser.add_argument(
        "task_names",
        help="Comma-separated list of task names (e.g., 'hellaswag,mmlu,truthfulqa')"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Layer to extract activations from (default: 15)"
    )
    
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)"
    )
    
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents per task (default: no limit)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./wg_harness_results",
        help="Output directory for results (default: ./wg_harness_results)"
    )
    
    parser.add_argument(
        "--classifier-type",
        type=str,
        choices=["logistic", "mlp"],
        default="logistic",
        help="Type of classifier to train (default: logistic)"
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum new tokens for generation (default: 50)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./wg_harness_cache",
        help="Directory for caching generation results (default: ./wg_harness_cache)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: auto-detect)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable generation caching"
    )
    
    return parser


def run_single_task(
    task_name: str,
    model_name: str,
    layer: int,
    shots: int = 0,
    split_ratio: float = 0.8,
    limit: int = None,
    classifier_type: str = "logistic",
    max_new_tokens: int = 50,
    batch_size: int = 8,
    cache_dir: str = "./wg_harness_cache",
    device: str = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run the complete pipeline for a single task.
    
    Args:
        task_name: Name of the benchmark task
        model_name: Language model name
        layer: Layer for activation extraction
        shots: Number of few-shot examples
        split_ratio: Train/test split ratio
        limit: Optional limit on documents
        classifier_type: Type of classifier
        max_new_tokens: Max tokens for generation
        batch_size: Generation batch size
        cache_dir: Cache directory
        device: Target device
        seed: Random seed
        
    Returns:
        Dictionary with all results
    """
    logger.info(f"Starting pipeline for task: {task_name}")
    
    # Step 1: Load task and split data
    logger.info("Step 1: Loading task and splitting data")
    task_data = load_task(task_name, shots=shots, limit=limit)
    train_docs, test_docs = split_data(task_data, split_ratio=split_ratio, random_seed=seed)
    
    logger.info(f"Split: {len(train_docs)} train, {len(test_docs)} test documents")
    
    # Step 2: Generate responses for training set
    logger.info("Step 2: Generating responses for training set")
    train_prompts = prepare_prompts_from_docs(task_data, train_docs)
    train_references = get_reference_answers(task_data, train_docs)
    
    cache_dir_to_use = cache_dir if not cache_dir.endswith("no-cache") else None
    
    train_responses, train_hidden_states = generate_responses(
        model_name=model_name,
        prompts=train_prompts,
        layer=layer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        cache_dir=cache_dir_to_use,
        device=device
    )
    
    # Step 3: Label responses as good/bad pairs
    logger.info("Step 3: Labeling responses for training")
    labeled_pairs = label_responses(
        task_data,
        docs=train_docs,
        responses=train_responses,
        references=train_references,
        model_name=model_name,
        layer=layer
    )
    
    # Validate labels
    valid_pairs = validate_labels(labeled_pairs)
    logger.info(f"Using {len(valid_pairs)} valid training pairs")
    
    if len(valid_pairs) == 0:
        logger.error("No valid training pairs found!")
        return {"error": "No valid training pairs", "task_name": task_name}
    
    # Step 4: Train guard classifier
    logger.info("Step 4: Training guard classifier")
    guard_pipeline = GuardPipeline(
        model_name=model_name,
        layer=layer,
        device=device,
        classifier_type=classifier_type
    )
    
    # Convert labeled pairs to training triples (prompt, good, bad)
    train_triples = []
    for i, (good, bad) in enumerate(valid_pairs):
        # Use the corresponding prompt, or a generic one
        prompt = train_prompts[i] if i < len(train_prompts) else "Question: Please respond."
        train_triples.append((prompt, good, bad))
    
    # Check if we have enough training data
    min_required_examples = 4  # Need at least 4 examples for proper train/validation split
    if len(train_triples) < min_required_examples:
        logger.warning(f"Only {len(train_triples)} training examples available, minimum {min_required_examples} required for stable training")
        logger.warning("Skipping classifier training and using random baseline")
        
        # Create a dummy result for insufficient data
        result = {
            'task': task_name,
            'model': model_name,
            'layer': layer,
            'train_samples': len(train_triples),
            'test_samples': 0,
            'detection_rate': 0.0,
            'false_positive_rate': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'status': 'insufficient_data',
            'error': f'Need at least {min_required_examples} training examples, got {len(train_triples)}'
        }
        return result
    
    logger.info(f"Using {len(train_triples)} valid training pairs")
    
    guard_pipeline.fit(train_triples)
    
    # Step 5: Generate responses for test set
    logger.info("Step 5: Generating responses for test set")
    test_prompts = prepare_prompts_from_docs(task_data, test_docs)
    test_references = get_reference_answers(task_data, test_docs)
    
    test_responses, test_hidden_states = generate_responses(
        model_name=model_name,
        prompts=test_prompts,
        layer=layer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        cache_dir=cache_dir_to_use,
        device=device
    )
    
    # Step 6: Label test responses for evaluation
    logger.info("Step 6: Labeling test responses")
    test_labeled_pairs = label_responses(
        task_data,
        docs=test_docs,
        responses=test_responses,
        references=test_references,
        model_name=model_name,
        layer=layer
    )
    
    # Create test labels (0=good, 1=bad) and corresponding hidden states
    test_labels = []
    test_eval_hidden_states = []
    
    for i, (good, bad) in enumerate(test_labeled_pairs):
        if i < len(test_hidden_states):
            # We have the hidden state for the generated response
            # Determine if the generated response was good or bad
            generated_response = test_responses[i] if i < len(test_responses) else ""
            
            # If generated response is similar to good response, label as 0 (good)
            # If similar to bad response, label as 1 (bad)
            if generated_response.strip().lower() == good.strip().lower():
                test_labels.append(0)  # Good
            else:
                test_labels.append(1)  # Assume bad if not matching good
            
            test_eval_hidden_states.append(test_hidden_states[i])
    
    # Step 7: Evaluate guard effectiveness
    logger.info("Step 7: Evaluating guard effectiveness")
    guard_results = evaluate_guard(
        guard_pipeline=guard_pipeline,
        test_hidden_states=test_eval_hidden_states,
        test_labels=test_labels,
        task_name=task_name,
        model_name=model_name,
        layer=layer
    )
    
    # Step 8: Evaluate with lm-harness
    logger.info("Step 8: Evaluating with lm-harness")
    lm_harness_results = evaluate_lm_harness_task(
        task_data,
        test_docs=test_docs,
        test_responses=test_responses,
        test_references=test_references
    )
    
    # Combine results
    combined_results = {
        **guard_results,
        'lm_harness_metrics': lm_harness_results.get('lm_harness_metrics', {}),
        'lm_harness_primary_metric': lm_harness_results.get('primary_metric', 0.0),
        'training_samples': len(valid_pairs),
        'test_samples': len(test_docs),
        'pipeline_config': {
            'shots': shots,
            'split_ratio': split_ratio,
            'classifier_type': classifier_type,
            'max_new_tokens': max_new_tokens,
            'layer': layer
        }
    }
    
    logger.info(f"Pipeline complete for {task_name}")
    return combined_results


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse task names
    task_names = [task.strip() for task in args.task_names.split(',')]
    
    logger.info(f"Starting wisent-guard pipeline for tasks: {task_names}")
    logger.info(f"Model: {args.model}, Layer: {args.layer}")
    
    # Set up cache directory
    cache_dir = None if args.no_cache else args.cache_dir
    
    # Run pipeline for each task
    all_results = []
    
    for task_name in tqdm(task_names, desc="Processing tasks"):
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing task: {task_name}")
            logger.info(f"{'='*50}")
            
            result = run_single_task(
                task_name=task_name,
                model_name=args.model,
                layer=args.layer,
                shots=args.shots,
                split_ratio=args.split_ratio,
                limit=args.limit,
                classifier_type=args.classifier_type,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                cache_dir=cache_dir,
                device=args.device,
                seed=args.seed
            )
            
            all_results.append(result)
            
            # Save individual result
            save_results_json(result, args.output, f"{task_name}_results.json")
            
        except Exception as e:
            logger.error(f"Failed to process task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add error result
            error_result = {
                'task_name': task_name,
                'model_name': args.model,
                'layer': args.layer,
                'error': str(e),
                'detection_rate': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0
            }
            all_results.append(error_result)
    
    # Save aggregated results
    if all_results:
        logger.info(f"\n{'='*50}")
        logger.info("Saving aggregated results")
        logger.info(f"{'='*50}")
        
        # Save CSV summary
        save_results_csv(all_results, args.output)
        
        # Create evaluation report
        create_evaluation_report(all_results, args.output)
        
        # Calculate and display aggregate metrics
        aggregate_metrics = calculate_aggregate_metrics(all_results)
        logger.info(f"\nAggregate Results:")
        logger.info(f"  Tasks processed: {aggregate_metrics.get('total_tasks', 0)}")
        logger.info(f"  Overall detection rate: {aggregate_metrics.get('overall_detection_rate', 0.0):.1%}")
        logger.info(f"  Mean F1 score: {aggregate_metrics.get('mean_f1_score', 0.0):.4f}")
        logger.info(f"  Mean accuracy: {aggregate_metrics.get('mean_accuracy', 0.0):.4f}")
    
    logger.info(f"\nResults saved to: {args.output}")
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main() 