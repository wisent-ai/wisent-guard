"""
Command-line interface for wisent-guard lm-evaluation-harness integration.
Clean implementation using enhanced core primitives.
"""

import argparse
import logging
import sys
import os
import json
import csv
from typing import List, Dict, Any

from .core import Model, ContrastivePairSet, SteeringMethod, SteeringType, Layer

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
        """
    )
    
    parser.add_argument("command", choices=["tasks"], help="Command to run")
    parser.add_argument("task_names", help="Comma-separated list of task names")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--layer", type=int, default=15, help="Layer to extract activations from")
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents per task")
    parser.add_argument("--output", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--classifier-type", type=str, choices=["logistic", "mlp"], default="logistic", help="Type of classifier")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum new tokens for generation")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser


def run_task_pipeline(
    task_name: str,
    model_name: str,
    layer: int,
    shots: int = 0,
    split_ratio: float = 0.8,
    limit: int = None,
    classifier_type: str = "logistic",
    max_new_tokens: int = 50,
    device: str = None,
    seed: int = 42,
    verbose: bool = False
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
        device: Target device
        seed: Random seed
        
    Returns:
        Dictionary with all results
    """
    logger.info(f"Running pipeline for task: {task_name}")
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🚀 STARTING PIPELINE FOR TASK: {task_name.upper()}")
        print(f"{'='*80}")
        print(f"📋 Configuration:")
        print(f"   • Model: {model_name}")
        print(f"   • Layer: {layer}")
        print(f"   • Classifier: {classifier_type}")
        print(f"   • Max tokens: {max_new_tokens}")
        print(f"   • Split ratio: {split_ratio}")
        print(f"   • Limit: {limit}")
        print(f"   • Seed: {seed}")
    
    try:
        # Initialize enhanced primitives
        if verbose:
            print(f"\n🔧 Initializing model and primitives...")
        model = Model(name=model_name, device=device)
        layer_obj = Layer(index=layer, type="transformer")
        
        # Load and prepare data using enhanced Model primitive
        if verbose:
            print(f"📚 Loading task data for {task_name}...")
        task_data = model.load_lm_eval_task(task_name, shots=shots, limit=limit)
        train_docs, test_docs = model.split_task_data(task_data, split_ratio=split_ratio, random_seed=seed)
        
        if verbose:
            print(f"📊 Data split: {len(train_docs)} training docs, {len(test_docs)} test docs")
        
        # Create training data
        train_prompts = model.prepare_prompts_from_docs(task_data, train_docs)
        train_references = model.get_reference_answers(task_data, train_docs)
        
        if verbose:
            print(f"\n📝 TRAINING DATA PREPARATION:")
            print(f"   • Total training examples: {len(train_prompts)}")
            print(f"\n🔍 Training Examples:")
            for i, (prompt, reference) in enumerate(zip(train_prompts, train_references)):
                print(f"\n   📋 Example {i+1}:")
                print(f"      🔸 Question: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                print(f"      ✅ Good Answer: {reference}")
                print(f"      ❌ Bad Answer: [INCORRECT]")
        
        # Create contrastive pairs using enhanced ContrastivePairSet
        phrase_pairs = []
        for i, (prompt, reference) in enumerate(zip(train_prompts, train_references)):
            harmful = f"{prompt} [INCORRECT]"
            harmless = f"{prompt} {reference}"
            phrase_pairs.append({
                "harmful": harmful,
                "harmless": harmless
            })
            
            if verbose:
                print(f"\n   🔄 Contrastive Pair {i+1}:")
                print(f"      🟢 Harmless: {harmless[:150]}{'...' if len(harmless) > 150 else ''}")
                print(f"      🔴 Harmful: {harmful[:150]}{'...' if len(harmful) > 150 else ''}")
        
        # Create and train ContrastivePairSet
        if verbose:
            print(f"\n🧠 Creating ContrastivePairSet with {len(phrase_pairs)} pairs...")
        pair_set = ContrastivePairSet.from_phrase_pairs(
            name=f"{task_name}_training",
            phrase_pairs=phrase_pairs,
            task_type="lm_evaluation"
        )
        
        # Extract activations for the pairs
        if verbose:
            print(f"🔬 Extracting activations from layer {layer}...")
        pair_set.extract_activations_with_model(model, layer_obj)
        
        # Train classifier
        if verbose:
            print(f"\n🎯 TRAINING CLASSIFIER:")
            print(f"   • Type: {classifier_type}")
            print(f"   • Training pairs: {len(pair_set)}")
        
        steering_type = SteeringType.LOGISTIC if classifier_type == "logistic" else SteeringType.MLP
        steering_method = SteeringMethod(method_type=steering_type, device=device)
        
        training_results = steering_method.train(pair_set)
        
        if verbose:
            print(f"✅ Training completed!")
            print(f"   • Accuracy: {training_results.get('accuracy', 'N/A'):.2%}")
            print(f"   • F1 Score: {training_results.get('f1', 'N/A'):.3f}")
        
        # Evaluate on test set
        if verbose:
            print(f"\n🧪 PREPARING TEST DATA:")
        test_prompts = model.prepare_prompts_from_docs(task_data, test_docs)
        test_references = model.get_reference_answers(task_data, test_docs)
        
        if verbose:
            print(f"   • Test examples: {len(test_prompts)}")
            print(f"\n🔍 Test Examples:")
            for i, (prompt, reference) in enumerate(zip(test_prompts, test_references)):
                print(f"\n   📋 Test Example {i+1}:")
                print(f"      🔸 Question: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                print(f"      ✅ Expected Answer: {reference}")
        
        test_phrase_pairs = []
        for i, (prompt, reference) in enumerate(zip(test_prompts, test_references)):
            harmful = f"{prompt} [INCORRECT]"
            harmless = f"{prompt} {reference}"
            test_phrase_pairs.append({
                "harmful": harmful,
                "harmless": harmless
            })
            
            if verbose:
                print(f"\n   🔄 Test Pair {i+1}:")
                print(f"      🟢 Harmless: {harmless[:150]}{'...' if len(harmless) > 150 else ''}")
                print(f"      🔴 Harmful: {harmful[:150]}{'...' if len(harmful) > 150 else ''}")
        
        test_pair_set = ContrastivePairSet.from_phrase_pairs(
            name=f"{task_name}_test",
            phrase_pairs=test_phrase_pairs,
            task_type="lm_evaluation"
        )
        
        # Extract activations for the test pairs
        if verbose:
            print(f"\n🔬 Extracting test activations from layer {layer}...")
        test_pair_set.extract_activations_with_model(model, layer_obj)
        
        if verbose:
            print(f"📊 Evaluating classifier on test set...")
        evaluation_results = steering_method.evaluate(test_pair_set)
        
        if verbose:
            print(f"✅ Evaluation completed!")
            print(f"   • Test Accuracy: {evaluation_results.get('accuracy', 'N/A'):.2%}")
            print(f"   • Test F1 Score: {evaluation_results.get('f1', 'N/A'):.3f}")
        
        # Generate sample responses
        if verbose:
            print(f"\n🎭 GENERATING SAMPLE RESPONSES:")
            print(f"   • Generating {min(5, len(test_prompts))} sample responses...")
        
        generated_responses = []
        for i, prompt in enumerate(test_prompts[:5]):  # Sample 5 responses
            if verbose:
                print(f"\n   🎯 Generating response {i+1}:")
                print(f"      📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            
            response, _ = model.generate(prompt, layer, max_new_tokens)
            generated_responses.append(response)
            
            if verbose:
                print(f"      🤖 Generated: {response[:150]}{'...' if len(response) > 150 else ''}")
        
        results = {
            "task_name": task_name,
            "model_name": model_name,
            "layer": layer,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "num_train": len(train_docs),
            "num_test": len(test_docs),
            "sample_responses": generated_responses
        }
        
        if verbose:
            print(f"\n🎉 PIPELINE COMPLETED FOR {task_name.upper()}!")
            print(f"{'='*80}")
            print(f"📊 FINAL RESULTS:")
            print(f"   • Training samples: {len(train_docs)}")
            print(f"   • Test samples: {len(test_docs)}")
            print(f"   • Training accuracy: {training_results.get('accuracy', 'N/A'):.2%}")
            print(f"   • Test accuracy: {evaluation_results.get('accuracy', 'N/A'):.2%}")
            print(f"   • Generated responses: {len(generated_responses)}")
            print(f"{'='*80}\n")
        
        logger.info(f"Pipeline completed for {task_name}")
        return results
        
    except Exception as e:
        logger.error(f"Error in pipeline for {task_name}: {e}")
        return {"task_name": task_name, "error": str(e)}


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")


def save_results_csv(results: Dict[str, Any], output_path: str) -> None:
    """Save results to CSV file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Flatten results for CSV
        rows = []
        for task_name, task_results in results.items():
            if isinstance(task_results, dict):
                row = {"task": task_name}
                row.update(task_results)
                rows.append(row)
        
        if rows:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"CSV results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save CSV to {output_path}: {e}")


def create_evaluation_report(results: Dict[str, Any], output_path: str) -> None:
    """Create a markdown evaluation report."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Wisent-Guard Evaluation Report\n\n")
            
            for task_name, task_results in results.items():
                f.write(f"## Task: {task_name}\n\n")
                
                if isinstance(task_results, dict):
                    for key, value in task_results.items():
                        if key != "task":
                            f.write(f"- **{key}**: {value}\n")
                    f.write("\n")
        
        logger.info(f"Evaluation report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create report at {output_path}: {e}")


def main():
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse task names
    task_names = [name.strip() for name in args.task_names.split(",")]
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"Starting wisent-guard harness for tasks: {task_names}")
    logger.info(f"Model: {args.model}, Layer: {args.layer}")
    
    try:
        # Run pipeline for each task
        all_results = {}
        
        for task_name in task_names:
            logger.info(f"Processing task: {task_name}")
            
            task_results = run_task_pipeline(
                task_name=task_name,
                model_name=args.model,
                layer=args.layer,
                shots=args.shots,
                split_ratio=args.split_ratio,
                limit=args.limit,
                classifier_type=args.classifier_type,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                seed=args.seed,
                verbose=args.verbose
            )
            
            all_results[task_name] = task_results
        
        # Save results
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_path = os.path.join(args.output, f"results_{timestamp}.json")
        save_results_json(all_results, json_path)
        
        # CSV results
        csv_path = os.path.join(args.output, f"results_{timestamp}.csv")
        save_results_csv(all_results, csv_path)
        
        # Markdown report
        report_path = os.path.join(args.output, f"report_{timestamp}.md")
        create_evaluation_report(all_results, report_path)
        
        logger.info("All tasks completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("WISENT-GUARD EVALUATION SUMMARY")
        print("="*50)
        
        for task_name, results in all_results.items():
            if "error" in results:
                print(f"{task_name}: ERROR - {results['error']}")
            else:
                training_acc = results.get("training_results", {}).get("accuracy", "N/A")
                eval_acc = results.get("evaluation_results", {}).get("accuracy", "N/A")
                print(f"{task_name}: Train={training_acc:.2%} | Test={eval_acc:.2%}")
        
        print(f"\nResults saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 