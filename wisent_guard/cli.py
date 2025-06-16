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
from typing import List, Dict, Any, Optional

from .core import Model, ContrastivePairSet, SteeringMethod, SteeringType, Layer
from .core.ground_truth_evaluator import GroundTruthEvaluator, GroundTruthMethod
from .core.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig
from .core.model import Model
from .core.classifier import Classifier
from .core.layer import Layer
from .optimize import run_smart_optimization, run_interactive_optimization, generate_with_all_layer_activations, compute_classification_score

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
  python -m wisent_guard tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B --token-aggregation final
  python -m wisent_guard tasks hellaswag,mmlu --layer 10 --model meta-llama/Llama-3.1-8B --shots 5 --token-aggregation max
  python -m wisent_guard tasks truthfulqa --optimize --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks truthfulqa --layer -1 --auto-optimize --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks truthfulqa --optimize --optimize-layers "10-20" --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks data.csv --from-csv --model meta-llama/Llama-3.1-8B
  python -m wisent_guard tasks data.json --from-json --model meta-llama/Llama-3.1-8B
        """
    )
    
    parser.add_argument("command", choices=["tasks"], help="Command to run")
    parser.add_argument("task_names", help="Comma-separated list of task names, or path to CSV/JSON file with --from-csv/--from-json")
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
    parser.add_argument("--token-aggregation", type=str, choices=["average", "final", "first", "max", "min"], 
                       default="average", help="How to aggregate token scores for classification")
    parser.add_argument("--ground-truth-method", type=str, 
                       choices=["none", "exact_match", "substring_match", "user_specified", "interactive", "manual_review", "good"],
                       default="none", 
                       help="Method for ground truth evaluation. 'none' skips evaluation, 'exact_match' and 'substring_match' are problematic for free-form generation, 'user_specified' allows manual labeling, 'interactive' prompts for y/n labeling, 'manual_review' marks for review, 'good' marks everything as truthful (for debugging)")
    parser.add_argument("--user-labels", type=str, nargs="*", default=None,
                       help="User-specified ground truth labels for responses ('truthful' or 'hallucination'). Used with --ground-truth-method user_specified")
    
    # File input arguments
    parser.add_argument("--from-csv", action="store_true", 
                       help="Load task data from CSV file. Requires columns: question, correct_answer, incorrect_answer")
    parser.add_argument("--from-json", action="store_true",
                       help="Load task data from JSON file. Expected format: list of objects with question, correct_answer, incorrect_answer")
    parser.add_argument("--question-col", type=str, default="question",
                       help="Column name for questions in CSV file (default: question)")
    parser.add_argument("--correct-col", type=str, default="correct_answer",
                       help="Column name for correct answers in CSV file (default: correct_answer)")
    parser.add_argument("--incorrect-col", type=str, default="incorrect_answer", 
                       help="Column name for incorrect answers in CSV file (default: incorrect_answer)")
    
    # Optimization arguments
    parser.add_argument("--optimize", action="store_true",
                       help="Enable hyperparameter optimization. When enabled, will find optimal layer, threshold, and aggregation method")
    parser.add_argument("--optimize-layers", type=str, default="all",
                       help="Layer range for optimization (e.g., '8-24' or '10,15,20' or 'all'). Default: all (uses all model layers)")
    parser.add_argument("--optimize-metric", type=str, choices=["accuracy", "f1", "precision", "recall", "auc"],
                       default="f1", help="Metric to optimize for. Default: f1")
    parser.add_argument("--optimize-max-combinations", type=int, default=100,
                       help="Maximum number of hyperparameter combinations to test. Default: 100")
    parser.add_argument("--auto-optimize", action="store_true",
                       help="Automatically enable optimization when layer is not specified or is -1")
    
    # Dataset validation arguments
    parser.add_argument("--allow-small-dataset", action="store_true",
                       help="Allow training with datasets smaller than 4 samples (may cause training issues)")
    
    return parser


def parse_layer_range(layer_range_str: str, model=None) -> Optional[List[int]]:
    """
    Parse layer range string into list of integers.
    
    Args:
        layer_range_str: String like "8-24", "10,15,20", or "all"
        model: Model object (needed for "all" option)
        
    Returns:
        List of layer indices, or None if "all" (will be auto-detected later)
    """
    if layer_range_str.lower() == "all":
        # Return None to signal auto-detection
        return None
    elif '-' in layer_range_str:
        # Range format: "8-24"
        start, end = map(int, layer_range_str.split('-'))
        return list(range(start, end + 1))
    elif ',' in layer_range_str:
        # Comma-separated format: "10,15,20"
        return [int(x.strip()) for x in layer_range_str.split(',')]
    else:
        # Single layer
        return [int(layer_range_str)]


def aggregate_token_scores(token_scores: List[float], method: str) -> float:
    """
    Aggregate token scores using the specified method.
    
    Args:
        token_scores: List of token scores (probabilities)
        method: Aggregation method ("average", "final", "first", "max", "min")
        
    Returns:
        Aggregated score
    """
    if not token_scores:
        return 0.5
    
    if method == "average":
        return sum(token_scores) / len(token_scores)
    elif method == "final":
        return token_scores[-1]
    elif method == "first":
        return token_scores[0]
    elif method == "max":
        return max(token_scores)
    elif method == "min":
        return min(token_scores)
    else:
        # Default to average if unknown method
        return sum(token_scores) / len(token_scores)


def generate_with_classification(model, prompt, layer, max_new_tokens, steering_method, token_aggregation="average", threshold=0.6, verbose=False):
    """
    Generate text with token-level hallucination classification.
    
    Args:
        model: Model object
        prompt: Input prompt
        layer: Layer index for activation extraction
        max_new_tokens: Maximum tokens to generate
        steering_method: Trained steering method with classifier
        token_aggregation: How to aggregate token scores ("average", "final", "first", "max", "min")
        verbose: Whether to print debug info
        
    Returns:
        Tuple of (response_text, token_scores, classification)
    """
    import torch
    from .core import Layer
    from .core.activations import Activations, ActivationAggregationMethod
    
    # Generate response and get token-by-token activations
    response, activations_dict = model.generate(prompt, layer, max_new_tokens)
    
    if not response.strip():
        return response, [], "UNKNOWN"
    
    # Tokenize the full prompt + response to get individual tokens
    full_text = f"{prompt}{response}"
    inputs = model.tokenizer(full_text, return_tensors="pt")
    tokens = model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Get prompt length to identify response tokens
    prompt_inputs = model.tokenizer(prompt, return_tensors="pt")
    prompt_length = len(prompt_inputs['input_ids'][0])
    
    # Extract activations for each response token
    token_scores = []
    layer_obj = Layer(index=layer, type="transformer")
    
    try:
        # Get model device
        model_device = next(model.hf_model.parameters()).device
        inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Get hidden states for the full sequence
        with torch.no_grad():
            outputs = model.hf_model(**inputs_on_device, output_hidden_states=True)
        
        # Extract activations for response tokens only
        if layer + 1 < len(outputs.hidden_states):
            hidden_states = outputs.hidden_states[layer + 1]  # [batch_size, seq_len, hidden_dim]
        else:
            hidden_states = outputs.hidden_states[-1]
        
        # Score each response token
        for token_idx in range(prompt_length, len(tokens)):
            if token_idx < hidden_states.shape[1]:
                # Extract activation for this token
                token_activation = hidden_states[0, token_idx, :].cpu()
                
                # Create Activations object
                activation_obj = Activations(
                    tensor=token_activation.unsqueeze(0),  # Add batch dimension
                    layer=layer_obj,
                    aggregation_method=ActivationAggregationMethod.LAST_TOKEN
                )
                
                # Get feature vector for classifier
                features = activation_obj.extract_features_for_classifier()
                
                # Get prediction probability from classifier
                if hasattr(steering_method, 'classifier') and steering_method.classifier:
                    try:
                        # Predict probability of being harmful (class 1)
                        # Our classifier returns a single float, not an array like sklearn
                        prob = steering_method.classifier.predict_proba([features.numpy()])
                        # Handle both single float and array returns
                        if isinstance(prob, (list, tuple)) or hasattr(prob, '__getitem__'):
                            prob = float(prob[0]) if len(prob) > 0 else 0.5
                        else:
                            prob = float(prob)
                        token_scores.append(prob)
                    except Exception as e:
                        if verbose:
                            print(f"      ⚠️  Classifier error for token {token_idx}: {e}")
                        token_scores.append(0.5)  # Neutral score
                else:
                    token_scores.append(0.5)  # Neutral score if no classifier
    
    except Exception as e:
        if verbose:
            print(f"      ⚠️  Error during token scoring: {e}")
        # Fallback: assign neutral scores
        response_tokens = model.tokenizer(response, return_tensors="pt")['input_ids'][0]
        token_scores = [0.5] * len(response_tokens)
    
    # Classify overall response using specified aggregation method
    if token_scores:
        aggregated_score = aggregate_token_scores(token_scores, token_aggregation)
        classification = "HALLUCINATION" if aggregated_score > threshold else "TRUTHFUL"
    else:
        aggregated_score = 0.5
        classification = "UNKNOWN"
    
    return response, token_scores, classification



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
    token_aggregation: str = "average",
    ground_truth_method: str = "none",
    user_labels: List[str] = None,
    optimize: bool = False,
    optimize_layers: str = "8-24",
    optimize_metric: str = "f1",
    optimize_max_combinations: int = 100,
    verbose: bool = False,
    from_csv: bool = False,
    from_json: bool = False,
    question_col: str = "question",
    correct_col: str = "correct_answer",
    incorrect_col: str = "incorrect_answer",
    allow_small_dataset: bool = False
) -> Dict[str, Any]:
    """
    Run the complete pipeline for a single task or file.
    
    Args:
        task_name: Name of the benchmark task or path to file
        model_name: Language model name
        layer: Layer for activation extraction
        shots: Number of few-shot examples
        split_ratio: Train/test split ratio
        limit: Optional limit on documents
        classifier_type: Type of classifier
        max_new_tokens: Max tokens for generation
        device: Target device
        seed: Random seed
        token_aggregation: How to aggregate token scores for classification
        from_csv: Whether task_name is a CSV file
        from_json: Whether task_name is a JSON file
        question_col: CSV column name for questions
        correct_col: CSV column name for correct answers
        incorrect_col: CSV column name for incorrect answers
        
    Returns:
        Dictionary with all results
    """
    logger.info(f"Running pipeline for task: {task_name}")
    
    display_name = task_name if not (from_csv or from_json) else f"file:{task_name}"
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🚀 STARTING PIPELINE FOR TASK: {display_name.upper()}")
        print(f"{'='*80}")
        print(f"📋 Configuration:")
        print(f"   • Model: {model_name}")
        print(f"   • Layer: {layer}")
        print(f"   • Classifier: {classifier_type}")
        print(f"   • Max tokens: {max_new_tokens}")
        print(f"   • Split ratio: {split_ratio}")
        print(f"   • Token aggregation: {token_aggregation}")
        print(f"   • Limit: {limit}")
        print(f"   • Seed: {seed}")
        if from_csv:
            print(f"   • Input: CSV file")
            print(f"   • Columns: {question_col}, {correct_col}, {incorrect_col}")
        elif from_json:
            print(f"   • Input: JSON file")
    
    try:
        # Initialize enhanced primitives
        if verbose:
            print(f"\n🔧 Initializing model and primitives...")
        model = Model(name=model_name, device=device)
        layer_obj = Layer(index=layer, type="transformer")
        
        if from_csv or from_json:
            # Load data from CSV/JSON file using ContrastivePairSet
            if verbose:
                print(f"\n📁 Loading data from {'CSV' if from_csv else 'JSON'} file...")
            
            # ContrastivePairSet is imported at the top of the file
            
            if from_csv:
                pair_set = ContrastivePairSet.from_csv_file(
                    name="csv_data",
                    csv_path=task_name,
                    question_col=question_col,
                    correct_col=correct_col,
                    incorrect_col=incorrect_col,
                    limit=limit
                )
            else:  # from_json
                pair_set = ContrastivePairSet.from_json_file(
                    name="json_data",
                    json_path=task_name,
                    limit=limit
                )
            
            # Convert ContrastivePairSet to qa_pairs format for existing pipeline
            all_qa_pairs = []
            for pair in pair_set.pairs:
                if hasattr(pair, 'question'):
                    all_qa_pairs.append({
                        'question': pair.question,
                        'correct_answer': pair.correct_answer,
                        'incorrect_answer': pair.incorrect_answer
                    })
            
            # Split the qa_pairs
            import random
            random.seed(seed)
            random.shuffle(all_qa_pairs)
            split_point = int(len(all_qa_pairs) * split_ratio)
            qa_pairs = all_qa_pairs[:split_point]  # Training data
            test_qa_pairs_source = all_qa_pairs[split_point:]  # Test data
            
            if verbose:
                print(f"📊 Data split: {len(qa_pairs)} training pairs, {len(test_qa_pairs_source)} test pairs")
            
        else:
            # Traditional lm-harness task loading
            if verbose:
                print(f"📚 Loading task data for {task_name}...")
            task_data = model.load_lm_eval_task(task_name, shots=shots, limit=limit)
            train_docs, test_docs = model.split_task_data(task_data, split_ratio=split_ratio, random_seed=seed)
            
            if verbose:
                print(f"📊 Data split: {len(train_docs)} training docs, {len(test_docs)} test docs")
            
            # Extract QA pairs from training documents  
            if verbose:
                print(f"\n📝 TRAINING DATA PREPARATION:")
                print(f"   • Loading TruthfulQA data with correct/incorrect answers...")
            
            qa_pairs = []
            for doc in train_docs:
                try:
                    # Extract question
                    if hasattr(task_data, 'doc_to_text'):
                        question = task_data.doc_to_text(doc)
                    else:
                        question = doc.get('question', str(doc))
                    
                    # Extract correct answer
                    correct_answers = doc.get('mc1_targets', {}).get('choices', [])
                    correct_labels = doc.get('mc1_targets', {}).get('labels', [])
                    
                    # Find the correct answer
                    correct_answer = None
                    for i, label in enumerate(correct_labels):
                        if label == 1 and i < len(correct_answers):
                            correct_answer = correct_answers[i]
                            break
                    
                    # Find an incorrect answer
                    incorrect_answer = None
                    for i, label in enumerate(correct_labels):
                        if label == 0 and i < len(correct_answers):
                            incorrect_answer = correct_answers[i]
                            break
                    
                    if correct_answer and incorrect_answer:
                        qa_pairs.append({
                            'question': question,
                            'correct_answer': correct_answer,
                            'incorrect_answer': incorrect_answer
                        })
                        
                except Exception as e:
                    # Skip problematic docs
                    continue
            
            test_qa_pairs_source = test_docs  # Keep original format for test docs
        
        if verbose:
            print(f"   • Successfully extracted {len(qa_pairs)} QA pairs")
            print(f"\n🔍 Training Examples:")
            for i, qa_pair in enumerate(qa_pairs[:4]):  # Show first 4
                print(f"\n   📋 Example {i+1}:")
                question_preview = qa_pair['question'][:100] + "..." if len(qa_pair['question']) > 100 else qa_pair['question']
                print(f"      🔸 Question: {question_preview}")
                print(f"      ✅ Correct Answer: {qa_pair['correct_answer']}")
                print(f"      ❌ Incorrect Answer: {qa_pair['incorrect_answer']}")

        # Validate dataset size before proceeding
        min_training_samples = 4
        if len(qa_pairs) < min_training_samples:
            error_msg = f"Insufficient training data: {len(qa_pairs)} pairs found, minimum {min_training_samples} required"
            if verbose:
                print(f"\n❌ ERROR: {error_msg}")
                print(f"   • Consider increasing --limit or using a larger dataset")
                print(f"   • CSV/JSON files should have at least {min_training_samples} rows")
                print(f"   • lm-harness tasks may need higher --limit values")
            
            if not allow_small_dataset:
                if verbose:
                    print(f"   • Use --allow-small-dataset flag to bypass this check (may cause training issues)")
                
                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "error": error_msg,
                    "training_samples": len(qa_pairs),
                    "minimum_required": min_training_samples,
                    "suggestion": "Increase dataset size, --limit parameter, or use --allow-small-dataset flag"
                }
            else:
                if verbose:
                    print(f"   ⚠️  WARNING: Proceeding with small dataset due to --allow-small-dataset flag")
                    print(f"   • Training may be unstable with only {len(qa_pairs)} samples")
        
        # Create contrastive pairs using proper activation collection logic
        from .core.activations import ActivationCollectionLogic, Activations, ActivationAggregationMethod
        collector = ActivationCollectionLogic(model=model)
        contrastive_pairs = collector.create_batch_contrastive_pairs(qa_pairs)
        
        if verbose:
            print(f"\n🔄 Created {len(contrastive_pairs)} contrastive pairs:")
            for i, pair in enumerate(contrastive_pairs[:3]):  # Show first 3
                print(f"\n   🔄 Contrastive Pair {i+1}:")
                print(f"      📝 Prompt: {pair.prompt[:100]}{'...' if len(pair.prompt) > 100 else ''}")
                print(f"      🟢 Positive (B): {pair.positive_response}")
                print(f"      🔴 Negative (A): {pair.negative_response}")
        
        # Check if optimization is needed
        original_layer = layer
        original_token_aggregation = token_aggregation
        optimization_result = None
        
        if optimize and ground_truth_method == "interactive":
            # Special case: Interactive optimization - SKIP normal pipeline
            # Generate test questions for optimization
            test_questions = []
            for doc in test_qa_pairs_source[:2]:  # Use first 2 test questions for optimization
                try:
                    if from_csv or from_json:
                        # For CSV/JSON, doc is already a qa_pair dict
                        question = doc['question']
                    else:
                        # For lm-harness tasks, extract from document
                        if hasattr(task_data, 'doc_to_text'):
                            question = task_data.doc_to_text(doc)
                        else:
                            question = doc.get('question', str(doc))
                    test_questions.append(question)
                except:
                    continue
            
            if test_questions:
                # Run interactive optimization ONLY
                optimization_result = run_interactive_optimization(
                    model=model,
                    questions=test_questions,
                    training_pairs=contrastive_pairs,
                    max_new_tokens=max_new_tokens,
                    max_combinations=optimize_max_combinations,
                    verbose=verbose
                )
                
                if optimization_result.get('optimization_performed'):
                    layer = optimization_result['best_layer']
                    token_aggregation = optimization_result['best_aggregation']
                    if verbose:
                        print(f"✅ Interactive optimization completed!")
                        print(f"   • Optimized layer: {layer} (was {original_layer})")
                        print(f"   • Optimized aggregation: {token_aggregation} (was {original_token_aggregation})")
                    
                    # Return results immediately - skip normal pipeline
                    return {
                        "task_name": task_name,
                        "model_name": model_name,
                        "layer": layer,
                        "original_layer": original_layer,
                        "token_aggregation": token_aggregation,
                        "original_token_aggregation": original_token_aggregation,
                        "optimization_performed": True,
                        "optimization_result": optimization_result,
                        "interactive_optimization_only": True
                    }
            else:
                if verbose:
                    print(f"⚠️ No test questions available for interactive optimization")
                return {"task_name": task_name, "error": "No test questions available"}
        
        elif optimize:
            # Load test data first for optimization
            test_qa_pairs = []
            for doc in test_qa_pairs_source:
                try:
                    if from_csv or from_json:
                        # For CSV/JSON, doc is already a qa_pair dict
                        test_qa_pairs.append({
                            'question': doc['question'],
                            'formatted_question': doc['question'], 
                            'correct_answer': doc['correct_answer']
                        })
                    else:
                        # For lm-harness tasks, extract from document
                        raw_question = doc.get('question', str(doc))
                        if hasattr(task_data, 'doc_to_text'):
                            formatted_question = task_data.doc_to_text(doc)
                        else:
                            formatted_question = raw_question
                        
                        # Extract correct answer for ground truth comparison
                        correct_answers = doc.get('mc1_targets', {}).get('choices', [])
                        correct_labels = doc.get('mc1_targets', {}).get('labels', [])
                        
                        correct_answer = None
                        for i, label in enumerate(correct_labels):
                            if label == 1 and i < len(correct_answers):
                                correct_answer = correct_answers[i]
                                break
                        
                        if correct_answer:
                            test_qa_pairs.append({
                                'question': raw_question,
                                'formatted_question': formatted_question,
                                'correct_answer': correct_answer
                            })
                        
                except Exception as e:
                    continue
            
            # Run smart optimization with caching
            optimization_result = run_smart_optimization(
                model=model,
                collector=collector,
                contrastive_pairs=contrastive_pairs,
                test_qa_pairs=test_qa_pairs,
                task_name=task_name,
                model_name=model_name,
                limit=limit,
                ground_truth_method=ground_truth_method,
                max_new_tokens=max_new_tokens,
                device=device,
                verbose=verbose,
                optimize_layers=optimize_layers
            )
            
            # Extract the best parameters from optimization
            if optimization_result.get('optimization_performed', False):
                layer = optimization_result.get('best_layer', layer)
                token_aggregation = optimization_result.get('best_aggregation', token_aggregation)
                optimized_classifier_type = optimization_result.get('best_classifier_type', classifier_type)
                optimized_threshold = optimization_result.get('best_threshold', 0.6)
                if verbose:
                    print(f"✅ Hyperparameter optimization completed!")
                    print(f"   • Best layer: {layer} + {token_aggregation} aggregation")
                    print(f"   • Best classifier: {optimized_classifier_type}")
                    print(f"   • Best threshold: {optimized_threshold}")
            else:
                if verbose:
                    print(f"⚠️ Optimization failed, using default layer {layer}")
                optimized_classifier_type = classifier_type
                optimized_threshold = 0.6
                optimization_result = {
                    'best_layer': layer,
                    'best_aggregation': token_aggregation,
                    'best_classifier_type': optimized_classifier_type,
                    'best_threshold': optimized_threshold,
                    'optimization_performed': False
                }
        
        # Extract activations from the choice tokens using the (possibly optimized) layer
        optimization_note = f" (optimized)" if optimize and layer != original_layer else ""
        if verbose:
            print(f"\n🔬 Extracting activations from layer {layer}{optimization_note} choice tokens...")
        
        processed_pairs = collector.collect_activations_batch(
            pairs=contrastive_pairs,
            layer_index=layer,
            device=device
        )
        
        # Convert to ContrastivePairSet format for training
        phrase_pairs = []
        for pair in processed_pairs:
            # Create the full prompts for the pair set
            positive_full = f"{pair.prompt}{pair.positive_response}"
            negative_full = f"{pair.prompt}{pair.negative_response}"
            
            phrase_pairs.append({
                "harmful": negative_full,  # A choice (incorrect)
                "harmless": positive_full  # B choice (correct)
            })
        
        # Create ContrastivePairSet with the real activations
        pair_set = ContrastivePairSet.from_phrase_pairs(
            name=f"{task_name}_training",
            phrase_pairs=phrase_pairs,
            task_type="lm_evaluation"
        )
        
        # Store the real activations in the pair set response objects
        for i, processed_pair in enumerate(processed_pairs):
            if i < len(pair_set.pairs):
                # Assign activations to the response objects, not the pair directly
                if hasattr(pair_set.pairs[i], 'positive_response') and pair_set.pairs[i].positive_response:
                    pair_set.pairs[i].positive_response.activations = processed_pair.positive_activations
                if hasattr(pair_set.pairs[i], 'negative_response') and pair_set.pairs[i].negative_response:
                    pair_set.pairs[i].negative_response.activations = processed_pair.negative_activations
        
        # Train classifier using optimized type (if optimization was performed)
        if optimize:
            final_classifier_type = optimized_classifier_type
            final_threshold = optimized_threshold
        else:
            final_classifier_type = classifier_type
            final_threshold = 0.6
        
        if verbose:
            print(f"\n🎯 TRAINING CLASSIFIER:")
            print(f"   • Type: {final_classifier_type}")
            print(f"   • Threshold: {final_threshold}")
            print(f"   • Training pairs: {len(pair_set)}")
        
        steering_type = SteeringType.LOGISTIC if final_classifier_type == "logistic" else SteeringType.MLP
        steering_method = SteeringMethod(method_type=steering_type, threshold=final_threshold, device=device)
        
        try:
            training_results = steering_method.train(pair_set)
            
            if verbose:
                print(f"✅ Training completed!")
                print(f"   • Accuracy: {training_results.get('accuracy', 'N/A'):.2%}")
                print(f"   • F1 Score: {training_results.get('f1', 'N/A'):.3f}")
                
        except ZeroDivisionError as e:
            error_msg = f"Classifier training failed due to insufficient or imbalanced data: {str(e)}"
            if verbose:
                print(f"\n❌ TRAINING ERROR: {error_msg}")
                print(f"   • This often happens with very small datasets")
                print(f"   • Try increasing the dataset size or using --limit with a higher value")
                print(f"   • Current training samples: {len(pair_set)}")
            
            return {
                "task_name": task_name,
                "model_name": model_name,
                "error": error_msg,
                "training_samples": len(pair_set),
                "error_type": "division_by_zero",
                "suggestion": "Increase dataset size or check data quality"
            }
            
        except Exception as e:
            error_msg = f"Classifier training failed: {str(e)}"
            if verbose:
                print(f"\n❌ TRAINING ERROR: {error_msg}")
                print(f"   • Training samples: {len(pair_set)}")
                print(f"   • Classifier type: {final_classifier_type}")
                
            return {
                "task_name": task_name,
                "model_name": model_name,
                "error": error_msg,
                "training_samples": len(pair_set),
                "error_type": "training_failure",
                "suggestion": "Check data quality or try a different classifier type"
            }
        
        # Test the optimized classifier by generating responses and classifying them
        if optimize:
            if verbose:
                print(f"\n🧪 TESTING OPTIMIZED CLASSIFIER ON GENERATED RESPONSES:")
                print(f"   • Generating responses to test questions...")
            
            # Get test questions for response generation
            test_qa_pairs = []
            for doc in test_qa_pairs_source:
                try:
                    if from_csv or from_json:
                        # For CSV/JSON, doc is already a qa_pair dict
                        test_qa_pairs.append({
                            'question': doc['question'],
                            'formatted_question': doc['question'], 
                            'correct_answer': doc['correct_answer']
                        })
                    else:
                        # For lm-harness tasks, extract from document
                        raw_question = doc.get('question', str(doc))
                        if hasattr(task_data, 'doc_to_text'):
                            formatted_question = task_data.doc_to_text(doc)
                        else:
                            formatted_question = raw_question
                        
                        # Extract correct answer for ground truth comparison
                        correct_answers = doc.get('mc1_targets', {}).get('choices', [])
                        correct_labels = doc.get('mc1_targets', {}).get('labels', [])
                        
                        correct_answer = None
                        for i, label in enumerate(correct_labels):
                            if label == 1 and i < len(correct_answers):
                                correct_answer = correct_answers[i]
                                break
                        
                        if correct_answer:
                            test_qa_pairs.append({
                                'question': raw_question,
                                'formatted_question': formatted_question,
                                'correct_answer': correct_answer
                            })
                        
                except Exception as e:
                    continue
            
            if verbose:
                print(f"   • Successfully extracted {len(test_qa_pairs)} test questions")
                print(f"\n🔍 Test Questions:")
                for i, qa_pair in enumerate(test_qa_pairs):
                    print(f"\n   📋 Question {i+1}:")
                    print(f"      🔸 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}")
                    print(f"      ✅ Correct Answer: {qa_pair['correct_answer']}")
            
            # Generate responses and classify them
            if verbose:
                print(f"\n🎭 GENERATING AND CLASSIFYING RESPONSES:")
                print(f"   • Generating responses with optimized layer {layer}...")
            
            generated_responses = []
            correct_classifications = 0
            total_classifications = 0
            
            for i, qa_pair in enumerate(test_qa_pairs):
                if verbose:
                    print(f"\n   🎯 Generating response {i+1}:")
                    print(f"      📝 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}")
                
                # Use the raw question for natural generation
                simple_prompt = qa_pair['question']
                
                # Generate response with token-level scoring
                response, token_scores, classification = generate_with_classification(
                    model, simple_prompt, layer, max_new_tokens, steering_method, token_aggregation, final_threshold, verbose
                )
                
                # Evaluate the generated response using the ground truth evaluator
                try:
                    # Create ground truth evaluator
                    evaluator = GroundTruthEvaluator.from_string(ground_truth_method)
                    
                    # Get correct answer for comparison
                    correct_answers = qa_pair.get('correct_answer', '')
                    
                    # Get user label if available
                    user_label = None
                    if user_labels and i < len(user_labels):
                        user_label = user_labels[i]
                    
                    # Evaluate the response
                    evaluation_result = evaluator.evaluate_response(response, correct_answers, user_label)
                    
                    ground_truth = evaluation_result["ground_truth"]
                    
                    # Check if our classification matches ground truth (only if ground truth is not UNKNOWN)
                    classification_correct = None
                    if ground_truth != "UNKNOWN":
                        expected_classification = "HALLUCINATION" if ground_truth == "HALLUCINATION" else "TRUTHFUL"
                        classification_correct = (classification == expected_classification)
                        
                        if classification_correct:
                            correct_classifications += 1
                        total_classifications += 1
                    
                    generated_responses.append({
                        'question': qa_pair['question'],
                        'response': response,
                        'classification': classification,
                        'ground_truth': ground_truth,
                        'correct': classification_correct,
                        'token_scores': token_scores
                    })
                    
                    if verbose:
                        print(f"      🤖 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
                        print(f"      🎯 Classification: {classification}")
                        print(f"      📊 Token Scores: {[f'{score:.3f}' for score in token_scores[:10]]}{'...' if len(token_scores) > 10 else ''}")
                        print(f"      🔢 Aggregated Score: {aggregate_token_scores(token_scores, token_aggregation):.3f}")
                        print(f"      ✅ Ground Truth: {ground_truth}")
                        if classification_correct is not None:
                            print(f"      {'✅' if classification_correct else '❌'} Match: {classification_correct}")
                        
                except Exception as e:
                    if verbose:
                        print(f"      ❌ Error evaluating response: {e}")
                    continue
            
            # Calculate evaluation results
            if total_classifications > 0:
                test_accuracy = correct_classifications / total_classifications
                evaluation_results = {
                    "accuracy": test_accuracy,
                    "correct_predictions": correct_classifications,
                    "total_predictions": total_classifications
                }
            else:
                evaluation_results = {
                    "accuracy": "N/A",
                    "correct_predictions": 0,
                    "total_predictions": 0
                }
            
            if verbose:
                print(f"\n✅ Response generation and classification completed!")
                if total_classifications > 0:
                    print(f"   • Test accuracy: {test_accuracy:.2%} ({correct_classifications}/{total_classifications})")
                else:
                    print(f"   • Test accuracy: Could not evaluate")
                print(f"   • Tested on generated responses, not pre-written choices")
            
            # Create results dictionary for optimization path
            results = {
                "task_name": task_name,
                "model_name": model_name,
                "layer": layer,
                "original_layer": original_layer,
                "token_aggregation": token_aggregation,
                "original_token_aggregation": original_token_aggregation,
                "optimization_performed": optimize,
                "optimization_result": optimization_result,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "num_train": len(contrastive_pairs),
                "num_test": len(test_qa_pairs),
                "sample_responses": generated_responses,
                "classification_accuracy": correct_classifications / total_classifications if total_classifications > 0 else None,
                "correct_classifications": correct_classifications,
                "total_classifications": total_classifications
            }
            
            if verbose:
                print(f"\n🎉 OPTIMIZATION PIPELINE COMPLETED FOR {task_name.upper()}!")
                print(f"{'='*80}")
                print(f"📊 FINAL RESULTS:")
                print(f"   • Training samples: {len(contrastive_pairs)}")
                print(f"   • Test samples: {len(test_qa_pairs)}")
                print(f"   • Training accuracy: {training_results.get('accuracy', 'N/A'):.2%}")
                if total_classifications > 0:
                    print(f"   • Test accuracy: {test_accuracy:.2%} ({correct_classifications}/{total_classifications})")
                else:
                    print(f"   • Test accuracy: Could not evaluate")
                print(f"   • Generated responses: {len(generated_responses)}")
                print(f"   • Optimization performed: Yes")
                print(f"   • Best layer: {layer}")
                print(f"   • Best aggregation: {token_aggregation}")
                print(f"{'='*80}\n")
            
            logger.info(f"Optimization pipeline completed for {task_name}")
            return results
        else:
            # Only do pre-written validation when NOT optimizing
            if verbose:
                print(f"\n🧪 PREPARING TEST DATA:")
                print(f"   • Loading TruthfulQA test data with correct/incorrect answers...")
            
            # Get the actual test data with correct and incorrect answers
            test_qa_pairs = []
            for doc in test_qa_pairs_source:
                try:
                    if from_csv or from_json:
                        # For CSV/JSON, doc is already a qa_pair dict
                        test_qa_pairs.append({
                            'question': doc['question'],
                            'formatted_question': doc['question'],
                            'correct_answer': doc['correct_answer'],
                            'incorrect_answer': doc['incorrect_answer']
                        })
                    else:
                        # For lm-harness tasks, extract from document
                        raw_question = doc.get('question', str(doc))
                        if hasattr(task_data, 'doc_to_text'):
                            formatted_question = task_data.doc_to_text(doc)
                        else:
                            formatted_question = raw_question
                        
                        # Extract correct answer
                        correct_answers = doc.get('mc1_targets', {}).get('choices', [])
                        correct_labels = doc.get('mc1_targets', {}).get('labels', [])
                        
                        # Find the correct answer
                        correct_answer = None
                        for i, label in enumerate(correct_labels):
                            if label == 1 and i < len(correct_answers):
                                correct_answer = correct_answers[i]
                                break
                        
                        # Find an incorrect answer
                        incorrect_answer = None
                        for i, label in enumerate(correct_labels):
                            if label == 0 and i < len(correct_answers):
                                incorrect_answer = correct_answers[i]
                                break
                        
                        if correct_answer and incorrect_answer:
                            test_qa_pairs.append({
                                'question': raw_question,
                                'formatted_question': formatted_question,
                                'correct_answer': correct_answer,
                                'incorrect_answer': incorrect_answer
                            })
                        
                except Exception as e:
                    # Skip problematic docs
                    continue
            
            if verbose:
                print(f"   • Successfully extracted {len(test_qa_pairs)} test QA pairs")
                print(f"\n🔍 Test Examples:")
                for i, qa_pair in enumerate(test_qa_pairs[:3]):  # Show first 3
                    print(f"\n   📋 Test Example {i+1}:")
                    print(f"      🔸 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}")
                    print(f"      ✅ Correct Answer: {qa_pair['correct_answer']}")
            
            # Create test contrastive pairs using proper activation collection logic
            test_contrastive_pairs = collector.create_batch_contrastive_pairs(test_qa_pairs)
            
            test_processed_pairs = collector.collect_activations_batch(
                pairs=test_contrastive_pairs,
                layer_index=layer,
                device=device
            )
            
            # Convert to ContrastivePairSet format for evaluation
            test_phrase_pairs = []
            for pair in test_processed_pairs:
                # Create the full prompts for the pair set
                positive_full = f"{pair.prompt}{pair.positive_response}"
                negative_full = f"{pair.prompt}{pair.negative_response}"
                
                test_phrase_pairs.append({
                    "harmful": negative_full,  # A choice (incorrect)
                    "harmless": positive_full  # B choice (correct)
                })
            
            test_pair_set = ContrastivePairSet.from_phrase_pairs(
                name=f"{task_name}_test",
                phrase_pairs=test_phrase_pairs,
                task_type="lm_evaluation"
            )
            
            # Store the real test activations in the pair set response objects
            for i, processed_pair in enumerate(test_processed_pairs):
                if i < len(test_pair_set.pairs):
                    # Assign activations to the response objects, not the pair directly
                    if hasattr(test_pair_set.pairs[i], 'positive_response') and test_pair_set.pairs[i].positive_response:
                        test_pair_set.pairs[i].positive_response.activations = processed_pair.positive_activations
                    if hasattr(test_pair_set.pairs[i], 'negative_response') and test_pair_set.pairs[i].negative_response:
                        test_pair_set.pairs[i].negative_response.activations = processed_pair.negative_activations
            
            if verbose:
                print(f"\n📊 Evaluating classifier on test set...")
            evaluation_results = steering_method.evaluate(test_pair_set)
            
            if verbose:
                print(f"✅ Classifier validation completed!")
                print(f"   • Validated on pre-written answer choices only")
            
            # Generate sample responses with token-level classification
            if verbose:
                if optimize:
                    print(f"\n🎭 GENERATING SAMPLE RESPONSES WITH OPTIMIZED CLASSIFIER:")
                    print(f"   • Generating {min(5, len(test_qa_pairs))} sample responses with optimized layer {layer}...")
                else:
                    print(f"\n🎭 GENERATING SAMPLE RESPONSES WITH HALLUCINATION DETECTION:")
                    print(f"   • Generating {min(5, len(test_qa_pairs))} sample responses...")
            
            generated_responses = []
            correct_classifications = 0
            total_classifications = 0
            
            for i, qa_pair in enumerate(test_qa_pairs[:5]):  # Sample 5 responses
                if verbose and not optimize:  # Only show detailed progress when not optimizing
                    print(f"\n   🎯 Generating response {i+1}:")
                    print(f"      📝 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}")
                
                # Use the raw question for natural generation
                # The formatted_question contains few-shot examples which are for training, not generation
                simple_prompt = qa_pair['question']
                
                # Generate response with token-level scoring
                response, token_scores, classification = generate_with_classification(
                    model, simple_prompt, layer, max_new_tokens, steering_method, token_aggregation, final_threshold, verbose and not optimize
                )
                
                # Evaluate the generated response using the ground truth evaluator
                try:
                    # Create ground truth evaluator
                    evaluator = GroundTruthEvaluator.from_string(ground_truth_method)
                    
                    # Get correct answer for comparison
                    correct_answers = qa_pair.get('correct_answer', '')
                    
                    # Get user label if available
                    user_label = None
                    if user_labels and i < len(user_labels):
                        user_label = user_labels[i]
                    
                    # Evaluate the response
                    evaluation_result = evaluator.evaluate_response(response, correct_answers, user_label)
                    
                    ground_truth = evaluation_result["ground_truth"]
                    
                    # Check if our classification matches ground truth (only if ground truth is not UNKNOWN)
                    classification_correct = None
                    if ground_truth != "UNKNOWN":
                        classification_correct = (classification == ground_truth)
                        if classification_correct:
                            correct_classifications += 1
                        total_classifications += 1
                    
                    generated_responses.append({
                        'response': response,
                        'token_scores': token_scores,
                        'classification': classification,
                        'ground_truth': ground_truth,
                        'ground_truth_method': evaluation_result["method_used"],
                        'ground_truth_confidence': evaluation_result["confidence"],
                        'ground_truth_details': evaluation_result["details"],
                        'classification_correct': classification_correct
                    })
                    
                    if verbose and not optimize:  # Only show detailed output when not optimizing
                        print(f"      🤖 Generated: {response[:150]}{'...' if len(response) > 150 else ''}")
                        print(f"      🔍 Token Scores: {[f'{score:.3f}' for score in token_scores[:10]]}")
                        if len(token_scores) > 10:
                            print(f"                    ... ({len(token_scores)} total tokens)")
                        aggregated_score = aggregate_token_scores(token_scores, token_aggregation)
                        print(f"      📊 Our Classification: {classification} ({token_aggregation} score: {aggregated_score:.3f})")
                        print(f"      🎯 Ground Truth: {ground_truth} (method: {evaluation_result['method_used']}, confidence: {evaluation_result['confidence']:.2f})")
                        if classification_correct is not None:
                            print(f"      {'✅' if classification_correct else '❌'} Classification {'CORRECT' if classification_correct else 'WRONG'}")
                        else:
                            print(f"      ❓ Classification accuracy not evaluated (ground truth method: {evaluation_result['method_used']})")
                        print(f"      ✅ Expected: {qa_pair['correct_answer']}")
                        print(f"      ❌ Incorrect: {qa_pair['correct_answer']}")
                        if evaluation_result["details"]:
                            print(f"      📝 Details: {evaluation_result['details']}")
                    
                except Exception as e:
                    if verbose and not optimize:
                        print(f"      ⚠️  Could not evaluate response: {e}")
                    generated_responses.append({
                        'response': response,
                        'token_scores': token_scores,
                        'classification': classification,
                        'ground_truth': 'UNKNOWN',
                        'ground_truth_method': 'error',
                        'ground_truth_confidence': 0.0,
                        'ground_truth_details': f'Error during evaluation: {str(e)}',
                        'classification_correct': None
                    })
            
            # Show summary for optimization
            if verbose and optimize:
                print(f"\n   ✅ Generated {len(generated_responses)} responses with optimized layer {layer}")
                if total_classifications > 0:
                    classification_acc = correct_classifications / total_classifications
                    print(f"   📊 Classification accuracy: {classification_acc:.2%} ({correct_classifications}/{total_classifications})")
            
            results = {
                "task_name": task_name,
                "model_name": model_name,
                "layer": layer,
                "original_layer": original_layer,
                "token_aggregation": token_aggregation,
                "original_token_aggregation": original_token_aggregation,
                "optimization_performed": optimize,
                "optimization_result": optimization_result,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "num_train": len(contrastive_pairs),
                "num_test": len(test_qa_pairs),
                "sample_responses": generated_responses,
                "classification_accuracy": correct_classifications / total_classifications if total_classifications > 0 else None,
                "correct_classifications": correct_classifications,
                "total_classifications": total_classifications
            }
            
            if verbose:
                print(f"\n🎉 PIPELINE COMPLETED FOR {task_name.upper()}!")
                print(f"{'='*80}")
                print(f"📊 FINAL RESULTS:")
                print(f"   • Training samples: {len(contrastive_pairs)}")
                print(f"   • Test samples: {len(test_qa_pairs)}")
                print(f"   • Training accuracy: {training_results.get('accuracy', 'N/A'):.2%}")
                print(f"   • Generated responses: {len(generated_responses)}")
                if total_classifications > 0:
                    classification_acc = correct_classifications / total_classifications
                    print(f"   • Classification accuracy on generated responses: {classification_acc:.2%} ({correct_classifications}/{total_classifications})")
                else:
                    print(f"   • Classification accuracy: Could not evaluate")
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
    """Create a comprehensive evaluation report in markdown format."""
    try:
        with open(output_path, 'w') as f:
            f.write("# Wisent-Guard Evaluation Report\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Task | Training Accuracy | Evaluation Accuracy | Optimization |\n")
            f.write("|------|------------------|--------------------|--------------|\n")
            
            for task_name, task_results in results.items():
                if task_results is None:
                    f.write(f"| {task_name} | NULL | NULL | N/A |\n")
                elif isinstance(task_results, dict) and "error" in task_results:
                    f.write(f"| {task_name} | ERROR | ERROR | N/A |\n")
                elif isinstance(task_results, dict):
                    train_acc = task_results.get("training_results", {}).get("accuracy", "N/A")
                    eval_acc = task_results.get("evaluation_results", {}).get("accuracy", "N/A")
                    optimized = "Yes" if task_results.get("optimization_performed", False) else "No"
                    
                    if isinstance(train_acc, float):
                        train_acc = f"{train_acc:.2%}"
                    if isinstance(eval_acc, float):
                        eval_acc = f"{eval_acc:.2%}"
                    
                    f.write(f"| {task_name} | {train_acc} | {eval_acc} | {optimized} |\n")
            
            # Detailed results for each task
            for task_name, task_results in results.items():
                f.write(f"\n## {task_name}\n\n")
                
                if task_results is None:
                    f.write(f"**Error**: Task results are None\n")
                elif isinstance(task_results, dict) and "error" in task_results:
                    f.write(f"**Error**: {task_results['error']}\n")
                elif isinstance(task_results, dict):
                    # Configuration
                    f.write("### Configuration\n")
                    f.write(f"- **Model**: {task_results.get('model_name', 'Unknown')}\n")
                    f.write(f"- **Layer**: {task_results.get('layer', 'Unknown')}\n")
                    f.write(f"- **Classifier**: {task_results.get('classifier_type', 'Unknown')}\n")
                    f.write(f"- **Token Aggregation**: {task_results.get('token_aggregation', 'Unknown')}\n")
                    f.write(f"- **Ground Truth Method**: {task_results.get('ground_truth_method', 'Unknown')}\n")
                    
                    # Training results
                    if "training_results" in task_results:
                        train_results = task_results["training_results"]
                        f.write("\n### Training Results\n")
                        train_acc = train_results.get('accuracy', 'N/A')
                        if isinstance(train_acc, float):
                            f.write(f"- **Accuracy**: {train_acc:.2%}\n")
                        else:
                            f.write(f"- **Accuracy**: {train_acc}\n")
                        
                        train_prec = train_results.get('precision', 'N/A')
                        if isinstance(train_prec, float):
                            f.write(f"- **Precision**: {train_prec:.2f}\n")
                        else:
                            f.write(f"- **Precision**: {train_prec}\n")
                        
                        train_recall = train_results.get('recall', 'N/A')
                        if isinstance(train_recall, float):
                            f.write(f"- **Recall**: {train_recall:.2f}\n")
                        else:
                            f.write(f"- **Recall**: {train_recall}\n")
                        
                        train_f1 = train_results.get('f1', 'N/A')
                        if isinstance(train_f1, float):
                            f.write(f"- **F1 Score**: {train_f1:.2f}\n")
                        else:
                            f.write(f"- **F1 Score**: {train_f1}\n")
                    
                    # Evaluation results
                    if "evaluation_results" in task_results:
                        eval_results = task_results["evaluation_results"]
                        f.write("\n### Evaluation Results\n")
                        eval_acc = eval_results.get('accuracy', 'N/A')
                        if isinstance(eval_acc, float):
                            f.write(f"- **Accuracy**: {eval_acc:.2%}\n")
                        else:
                            f.write(f"- **Accuracy**: {eval_acc}\n")
                        f.write(f"- **Total Predictions**: {eval_results.get('total_predictions', 'N/A')}\n")
                        f.write(f"- **Correct Predictions**: {eval_results.get('correct_predictions', 'N/A')}\n")
                    
                    # Optimization results
                    if task_results.get("optimization_performed", False):
                        f.write("\n### Optimization Results\n")
                        f.write(f"- **Best Layer**: {task_results.get('best_layer', 'Unknown')}\n")
                        f.write(f"- **Best Aggregation**: {task_results.get('best_aggregation', 'Unknown')}\n")
                        best_acc = task_results.get('best_accuracy', 'Unknown')
                        if isinstance(best_acc, float):
                            f.write(f"- **Best Accuracy**: {best_acc:.2%}\n")
                        else:
                            f.write(f"- **Best Accuracy**: {best_acc}\n")
            
            f.write(f"\n---\n\n*Report generated on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        logger.info(f"Evaluation report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create report at {output_path}: {e}")


def main():
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate file input arguments
    if args.from_csv and args.from_json:
        print("Error: Cannot specify both --from-csv and --from-json")
        sys.exit(1)
    
    # Parse task names or file paths
    if args.from_csv or args.from_json:
        # Single file input
        file_path = args.task_names
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        task_sources = [file_path]
    else:
        # Traditional task names
        task_sources = [name.strip() for name in args.task_names.split(",")]
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"Starting wisent-guard harness for sources: {task_sources}")
    logger.info(f"Model: {args.model}, Layer: {args.layer}")
    
    try:
        # Run pipeline for each source
        all_results = {}
        
        for source in task_sources:
            logger.info(f"Processing source: {source}")
            
            # Determine if optimization should be enabled
            should_optimize = args.optimize or (args.auto_optimize and args.layer == -1)
            
            # Use layer -1 as a signal for auto-optimization
            if should_optimize:
                layer_to_use = -1  # Signal to the pipeline to optimize
            else:
                layer_to_use = args.layer if args.layer != -1 else 15  # Default to 15 if not specified
            
            # For file inputs, use the file path as task name
            if args.from_csv or args.from_json:
                display_name = f"file_{os.path.basename(source)}"
            else:
                display_name = source
            
            task_results = run_task_pipeline(
                task_name=source,
                model_name=args.model,
                layer=layer_to_use,
                shots=args.shots,
                split_ratio=args.split_ratio,
                limit=args.limit,
                classifier_type=args.classifier_type,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                seed=args.seed,
                token_aggregation=args.token_aggregation,
                ground_truth_method=args.ground_truth_method,
                user_labels=args.user_labels,
                optimize=should_optimize,
                optimize_layers=args.optimize_layers,
                optimize_metric=args.optimize_metric,
                optimize_max_combinations=args.optimize_max_combinations,
                verbose=args.verbose,
                from_csv=args.from_csv,
                from_json=args.from_json,
                question_col=args.question_col,
                correct_col=args.correct_col,
                incorrect_col=args.incorrect_col,
                allow_small_dataset=args.allow_small_dataset
            )
            
            all_results[display_name] = task_results
        
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
            if results is None:
                print(f"{task_name}: ERROR - Results are None")
            elif isinstance(results, dict) and "error" in results:
                print(f"{task_name}: ERROR - {results['error']}")
            elif isinstance(results, dict):
                training_acc = results.get("training_results", {}).get("accuracy", "N/A")
                eval_acc = results.get("evaluation_results", {}).get("accuracy", "N/A")
                if isinstance(training_acc, float) and isinstance(eval_acc, float):
                    print(f"{task_name}: Train={training_acc:.2%} | Test={eval_acc:.2%}")
                else:
                    print(f"{task_name}: Train={training_acc} | Test={eval_acc}")
            else:
                print(f"{task_name}: ERROR - Invalid results type: {type(results)}")
        
        print(f"\nResults saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 