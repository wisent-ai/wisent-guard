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
  python -m wisent_guard tasks truthfulqa --layer 15 --model meta-llama/Llama-3.1-8B --token-aggregation final
  python -m wisent_guard tasks hellaswag,mmlu --layer 10 --model meta-llama/Llama-3.1-8B --shots 5 --token-aggregation max
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
    parser.add_argument("--token-aggregation", type=str, choices=["average", "final", "first", "max", "min"], 
                       default="average", help="How to aggregate token scores for classification")
    
    return parser


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


def generate_with_classification(model, prompt, layer, max_new_tokens, steering_method, token_aggregation="average", verbose=False):
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
        classification = "HALLUCINATION" if aggregated_score > 0.6 else "TRUTHFUL"
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
        token_aggregation: How to aggregate token scores for classification
        
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
        print(f"   • Token aggregation: {token_aggregation}")
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
        
        # Create training data using proper activation collection
        from .core.activations import ActivationCollectionLogic, Activations, ActivationAggregationMethod
        
        if verbose:
            print(f"\n📝 TRAINING DATA PREPARATION:")
            print(f"   • Loading TruthfulQA data with correct/incorrect answers...")
        
        # Get the actual TruthfulQA data with correct and incorrect answers
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
        
        if verbose:
            print(f"   • Successfully extracted {len(qa_pairs)} QA pairs")
            print(f"\n🔍 Training Examples:")
            for i, qa_pair in enumerate(qa_pairs[:5]):  # Show first 5
                print(f"\n   📋 Example {i+1}:")
                print(f"      🔸 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}")
                print(f"      ✅ Correct Answer: {qa_pair['correct_answer']}")
                print(f"      ❌ Incorrect Answer: {qa_pair['incorrect_answer']}")
        
        # Create contrastive pairs using proper activation collection logic
        collector = ActivationCollectionLogic(model=model)
        contrastive_pairs = collector.create_batch_contrastive_pairs(qa_pairs)
        
        if verbose:
            print(f"\n🔄 Created {len(contrastive_pairs)} contrastive pairs:")
            for i, pair in enumerate(contrastive_pairs[:3]):  # Show first 3
                print(f"\n   🔄 Contrastive Pair {i+1}:")
                print(f"      📝 Prompt: {pair.prompt[:100]}{'...' if len(pair.prompt) > 100 else ''}")
                print(f"      🟢 Positive (B): {pair.positive_response}")
                print(f"      🔴 Negative (A): {pair.negative_response}")
        
        # Extract activations from the choice tokens
        if verbose:
            print(f"\n🔬 Extracting activations from layer {layer} choice tokens...")
        
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
        
        # Evaluate on test set using proper activation collection
        if verbose:
            print(f"\n🧪 PREPARING TEST DATA:")
            print(f"   • Loading TruthfulQA test data with correct/incorrect answers...")
        
        # Get the actual TruthfulQA test data with correct and incorrect answers
        test_qa_pairs = []
        for doc in test_docs:
            try:
                # Extract question - use both raw and formatted versions
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
                        'question': raw_question,  # Raw question for display
                        'formatted_question': formatted_question,  # Formatted for generation
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
                print(f"      ❌ Incorrect Answer: {qa_pair['incorrect_answer']}")
        
        # Create test contrastive pairs using proper activation collection logic
        test_contrastive_pairs = collector.create_batch_contrastive_pairs(test_qa_pairs)
        
        if verbose:
            print(f"\n🔄 Created {len(test_contrastive_pairs)} test contrastive pairs:")
            for i, pair in enumerate(test_contrastive_pairs[:2]):  # Show first 2
                print(f"\n   🔄 Test Pair {i+1}:")
                print(f"      📝 Prompt: {pair.prompt[:100]}{'...' if len(pair.prompt) > 100 else ''}")
                print(f"      🟢 Positive (B): {pair.positive_response}")
                print(f"      🔴 Negative (A): {pair.negative_response}")
        
        # Extract activations from the test choice tokens
        if verbose:
            print(f"\n🔬 Extracting test activations from layer {layer} choice tokens...")
        
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
            print(f"\n🎭 GENERATING SAMPLE RESPONSES WITH HALLUCINATION DETECTION:")
            print(f"   • Generating {min(5, len(test_qa_pairs))} sample responses...")
        
        generated_responses = []
        correct_classifications = 0
        total_classifications = 0
        
        for i, qa_pair in enumerate(test_qa_pairs[:5]):  # Sample 5 responses
            if verbose:
                print(f"\n   🎯 Generating response {i+1}:")
                print(f"      📝 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}")
            
            # Use the raw question for natural generation
            # The formatted_question contains few-shot examples which are for training, not generation
            simple_prompt = qa_pair['question']
            
            # Generate response with token-level scoring
            response, token_scores, classification = generate_with_classification(
                model, simple_prompt, layer, max_new_tokens, steering_method, token_aggregation, verbose
            )
            
            # Evaluate the generated response using lm-eval harness
            try:
                # Use lm-eval's evaluation logic to check if response is correct
                from lm_eval.api.metrics import exact_match_hf_evaluate
                
                # Get the correct answers for comparison
                correct_answers = qa_pair.get('correct_answer', '')
                
                # Evaluate if the generated response matches the correct answer
                # Handle the case where exact_match_hf_evaluate might return different formats
                try:
                    is_actually_correct = exact_match_hf_evaluate([response], [correct_answers])
                    
                    # Handle different return formats from exact_match_hf_evaluate
                    if isinstance(is_actually_correct, dict):
                        # It returns {'exact_match': 0.0 or 1.0}
                        is_correct = bool(is_actually_correct.get('exact_match', 0))
                    elif isinstance(is_actually_correct, (list, tuple)) and len(is_actually_correct) > 0:
                        is_correct = bool(is_actually_correct[0])
                    else:
                        is_correct = bool(is_actually_correct)
                except:
                    # Fallback: simple string matching
                    is_correct = correct_answers.lower() in response.lower()
                
                ground_truth = "TRUTHFUL" if is_correct else "HALLUCINATION"
                
                # Check if our classification matches ground truth
                classification_correct = (classification == ground_truth)
                if classification_correct:
                    correct_classifications += 1
                total_classifications += 1
                
                generated_responses.append({
                    'response': response,
                    'token_scores': token_scores,
                    'classification': classification,
                    'ground_truth': ground_truth,
                    'classification_correct': classification_correct
                })
                
                if verbose:
                    print(f"      🤖 Generated: {response[:150]}{'...' if len(response) > 150 else ''}")
                    print(f"      🔍 Token Scores: {[f'{score:.3f}' for score in token_scores[:10]]}")
                    if len(token_scores) > 10:
                        print(f"                    ... ({len(token_scores)} total tokens)")
                    aggregated_score = aggregate_token_scores(token_scores, token_aggregation)
                    print(f"      📊 Our Classification: {classification} ({token_aggregation} score: {aggregated_score:.3f})")
                    print(f"      🎯 Ground Truth: {ground_truth}")
                    print(f"      {'✅' if classification_correct else '❌'} Classification {'CORRECT' if classification_correct else 'WRONG'}")
                    print(f"      ✅ Expected: {qa_pair['correct_answer']}")
                    print(f"      ❌ Incorrect: {qa_pair['incorrect_answer']}")
                    
            except Exception as e:
                if verbose:
                    print(f"      ⚠️  Could not evaluate response: {e}")
                generated_responses.append({
                    'response': response,
                    'token_scores': token_scores,
                    'classification': classification,
                    'ground_truth': 'UNKNOWN',
                    'classification_correct': None
                })
        
        results = {
            "task_name": task_name,
            "model_name": model_name,
            "layer": layer,
            "token_aggregation": token_aggregation,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "num_train": len(train_docs),
            "num_test": len(test_docs),
            "sample_responses": generated_responses,
            "classification_accuracy": correct_classifications / total_classifications if total_classifications > 0 else None,
            "correct_classifications": correct_classifications,
            "total_classifications": total_classifications
        }
        
        if verbose:
            print(f"\n🎉 PIPELINE COMPLETED FOR {task_name.upper()}!")
            print(f"{'='*80}")
            print(f"📊 FINAL RESULTS:")
            print(f"   • Training samples: {len(train_docs)}")
            print(f"   • Test samples: {len(test_docs)}")
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
                token_aggregation=args.token_aggregation,
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