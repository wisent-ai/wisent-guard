#!/usr/bin/env python
"""
Evaluation script for meta-llama/Llama-3.1-8B-Instruct using TruthfulQA dataset.

This script:
1. Loads a TruthfulQA dataset split into training and evaluation sets
2. Trains the wisent-guard activation monitor on hallucination/non-hallucination pairs
3. Can train a classifier directly for hallucination detection
4. Evaluates the model's performance on the test set
5. Reports metrics on hallucination detection and prevention

The script supports both threshold-based and classifier-based approaches
for hallucination detection.
"""

import argparse
import csv
import os
import time
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Now import wisent_guard
from wisent_guard import ActivationGuard
from wisent_guard.classifier import ActivationClassifier
from .evaluate_string import call_claude_and_evaluate

def format_prompt(message, system_prompt=None):
    """Format prompt for Llama 3.1 Instruct models using the special tokens"""
    # Llama 3.1 special tokens
    BEGIN_TEXT = "<|begin_of_text|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    EOT = "<|eot_id|>"
    
    if system_prompt:
        # Format with system prompt
        formatted = f"{BEGIN_TEXT}{START_HEADER}system{END_HEADER}\n{system_prompt}{EOT}{START_HEADER}user{END_HEADER}\n{message}{EOT}{START_HEADER}assistant{END_HEADER}"
    else:
        # Format without system prompt
        formatted = f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{message}{EOT}{START_HEADER}assistant{END_HEADER}"
    
    return formatted

def load_truthfulqa_data(filepath):
    """Load TruthfulQA dataset"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples from {filepath}")
    return df

def create_training_pairs(train_df, max_pairs=0):
    """
    Create training pairs for the activation guard.
    Each pair consists of:
    - harmful: A question-incorrect answer pair (hallucination)
    - harmless: The same question with a correct answer
    
    Using proper Llama chat formatting with user/assistant tags.
    
    Args:
        train_df: DataFrame containing TruthfulQA training data
        max_pairs: Maximum number of pairs to create (0 for all)
    """
    training_pairs = []
    
    # If max_pairs is set, sample the dataframe
    if max_pairs > 0 and max_pairs < len(train_df):
        train_df = train_df.sample(max_pairs, random_state=42)
        print(f"Sampling {max_pairs} examples from training data")
    
    for _, row in train_df.iterrows():
        question = row['Question']
        
        # Get correct answers
        correct_answers = row['Correct Answers'].split(';')
        best_answer = row['Best Answer']
        
        # Get incorrect answers
        incorrect_answers = row['Incorrect Answers'].split(';')
        
        # Use the best answer if available, otherwise use the first correct answer
        harmless_answer = best_answer if best_answer else correct_answers[0]
        
        # Use the first incorrect answer
        harmful_answer = incorrect_answers[0] if incorrect_answers else None
        
        if harmful_answer and harmless_answer:
            # Format with proper user/assistant tags to match Llama 3.1 chat format
            training_pairs.append({
                "harmful": f"{question}{harmful_answer}",
                "harmless": f"{question}{harmless_answer}"
            })
    
    print(f"Created {len(training_pairs)} training pairs")
    return training_pairs

def evaluate_model(model, tokenizer, eval_df, device, max_tokens=100, temperature=0.1, max_eval_samples=0, use_claude=True):
    """
    Evaluate model on TruthfulQA dataset without using the guard.
    This serves as a baseline for comparison.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        eval_df: DataFrame containing TruthfulQA evaluation data
        device: Device to run inference on
        max_tokens: Maximum tokens to generate per response
        temperature: Temperature for sampling
        max_eval_samples: Maximum number of samples to evaluate (0 for all)
        use_claude: Whether to use Claude for evaluation (if False, leaves human_score empty)
    """
    # Check for Claude API key if using Claude
    if use_claude and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set. Claude evaluation is required.")
        print("Please set the environment variable with: export ANTHROPIC_API_KEY='your-api-key-here'")
        return []
    
    # If max_eval_samples is specified, sample the dataframe
    if max_eval_samples > 0 and max_eval_samples < len(eval_df):
        eval_df = eval_df.sample(max_eval_samples, random_state=42)
        print(f"Sampling {max_eval_samples} examples for baseline evaluation")
    
    results = []
    
    for i, row in eval_df.iterrows():
        question = row['Question']
        best_answer = row['Best Answer'] if not pd.isna(row['Best Answer']) else row['Correct Answers'].split(';')[0]
        incorrect_answer = row['Incorrect Answers'].split(';')[0] if not pd.isna(row['Incorrect Answers']) else "No answer available"
        
        # Format prompt using our format_prompt function instead of hardcoding
        prompt_text = f"Answer the following question accurately: {question}"
        formatted_prompt = format_prompt(prompt_text)
        
        # Generate response using same parameters as guard
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
        
        # Set up generation parameters to match the guard
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Only add temperature if it's not the default
        if temperature != 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                **gen_kwargs
            )
        
        # Decode and extract response - matching the guard's extraction method
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1].strip()
        else:
            response = generated_text
        
        # Get evaluation score - either from Claude or empty for human annotation
        if use_claude:
            try:
                human_score = call_claude_and_evaluate(
                    actual_response=response,
                    good_response=best_answer,
                    bad_response=incorrect_answer
                )
                print(f"Claude evaluation for question {i+1}: {human_score}")
            except Exception as e:
                print(f"Error during Claude evaluation: {e}")
                human_score = -1  # Error case
        else:
            human_score = ""
            print(f"Added empty human_score for question {i+1}")
        
        results.append({
            'question': question,
            'response': response,
            'human_score': human_score,
            'category': row['Category']
        })
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(eval_df)} samples")
    
    return results

def evaluate_model_with_guard(guard, eval_df, max_tokens=100, max_eval_samples=0, use_claude=True):
    """
    Evaluate model on TruthfulQA dataset using the guard to detect and prevent hallucinations.
    
    Args:
        guard: The ActivationGuard instance
        eval_df: DataFrame containing TruthfulQA evaluation data
        max_tokens: Maximum tokens to generate per response
        max_eval_samples: Maximum number of samples to evaluate (0 for all)
        use_claude: Whether to use Claude for evaluation (if False, leaves human_score empty)
    """
    # Check for Claude API key if using Claude
    if use_claude and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set. Claude evaluation is required.")
        print("Please set the environment variable with: export ANTHROPIC_API_KEY='your-api-key-here'")
        return []
    
    # If max_eval_samples is specified, sample the dataframe
    if max_eval_samples > 0 and max_eval_samples < len(eval_df):
        eval_df = eval_df.sample(max_eval_samples, random_state=42)
        print(f"Sampling {max_eval_samples} examples for evaluation with guard")
    
    results = []
    
    for i, row in eval_df.iterrows():
        question = row['Question']
        best_answer = row['Best Answer'] if not pd.isna(row['Best Answer']) else row['Correct Answers'].split(';')[0]
        incorrect_answer = row['Incorrect Answers'].split(';')[0] if not pd.isna(row['Incorrect Answers']) else "No answer available"
        
        # Format prompt using our format_prompt function instead of hardcoding
        prompt_text = f"Answer the following question accurately: {question}"
        formatted_prompt = format_prompt(prompt_text)
        
        # Check if the guard detects potential hallucination in the formatted prompt
        is_harmful = guard.is_harmful(formatted_prompt)
        similarity_score = guard.get_similarity(formatted_prompt) if is_harmful else None
        
        # Generate safe response - ensure token scores are captured
        result = guard.generate_safe_response(
            prompt_text,
            max_new_tokens=max_tokens
        )
        
        response = result.get('response', '')
        blocked = result.get('blocked', False)
        reason = result.get('reason', '') if blocked else ''
        token_scores = result.get('token_scores', [])  # Get token scores (if available by default)
        
        # Get evaluation score - either from Claude or empty for human annotation
        if use_claude:
            try:
                human_score = call_claude_and_evaluate(
                    actual_response=response,
                    good_response=best_answer,
                    bad_response=incorrect_answer
                )
                print(f"Claude evaluation for guarded question {i+1}: {human_score}")
            except Exception as e:
                print(f"Error during Claude evaluation: {e}")
                human_score = -1  # Error case
        else:
            human_score = ""
            print(f"Added empty human_score for guarded question {i+1}")
        
        results.append({
            'question': question,
            'response': response,
            'human_score': human_score,
            'category': row['Category'],
            'is_harmful': is_harmful,
            'similarity_score': similarity_score,
            'blocked': blocked,
            'reason': reason,
            'token_scores': token_scores  # Store token scores
        })
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(eval_df)} samples with guard")
    
    return results

def compute_metrics(results):
    """Compute evaluation metrics"""
    if not results:
        return {
            'total_samples': 0,
            'human_correct': 0,
            'human_incorrect': 0,
            'human_ambiguous': 0,
            'human_correct_rate': 0,
            'human_incorrect_rate': 0,
            'categories': {}
        }
        
    total = len(results)
    
    # Calculate human-based metrics (may be populated by Claude)
    human_correct = sum(1 for r in results if r['human_score'] == 1)
    human_incorrect = sum(1 for r in results if r['human_score'] == 0)
    human_ambiguous = sum(1 for r in results if r['human_score'] == -1 or r['human_score'] == "")
    
    metrics = {
        'total_samples': total,
        'human_correct': human_correct,
        'human_incorrect': human_incorrect,
        'human_ambiguous': human_ambiguous,
        'human_correct_rate': human_correct / total if total > 0 else 0,
        'human_incorrect_rate': human_incorrect / total if total > 0 else 0,
    }
    
    # Compute metrics by category
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            category_metrics = {
                'total': 0,
                'human_correct': 0,
                'human_incorrect': 0,
                'human_ambiguous': 0
            }
            categories[cat] = category_metrics
        
        categories[cat]['total'] += 1
        
        # Add human metrics by category
        human_score = r['human_score']
        if human_score == 1:
            categories[cat]['human_correct'] += 1
        elif human_score == 0:
            categories[cat]['human_incorrect'] += 1
        else:
            categories[cat]['human_ambiguous'] += 1
    
    # Calculate rates for each category
    for cat in categories:
        cat_total = categories[cat]['total']
        categories[cat]['human_correct_rate'] = categories[cat]['human_correct'] / cat_total if cat_total > 0 else 0
        categories[cat]['human_incorrect_rate'] = categories[cat]['human_incorrect'] / cat_total if cat_total > 0 else 0
    
    metrics['categories'] = categories
    
    # For guard results, compute additional metrics
    if 'is_harmful' in results[0]:
        harmful_detected = sum(1 for r in results if r['is_harmful'])
        blocked_count = sum(1 for r in results if r.get('blocked', False))
        
        # Compute correlation between detection and human scores
        harmful_and_human_incorrect = sum(1 for r in results if r['is_harmful'] and r['human_score'] == 0)
        not_harmful_and_human_correct = sum(1 for r in results if not r['is_harmful'] and r['human_score'] == 1)
        
        metrics['harmful_detected'] = harmful_detected
        metrics['harmful_rate'] = harmful_detected / total if total > 0 else 0
        metrics['blocked_count'] = blocked_count
        metrics['blocked_rate'] = blocked_count / total if total > 0 else 0
        
        metrics['harmful_and_human_incorrect'] = harmful_and_human_incorrect
        metrics['harmful_and_human_incorrect_rate'] = harmful_and_human_incorrect / human_incorrect if human_incorrect > 0 else 0
        
        metrics['not_harmful_and_human_correct'] = not_harmful_and_human_correct
        metrics['not_harmful_and_human_correct_rate'] = not_harmful_and_human_correct / human_correct if human_correct > 0 else 0
    
    return metrics

def print_metrics(metrics, title="Model Evaluation Metrics"):
    """Print metrics in a readable format"""
    print(f"\n===== {title} =====")
    print(f"Total samples: {metrics['total_samples']}")
    
    # Human-based metrics (placeholders)
    print("\n* Human-based Evaluation Metrics *")
    print(f"Correct responses (Human): {metrics['human_correct']} ({metrics['human_correct_rate']:.2%})")
    print(f"Incorrect responses (Human): {metrics['human_incorrect']} ({metrics['human_incorrect_rate']:.2%})")
    print(f"Ambiguous responses: {metrics['human_ambiguous']} ({metrics['human_ambiguous'] / metrics['total_samples']:.2%})")
    
    if 'harmful_detected' in metrics:
        print(f"\nHarmful (hallucination) detection:")
        print(f"Hallucinations detected: {metrics['harmful_detected']} ({metrics['harmful_rate']:.2%})")
        print(f"Responses blocked: {metrics['blocked_count']} ({metrics['blocked_rate']:.2%})")
    
    # Show category results
    print("\nResults by category (Human-based):")
    for cat, values in metrics['categories'].items():
        print(f"  {cat}: {values['human_correct']} correct / {values['total']} total ({values['human_correct_rate']:.2%})")

def save_results(results, filename):
    """Save results to a CSV file with token scores"""
    # Create a list of dictionaries for CSV output
    csv_results = []
    
    for result in results:
        # Create a copy of the result dictionary without token_scores
        result_copy = {k: v for k, v in result.items() if k != 'token_scores'}
        
        # Process token scores if they exist
        if 'token_scores' in result and result['token_scores']:
            # Format token scores as string
            token_data = []
            for token in result['token_scores']:
                # Format: position:token_id:token_text:similarity:category:is_harmful
                token_text = token.get('token_text', '').replace('\n', '\\n').replace(',', '\\,')
                token_info = f"{token.get('position', '')}:{token.get('token_id', '')}:{token_text}:{token.get('similarity', 0.0):.6f}:{token.get('category', '')}:{token.get('is_harmful', False)}"
                token_data.append(token_info)
            
            # Add formatted token data to result
            result_copy['token_scores'] = '|'.join(token_data)
        else:
            result_copy['token_scores'] = ''
        
        csv_results.append(result_copy)
    
    # Write to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    
    print(f"Results saved to {filename}")

def save_combined_results(baseline_results, guard_results, filename):
    """Save combined baseline and guard results to a CSV file"""
    combined_results = []
    
    # Create a mapping of questions to their indices
    question_map = {result['question']: i for i, result in enumerate(baseline_results)}
    
    for guard_result in guard_results:
        question = guard_result['question']
        if question in question_map:
            baseline_idx = question_map[question]
            baseline_result = baseline_results[baseline_idx]
            
            # Process token scores if they exist
            token_scores_formatted = ""
            if 'token_scores' in guard_result and guard_result['token_scores']:
                # Format token scores as string
                token_data = []
                for token in guard_result['token_scores']:
                    # Format: position:token_id:token_text:similarity:category:is_harmful
                    token_text = token.get('token_text', '').replace('\n', '\\n').replace(',', '\\,')
                    token_info = f"{token.get('position', '')}:{token.get('token_id', '')}:{token_text}:{token.get('similarity', 0.0):.6f}:{token.get('category', '')}:{token.get('is_harmful', False)}"
                    token_data.append(token_info)
                
                token_scores_formatted = '|'.join(token_data)
            
            # Create combined entry with both baseline and guard results
            combined_entry = {
                'question': question,
                'category': guard_result['category'],
                'baseline_response': baseline_result['response'],
                'guard_response': guard_result['response'],
                'baseline_human_score': baseline_result['human_score'],
                'guard_human_score': guard_result['human_score'],
                'is_harmful': guard_result.get('is_harmful', False),
                'similarity_score': guard_result.get('similarity_score', None),
                'blocked': guard_result.get('blocked', False),
                'block_reason': guard_result.get('reason', ''),
                'token_scores': token_scores_formatted
            }
                
            combined_results.append(combined_entry)
    
    # Save the combined results
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=combined_results[0].keys())
        writer.writeheader()
        writer.writerows(combined_results)
    print(f"Combined results saved to {filename}")

def save_metrics(metrics, filename):
    """Save metrics to a JSON file"""
    import json
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filename}")

def collect_activations_for_training(guard, train_df, max_pairs=0):
    """
    Collect activations from the model for training a classifier.
    
    Args:
        guard: The ActivationGuard instance
        train_df: DataFrame containing TruthfulQA training data
        max_pairs: Maximum number of pairs to use (0 for all)
    
    Returns:
        Dictionary with harmful and harmless activation lists
    """
    # Ensure monitor is initialized
    if guard.monitor is None:
        guard._initialize_monitor_and_inference()
    
    # Use the same training pairs creation logic but collect activations
    training_pairs = create_training_pairs(train_df, max_pairs)
    
    print(f"\nCollecting activations for {len(training_pairs)} training pairs...")
    
    harmful_activations = []
    harmless_activations = []
    
    for i, pair in enumerate(training_pairs):
        # Process harmful example
        try:
            # Reset monitor before collecting activations
            guard.monitor.reset()
            
            # Use _prepare_activations to generate tokens and capture activations
            harmful_text = pair["harmful"]
            success = guard._prepare_activations(harmful_text)
            
            if not success:
                print(f"Failed to generate activations for harmful example {i+1}")
                continue
            
            # Get activations for all monitored layers
            activations = guard.monitor.get_activations()
            if not activations:
                print(f"No activations collected for harmful example {i+1}")
                continue
                
            for layer in guard.layers:
                # Get activation tensor for this layer - only using the "last" token
                activation_tensor = activations.get(layer)
                
                if activation_tensor is not None:
                    # Convert tensor to flattened numpy array for classification
                    tensor_data = activation_tensor.detach().cpu().numpy().flatten()
                    
                    # Create a dictionary with activation data for the classifier
                    activation_dict = {
                        "activations": tensor_data,
                        "layer": layer,
                        "token_text": "harmful",  # Label for classification
                        "is_harmful": True
                    }
                    harmful_activations.append(activation_dict)
                else:
                    print(f"Warning: No activations for layer {layer} in harmful example")
        
        except Exception as e:
            print(f"Error processing harmful example {i+1}: {e}")
            continue
        
        # Process harmless example
        try:
            # Reset monitor before collecting activations
            guard.monitor.reset()
            
            # Use _prepare_activations to generate tokens and capture activations
            harmless_text = pair["harmless"]
            success = guard._prepare_activations(harmless_text)
            
            if not success:
                print(f"Failed to generate activations for harmless example {i+1}")
                continue
            
            # Get activations for all monitored layers
            activations = guard.monitor.get_activations()
            if not activations:
                print(f"No activations collected for harmless example {i+1}")
                continue
                
            for layer in guard.layers:
                # Get activation tensor for this layer - only using the "last" token
                activation_tensor = activations.get(layer)
                
                if activation_tensor is not None:
                    # Convert tensor to flattened numpy array for classification
                    tensor_data = activation_tensor.detach().cpu().numpy().flatten()
                    
                    # Create a dictionary with activation data for the classifier
                    activation_dict = {
                        "activations": tensor_data,
                        "layer": layer,
                        "token_text": "harmless",  # Label for classification
                        "is_harmful": False
                    }
                    harmless_activations.append(activation_dict)
                else:
                    print(f"Warning: No activations for layer {layer} in harmless example")
        
        except Exception as e:
            print(f"Error processing harmless example {i+1}: {e}")
            continue
        
        # Print progress
        if (i + 1) % 10 == 0 or i == len(training_pairs) - 1:
            print(f"Collected activations for {i + 1}/{len(training_pairs)} pairs. "
                  f"Total: {len(harmful_activations)} harmful, {len(harmless_activations)} harmless activations")
    
    return {
        "harmful_activations": harmful_activations,
        "harmless_activations": harmless_activations
    }

def train_classifier(harmful_activations, harmless_activations, args):
    """
    Train a classifier for hallucination detection using collected activations.
    
    Args:
        harmful_activations: List of activation dictionaries from harmful examples
        harmless_activations: List of activation dictionaries from harmless examples
        args: Command-line arguments
        
    Returns:
        Trained ActivationClassifier instance
    """
    print(f"\nTraining classifier with {len(harmful_activations)} harmful and "
          f"{len(harmless_activations)} harmless activations...")
    
    # Ensure the model directory exists
    classifier_dir = os.path.dirname(args.classifier_path)
    if classifier_dir:
        os.makedirs(classifier_dir, exist_ok=True)
    
    # Create and train the classifier
    classifier = ActivationClassifier.create_from_activations(
        harmful_activations=harmful_activations,
        harmless_activations=harmless_activations,
        model_type=args.classifier_model,
        save_path=args.classifier_path,
        threshold=args.classifier_threshold,
        positive_class_label="harmful",
        test_size=0.2,  # Use 20% for validation
        random_state=42  # For reproducibility
    )
    
    print(f"Classifier trained and saved to {args.classifier_path}")
    return classifier

def test_classifier(args, model_name, model, tokenizer, device, eval_df=None):
    """Test the classifier with real examples from the dataset."""
    print("\n----- TESTING CLASSIFIER WITH REAL EXAMPLES -----")
    
    # Load the classifier
    classifier_path = args.classifier_path
    classifier_threshold = args.classifier_threshold
    
    # Create a guard with the classifier
    guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=[args.layer_number],
        device=device,
        threshold=args.threshold,
        save_dir=args.save_dir,
        use_classifier=True,
        classifier_path=classifier_path,
        classifier_threshold=classifier_threshold,
        log_level="info"
    )
    
    # If eval_df wasn't provided, try to load it
    if eval_df is None:
        try:
            eval_df = load_truthfulqa_data(args.eval_data)
        except Exception as e:
            print(f"Error loading evaluation data: {e}")
            return False
    
    # Select 2 random examples from the dataset
    if len(eval_df) > 1:
        test_samples = eval_df.sample(2, random_state=42)
    else:
        test_samples = eval_df
    
    print("\n==== TESTING WITH ACTUAL DATASET EXAMPLES ====")
    
    for idx, row in test_samples.iterrows():
        question = row['Question']
        
        # Get correct answers
        correct_answers = row['Correct Answers'].split(';')
        best_answer = row['Best Answer'] if not pd.isna(row['Best Answer']) else correct_answers[0]
        
        # Get incorrect answers
        incorrect_answers = row['Incorrect Answers'].split(';')
        if len(incorrect_answers) > 0 and not pd.isna(incorrect_answers[0]):
            incorrect_answer = incorrect_answers[0]
        else:
            incorrect_answer = "No incorrect answer available"
        
        print(f"\nQuestion: {question}")
        print(f"Category: {row['Category']}")
        print(f"Best Answer (truthful): {best_answer}")
        print(f"Incorrect Answer (hallucination): {incorrect_answer}")
        
        # Test with truthful example
        print(f"\nTesting truthful example...")
        truthful_result = guard.is_harmful(best_answer)
        print(f"Classification: {'HARMFUL' if truthful_result else 'NOT HARMFUL'}")
        
        # Test with hallucinated example
        print(f"\nTesting hallucinated example...")
        hallucinated_result = guard.is_harmful(incorrect_answer)
        print(f"Classification: {'HARMFUL' if hallucinated_result else 'NOT HARMFUL'}")
        
        # Test with response generation
        print("\nTesting response generation:")
        print(f"Prompt: '{question}'")
        
        try:
            response = guard.generate_safe_response(
                prompt=question,
                max_new_tokens=50
            )
            
            print(f"Response: '{response['response']}'")
            print(f"Blocked: {response.get('blocked', False)}")
            if response.get('blocked', False):
                print(f"Reason: {response.get('reason', 'Unknown')}")
                
            # Compare to truthful answer
            print(f"\nComparison to best answer:")
            print(f"Best answer: '{best_answer}'")
            print(f"Generated: '{response['response']}'")
        except Exception as e:
            print(f"Error generating response: {e}")
    
    return True

def main(args):
    # Check for Claude API key, unless --no-eval or --no-claude is set
    if not args.no_eval and not args.no_claude and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set. Claude evaluation is required.")
        print("Please set the environment variable with: export ANTHROPIC_API_KEY='your-api-key-here'")
        print("Or use --no-eval flag to skip evaluation (for testing purposes only)")
        print("Or use --no-claude flag to use empty human scores instead of Claude")
        return
        
    # Use Claude for evaluation unless --no-claude is specified
    use_claude = not args.no_claude
    
    # Device configuration
    if args.cpu_only:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available() and not args.no_mps:
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Model configuration
    print(f"Loading model: {args.model_name}")
    load_kwargs = {
        "torch_dtype": torch.float16 if args.half_precision else torch.float32,
    }
    
    # Configure device mapping based on device type
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    elif device == "mps":
        # For MPS, we'll load on CPU first then move to MPS
        # as direct loading to MPS can cause issues with some models
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = device
    
    if args.load_in_8bit and device != "mps":  # 8-bit quantization not supported on MPS
        load_kwargs["load_in_8bit"] = True
    elif args.load_in_8bit and device == "mps":
        print("Warning: 8-bit quantization not supported on MPS. Falling back to standard precision.")
        
    # Initialize model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **load_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Move model to MPS if needed (after loading)
        if device == "mps" and load_kwargs["device_map"] == "cpu":
            model = model.to(device)
            
        print(f"Model loaded successfully: {model.__class__.__name__}")
        
        # Calculate model parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {(param_count / 1e9):.2f} billion parameters")
        
        # Get number of layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            print(f"Number of layers: {num_layers}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load TruthfulQA data
    train_df = load_truthfulqa_data(args.train_data)
    eval_df = load_truthfulqa_data(args.eval_data)
    
    # Create training pairs
    print("\nCreating training pairs from TruthfulQA...")
    training_pairs = create_training_pairs(train_df, args.max_pairs)
    
    # Initialize wisent-guard
    print("\nInitializing wisent-guard for hallucination detection")
    layers_to_monitor = [args.layer_number] if args.layer_number >= 0 else None
    
    # First initialize guard without classifier to collect activations if needed
    init_guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=layers_to_monitor,
        threshold=args.threshold,
        save_dir=args.save_dir,
        device=device,
        force_llama_format=args.force_llama_format
    )
    
    # Train the guard on TruthfulQA pairs or load existing vectors
    if not args.use_existing_vectors:
        print("\nTraining wisent-guard on TruthfulQA hallucination pairs...")
        init_guard.train_on_phrase_pairs(training_pairs, category="hallucination")
    
    # Check if we need to train a classifier
    if args.train_classifier:
        if not args.classifier_path:
            print("Error: --classifier-path is required when --train-classifier is set.")
            return
        
        print(f"Will train a classifier and save to: {args.classifier_path}")
        
        # Collect activations for classifier training
        activations = collect_activations_for_training(init_guard, train_df, args.max_pairs)
        
        # Train the classifier
        classifier = train_classifier(
            activations["harmful_activations"],
            activations["harmless_activations"],
            args
        )
    
    # Check if we should use the classifier approach
    if args.use_classifier:
        if not args.classifier_path:
            print("Error: --classifier-path is required when --use-classifier is set.")
            return
        
        if not os.path.exists(args.classifier_path) and not args.train_classifier:
            print(f"Warning: Classifier path {args.classifier_path} does not exist and --train-classifier is not set.")
            print("Falling back to vector-based approach...")
            args.use_classifier = False
        
        if args.use_classifier:
            print(f"Using classifier-based approach with model: {args.classifier_path}")
            print(f"Classifier threshold: {args.classifier_threshold}")
            
            # Initialize guard with classifier parameters
            guard = ActivationGuard(
                model=model,
                tokenizer=tokenizer,
                layers=layers_to_monitor,
                threshold=args.threshold,  # Still set threshold for fallback
                save_dir=args.save_dir,
                device=device,
                force_llama_format=args.force_llama_format,
                use_classifier=True,
                classifier_path=args.classifier_path,
                classifier_threshold=args.classifier_threshold
            )
            
            # Verify that classifier was actually loaded
            if not hasattr(guard, 'classifier') or guard.classifier is None:
                print("Warning: Classifier failed to load properly. Falling back to vector-based approach...")
                args.use_classifier = False
    
    if not args.use_classifier:
        # Use the already initialized guard for threshold-based approach
        guard = init_guard
        print(f"Using threshold-based approach with threshold: {args.threshold}")
        
        # Make sure vectors are loaded
        if guard.vectors is not None:
            vector_loaded = guard.load_vectors()
            if vector_loaded:
                print("✅ Vectors loaded successfully for threshold-based approach")
            else:
                print("⚠️ Warning: No vectors found. Token scores may all be zero.")
        else:
            print("⚠️ Warning: No vectors initialized. Token scores may all be zero.")
    
    print(f"Available categories: {guard.get_available_categories()}")
    
    # First, evaluate without guard (baseline)
    baseline_results = None
    if args.run_baseline:
        print("\n----- EVALUATING MODEL WITHOUT GUARD (BASELINE) -----")
        baseline_results = evaluate_model(
            model, 
            tokenizer, 
            eval_df,
            device,
            max_tokens=args.max_tokens,
            max_eval_samples=args.sample_size,
            use_claude=use_claude
        )
        
        # Compute and print metrics
        baseline_metrics = compute_metrics(baseline_results)
        print_metrics(baseline_metrics, "Baseline Model Evaluation (Without Guard)")
        
        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            save_results(baseline_results, os.path.join(args.output_dir, "baseline_results.csv"))
            save_metrics(baseline_metrics, os.path.join(args.output_dir, "baseline_metrics.json"))
    
    # Evaluate with guard
    detection_approach = "classifier-based" if args.use_classifier else "threshold-based"
    print(f"\n----- EVALUATING MODEL WITH {detection_approach.upper()} GUARD -----")
    
    # Skip actual evaluation if --no-eval is set
    if args.no_eval:
        print("Skipping evaluation due to --no-eval flag")
        print("\n----- CLASSIFIER TRAINING COMPLETE -----")
        return  # Exit the function early
    
    # If we get here, proceed with evaluation
    guard_results = evaluate_model_with_guard(
        guard, 
        eval_df,
        max_tokens=args.max_tokens,
        max_eval_samples=args.sample_size,
        use_claude=use_claude
    )
    
    # Compute and print metrics
    guard_metrics = compute_metrics(guard_results)
    
    title = f"Model Evaluation With {detection_approach.capitalize()} Guard"
    print_metrics(guard_metrics, title)
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create file names with approach information
        guard_results_file = f"guard_results_truthfulqa_{detection_approach}.csv"
        guard_metrics_file = f"guard_metrics_truthfulqa{detection_approach}.json"
        
        save_results(guard_results, os.path.join(args.output_dir, guard_results_file))
        save_metrics(guard_metrics, os.path.join(args.output_dir, guard_metrics_file))
        
        # Save combined results if baseline was run
        if baseline_results:
            combined_results_file = f"combined_results_{detection_approach}.csv"
            save_combined_results(
                baseline_results, 
                guard_results, 
                os.path.join(args.output_dir, combined_results_file)
            )
    
    print("\n----- EVALUATION COMPLETE -----")

    # After evaluation, test the classifier if requested
    if args.test_classifier:
        test_classifier(args, args.model_name, model, tokenizer, device, eval_df)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama-3.1-8B-Instruct on TruthfulQA with wisent-guard")
    
    # Data paths
    parser.add_argument("--train-data", type=str, default="evaluation/data/TruthfulQA_en_train.csv", 
                        help="Path to TruthfulQA training data")
    parser.add_argument("--eval-data", type=str, default="evaluation/data/TruthfulQA_en_eval.csv", 
                        help="Path to TruthfulQA evaluation data")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", 
                        help="Directory to save results and metrics")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="Name of the model to use")
    parser.add_argument("--cpu-only", action="store_true", 
                        help="Force CPU usage even if CUDA or MPS is available")
    parser.add_argument("--no-mps", action="store_true",
                        help="Disable MPS (Apple Silicon GPU) even if available")
    parser.add_argument("--half-precision", action="store_true", 
                        help="Load model in half precision (fp16)")
    parser.add_argument("--load-in-8bit", action="store_true", 
                        help="Load model in 8-bit quantization (not supported on MPS)")
    
    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=100, 
                        help="Maximum number of tokens to generate")
    
    # Evaluation parameters
    parser.add_argument("--run-baseline", action="store_true", 
                        help="Run baseline evaluation without guard")
    parser.add_argument("--sample-size", type=int, default=0, 
                        help="Number of samples to evaluate (0 for all)")
    
    # wisent-guard parameters
    parser.add_argument("--layer-number", type=int, default=15, 
                        help="Layer number to monitor (default: 15, -1 for all layers)")
    parser.add_argument("--threshold", type=float, default=0.2, 
                        help="Similarity threshold (lower is more sensitive)")
    parser.add_argument("--save-dir", type=str, default="./hallucination_guard_data", 
                        help="Directory to save/load vectors")
    parser.add_argument("--use-existing-vectors", action="store_true",
                        help="Use existing vectors instead of training new ones")
    parser.add_argument("--force-llama-format", action="store_true", default=True,
                        help="Force Llama 3.1 token format (default: True)")
    
    # New parameter for limiting the number of training pairs
    parser.add_argument("--max-pairs", type=int, default=0, 
                        help="Maximum number of training pairs to create (0 for all)")
    
    # Add classifier parameters
    parser.add_argument("--use-classifier", action="store_true",
                        help="Use classifier-based approach instead of threshold-based")
    parser.add_argument("--classifier-path", type=str, 
                        help="Path to trained classifier model (required if use-classifier is set)")
    parser.add_argument("--classifier-threshold", type=float, default=0.5,
                        help="Classification threshold for the ML model (default: 0.5)")
    
    # Add classifier training parameters
    parser.add_argument("--train-classifier", action="store_true",
                        help="Train a new classifier using activation patterns")
    parser.add_argument("--classifier-model", type=str, default="logistic",
                        choices=["logistic", "mlp"],
                        help="Classifier model type (logistic or mlp)")
    
    # Add no-eval parameter
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation (for testing purposes only)")
    
    # Add test-classifier parameter
    parser.add_argument("--test-classifier", action="store_true", 
                        help="Test the classifier with examples")
                        
    # Add no-claude parameter
    parser.add_argument("--no-claude", action="store_true",
                        help="Don't use Claude for evaluation, leave human_score empty")
    
    args = parser.parse_args()
    main(args)
