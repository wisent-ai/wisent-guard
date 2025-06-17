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
from .core.detection_handling import DetectionHandler, DetectionAction
from .core.save_results import save_results_json, save_results_csv, save_classification_results_csv, create_evaluation_report
from .core.parser import setup_parser, parse_layer_range, aggregate_token_scores, parse_layers_from_arg
from .inference import (generate_with_classification_and_handling, generate_with_classification,
                        generate_with_multi_layer_classification, generate_with_multi_layer_classification_and_handling)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def run_task_pipeline(
    task_name: str,
    model_name: str,
    layer: str,
    shots: int = 0,
    split_ratio: float = 0.8,
    limit: int = None,
    classifier_type: str = "logistic",
    max_new_tokens: int = 300,
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
    allow_small_dataset: bool = False,
    detection_action: str = "pass_through",
    placeholder_message: str = None,
    max_regeneration_attempts: int = 3,
    detection_threshold: float = 0.6,
    log_detections: bool = False,
    steering_mode: bool = False,
    steering_strength: float = 1.0,
    save_steering_vector: str = None,
    load_steering_vector: str = None,
    train_only: bool = False,
    inference_only: bool = False,
    save_classifier: str = None,
    load_classifier: str = None,
    classifier_dir: str = "./models"
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
        # Parse layers from argument
        layers = parse_layers_from_arg(layer)
        is_multi_layer = len(layers) > 1
        
        # Initialize enhanced primitives
        if verbose:
            print(f"\n🔧 Initializing model and primitives...")
            if is_multi_layer:
                print(f"   • Multi-layer mode: {layers}")
            else:
                print(f"   • Single layer mode: {layers[0]}")
        model = Model(name=model_name, device=device)
        layer_obj = Layer(index=layers[0], type="transformer")
        
        # Create detection handler based on CLI arguments
        if verbose and detection_action != "pass_through":
            print(f"\n🛡️  Setting up detection handling:")
            print(f"   • Action: {detection_action}")
            if placeholder_message:
                print(f"   • Custom placeholder: {placeholder_message}")
            if detection_action == "regenerate_until_safe":
                print(f"   • Max regeneration attempts: {max_regeneration_attempts}")
            print(f"   • Detection threshold: {detection_threshold}")
            print(f"   • Logging enabled: {log_detections}")
        
        detection_handler = None
        if detection_action != "pass_through":
            from .core.detection_handling import DetectionHandler, DetectionAction
            
            # Map string to enum
            action_mapping = {
                "pass_through": DetectionAction.PASS_THROUGH,
                "replace_with_placeholder": DetectionAction.REPLACE_WITH_PLACEHOLDER,
                "regenerate_until_safe": DetectionAction.REGENERATE_UNTIL_SAFE
            }
            
            detection_handler = DetectionHandler(
                action=action_mapping[detection_action],
                placeholder_message=placeholder_message,
                max_regeneration_attempts=max_regeneration_attempts,
                log_detections=log_detections
            )
        
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
                print(f"   • Loading {task_name} data with correct/incorrect answers...")
            
            # Use the proper extraction method from ContrastivePairSet
            from .core.contrastive_pair_set import ContrastivePairSet
            qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(task_name, task_data, train_docs)
                
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
        
        # Validate mode combinations
        if train_only and inference_only:
            error_msg = "Cannot specify both --train-only and --inference-only modes"
            if verbose:
                print(f"\n❌ ERROR: {error_msg}")
            return {"task_name": task_name, "error": error_msg}
        
        if inference_only and not load_classifier and not load_steering_vector:
            error_msg = "Inference-only mode requires --load-classifier or --load-steering-vector"
            if verbose:
                print(f"\n❌ ERROR: {error_msg}")
                print(f"   • Use --load-classifier to load pre-trained classifiers")
                print(f"   • Use --load-steering-vector to load pre-trained steering vectors")
            return {"task_name": task_name, "error": error_msg}
        
        # Handle inference-only mode
        if inference_only:
            from .core.model_persistence import ModelPersistence
            
            if verbose:
                print(f"\n🔄 INFERENCE-ONLY MODE:")
                print(f"   • Loading pre-trained models for inference...")
            
            # Parse layers to know what to load
            layers = parse_layers_from_arg(layer)
            
            # Load classifiers or steering vectors
            steering_methods = {}
            loaded_models = {}
            
            if load_classifier:
                if verbose:
                    print(f"   • Loading classifiers from: {load_classifier}")
                
                if len(layers) > 1:
                    # Multi-layer mode
                    classifiers_data = ModelPersistence.load_multi_layer_classifiers(load_classifier, layers)
                    for layer_idx, (classifier, metadata) in classifiers_data.items():
                        steering_methods[layer_idx] = type('SteeringMethod', (), {'classifier': classifier})()
                        loaded_models[layer_idx] = metadata
                        if verbose:
                            print(f"     ✅ Layer {layer_idx}: {metadata.get('classifier_type', 'unknown')} classifier")
                else:
                    # Single layer mode
                    classifier, metadata = ModelPersistence.load_classifier(load_classifier, layers[0])
                    steering_methods[layers[0]] = type('SteeringMethod', (), {'classifier': classifier})()
                    loaded_models[layers[0]] = metadata
                    if verbose:
                        print(f"     ✅ Layer {layers[0]}: {metadata.get('classifier_type', 'unknown')} classifier")
            
            if load_steering_vector:
                if verbose:
                    print(f"   • Loading steering vectors from: {load_steering_vector}")
                # TODO: Implement steering vector loading
                print(f"   ⚠️  Steering vector loading not yet implemented")
            
            if not steering_methods:
                error_msg = "No models could be loaded for inference"
                if verbose:
                    print(f"\n❌ ERROR: {error_msg}")
                return {"task_name": task_name, "error": error_msg}
            
            # Set up inference with loaded models
            if verbose:
                print(f"   • Inference setup complete")
                print(f"   • Available models: {list(loaded_models.keys())}")
            
            # Continue with inference pipeline using loaded models
            # (The rest of the pipeline will use the loaded steering_methods)
        
        # Handle training-only mode  
        elif train_only:
            if verbose:
                print(f"\n🎓 TRAINING-ONLY MODE:")
                print(f"   • Training classifiers/vectors and saving, skipping inference...")
                
            # Continue with training but return early before inference
            # (Training will happen in the normal flow, but we'll return after training)
            pass
        
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
            layer_index=layers[0],
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
        
        # STEERING MODE vs CLASSIFICATION MODE
        if steering_mode:
            if verbose:
                print(f"\n🎯 STEERING MODE: Computing CAA Vector")
                print(f"   • Method: Contrastive Activation Addition (CAA)")
                print(f"   • Target layer: {layer}")
                print(f"   • Training pairs: {len(pair_set)}")
                print(f"   • Steering strength: {steering_strength}")
            
            # Import CAA from steering_method
            from .core.steering_method import CAA
            
            # Create CAA steering method
            caa_steering = CAA(device=device)
            
            # Train CAA to compute steering vector
            try:
                training_stats = caa_steering.train(pair_set, layers[0])
                
                if verbose:
                    print(f"✅ CAA vector computed successfully!")
                    print(f"   • Vector norm: {training_stats['vector_norm']:.4f}")
                    print(f"   • Vector shape: {training_stats['vector_shape']}")
                    print(f"   • Training pairs used: {training_stats['num_pairs']}")
                
                # Save steering vector if requested
                if save_steering_vector:
                    success = caa_steering.save_steering_vector(save_steering_vector)
                    if verbose:
                        if success:
                            print(f"   • Saved steering vector to: {save_steering_vector}")
                        else:
                            print(f"   • Failed to save steering vector to: {save_steering_vector}")
                
                # TEST THE STEERING by generating responses with and without the vector
                if verbose:
                    print(f"\n🧪 TESTING CAA STEERING:")
                    print(f"   • Generating responses with steering applied...")
                
                # Get test questions
                test_qa_pairs = []
                for doc in test_qa_pairs_source[:3]:  # Test on first 3 questions
                    try:
                        if from_csv or from_json:
                            test_qa_pairs.append({
                                'question': doc['question'],
                                'correct_answer': doc.get('correct_answer', 'N/A')
                            })
                        else:
                            # Extract from lm-harness format
                            if hasattr(task_data, 'doc_to_text'):
                                question = task_data.doc_to_text(doc)
                            else:
                                question = doc.get('question', str(doc))
                            
                            # Extract correct answer
                            correct_answers = doc.get('mc1_targets', {}).get('choices', [])
                            correct_labels = doc.get('mc1_targets', {}).get('labels', [])
                            correct_answer = "N/A"
                            for i, label in enumerate(correct_labels):
                                if label == 1 and i < len(correct_answers):
                                    correct_answer = correct_answers[i]
                                    break
                            
                            test_qa_pairs.append({
                                'question': question,
                                'correct_answer': correct_answer
                            })
                    except:
                        continue
                
                steered_responses = []
                
                for i, qa_pair in enumerate(test_qa_pairs):
                    if verbose:
                        print(f"\n   🎯 Test Question {i+1}:")
                        print(f"      📝 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}")
                        print(f"      ✅ Expected: {qa_pair['correct_answer']}")
                    
                    # Generate UNSTEERED response (baseline)
                    unsteered_response, _ = model.generate(qa_pair['question'], layers[0], max_new_tokens)
                    
                    # Generate STEERED response using activation hooks
                    def steering_hook(module, input, output):
                        """Hook function that applies CAA steering to activations"""
                        if isinstance(output, tuple):
                            hidden_states = output[0]
                        else:
                            hidden_states = output
                        
                        if isinstance(hidden_states, torch.Tensor):
                            # Apply steering to the last token
                            steered_hidden = caa_steering.apply_steering(
                                hidden_states[:, -1:, :], strength=steering_strength
                            )
                            # Replace the last token's activations
                            new_hidden_states = hidden_states.clone()
                            new_hidden_states[:, -1:, :] = steered_hidden
                            
                            if isinstance(output, tuple):
                                return (new_hidden_states,) + output[1:]
                            else:
                                return new_hidden_states
                        return output
                    
                    # Apply steering hook to the target layer
                    import torch
                    layer_module = None
                    if hasattr(model.hf_model, 'model') and hasattr(model.hf_model.model, 'layers'):
                        # Llama-style model
                        if layers[0] < len(model.hf_model.model.layers):
                            layer_module = model.hf_model.model.layers[layers[0]]
                    elif hasattr(model.hf_model, 'transformer') and hasattr(model.hf_model.transformer, 'h'):
                        # GPT-style model
                        if layers[0] < len(model.hf_model.transformer.h):
                            layer_module = model.hf_model.transformer.h[layers[0]]
                    
                    if layer_module:
                        # Register the hook
                        handle = layer_module.register_forward_hook(steering_hook)
                        
                        try:
                            # Generate with steering
                            steered_response, _ = model.generate(qa_pair['question'], layers[0], max_new_tokens)
                        finally:
                            # Always remove the hook
                            handle.remove()
                    else:
                        steered_response = "ERROR: Could not find target layer for steering"
                    
                    # Store results
                    steered_responses.append({
                        'question': qa_pair['question'],
                        'expected_answer': qa_pair['correct_answer'],
                        'unsteered_response': unsteered_response,
                        'steered_response': steered_response,
                        'steering_strength': steering_strength
                    })
                    
                    if verbose:
                        print(f"      🔄 Unsteered: {unsteered_response[:100]}{'...' if len(unsteered_response) > 100 else ''}")
                        print(f"      🎯 Steered:   {steered_response[:100]}{'...' if len(steered_response) > 100 else ''}")
                        print(f"      📊 Steering strength: {steering_strength}")
                
                if verbose:
                    print(f"\n✅ CAA steering test completed!")
                    print(f"   • Generated {len(steered_responses)} steered responses")
                    print(f"   • Vector applied at layer {layers[0]} with strength {steering_strength}")
                
                # Return steering mode results with test data
                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "layer": layers[0],
                    "steering_mode": True,
                    "steering_method": "CAA",
                    "steering_strength": steering_strength,
                    "training_stats": training_stats,
                    "training_pairs": len(pair_set),
                    "vector_saved": save_steering_vector is not None,
                    "test_responses": steered_responses,
                    "tests_performed": len(steered_responses)
                }
                
            except Exception as e:
                error_msg = f"CAA steering vector computation failed: {str(e)}"
                if verbose:
                    print(f"\n❌ STEERING ERROR: {error_msg}")
                    print(f"   • Training pairs: {len(pair_set)}")
                    print(f"   • Layer: {layers[0]}")
                
                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "error": error_msg,
                    "steering_mode": True,
                    "error_type": "steering_failure",
                    "suggestion": "Check activation extraction and data quality"
                }
        
        # CLASSIFICATION MODE (single or multi-layer)
        # Train classifier(s) using optimized type (if optimization was performed)
        if optimize:
            final_classifier_type = optimized_classifier_type
            final_threshold = optimized_threshold
        else:
            final_classifier_type = classifier_type
            final_threshold = 0.6
        
        # Train classifiers for each layer
        steering_methods = {}
        layer_training_results = {}
        
        if is_multi_layer:
            if verbose:
                print(f"\n🎯 TRAINING MULTI-LAYER CLASSIFIERS:")
                print(f"   • Layers: {layers}")
                print(f"   • Type: {final_classifier_type}")
                print(f"   • Threshold: {final_threshold}")
                print(f"   • Training pairs: {len(contrastive_pairs)}")
            
            # Train a classifier for each layer
            for layer_idx in layers:
                if verbose:
                    print(f"\n   🔬 Training classifier for layer {layer_idx}...")
                
                # Extract activations for this specific layer
                layer_processed_pairs = collector.collect_activations_batch(
                    pairs=contrastive_pairs,
                    layer_index=layer_idx,
                    device=device
                )
                
                # Create layer-specific ContrastivePairSet
                layer_phrase_pairs = []
                for pair in layer_processed_pairs:
                    positive_full = f"{pair.prompt}{pair.positive_response}"
                    negative_full = f"{pair.prompt}{pair.negative_response}"
                    
                    layer_phrase_pairs.append({
                        "harmful": negative_full,
                        "harmless": positive_full
                    })
                
                layer_pair_set = ContrastivePairSet.from_phrase_pairs(
                    name=f"{task_name}_layer_{layer_idx}",
                    phrase_pairs=layer_phrase_pairs,
                    task_type="lm_evaluation"
                )
                
                # Store activations in the layer pair set
                for i, processed_pair in enumerate(layer_processed_pairs):
                    if i < len(layer_pair_set.pairs):
                        if hasattr(layer_pair_set.pairs[i], 'positive_response') and layer_pair_set.pairs[i].positive_response:
                            layer_pair_set.pairs[i].positive_response.activations = processed_pair.positive_activations
                        if hasattr(layer_pair_set.pairs[i], 'negative_response') and layer_pair_set.pairs[i].negative_response:
                            layer_pair_set.pairs[i].negative_response.activations = processed_pair.negative_activations
                
                # Train classifier for this layer
                steering_type = SteeringType.LOGISTIC if final_classifier_type == "logistic" else SteeringType.MLP
                layer_steering_method = SteeringMethod(method_type=steering_type, threshold=final_threshold, device=device)
                
                try:
                    layer_training_results[layer_idx] = layer_steering_method.train(layer_pair_set)
                    steering_methods[layer_idx] = layer_steering_method
                    
                    if verbose:
                        accuracy = layer_training_results[layer_idx].get('accuracy', 'N/A')
                        f1_score = layer_training_results[layer_idx].get('f1', 'N/A')
                        print(f"      ✅ Layer {layer_idx}: Accuracy={accuracy:.2%}, F1={f1_score:.3f}")
                        
                except Exception as e:
                    if verbose:
                        print(f"      ❌ Layer {layer_idx}: Training failed - {str(e)}")
                    layer_training_results[layer_idx] = {"error": str(e)}
            
            # Use the first successfully trained layer as the primary one for compatibility
            primary_layer = layers[0]
            if primary_layer in steering_methods:
                steering_method = steering_methods[primary_layer]
                training_results = layer_training_results[primary_layer]
            else:
                # If primary layer failed, try to find any successful layer
                successful_layers = [l for l in layers if l in steering_methods]
                if successful_layers:
                    primary_layer = successful_layers[0]
                    steering_method = steering_methods[primary_layer]
                    training_results = layer_training_results[primary_layer]
                else:
                    # All layers failed
                    error_msg = "All layer classifiers failed to train"
                    if verbose:
                        print(f"\n❌ MULTI-LAYER TRAINING ERROR: {error_msg}")
                    return {
                        "task_name": task_name,
                        "model_name": model_name,
                        "error": error_msg,
                        "layers": layers,
                        "layer_results": layer_training_results,
                        "error_type": "multi_layer_training_failure"
                    }
        else:
            # Single layer mode (original logic)
            if verbose:
                print(f"\n🎯 TRAINING CLASSIFIER:")
                print(f"   • Layer: {layers[0]}")
                print(f"   • Type: {final_classifier_type}")
                print(f"   • Threshold: {final_threshold}")
                print(f"   • Training pairs: {len(pair_set)}")
            
            steering_type = SteeringType.LOGISTIC if final_classifier_type == "logistic" else SteeringType.MLP
            steering_method = SteeringMethod(method_type=steering_type, threshold=final_threshold, device=device)
            
            try:
                training_results = steering_method.train(pair_set)
                steering_methods[layers[0]] = steering_method
                layer_training_results[layers[0]] = training_results
            
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
        
        # Save trained classifiers if requested
        saved_classifier_paths = []
        if save_classifier or train_only:
            from .core.model_persistence import ModelPersistence, create_classifier_metadata
            
            # Determine save path
            if save_classifier:
                save_path = save_classifier
            else:
                # Default path for train-only mode
                safe_model_name = model_name.replace('/', '_').replace('-', '_')
                save_path = os.path.join(classifier_dir, f"{task_name}_{safe_model_name}_classifier")
            
            if verbose:
                print(f"\n💾 SAVING TRAINED CLASSIFIERS:")
                print(f"   • Save path: {save_path}")
            
            try:
                if is_multi_layer:
                    # Save multiple classifiers
                    for layer_idx in layers:
                        if layer_idx in steering_methods:
                            classifier = steering_methods[layer_idx].classifier
                            training_result = layer_training_results[layer_idx]
                            
                            # Create metadata
                            metadata = create_classifier_metadata(
                                model_name=model_name,
                                task_name=task_name,
                                layer=layer_idx,
                                classifier_type=final_classifier_type,
                                training_accuracy=training_result.get('accuracy', 0.0),
                                training_samples=len(contrastive_pairs),
                                token_aggregation=token_aggregation,
                                detection_threshold=final_threshold
                            )
                            
                            path = ModelPersistence.save_classifier(classifier, layer_idx, save_path, metadata)
                            saved_classifier_paths.append(path)
                            
                            if verbose:
                                print(f"     ✅ Layer {layer_idx}: {path}")
                else:
                    # Save single classifier
                    classifier = steering_method.classifier
                    metadata = create_classifier_metadata(
                        model_name=model_name,
                        task_name=task_name,
                        layer=layers[0],
                        classifier_type=final_classifier_type,
                        training_accuracy=training_results.get('accuracy', 0.0),
                        training_samples=len(contrastive_pairs),
                        token_aggregation=token_aggregation,
                        detection_threshold=final_threshold
                    )
                    
                    path = ModelPersistence.save_classifier(classifier, layers[0], save_path, metadata)
                    saved_classifier_paths.append(path)
                    
                    if verbose:
                        print(f"     ✅ Saved: {path}")
                        
            except Exception as e:
                if verbose:
                    print(f"     ❌ Error saving classifiers: {e}")
        
        # Handle train-only mode - return early after training and saving
        if train_only:
            if verbose:
                print(f"\n🎓 TRAINING-ONLY MODE COMPLETED!")
                print(f"   • Trained classifiers for layers: {list(steering_methods.keys())}")
                if saved_classifier_paths:
                    print(f"   • Saved {len(saved_classifier_paths)} classifier files")
                print(f"   • Skipping inference phase")
            
            return {
                "task_name": task_name,
                "model_name": model_name,
                "mode": "train_only",
                "layers": layers,
                "trained_layers": list(steering_methods.keys()),
                "training_results": layer_training_results if is_multi_layer else {layers[0]: training_results},
                "saved_classifier_paths": saved_classifier_paths,
                "classifier_type": final_classifier_type,
                "training_samples": len(contrastive_pairs),
                "success": True
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
                
                # Generate response with token-level scoring and detection handling
                if len(layers) > 1:
                    # Multi-layer mode: use multi-layer generation function
                    response, layer_results, was_handled = generate_with_multi_layer_classification_and_handling(
                        model, simple_prompt, layers, max_new_tokens, steering_methods, 
                        token_aggregation, detection_threshold, verbose and not optimize, detection_handler
                    )
                    # For backward compatibility, use primary layer's results for main fields
                    primary_layer = layers[0]
                    token_scores = layer_results[primary_layer]['token_scores'] if primary_layer in layer_results else []
                    classification = layer_results[primary_layer]['classification'] if primary_layer in layer_results else 'UNKNOWN'
                    aggregated_score = layer_results[primary_layer]['aggregated_score'] if primary_layer in layer_results else 0.0
                else:
                    # Single-layer mode: use original function
                    response, token_scores, classification, was_handled = generate_with_classification_and_handling(
                        model, simple_prompt, layers[0], max_new_tokens, steering_method, 
                        token_aggregation, detection_threshold, verbose and not optimize, detection_handler
                    )
                    layer_results = None
                    aggregated_score = aggregate_token_scores(token_scores, token_aggregation) if token_scores else 0.0
                
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
                    
                    # Create response entry with layer results if available
                    response_entry = {
                        'question': qa_pair['question'],  # Add the question
                        'response': response,
                        'token_scores': token_scores,
                        'aggregated_score': aggregated_score,
                        'classification': classification,
                        'ground_truth': ground_truth,
                        'ground_truth_method': evaluation_result["method_used"],
                        'ground_truth_confidence': evaluation_result["confidence"],
                        'ground_truth_details': evaluation_result["details"],
                        'classification_correct': classification_correct,
                        'was_handled': was_handled
                    }
                    
                    # Add layer-specific results if multi-layer
                    if layer_results:
                        response_entry['layer_results'] = layer_results
                    
                    generated_responses.append(response_entry)
                    
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
                        print(f"      ❌ Incorrect: {qa_pair['incorrect_answer']}")
                        if evaluation_result["details"]:
                            print(f"      📝 Details: {evaluation_result['details']}")
                    
                except Exception as e:
                    if verbose and not optimize:
                        print(f"      ⚠️  Could not evaluate response: {e}")
                    generated_responses.append({
                        'question': qa_pair['question'],  # Add the question
                        'response': response,
                        'token_scores': token_scores,
                        'classification': classification,
                        'ground_truth': 'UNKNOWN',
                        'ground_truth_method': 'error',
                        'ground_truth_confidence': 0.0,
                        'ground_truth_details': f'Error during evaluation: {str(e)}',
                        'classification_correct': None,
                        'was_handled': was_handled
                    })
            
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
                if total_classifications > 0:
                    classification_acc = correct_classifications / total_classifications
                    print(f"   • Classification accuracy on generated responses: {classification_acc:.2%} ({correct_classifications}/{total_classifications})")
                else:
                    print(f"   • Classification accuracy: Could not evaluate")
                print(f"{'='*80}\n")
            
            logger.info(f"Optimization pipeline completed for {task_name}")
            return results
        else:
            # Only do pre-written validation when NOT optimizing
            if verbose:
                print(f"\n🧪 PREPARING TEST DATA:")
                print(f"   • Loading {task_name} test data with correct/incorrect answers...")
            
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
                    print(f"      ❌ Incorrect Answer: {qa_pair['incorrect_answer']}")
            
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
            
            # NOTE: Removed pointless "classifier validation" step that just tested on pre-written answers
            # The real evaluation happens when we test on actual generated responses below
            evaluation_results = {"accuracy": "N/A", "note": "Validation on pre-written answers removed as misleading"}
            
            # Generate sample responses with token-level classification
            if verbose:
                if optimize:
                    print(f"\n🎭 GENERATING SAMPLE RESPONSES WITH OPTIMIZED CLASSIFIER:")
                    print(f"   • Generating {len(test_qa_pairs)} sample responses with optimized layer {layer}...")
                else:
                    print(f"\n🎭 GENERATING SAMPLE RESPONSES WITH HALLUCINATION DETECTION:")
                    print(f"   • Generating {len(test_qa_pairs)} sample responses...")
            
            generated_responses = []
            correct_classifications = 0
            total_classifications = 0
            
            for i, qa_pair in enumerate(test_qa_pairs):
                if verbose and not optimize:  # Only show detailed progress when not optimizing
                    print(f"\n   🎯 Generating response {i+1}:")
                    print(f"      📝 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}")
                
                # Use the raw question for natural generation
                # The formatted_question contains few-shot examples which are for training, not generation
                simple_prompt = qa_pair['question']
                
                # Generate response with token-level scoring and detection handling
                if len(layers) > 1:
                    # Multi-layer mode: use multi-layer generation function
                    response, layer_results, was_handled = generate_with_multi_layer_classification_and_handling(
                        model, simple_prompt, layers, max_new_tokens, steering_methods, 
                        token_aggregation, detection_threshold, verbose and not optimize, detection_handler
                    )
                    # For backward compatibility, use primary layer's results for main fields
                    primary_layer = layers[0]
                    token_scores = layer_results[primary_layer]['token_scores'] if primary_layer in layer_results else []
                    classification = layer_results[primary_layer]['classification'] if primary_layer in layer_results else 'UNKNOWN'
                    aggregated_score = layer_results[primary_layer]['aggregated_score'] if primary_layer in layer_results else 0.0
                else:
                    # Single-layer mode: use original function
                    response, token_scores, classification, was_handled = generate_with_classification_and_handling(
                        model, simple_prompt, layers[0], max_new_tokens, steering_method, 
                        token_aggregation, detection_threshold, verbose and not optimize, detection_handler
                    )
                    layer_results = None
                    aggregated_score = aggregate_token_scores(token_scores, token_aggregation) if token_scores else 0.0
                
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
                    
                    # Create response entry with layer results if available
                    response_entry = {
                        'question': qa_pair['question'],  # Add the question
                        'response': response,
                        'token_scores': token_scores,
                        'aggregated_score': aggregated_score,
                        'classification': classification,
                        'ground_truth': ground_truth,
                        'ground_truth_method': evaluation_result["method_used"],
                        'ground_truth_confidence': evaluation_result["confidence"],
                        'ground_truth_details': evaluation_result["details"],
                        'classification_correct': classification_correct,
                        'was_handled': was_handled
                    }
                    
                    # Add layer-specific results if multi-layer
                    if layer_results:
                        response_entry['layer_results'] = layer_results
                    
                    generated_responses.append(response_entry)
                    
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
                        print(f"      ❌ Incorrect: {qa_pair['incorrect_answer']}")
                        if evaluation_result["details"]:
                            print(f"      📝 Details: {evaluation_result['details']}")
                    
                except Exception as e:
                    if verbose and not optimize:
                        print(f"      ⚠️  Could not evaluate response: {e}")
                    generated_responses.append({
                        'question': qa_pair['question'],  # Add the question
                        'response': response,
                        'token_scores': token_scores,
                        'classification': classification,
                        'ground_truth': 'UNKNOWN',
                        'ground_truth_method': 'error',
                        'ground_truth_confidence': 0.0,
                        'ground_truth_details': f'Error during evaluation: {str(e)}',
                        'classification_correct': None,
                        'was_handled': was_handled
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
            should_optimize = args.optimize or (args.auto_optimize and args.layer == "-1")
            
            # Use layer -1 as a signal for auto-optimization
            if should_optimize:
                layer_to_use = "-1"  # Signal to the pipeline to optimize
            else:
                layer_to_use = args.layer if args.layer != "-1" else "15"  # Default to 15 if not specified
            
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
                allow_small_dataset=args.allow_small_dataset,
                detection_action=args.detection_action,
                placeholder_message=args.placeholder_message,
                max_regeneration_attempts=args.max_regeneration_attempts,
                detection_threshold=args.detection_threshold,
                log_detections=args.log_detections,
                steering_mode=args.steering_mode,
                steering_strength=args.steering_strength,
                save_steering_vector=args.save_steering_vector,
                load_steering_vector=args.load_steering_vector,
                train_only=args.train_only,
                inference_only=args.inference_only,
                save_classifier=args.save_classifier,
                load_classifier=args.load_classifier,
                classifier_dir=args.classifier_dir
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
        
        # Classification results CSV (detailed token-level data for manual evaluation)
        classification_csv_path = os.path.join(args.output, f"classification_results_{timestamp}.csv")
        save_classification_results_csv(all_results, classification_csv_path)
        
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
                # Check if this is steering mode
                if results.get("steering_mode", False):
                    steering_method = results.get("steering_method", "Unknown")
                    steering_strength = results.get("steering_strength", "N/A")
                    tests_performed = results.get("tests_performed", 0)
                    training_pairs = results.get("training_pairs", 0)
                    print(f"{task_name}: Steering={steering_method} | Strength={steering_strength} | Trained={training_pairs} | Tested={tests_performed}")
                else:
                    # Classification mode
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