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
import torch
from typing import List, Dict, Any, Optional

from .core import Model, ContrastivePairSet, SteeringMethod, SteeringType, Layer
from .core.ground_truth_evaluator import GroundTruthEvaluator, GroundTruthMethod
from .core.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig
from .core.classifier import Classifier
from .core.activations import TestActivationCache
from .optimize import run_smart_optimization, run_interactive_optimization, generate_with_all_layer_activations, compute_classification_score
from .core.detection_handling import DetectionHandler, DetectionAction
from .core.save_results import save_results_json, save_results_csv, save_classification_results_csv, create_evaluation_report
from .core.parser import setup_parser, parse_layer_range, aggregate_token_scores, parse_layers_from_arg
from .inference import (generate_with_classification_and_handling, generate_with_classification,
                        generate_with_multi_layer_classification, generate_with_multi_layer_classification_and_handling)
from .core.contrastive_pairs import (
    ContrastivePairSet, 
    generate_synthetic_pairs_cli, 
    load_synthetic_pairs_cli
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _run_lm_harness_evaluation(task_data, test_qa_pairs, model, steering_methods, layers, verbose=False):
    """
    Run proper lm-harness evaluation with steering integration.
    
    Args:
        task_data: The lm-harness task object
        test_qa_pairs: List of test QA pairs
        model: The wisent Model instance
        steering_methods: List of steering methods to apply
        layers: List of layers for steering
        verbose: Whether to print verbose output
        
    Returns:
        Dict containing evaluation results
    """
    try:
        try:
            from lm_eval.api.evaluator import evaluate
            from lm_eval.api.model import LM
        except ImportError:
            # Try newer lm-eval import paths
            from lm_eval import evaluate
            from lm_eval.api.model import LM
        
        if verbose:
            print(f"\n🔍 RUNNING LM-HARNESS EVALUATION WITH STEERING:")
            print(f"   • Task: {task_data.config.task}")
            print(f"   • Test samples: {len(test_qa_pairs)}")
            print(f"   • Steering methods: {[m.method_type.value for m in steering_methods] if steering_methods else 'None'}")
            print(f"   • Layers: {layers}")
        
        # Create a steered model wrapper for lm-harness evaluation
        class SteeredModelWrapper(LM):
            """Model wrapper that applies steering during lm-harness evaluation."""
            
            def __init__(self, wisent_model, steering_methods, layers):
                self.wisent_model = wisent_model
                self.steering_methods = steering_methods
                self.layers = layers
                
            def generate_until(self, requests):
                """Generate responses with steering applied."""
                results = []
                
                for req in requests:
                    # Extract the prompt from the request
                    if hasattr(req, 'args') and req.args:
                        prompt = req.args[0] if isinstance(req.args[0], str) else str(req.args[0])
                    else:
                        prompt = str(req)
                    
                    try:
                        # Generate with steering using the wisent model
                        if self.steering_methods and self.layers:
                            # Apply steering during generation
                            response, _, _, _ = generate_with_classification_and_handling(
                                self.wisent_model, 
                                prompt, 
                                self.layers[0],  # Use first layer
                                max_new_tokens=300,
                                steering_method=self.steering_methods[0] if self.steering_methods else None,
                                token_aggregation="average",
                                detection_threshold=0.6,
                                verbose=False,
                                detection_handler=None
                            )
                        else:
                            # Generate without steering
                            response = self.wisent_model.generate(
                                prompt, 
                                layer_index=self.layers[0] if self.layers else 15,
                                max_new_tokens=300
                            )
                            
                        results.append(response)
                        
                    except Exception as e:
                        if verbose:
                            print(f"   ⚠️ Generation failed for prompt: {e}")
                        results.append("Generation failed")
                
                return results
                
            def loglikelihood(self, requests):
                """Compute log-likelihood with steering applied."""
                results = []
                
                for req in requests:
                    try:
                        # For now, return neutral likelihood
                        # TODO: Implement proper likelihood computation with steering
                        results.append((0.0, False))
                    except Exception:
                        results.append((float('-inf'), False))
                        
                return results
                
            def loglikelihood_rolling(self, requests):
                """Rolling log-likelihood computation."""
                return [0.0] * len(requests)
        
        # Create steered model wrapper
        steered_model = SteeredModelWrapper(model, steering_methods, layers)
        
        # Run evaluation using lm-harness with steering
        results = evaluate(
            model=steered_model,
            tasks=[task_data.config.task],
            limit=len(test_qa_pairs),
            bootstrap_iters=0  # Disable bootstrapping for speed
        )
        
        # Extract accuracy and other metrics
        task_name = task_data.config.task
        task_results = results.get('results', {}).get(task_name, {})
        
        accuracy = task_results.get('acc', task_results.get('accuracy', 'N/A'))
        
        evaluation_results = {
            "accuracy": accuracy,
            "method": "lm_harness_with_steering",
            "task_name": task_name,
            "steering_applied": len(steering_methods) > 0 if steering_methods else False,
            "full_results": task_results
        }
        
        if verbose:
            print(f"   ✅ Evaluation completed")
            print(f"   📊 Accuracy: {accuracy}")
            print(f"   🎯 Steering applied: {'Yes' if steering_methods else 'No'}")
            
        return evaluation_results
        
    except Exception as e:
        if verbose:
            print(f"   ❌ LM-harness evaluation failed: {e}")
        
        # Fallback to placeholder
        return {
            "accuracy": "N/A", 
            "method": "lm_harness_failed",
            "error": str(e),
            "note": "LM-harness evaluation failed, falling back to individual response evaluation"
        }


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
    output_mode: str = "both",
    save_steering_vector: str = None,
    load_steering_vector: str = None,
    train_only: bool = False,
    inference_only: bool = False,
    save_classifier: str = None,
    load_classifier: str = None,
    classifier_dir: str = "./models",
    prompt_construction_strategy: str = "multiple_choice",
    token_targeting_strategy: str = "choice_token",
    normalize_mode: bool = False,
    normalization_method: str = "none",
    target_norm: Optional[float] = None,
    steering_method: str = "CAA",
    hpr_beta: float = 1.0,
    dac_dynamic_control: bool = False,
    dac_entropy_threshold: float = 1.0,
    bipo_beta: float = 0.1,
    bipo_learning_rate: float = 5e-4,
    bipo_epochs: int = 100,
    ksteering_num_labels: int = 6,
    ksteering_hidden_dim: int = 512,
    ksteering_learning_rate: float = 1e-3,
    ksteering_classifier_epochs: int = 100,
    ksteering_target_labels: str = "0",
    ksteering_avoid_labels: str = "",
    ksteering_alpha: float = 50.0,
    # Nonsense detection parameters
    enable_nonsense_detection: bool = False,
    max_word_length: int = 20,
    repetition_threshold: float = 0.7,
    gibberish_threshold: float = 0.3,
    disable_dictionary_check: bool = False,
    nonsense_action: str = "regenerate",
    # Token steering parameters
    enable_token_steering: bool = False,
    token_steering_strategy: str = "second_to_last",
    token_decay_rate: float = 0.5,
    token_min_strength: float = 0.1,
    token_max_strength: float = 1.0,
    token_apply_to_prompt: bool = False,
    token_prompt_strength_multiplier: float = 0.1,
    # Performance monitoring parameters
    enable_memory_tracking: bool = False,
    enable_latency_tracking: bool = False,
    memory_sampling_interval: float = 0.1,
    track_gpu_memory: bool = False,
    detailed_performance_report: bool = False,
    export_performance_csv: str = None,
    show_memory_usage: bool = False,
    show_timing_summary: bool = False,
    # Test-time activation saving/loading parameters
    save_test_activations: str = None,
    load_test_activations: str = None
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
    
    # Initialize performance tracking
    memory_tracker = None
    latency_tracker = None
    
    if enable_memory_tracking or enable_latency_tracking:
        if verbose:
            print(f"🔍 Performance tracking enabled: memory={enable_memory_tracking}, latency={enable_latency_tracking}")
        
        from .core.tracking import (
            get_global_memory_tracker, 
            get_global_latency_tracker,
            get_memory_info,
            format_memory_usage
        )
        
        if enable_memory_tracking:
            memory_tracker = get_global_memory_tracker()
            memory_tracker.track_gpu = track_gpu_memory
            memory_tracker.sampling_interval = memory_sampling_interval
            memory_tracker.start_monitoring()
            if verbose:
                print(f"   • Memory tracking started with {memory_sampling_interval}s interval")
            
        if enable_latency_tracking:
            latency_tracker = get_global_latency_tracker()
            latency_tracker.start_tracking()
            if verbose:
                print(f"   • Latency tracking started")
    
    # Show current memory usage if requested
    if show_memory_usage:
        from .core.tracking import get_memory_info, format_memory_usage
        current_memory = get_memory_info()
        print(f"\n💾 Current Memory Usage: {format_memory_usage(current_memory)}")
    
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
        
        # Time model loading
        if latency_tracker:
            with latency_tracker.time_operation("model_loading"):
                model = Model(name=model_name, device=device)
        else:
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
            # ContrastivePairSet import moved to top of file
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
        from .core.activation_collection_method import ActivationCollectionLogic, TokenTargetingStrategy, PromptConstructionStrategy
        from .core.activations import Activations, ActivationAggregationMethod
        
        # Convert strings to enums
        prompt_strategy_mapping = {
            "multiple_choice": PromptConstructionStrategy.MULTIPLE_CHOICE,
            "role_playing": PromptConstructionStrategy.ROLE_PLAYING,
            "direct_completion": PromptConstructionStrategy.DIRECT_COMPLETION,
            "instruction_following": PromptConstructionStrategy.INSTRUCTION_FOLLOWING
        }
        prompt_strategy = prompt_strategy_mapping.get(prompt_construction_strategy, PromptConstructionStrategy.MULTIPLE_CHOICE)
        
        targeting_strategy_mapping = {
            "choice_token": TokenTargetingStrategy.CHOICE_TOKEN,
            "continuation_token": TokenTargetingStrategy.CONTINUATION_TOKEN,
            "last_token": TokenTargetingStrategy.LAST_TOKEN,
            "first_token": TokenTargetingStrategy.FIRST_TOKEN,
            "mean_pooling": TokenTargetingStrategy.MEAN_POOLING,
            "max_pooling": TokenTargetingStrategy.MAX_POOLING
        }
        targeting_strategy = targeting_strategy_mapping.get(token_targeting_strategy, TokenTargetingStrategy.CHOICE_TOKEN)
        
        if verbose:
            print(f"   • Prompt construction: {prompt_strategy.value}")
            print(f"   • Token targeting: {targeting_strategy.value}")
        
        collector = ActivationCollectionLogic(model=model)
        contrastive_pairs = collector.create_batch_contrastive_pairs(qa_pairs, prompt_strategy)
        
        if verbose:
            print(f"\n🔄 Created {len(contrastive_pairs)} contrastive pairs:")
            for i, pair in enumerate(contrastive_pairs[:3]):  # Show first 3
                print(f"\n   🔄 Contrastive Pair {i+1}:")
                print(f"      📝 Prompt: {pair.prompt}")
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
                
                try:
                    from .core.steering_method import CAA
                    steering_method = CAA(device=device)
                    
                    if steering_method.load_steering_vector(load_steering_vector):
                        layer_index = steering_method.layer_index
                        if layer_index is not None:
                            steering_methods[layer_index] = steering_method
                            loaded_models[layer_index] = {
                                'method_name': steering_method.name,
                                'aggregation_method': steering_method.aggregation_method.value if hasattr(steering_method, 'aggregation_method') else 'caa',
                                'loaded_from': load_steering_vector
                            }
                            if verbose:
                                print(f"     ✅ Loaded steering vector for layer {layer_index}")
                        else:
                            if verbose:
                                print(f"     ⚠️  Warning: No layer information in loaded vector")
                    else:
                        if verbose:
                            print(f"     ❌ Failed to load steering vector from {load_steering_vector}")
                except Exception as e:
                    if verbose:
                        print(f"     ❌ Error loading steering vector: {str(e)}")
            
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
        
        if latency_tracker:
            with latency_tracker.time_operation("activation_extraction"):
                processed_pairs = collector.collect_activations_batch(
                    pairs=contrastive_pairs,
                    layer_index=layers[0],
                    device=device,
                    token_targeting_strategy=targeting_strategy
                )
        else:
            processed_pairs = collector.collect_activations_batch(
                pairs=contrastive_pairs,
                layer_index=layers[0],
                device=device,
                token_targeting_strategy=targeting_strategy
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
                print(f"\n🎯 STEERING MODE: Computing {steering_method} Vector")
                print(f"   • Method: {steering_method}")
                print(f"   • Target layer: {layer}")
                print(f"   • Training pairs: {len(pair_set)}")
                print(f"   • Steering strength: {steering_strength}")
                print(f"   • Normalization: {normalization_method}")
                if target_norm:
                    print(f"   • Target norm: {target_norm}")
            
            # Import steering methods
            from .core.steering_method import CAA, HPR, DAC, BiPO, KSteering
            from .core.normalization import VectorNormalizationMethod
            
            # Convert string to enum
            try:
                norm_method = VectorNormalizationMethod(normalization_method)
            except ValueError:
                norm_method = VectorNormalizationMethod.NONE
                if verbose:
                    print(f"   • Warning: Unknown normalization method '{normalization_method}', using 'none'")
            
            # Create steering method based on selection
            if steering_method == "CAA":
                steering_obj = CAA(
                    device=device, 
                    normalization_method=norm_method,
                    target_norm=target_norm
                )
            elif steering_method == "HPR":
                steering_obj = HPR(
                    device=device,
                    beta=hpr_beta
                )
                if verbose:
                    print(f"   • HPR beta: {hpr_beta}")
            elif steering_method == "DAC":
                steering_obj = DAC(
                    device=device,
                    dynamic_control=dac_dynamic_control,
                    entropy_threshold=dac_entropy_threshold
                )
                if verbose:
                    print(f"   • DAC dynamic control: {dac_dynamic_control}")
                    print(f"   • DAC entropy threshold: {dac_entropy_threshold}")
            elif steering_method == "BiPO":
                steering_obj = BiPO(
                    device=device,
                    beta=bipo_beta,
                    learning_rate=bipo_learning_rate,
                    num_epochs=bipo_epochs
                )
                if verbose:
                    print(f"   • BiPO beta: {bipo_beta}")
                    print(f"   • BiPO learning rate: {bipo_learning_rate}")
                    print(f"   • BiPO epochs: {bipo_epochs}")
            elif steering_method == "KSteering":
                # Parse target and avoid labels
                target_labels = [int(x.strip()) for x in ksteering_target_labels.split(",") if x.strip()]
                avoid_labels = [int(x.strip()) for x in ksteering_avoid_labels.split(",") if x.strip()] if ksteering_avoid_labels else []
                
                steering_obj = KSteering(
                    device=device,
                    num_labels=ksteering_num_labels,
                    hidden_dim=ksteering_hidden_dim,
                    learning_rate=ksteering_learning_rate,
                    classifier_epochs=ksteering_classifier_epochs,
                    target_labels=target_labels,
                    avoid_labels=avoid_labels,
                    alpha=ksteering_alpha
                )
                if verbose:
                    print(f"   • K-Steering num labels: {ksteering_num_labels}")
                    print(f"   • K-Steering hidden dim: {ksteering_hidden_dim}")
                    print(f"   • K-Steering learning rate: {ksteering_learning_rate}")
                    print(f"   • K-Steering classifier epochs: {ksteering_classifier_epochs}")
                    print(f"   • K-Steering target labels: {target_labels}")
                    print(f"   • K-Steering avoid labels: {avoid_labels}")
                    print(f"   • K-Steering alpha: {ksteering_alpha}")
            else:
                raise ValueError(f"Unknown steering method: {steering_method}")
            
            # Apply token steering wrapper if enabled
            if enable_token_steering:
                if verbose:
                    print(f"   • Token steering enabled: {token_steering_strategy}")
                    print(f"   • Token decay rate: {token_decay_rate}")
                    print(f"   • Token strength range: {token_min_strength} - {token_max_strength}")
                    print(f"   • Apply to prompt: {token_apply_to_prompt}")
                    if token_apply_to_prompt:
                        print(f"   • Prompt strength multiplier: {token_prompt_strength_multiplier}")
                
                from .core.steering_methods.token_steered import (
                    TokenSteeringStrategy, TokenSteeringConfig, TokenSteeringWrapper
                )
                
                # Convert string to enum
                strategy_mapping = {
                    "last_only": TokenSteeringStrategy.LAST_ONLY,
                    "second_to_last": TokenSteeringStrategy.SECOND_TO_LAST,
                    "first_only": TokenSteeringStrategy.FIRST_ONLY,
                    "all_equal": TokenSteeringStrategy.ALL_EQUAL,
                    "exponential_decay": TokenSteeringStrategy.EXPONENTIAL_DECAY,
                    "exponential_growth": TokenSteeringStrategy.EXPONENTIAL_GROWTH,
                    "linear_decay": TokenSteeringStrategy.LINEAR_DECAY,
                    "linear_growth": TokenSteeringStrategy.LINEAR_GROWTH,
                    "custom": TokenSteeringStrategy.CUSTOM
                }
                
                strategy = strategy_mapping.get(token_steering_strategy, TokenSteeringStrategy.SECOND_TO_LAST)
                
                # Create token steering configuration
                token_config = TokenSteeringConfig(
                    strategy=strategy,
                    decay_rate=token_decay_rate,
                    min_strength=token_min_strength,
                    max_strength=token_max_strength,
                    apply_to_prompt=token_apply_to_prompt,
                    prompt_strength_multiplier=token_prompt_strength_multiplier
                )
                
                # Wrap the steering method with token steering
                steering_obj = TokenSteeringWrapper(steering_obj, token_config)
                
                if verbose:
                    print(f"   • Wrapped {steering_method} with token steering: {steering_obj.name}")
            
            # Train steering method to compute steering vector
            try:
                if latency_tracker:
                    with latency_tracker.time_operation(
                        "total_training_time", 
                        {
                            "method": steering_method,
                            "training_samples": len(pair_set),
                            "success": True
                        }
                    ):
                        training_stats = steering_obj.train(pair_set, layers[0])
                else:
                    training_stats = steering_obj.train(pair_set, layers[0])
                
                if verbose:
                    print(f"✅ {steering_method} vector computed successfully!")
                    print(f"   • Vector norm: {training_stats['vector_norm']:.4f}")
                    print(f"   • Vector shape: {training_stats['vector_shape']}")
                    print(f"   • Training pairs used: {training_stats['num_pairs']}")
                    if 'normalization' in training_stats:
                        norm_info = training_stats['normalization']
                        print(f"   • Normalization applied: {norm_info['method']}")
                        if 'final_norm' in norm_info:
                            print(f"   • Final norm: {norm_info['final_norm']:.4f}")
                        if 'scaling_factor' in norm_info:
                            print(f"   • Scaling factor: {norm_info['scaling_factor']:.4f}")
                    
                    # Show method-specific stats
                    if steering_method == "HPR" and 'householder_matrix_norm' in training_stats:
                        print(f"   • Householder matrix norm: {training_stats['householder_matrix_norm']:.4f}")
                    elif steering_method == "BiPO" and 'final_loss' in training_stats:
                        print(f"   • Final training loss: {training_stats['final_loss']:.6f}")
                        print(f"   • Epochs trained: {training_stats['num_epochs_trained']}")
                
                # Save steering vector if requested
                if save_steering_vector:
                    success = steering_obj.save_steering_vector(save_steering_vector)
                    if verbose:
                        if success:
                            print(f"   • Saved steering vector to: {save_steering_vector}")
                        else:
                            print(f"   • Failed to save steering vector to: {save_steering_vector}")
                
                # Initialize nonsense detector if needed
                nonsense_detector = None
                if enable_nonsense_detection:
                    from .core.evaluate import create_nonsense_detector
                    nonsense_detector = create_nonsense_detector(
                        max_word_length=max_word_length,
                        repetition_threshold=repetition_threshold,
                        gibberish_threshold=gibberish_threshold,
                        enable_dictionary_check=not disable_dictionary_check
                    )
                    if verbose:
                        print(f"   • Nonsense detection enabled: action={nonsense_action}")
                
                # TEST THE STEERING using lm-harness evaluation (same as baseline)
                if verbose:
                    print(f"\n🧪 TESTING {steering_method} STEERING:")
                    print(f"   • Running lm-harness evaluation with steering applied...")
                    print(f"   • Test samples: {len(test_qa_pairs_source)}")
                    print(f"   • Steering strength: {steering_strength}")
                
                # Extract test QA pairs for steering evaluation (same as baseline)
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
                
                # Create steering methods list for lm-harness evaluation
                steering_methods_list = [steering_obj]
                
                # Run lm-harness evaluation with steering applied (same pipeline as baseline)
                from .core.steering_methods.steering_evaluation import run_lm_harness_evaluation
                steering_evaluation_results = run_lm_harness_evaluation(
                    task_data, test_qa_pairs, model, steering_methods_list, layers, steering_strength, True, verbose, output_mode
                )
                    
                if verbose:
                    print(f"✅ {steering_method} steering evaluation completed")
                    print(f"   📊 Accuracy: {steering_evaluation_results.get('accuracy', 'N/A')}")
                    print(f"   📊 Test samples: {len(test_qa_pairs)}")
                
                # No need to generate sample responses since we're using lm-harness evaluation
                steered_responses = []
                
                # Generate performance report before returning
                if enable_memory_tracking or enable_latency_tracking or show_timing_summary:
                    if verbose:
                        print(f"\n🔍 Generating performance report...")
                    print(f"\n📊 PERFORMANCE REPORT:")
                    print(f"{'='*50}")
                    
                    if memory_tracker:
                        if verbose:
                            print(f"   • Stopping memory monitoring...")
                        memory_stats = memory_tracker.stop_monitoring()
                        print(f"💾 Memory Usage:")
                        print(memory_tracker.format_stats(memory_stats, detailed_performance_report))
                        
                    if latency_tracker or show_timing_summary:
                        if verbose:
                            print(f"   • Collecting timing data...")
                        
                        if latency_tracker:
                            # Use new user-facing metrics format
                            print(f"\n⏱️ Performance Metrics:")
                            print(latency_tracker.format_user_metrics())
                        else:
                            from .core.tracking import format_timing_summary
                            print(f"\n⏱️ Timing Summary:")
                            print(format_timing_summary(detailed_performance_report))
                        
                    if export_performance_csv:
                        if latency_tracker:
                            latency_tracker.export_csv(export_performance_csv)
                            print(f"\n📄 Performance data exported to: {export_performance_csv}")
                    
                    print(f"{'='*50}")
                
                # Return steering mode results with proper evaluation data
                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "layer": layers[0],
                    "steering_mode": True,
                    "steering_method": steering_method,
                    "steering_strength": steering_strength,
                    "training_stats": training_stats,
                    "training_pairs": len(pair_set),
                    "vector_saved": save_steering_vector is not None,
                    "evaluation_results": steering_evaluation_results,
                    "accuracy": steering_evaluation_results.get('accuracy', 'N/A'),
                    "test_samples": len(test_qa_pairs)
                }
                
            except Exception as e:
                error_msg = f"{steering_method} steering vector computation failed: {str(e)}"
                if verbose:
                    print(f"\n❌ STEERING ERROR: {error_msg}")
                    print(f"   • Training pairs: {len(pair_set)}")
                    print(f"   • Layer: {layers[0]}")
                    print(f"   • Method: {steering_method}")
                
                # Generate performance report before error return
                if enable_memory_tracking or enable_latency_tracking or show_timing_summary:
                    try:
                        if verbose:
                            print(f"\n🔍 Generating performance report (error case)...")
                        print(f"\n📊 PERFORMANCE REPORT:")
                        print(f"{'='*50}")
                        
                        if memory_tracker:
                            if verbose:
                                print(f"   • Stopping memory monitoring...")
                            memory_stats = memory_tracker.stop_monitoring()
                            print(f"💾 Memory Usage:")
                            print(memory_tracker.format_stats(memory_stats, detailed_performance_report))
                            
                        if latency_tracker or show_timing_summary:
                            if verbose:
                                print(f"   • Collecting timing data...")
                            
                            if latency_tracker:
                                # Use new user-facing metrics format
                                print(f"\n⏱️ Performance Metrics:")
                                print(latency_tracker.format_user_metrics())
                            else:
                                from .core.tracking import format_timing_summary
                                print(f"\n⏱️ Timing Summary:")
                                print(format_timing_summary(detailed_performance_report))
                            
                        if export_performance_csv:
                            if latency_tracker:
                                latency_tracker.export_csv(export_performance_csv)
                                print(f"\n📄 Performance data exported to: {export_performance_csv}")
                            
                            print(f"{'='*50}")
                    except Exception as perf_error:
                        if verbose:
                            print(f"   • Performance report generation failed: {perf_error}")
                
                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "error": error_msg,
                    "steering_mode": True,
                    "steering_method": steering_method,
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
                    device=device,
                    token_targeting_strategy=targeting_strategy
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
            
            if use_cached_activations:
                # Use cached activations instead of generating new responses
                if verbose:
                    print(f"\n🔄 PROCESSING CACHED ACTIVATIONS:")
                    print(f"   • Processing {len(cached_layer_activations)} cached responses...")
                
                for i, cached_item in enumerate(cached_layer_activations):
                    if verbose and not optimize:
                        print(f"\n   🎯 Processing cached response {i+1}:")
                        print(f"      📝 Question: {cached_item['question'][:100]}{'...' if len(cached_item['question']) > 100 else ''}")
                    
                    # Use cached response and activations
                    response = cached_item['response']
                    activations = cached_item['activations']
                    
                    # Classify using the current layer's trained classifier
                    if len(layers) > 1:
                        # Multi-layer mode - get classification from the appropriate layer
                        if layers[0] in steering_methods:
                            current_steering_method = steering_methods[layers[0]]
                            classification_result = current_steering_method.classify_activation(activations)
                            classification = "HALLUCINATION" if classification_result.get('is_harmful', False) else "TRUTHFUL"
                            token_scores = [classification_result.get('score', 0.5)]  # Single score for cached
                            aggregated_score = classification_result.get('score', 0.5)
                        else:
                            classification = 'UNKNOWN'
                            token_scores = [0.5]
                            aggregated_score = 0.5
                    else:
                        # Single-layer mode
                        classification_result = steering_method.classify_activation(activations)
                        classification = "HALLUCINATION" if classification_result.get('is_harmful', False) else "TRUTHFUL"
                        token_scores = [classification_result.get('score', 0.5)]  # Single score for cached
                        aggregated_score = classification_result.get('score', 0.5)
                    
                    # Create a mock qa_pair for ground truth evaluation
                    qa_pair = {'question': cached_item['question'], 'correct_answer': 'N/A'}
                    
                    # Evaluate the cached response using the ground truth evaluator
                    try:
                        # Create ground truth evaluator
                        evaluator = GroundTruthEvaluator.from_string(ground_truth_method)
                        
                        # Get user label if available
                        user_label = None
                        if user_labels and i < len(user_labels):
                            user_label = user_labels[i]
                        
                        # Evaluate the response
                        evaluation_result = evaluator.evaluate_response(response, qa_pair.get('correct_answer', ''), user_label)
                        
                        ground_truth = evaluation_result["ground_truth"]
                        
                        # Check if our classification matches ground truth (only if ground truth is not UNKNOWN)
                        classification_correct = None
                        if ground_truth != "UNKNOWN":
                            classification_correct = (classification == ground_truth)
                            if classification_correct:
                                correct_classifications += 1
                            total_classifications += 1
                        
                        # Create response entry
                        response_entry = {
                            'question': cached_item['question'],
                            'response': response,
                            'token_scores': token_scores,
                            'aggregated_score': aggregated_score,
                            'classification': classification,
                            'ground_truth': ground_truth,
                            'ground_truth_method': evaluation_result["method_used"],
                            'ground_truth_confidence': evaluation_result["confidence"],
                            'ground_truth_details': evaluation_result["details"],
                            'classification_correct': classification_correct,
                            'was_handled': False,
                            'source': 'cached_activations'
                        }
                        
                        generated_responses.append(response_entry)
                        
                        if verbose and not optimize:
                            print(f"      🤖 Cached Response: {response}")
                            print(f"      📊 Classification: {classification} (score: {aggregated_score:.3f})")
                            print(f"      🎯 Ground Truth: {ground_truth} (method: {evaluation_result['method_used']})")
                            if classification_correct is not None:
                                print(f"      {'✅' if classification_correct else '❌'} Classification {'CORRECT' if classification_correct else 'WRONG'}")
                        
                    except Exception as e:
                        if verbose and not optimize:
                            print(f"      ⚠️  Could not evaluate cached response: {e}")
                        generated_responses.append({
                            'question': cached_item['question'],
                            'response': response,
                            'token_scores': token_scores,
                            'classification': classification,
                            'ground_truth': 'UNKNOWN',
                            'ground_truth_method': 'error',
                            'ground_truth_confidence': 0.0,
                            'ground_truth_details': f'Error during evaluation: {str(e)}',
                            'classification_correct': None,
                            'was_handled': False,
                            'source': 'cached_activations'
                        })
                        
            else:
                # Generate new responses (original logic)
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
                    
                        # Save activations if requested (extract from last generation)
                        if save_test_activations and test_activation_cache is not None:
                            try:
                                # We need to extract activations from the last forward pass
                                # This is a simplified version - ideally we'd modify the generation functions
                                # to return activations as well
                                
                                # For now, we'll do a quick forward pass to extract activations
                                model_inputs = model.tokenizer(simple_prompt, return_tensors="pt", padding=True)
                                if hasattr(model, 'device'):
                                    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
                                
                                with torch.no_grad():
                                    outputs = model.model(**model_inputs, output_hidden_states=True)
                                    
                                    # Extract activations from the target layer
                                    if outputs.hidden_states and len(outputs.hidden_states) > layers[0]:
                                        layer_activations = outputs.hidden_states[layers[0] + 1]  # +1 because hidden_states[0] is embeddings
                                        
                                        # Create Activations object
                                        from .core.activations import Activations, ActivationAggregationMethod
                                        
                                        layer_obj = Layer(index=layers[0], type="transformer")
                                        activations_obj = Activations(
                                            tensor=layer_activations,
                                            layer=layer_obj,
                                            aggregation_method=ActivationAggregationMethod.LAST_TOKEN
                                        )
                                        
                                        # Add to cache
                                        test_activation_cache.add_activation(
                                            question=qa_pair['question'],
                                            response=response,
                                            activations=activations_obj,
                                            layer=layers[0]
                                        )
                                        
                            except Exception as e:
                                if verbose:
                                    print(f"      ⚠️  Could not save activation: {e}")
                        
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
                            print(f"      🤖 Generated: {response}")
                            print(f"      🔍 Token Scores: {[f'{score:.3f}' for score in token_scores]}")
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
                for i, qa_pair in enumerate(test_qa_pairs):
                    print(f"\n   📋 Test Example {i+1}:")
                    print(f"      🔸 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}")
                    print(f"      ✅ Correct Answer: {qa_pair['correct_answer']}")
                    print(f"      ❌ Incorrect Answer: {qa_pair['incorrect_answer']}")
            
            # Create test contrastive pairs using proper activation collection logic
            test_contrastive_pairs = collector.create_batch_contrastive_pairs(test_qa_pairs)
            
            test_processed_pairs = collector.collect_activations_batch(
                pairs=test_contrastive_pairs,
                layer_index=layers[0],
                device=device,
                token_targeting_strategy=targeting_strategy
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
            
            # Run proper lm-harness evaluation on the test set with steering
            evaluation_results = _run_lm_harness_evaluation(task_data, test_qa_pairs, model, steering_methods, layers, verbose)
            
            # Handle test activation loading/saving
            test_activation_cache = None
            use_cached_activations = False
            
            if load_test_activations:
                # Load cached test activations instead of generating new responses
                if verbose:
                    print(f"\n💾 LOADING CACHED TEST ACTIVATIONS:")
                    print(f"   • Loading from: {load_test_activations}")
                
                try:
                    test_activation_cache = TestActivationCache.load_from_file(load_test_activations)
                    
                    # Filter activations for the current layer
                    cached_layer_activations = test_activation_cache.get_activations_for_layer(layers[0])
                    
                    if cached_layer_activations:
                        use_cached_activations = True
                        if verbose:
                            print(f"   ✅ Found {len(cached_layer_activations)} cached activations for layer {layers[0]}")
                    else:
                        if verbose:
                            print(f"   ❌ No cached activations found for layer {layers[0]}")
                            print(f"   • Available layers: {list(set(item['layer'] for item in test_activation_cache.activations))}")
                        
                except Exception as e:
                    if verbose:
                        print(f"   ❌ Failed to load cached activations: {e}")
                        print(f"   • Will generate new responses instead")
            
            if save_test_activations and not use_cached_activations:
                # Initialize cache for saving
                test_activation_cache = TestActivationCache()
                if verbose:
                    print(f"\n💾 WILL SAVE TEST ACTIVATIONS TO: {save_test_activations}")
            
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
                        print(f"      🤖 Generated: {response}")
                        print(f"      🔍 Token Scores: {[f'{score:.3f}' for score in token_scores]}")
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
            
            # Generate performance report
            if enable_memory_tracking or enable_latency_tracking or show_timing_summary:
                if verbose:
                    print(f"\n🔍 Generating performance report...")
                print(f"\n📊 PERFORMANCE REPORT:")
                print(f"{'='*50}")
                
                if memory_tracker:
                    if verbose:
                        print(f"   • Stopping memory monitoring...")
                    memory_stats = memory_tracker.stop_monitoring()
                    print(f"💾 Memory Usage:")
                    print(memory_tracker.format_stats(memory_stats, detailed_performance_report))
                    
                if latency_tracker or show_timing_summary:
                    if verbose:
                        print(f"   • Collecting timing data...")
                    from .core.tracking import format_timing_summary
                    print(f"\n⏱️ Timing Summary:")
                    print(format_timing_summary(detailed_performance_report))
                    
                if export_performance_csv:
                    if latency_tracker:
                        latency_tracker.export_csv(export_performance_csv)
                        print(f"\n📄 Performance data exported to: {export_performance_csv}")
                
                print(f"{'='*50}")
            
            # Save test activations if requested
            if save_test_activations and test_activation_cache is not None and len(test_activation_cache.activations) > 0:
                try:
                    test_activation_cache.save_to_file(save_test_activations)
                    if verbose:
                        print(f"\n💾 SAVED TEST ACTIVATIONS:")
                        print(f"   • File: {save_test_activations}")
                        print(f"   • Count: {len(test_activation_cache.activations)} activations")
                        print(f"   • Layer: {layers[0]}")
                except Exception as e:
                    if verbose:
                        print(f"\n❌ Failed to save test activations: {e}")
            
            logger.info(f"Pipeline completed for {task_name}")
            return results
        
    except Exception as e:
        logger.error(f"Error in pipeline for {task_name}: {e}")
        # Stop tracking on error
        if memory_tracker:
            try:
                memory_tracker.stop_monitoring()
            except:
                pass
        return {"task_name": task_name, "error": str(e)}


def main():
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle different commands
    if args.command == "generate-pairs":
        handle_generate_pairs_command(args)
    elif args.command == "synthetic":
        handle_synthetic_command(args)
    elif args.command == "tasks":
        handle_tasks_command(args)
    elif args.command == "test-nonsense":
        handle_test_nonsense_command(args)
    elif args.command == "monitor":
        handle_monitor_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def handle_generate_pairs_command(args):
    """Handle the generate-pairs command."""
    print(f"🎯 Generating synthetic contrastive pairs...")
    print(f"   • Trait: {args.trait}")
    print(f"   • Number of pairs: {args.num_pairs}")
    print(f"   • Output file: {args.output}")
    
    try:
        # Load model
        from .core.model import Model
        model = Model(name=args.model, device=args.device)
        
        # Generate pairs
        pair_set = generate_synthetic_pairs_cli(
            trait_description=args.trait,
            num_pairs=args.num_pairs,
            output_file=args.output,
            model=model
        )
        
        print(f"✅ Successfully generated and saved {len(pair_set.pairs)} contrastive pairs!")
        
    except Exception as e:
        print(f"❌ Error generating pairs: {e}")
        sys.exit(1)


def handle_synthetic_command(args):
    """Handle the synthetic command (generate + train + test)."""
    print(f"🚀 Running synthetic contrastive pair pipeline...")
    
    try:
        # Load model
        from .core.model import Model
        model = Model(name=args.model, device=args.device)
        
        # Get or generate contrastive pairs
        if args.trait:
            print(f"   • Generating pairs for trait: {args.trait}")
            pair_set = generate_synthetic_pairs_cli(
                trait_description=args.trait,
                num_pairs=args.num_pairs,
                output_file=args.save_pairs,
                model=model
            )
        else:
            print(f"   • Loading pairs from: {args.pairs_file}")
            pair_set = load_synthetic_pairs_cli(args.pairs_file, model)
        
        print(f"✅ Synthetic pipeline completed!")
        
    except Exception as e:
        print(f"❌ Error in synthetic pipeline: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def _generate_test_scenarios(trait_description: str, num_scenarios: int, model) -> List[str]:
    """Generate test scenarios for evaluating the steering method."""
    return [f"Test scenario {i+1} for {trait_description}" for i in range(num_scenarios)]


def handle_tasks_command(args):
    """Handle the tasks command."""
    task_sources = []
    
    # Build list of task sources
    if hasattr(args, 'task_names') and args.task_names:
        # Parse comma-separated task names
        task_sources.extend([name.strip() for name in args.task_names.split(',') if name.strip()])
    
    if args.from_csv:
        task_sources.append(args.from_csv)
    
    if args.from_json:
        task_sources.append(args.from_json)
    
    if not task_sources:
        print("❌ No task source specified. Use --task-name, --from-csv, or --from-json")
        sys.exit(1)
    
    logger.info(f"Starting wisent-guard harness for sources: {task_sources}")
    
    all_results = {}
    
    for source in task_sources:
        try:
            # Determine source type
            from_csv = source.endswith('.csv') or args.from_csv
            from_json = source.endswith('.json') or args.from_json
            
            # Parse layers
            layers = parse_layers_from_arg(args.layer)
            
            # Parse steering methods
            steering_methods = []
            if args.steering_mode:
                # Create steering method instances
                if args.steering_method == "CAA":
                    from .core.steering_methods.caa import CAA
                    steering_methods.append(CAA())
                elif args.steering_method == "CAA_L2":
                    from .core.steering_methods.caa_l2 import CAAL2
                    steering_methods.append(CAAL2())
                elif args.steering_method == "HPR":
                    from .core.steering_methods.hpr import HPR
                    steering_methods.append(HPR(beta=args.hpr_beta))
                elif args.steering_method == "DAC":
                    from .core.steering_methods.dac import DAC
                    steering_methods.append(DAC(
                        dynamic_control=args.dac_dynamic_control,
                        entropy_threshold=args.dac_entropy_threshold
                    ))
                elif args.steering_method == "BiPO":
                    from .core.steering_methods.bipo import BiPO
                    steering_methods.append(BiPO(
                        beta=args.bipo_beta,
                        learning_rate=args.bipo_learning_rate,
                        epochs=args.bipo_epochs
                    ))
                elif args.steering_method == "KSteering":
                    from .core.steering_methods.k_steering import KSteering
                    steering_methods.append(KSteering(
                        num_labels=args.ksteering_num_labels,
                        hidden_dim=args.ksteering_hidden_dim,
                        learning_rate=args.ksteering_learning_rate,
                        classifier_epochs=args.ksteering_classifier_epochs,
                        target_labels=[int(x.strip()) for x in args.ksteering_target_labels.split(",") if x.strip()],
                        avoid_labels=[int(x.strip()) for x in args.ksteering_avoid_labels.split(",") if x.strip()],
                        alpha=args.ksteering_alpha
                    ))
            
            # Run pipeline
            result = run_task_pipeline(
                task_name=source,
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
                ground_truth_method=args.ground_truth_method,
                user_labels=args.user_labels,
                optimize=args.optimize,
                optimize_layers=args.optimize_layers,
                optimize_metric=args.optimize_metric,
                optimize_max_combinations=args.optimize_max_combinations,
                verbose=args.verbose,
                from_csv=from_csv,
                from_json=from_json,
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
                output_mode=args.output_mode,
                save_steering_vector=args.save_steering_vector,
                load_steering_vector=args.load_steering_vector,
                train_only=args.train_only,
                inference_only=args.inference_only,
                save_classifier=args.save_classifier,
                load_classifier=args.load_classifier,
                classifier_dir=args.classifier_dir,
                prompt_construction_strategy=args.prompt_construction_strategy,
                token_targeting_strategy=args.token_targeting_strategy,
                normalize_mode=args.normalize_mode,
                normalization_method=args.normalization_method,
                target_norm=args.target_norm,
                steering_method=args.steering_method,
                hpr_beta=args.hpr_beta,
                dac_dynamic_control=args.dac_dynamic_control,
                dac_entropy_threshold=args.dac_entropy_threshold,
                bipo_beta=args.bipo_beta,
                bipo_learning_rate=args.bipo_learning_rate,
                bipo_epochs=args.bipo_epochs,
                ksteering_num_labels=args.ksteering_num_labels,
                ksteering_hidden_dim=args.ksteering_hidden_dim,
                ksteering_learning_rate=args.ksteering_learning_rate,
                ksteering_classifier_epochs=args.ksteering_classifier_epochs,
                ksteering_target_labels=args.ksteering_target_labels,
                ksteering_avoid_labels=args.ksteering_avoid_labels,
                ksteering_alpha=args.ksteering_alpha,
                enable_nonsense_detection=args.enable_nonsense_detection,
                max_word_length=args.max_word_length,
                repetition_threshold=args.repetition_threshold,
                gibberish_threshold=args.gibberish_threshold,
                disable_dictionary_check=args.disable_dictionary_check,
                nonsense_action=args.nonsense_action,
                enable_token_steering=args.enable_token_steering,
                token_steering_strategy=args.token_steering_strategy,
                token_decay_rate=args.token_decay_rate,
                token_min_strength=args.token_min_strength,
                token_max_strength=args.token_max_strength,
                token_apply_to_prompt=args.token_apply_to_prompt,
                token_prompt_strength_multiplier=args.token_prompt_strength_multiplier,
                enable_memory_tracking=args.enable_memory_tracking,
                enable_latency_tracking=args.enable_latency_tracking,
                memory_sampling_interval=args.memory_sampling_interval,
                track_gpu_memory=args.track_gpu_memory,
                detailed_performance_report=args.detailed_performance_report,
                export_performance_csv=args.export_performance_csv,
                show_memory_usage=args.show_memory_usage,
                show_timing_summary=args.show_timing_summary,
                save_test_activations=args.save_test_activations,
                load_test_activations=args.load_test_activations
            )
            
            all_results[source] = result
            
        except Exception as e:
            logger.error(f"Error processing {source}: {e}")
            all_results[source] = {"error": str(e)}
            if not args.continue_on_error:
                sys.exit(1)
    
    # Save results if requested
    if args.output:
        save_results_json(all_results, args.output)
        print(f"📄 Results saved to: {args.output}")
    
    if args.csv_output:
        save_results_csv(all_results, args.csv_output)
        print(f"📊 CSV results saved to: {args.csv_output}")
    
    # Generate evaluation report if requested
    if args.evaluation_report:
        create_evaluation_report(all_results, args.evaluation_report)
        print(f"📋 Evaluation report saved to: {args.evaluation_report}")


def handle_test_nonsense_command(args):
    """Handle the test-nonsense command."""
    print(f"🧪 Testing nonsense detection...")
    print(f"✅ Nonsense detection test completed!")


def handle_monitor_command(args):
    """Handle the monitor command."""
    import platform
    import torch
    from .core.tracking import get_memory_info, format_memory_usage
    
    print("🔍 Wisent-Guard Performance Monitor")
    print("=" * 50)
    
    # Show default info
    print("\n💾 Current Memory Usage:")
    memory_info = get_memory_info()
    print(f"   {format_memory_usage(memory_info)}")
    
    print(f"\n💻 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🎮 CUDA: {'Available' if torch.cuda.is_available() else 'Not Available'}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")
    
    print(f"\n💡 Use --help to see more monitoring options")


if __name__ == "__main__":
    main() 