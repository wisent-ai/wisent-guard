"""
Intelligent hyperparameter optimization with caching for wisent-guard.

This module handles:
1. Layer optimization using contrastive pairs (training data)
2. Aggregation method optimization using ground truth (validation data)
3. Caching of results to avoid recomputation
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_cache_key(model_name: str, task_name: str, limit: int, ground_truth_method: str) -> str:
    """Generate a unique cache key for optimization results."""
    cache_input = f"{model_name}_{task_name}_{limit}_{ground_truth_method}"
    return hashlib.md5(cache_input.encode()).hexdigest()


def get_cache_path(cache_key: str) -> Path:
    """Get the cache file path for storing optimization results."""
    cache_dir = Path("opt_param")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{cache_key}.json"


def load_cached_results(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load cached optimization results if they exist and are valid."""
    cache_path = get_cache_path(cache_key)
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached_data = json.load(f)

            # Validate that cached results have valid (non-None) values
            required_fields = ["best_layer", "best_aggregation", "best_classifier_type", "best_threshold"]
            for field in required_fields:
                if cached_data.get(field) is None:
                    print(f"⚠️ Invalid cached results (field '{field}' is None), ignoring cache")
                    # Delete the corrupted cache file
                    cache_path.unlink()
                    return None

            # Additional validation: ensure best_accuracy is reasonable
            if cached_data.get("best_accuracy", 0.0) <= 0.0:
                print(f"⚠️ Invalid cached results (accuracy {cached_data.get('best_accuracy', 0.0)}), ignoring cache")
                cache_path.unlink()
                return None

            return cached_data

        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
            # Delete corrupted cache file
            try:
                cache_path.unlink()
            except:
                pass
    return None


def save_cached_results(cache_key: str, results: Dict[str, Any]) -> None:
    """Save optimization results to cache."""
    cache_path = get_cache_path(cache_key)
    try:
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Optimization results cached to {cache_path}")
    except Exception as e:
        print(f"⚠️ Error saving cache: {e}")


def optimize_layers_on_contrastive_pairs(
    model, collector, contrastive_pairs: List, device: str, verbose: bool = False, optimize_layers: str = "all"
) -> Dict[int, Dict[str, Any]]:
    """
    Optimize layers using contrastive pairs (training data).

    Returns:
        Dict mapping layer_idx -> {'classifier': classifier, 'train_accuracy': float}
    """
    from wisent_guard.core.classifier.classifier import Classifier

    from .core.hyperparameter_optimizer import detect_model_layers
    from .core.parser import parse_layer_range

    total_layers = detect_model_layers(model)

    # Parse layer range
    if optimize_layers == "all":
        candidate_layers = list(range(total_layers))
    else:
        candidate_layers = parse_layer_range(optimize_layers, model)
        if candidate_layers is None:
            candidate_layers = list(range(total_layers))

    if verbose:
        print("🔬 LAYER OPTIMIZATION ON CONTRASTIVE PAIRS:")
        print(f"   • Total layers: {total_layers}")
        print(f"   • Testing candidate layers: {candidate_layers}")

    layer_results = {}

    for layer_idx in candidate_layers:
        try:
            if verbose:
                print(f"   • Training classifier for layer {layer_idx}...")

            # Extract training activations for this layer
            processed_pairs = collector.collect_activations_batch(
                pairs=contrastive_pairs, layer_index=layer_idx, device=device
            )

            # Prepare training data
            X_train = []
            y_train = []

            for pair in processed_pairs:
                if hasattr(pair, "positive_activations") and pair.positive_activations is not None:
                    X_train.append(pair.positive_activations.detach().cpu().flatten().numpy())
                    y_train.append(0)  # truthful
                if hasattr(pair, "negative_activations") and pair.negative_activations is not None:
                    X_train.append(pair.negative_activations.detach().cpu().flatten().numpy())
                    y_train.append(1)  # hallucination

            if len(X_train) < 2 or len(set(y_train)) != 2:
                if verbose:
                    print(f"     ⚠️ Insufficient data for layer {layer_idx}")
                continue

            # Train classifier
            classifier = Classifier(model_type="logistic", threshold=0.5)
            training_results = classifier.fit(X_train, y_train, test_size=0.2, random_state=42)
            train_accuracy = training_results.get("accuracy", 0.0)

            layer_results[layer_idx] = {
                "classifier": classifier,
                "train_accuracy": train_accuracy,
                "training_samples": len(X_train),
            }

            if verbose:
                print(f"     ✅ Layer {layer_idx}: Train accuracy = {train_accuracy:.3f}")

        except Exception as e:
            if verbose:
                print(f"     ❌ Error training layer {layer_idx}: {e}")
            continue

    return layer_results


def optimize_aggregation_on_ground_truth(
    model,
    collector,
    contrastive_pairs: List,
    layer_results: Dict[int, Dict[str, Any]],
    test_qa_pairs: List[Dict],
    ground_truth_method: str,
    task_name: str,
    max_new_tokens: int,
    device: str,
    verbose: bool = False,
    classifier_types: List[str] = None,
    thresholds: List[float] = None,
) -> Dict[str, Any]:
    """
    Optimize aggregation methods + classifier types + thresholds using ground truth labels ONLY.

    This is the validation phase - no training data used here.
    """
    from .core import ContrastivePairSet, SteeringMethod, SteeringType
    from .core.ground_truth_evaluator import GroundTruthEvaluator

    # Set defaults
    if classifier_types is None:
        classifier_types = ["logistic", "mlp"]
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    aggregation_methods = ["average", "final", "first", "max", "min"]

    if verbose:
        print("\n🎯 HYPERPARAMETER OPTIMIZATION ON GROUND TRUTH:")
        print(f"   • Testing layers: {list(layer_results.keys())}")
        print(f"   • Testing aggregations: {aggregation_methods}")
        print(f"   • Testing classifier types: {classifier_types}")
        print(f"   • Testing thresholds: {thresholds}")
        print(
            f"   • Total combinations: {len(layer_results) * len(aggregation_methods) * len(classifier_types) * len(thresholds)}"
        )
        print(f"   • Ground truth method: {ground_truth_method}")
        print(f"   • Test QA pairs: {len(test_qa_pairs)}")
        if len(test_qa_pairs) > 0:
            print(f"   • First test question: {test_qa_pairs[0].get('formatted_question', 'N/A')[:100]}...")
        else:
            print("   ❌ CRITICAL: NO TEST QA PAIRS! Cannot perform ground truth optimization.")
            print("      This means test data loading failed completely.")

    # 🚨 HARD ERROR CHECK: No test data
    if len(test_qa_pairs) == 0:
        raise ValueError(
            f"❌ CRITICAL ERROR: No test QA pairs available for ground truth optimization!\n"
            f"   📊 Test data size: {len(test_qa_pairs)}\n"
            f"   💡 This indicates the test data loading phase failed.\n"
            f"   🛠️  Check test data extraction and QA pair formatting."
        )

    evaluator = GroundTruthEvaluator.from_string(ground_truth_method, task_name=task_name)
    if verbose:
        print(f"   • Ground truth evaluator: {evaluator.method.value}")
        print("   • Evaluator initialized successfully")

    best_combination = {
        "layer": None,
        "aggregation": None,
        "classifier_type": None,
        "threshold": None,
        "accuracy": 0.0,
        "correct": 0,
        "total": 0,
    }

    combination_results = {}

    # For each layer, we need to retrain classifiers with different types
    for layer_idx in layer_results:
        if verbose:
            print(f"   • Testing layer {layer_idx}...")

        # Train classifiers of different types for this layer
        layer_classifiers = {}
        for classifier_type in classifier_types:
            try:
                if verbose:
                    print(f"     🎯 Training {classifier_type} classifier for layer {layer_idx}...")

                # Extract activations for this layer and classifier type
                processed_pairs = collector.collect_activations_batch(
                    pairs=contrastive_pairs, layer_index=layer_idx, device=device
                )

                # Convert to ContrastivePairSet format for training
                phrase_pairs = []
                for pair in processed_pairs:
                    # Create the full prompts for the pair set
                    positive_full = f"{pair.prompt}{pair.positive_response}"
                    negative_full = f"{pair.prompt}{pair.negative_response}"

                    phrase_pairs.append(
                        {
                            "harmful": negative_full,  # A choice (incorrect)
                            "harmless": positive_full,  # B choice (correct)
                        }
                    )

                # Create ContrastivePairSet with the real activations
                pair_set = ContrastivePairSet.from_phrase_pairs(
                    name=f"layer_{layer_idx}_{classifier_type}_training",
                    phrase_pairs=phrase_pairs,
                    task_type="lm_evaluation",
                )

                # Store the real activations in the pair set response objects
                for i, processed_pair in enumerate(processed_pairs):
                    if i < len(pair_set.pairs):
                        # Assign activations to the response objects
                        if hasattr(pair_set.pairs[i], "positive_response") and pair_set.pairs[i].positive_response:
                            pair_set.pairs[i].positive_response.activations = processed_pair.positive_activations
                        if hasattr(pair_set.pairs[i], "negative_response") and pair_set.pairs[i].negative_response:
                            pair_set.pairs[i].negative_response.activations = processed_pair.negative_activations

                # Create steering method with this classifier type
                steering_type = SteeringType.LOGISTIC if classifier_type == "logistic" else SteeringType.MLP
                steering_method = SteeringMethod(
                    method_type=steering_type, threshold=0.5, device=device
                )  # Threshold will be tested separately

                # Train the classifier
                training_results = steering_method.train(pair_set)
                layer_classifiers[classifier_type] = steering_method

                if verbose:
                    print(f"       ✅ {classifier_type} trained: {training_results.get('accuracy', 'N/A'):.3f}")

            except Exception as e:
                if verbose:
                    print(f"       ❌ Error training {classifier_type}: {e}")
                continue

        # Generate responses and compute token scores ONCE per question per classifier type
        for classifier_type, steering_method in layer_classifiers.items():
            if verbose:
                print(f"     🎭 Generating responses with {classifier_type} classifier...")

            question_results = []
            for i, qa_pair in enumerate(test_qa_pairs):
                try:
                    question = qa_pair["formatted_question"]
                    if verbose:
                        print(f"       📝 Processing question {i + 1}/{len(test_qa_pairs)}: {question[:100]}...")

                    # Generate response with token scores
                    from .inference import generate_with_classification

                    response_text, token_scores, _ = generate_with_classification(
                        model,
                        question,
                        layer_idx,
                        max_new_tokens,
                        steering_method,
                        "average",
                        threshold=0.6,
                        verbose=False,  # Use any aggregation, we'll recalculate
                    )

                    if verbose:
                        print(f"       ✅ Generated response: {response_text[:100]}...")
                        print(f"       📊 Token scores length: {len(token_scores) if token_scores else 'None'}")

                    # Get ground truth label ONCE
                    gt_result = evaluator.evaluate_response(
                        response=response_text, expected_answer=qa_pair.get("correct_answer", None)
                    )
                    ground_truth_label = gt_result["ground_truth"]

                    if verbose:
                        print(f"       🎯 Ground truth: {ground_truth_label}")
                        print(f"       🔍 Ground truth details: {gt_result}")

                    # 🚨 HARD STOP: Ground truth evaluation failed
                    if ground_truth_label == "UNKNOWN":
                        print("\n💥💥💥 HARD STOP - GROUND TRUTH EVALUATION FAILED 💥💥💥")
                        print(f"Question: {question[:200]}...")
                        print(f"Response: {response_text[:200]}...")
                        print(f"Expected Answer: {qa_pair.get('correct_answer', 'N/A')}")
                        print(f"Ground Truth Method: {ground_truth_method}")
                        print(f"Evaluator Details: {gt_result}")
                        print("💡 Ground truth evaluation returned UNKNOWN instead of TRUTHFUL/HALLUCINATION")
                        print("🛠️  This means the ground truth method cannot evaluate this type of response")
                        print("💥💥💥 STOPPING OPTIMIZATION IMMEDIATELY 💥💥💥\n")

                        # HARD STOP - crash the entire process immediately
                        raise ValueError(
                            f"❌ CRITICAL ERROR: Ground truth evaluation failed!\n"
                            f"   📊 Ground truth: {ground_truth_label}\n"
                            f"   📝 Question: {question[:100]}...\n"
                            f"   📝 Response: {response_text[:100]}...\n"
                            f"   📝 Expected: {qa_pair.get('correct_answer', 'N/A')}\n"
                            f"   📝 Method: {ground_truth_method}\n"
                            f"   📝 Details: {gt_result}\n"
                            f"   💡 Ground truth evaluation must return TRUTHFUL or HALLUCINATION for optimization to work\n"
                            f"   🛠️  Check if the ground truth method is appropriate for this task type"
                        )

                    # 🚨 HARD STOP: Ground truth evaluation didn't return expected values
                    if ground_truth_label not in ["TRUTHFUL", "HALLUCINATION"]:
                        print("\n💥💥💥 HARD STOP - UNEXPECTED GROUND TRUTH LABEL 💥💥💥")
                        print(f"Question: {question[:200]}...")
                        print(f"Response: {response_text[:200]}...")
                        print(f"Expected Answer: {qa_pair.get('correct_answer', 'N/A')}")
                        print(f"Ground Truth Method: {ground_truth_method}")
                        print(f"Actual Ground Truth Label: '{ground_truth_label}'")
                        print(f"Evaluator Details: {gt_result}")
                        print(
                            f"💡 Ground truth evaluation returned '{ground_truth_label}' but optimization needs TRUTHFUL or HALLUCINATION"
                        )
                        print("🛠️  This means the ground truth method is not compatible with optimization")
                        print("💥💥💥 STOPPING OPTIMIZATION IMMEDIATELY 💥💥💥\n")

                        # HARD STOP - crash the entire process immediately
                        raise ValueError(
                            f"❌ CRITICAL ERROR: Ground truth evaluation returned unexpected label!\n"
                            f"   📊 Ground truth: '{ground_truth_label}'\n"
                            f"   📝 Expected: 'TRUTHFUL' or 'HALLUCINATION'\n"
                            f"   📝 Question: {question[:100]}...\n"
                            f"   📝 Response: {response_text[:100]}...\n"
                            f"   📝 Expected: {qa_pair.get('correct_answer', 'N/A')}\n"
                            f"   📝 Method: {ground_truth_method}\n"
                            f"   📝 Details: {gt_result}\n"
                            f"   💡 Ground truth evaluation must return TRUTHFUL or HALLUCINATION for optimization to work\n"
                            f"   🛠️  Check if the ground truth method is appropriate for this task type"
                        )

                    if ground_truth_label in ["TRUTHFUL", "HALLUCINATION"]:
                        question_results.append(
                            {
                                "response": response_text,
                                "token_scores": token_scores,
                                "ground_truth": ground_truth_label,
                            }
                        )
                        if verbose:
                            print("       ✅ Added to question_results")
                    else:
                        if verbose:
                            print(f"       ⚠️ Skipping - ground truth not TRUTHFUL/HALLUCINATION: {ground_truth_label}")

                except Exception as e:
                    if verbose:
                        print(f"         ❌ Error generating response: {e}")
                        import traceback

                        print(f"         🔍 Full traceback: {traceback.format_exc()}")
                    continue

            if verbose:
                print(f"     📊 Total question_results for {classifier_type}: {len(question_results)}")

            # Now test all combinations of aggregation + threshold on the SAME token scores
            if verbose:
                print(
                    f"     🔄 Testing {len(aggregation_methods)} aggregations × {len(thresholds)} thresholds = {len(aggregation_methods) * len(thresholds)} combinations..."
                )

            if len(question_results) == 0:
                if verbose:
                    print(f"     ❌ CRITICAL: No question_results for {classifier_type}! Cannot test combinations.")
                continue

            for aggregation in aggregation_methods:
                for threshold in thresholds:
                    if verbose and len(thresholds) <= 3:  # Only show details for small threshold sets
                        print(f"       ➤ Testing {aggregation} + threshold {threshold}...")

                    correct_predictions = 0
                    total_predictions = 0

                    for result in question_results:
                        try:
                            # Apply aggregation method to existing token scores
                            from .core.parser import aggregate_token_scores

                            aggregated_score = aggregate_token_scores(result["token_scores"], aggregation)
                            classification = "HALLUCINATION" if aggregated_score > threshold else "TRUTHFUL"

                            # Compare with ground truth
                            if classification == result["ground_truth"]:
                                correct_predictions += 1
                            total_predictions += 1

                        except Exception as e:
                            if verbose:
                                print(f"         ⚠️ Error applying {aggregation} + {threshold}: {e}")
                            continue

                    # Calculate accuracy for this combination
                    if total_predictions > 0:
                        accuracy = correct_predictions / total_predictions

                        combo_key = f"layer_{layer_idx}_{aggregation}_{classifier_type}_thresh_{threshold}"
                        combination_results[combo_key] = {
                            "layer": layer_idx,
                            "aggregation": aggregation,
                            "classifier_type": classifier_type,
                            "threshold": threshold,
                            "accuracy": accuracy,
                            "correct": correct_predictions,
                            "total": total_predictions,
                        }

                        if verbose and len(thresholds) <= 3:
                            print(
                                f"         ✅ {combo_key}: {accuracy:.3f} ({correct_predictions}/{total_predictions})"
                            )

                        # Update best combination
                        if accuracy > best_combination["accuracy"]:
                            best_combination.update(
                                {
                                    "layer": layer_idx,
                                    "aggregation": aggregation,
                                    "classifier_type": classifier_type,
                                    "threshold": threshold,
                                    "accuracy": accuracy,
                                    "correct": correct_predictions,
                                    "total": total_predictions,
                                }
                            )
                    else:
                        if verbose:
                            print(f"         ⚠️ No predictions for {aggregation} + {threshold}")

    # 🚨 HARD ERROR CHECK: Optimization returned None values
    if (
        best_combination["layer"] is None
        or best_combination["aggregation"] is None
        or best_combination["classifier_type"] is None
        or best_combination["threshold"] is None
    ):
        raise ValueError(
            f"❌ CRITICAL ERROR: Ground truth optimization found NO valid combinations!\n"
            f"   📊 Best combination: {best_combination}\n"
            f"   🔍 Tested {len(combination_results)} combinations total\n"
            f"   💡 This indicates either:\n"
            f"      • No test responses were generated successfully\n"
            f"      • Ground truth evaluation failed completely\n"
            f"      • All combinations scored 0.0 accuracy\n"
            f"   🛠️  Check generate_with_classification() and ground truth evaluation logic"
        )

    if verbose:
        print("\n   🏆 BEST COMBINATION:")
        print(f"     • Layer: {best_combination['layer']}")
        print(f"     • Aggregation: {best_combination['aggregation']}")
        print(f"     • Classifier: {best_combination['classifier_type']}")
        print(f"     • Threshold: {best_combination['threshold']}")
        print(
            f"     • Accuracy: {best_combination['accuracy']:.3f} ({best_combination['correct']}/{best_combination['total']})"
        )

    return {"best_combination": best_combination, "all_combinations": combination_results}


def run_smart_optimization(
    model,
    collector,
    contrastive_pairs: List,
    test_qa_pairs: List[Dict],
    task_name: str,
    model_name: str,
    limit: int,
    ground_truth_method: str,
    max_new_tokens: int,
    device: str,
    verbose: bool = False,
    classifier_types: List[str] = None,
    thresholds: List[float] = None,
    optimize_layers: str = "all",
) -> Dict[str, Any]:
    """
    Run complete smart optimization with caching.

    Args:
        model: Model instance
        collector: Activation collector
        contrastive_pairs: Training data (for layer optimization)
        test_qa_pairs: Test data (for aggregation optimization)
        task_name: Name of the task
        model_name: Name of the model
        limit: Data limit used
        ground_truth_method: Ground truth evaluation method
        max_new_tokens: Max tokens for generation
        device: Device to use
        verbose: Whether to print progress

    Returns:
        Dict with optimization results
    """
    # Set defaults for new hyperparameters
    if classifier_types is None:
        classifier_types = ["logistic", "mlp"]
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Generate cache key including new hyperparameters
    import hashlib

    cache_key_parts = [
        model_name,
        task_name,
        str(limit),
        ground_truth_method,
        str(sorted(classifier_types)),
        str(sorted(thresholds)),
    ]
    cache_key = hashlib.md5("_".join(cache_key_parts).encode()).hexdigest()

    if verbose:
        print("🔍 SMART HYPERPARAMETER OPTIMIZATION:")
        print(f"   • Cache key: {cache_key}")
        print(f"   • Classifier types: {classifier_types}")
        print(f"   • Thresholds: {thresholds}")

    # Check if results are already cached
    cached_results = load_cached_results(cache_key)
    if cached_results is not None:
        if verbose:
            print(
                f"   💾 Found cached results! Using layer {cached_results['best_layer']} + {cached_results['best_aggregation']} + {cached_results['best_classifier_type']} + threshold {cached_results['best_threshold']}"
            )
        return cached_results

    if verbose:
        print("   🆕 No cached results found, running optimization...")

    # Step 1: Optimize layers using contrastive pairs (training data)
    layer_results = optimize_layers_on_contrastive_pairs(
        model, collector, contrastive_pairs, device, verbose, optimize_layers=optimize_layers
    )

    # 🚨 HARD ERROR CHECK: All layers failed to train
    if not layer_results:
        raise ValueError(
            f"❌ CRITICAL ERROR: All layers failed to train classifiers for task '{task_name}'!\n"
            f"   📊 Input: {len(contrastive_pairs)} contrastive pairs\n"
            f"   🔍 Layers tested: {optimize_layers}\n"
            f"   💡 This indicates either:\n"
            f"      • Insufficient training data quality\n"
            f"      • Model activation extraction issues\n"
            f"      • Classifier training configuration problems\n"
            f"   🛠️  Check optimize_layers_on_contrastive_pairs() training logic"
        )

    # Step 2: Optimize aggregation methods + classifier types + thresholds using ground truth (validation data)
    aggregation_results = optimize_aggregation_on_ground_truth(
        model,
        collector,
        contrastive_pairs,
        layer_results,
        test_qa_pairs,
        ground_truth_method,
        task_name,
        max_new_tokens,
        device,
        verbose,
        classifier_types=classifier_types,
        thresholds=thresholds,
    )

    best_combo = aggregation_results["best_combination"]

    # Validate that we actually found a valid combination
    if (
        best_combo["layer"] is None
        or best_combo["aggregation"] is None
        or best_combo["classifier_type"] is None
        or best_combo["threshold"] is None
    ):
        raise ValueError(
            f"Optimization failed to find valid parameters! best_combination: {best_combo}. "
            f"This indicates no valid combinations were tested successfully. "
            f"Check ground truth evaluation and classifier training."
        )

    # Prepare final results
    optimization_results = {
        "best_layer": best_combo["layer"],
        "best_aggregation": best_combo["aggregation"],
        "best_classifier_type": best_combo["classifier_type"],
        "best_threshold": best_combo["threshold"],
        "best_accuracy": best_combo["accuracy"],
        "optimization_performed": True,
        "layer_results": {k: {"train_accuracy": v["train_accuracy"]} for k, v in layer_results.items()},
        "combination_results": aggregation_results["all_combinations"],
        "cache_key": cache_key,
    }

    # Cache the results
    save_cached_results(cache_key, optimization_results)

    if verbose:
        print("\n   🏆 OPTIMAL COMBINATION FOUND:")
        print(f"      • Best layer: {best_combo['layer']}")
        print(f"      • Best aggregation: {best_combo['aggregation']}")
        print(f"      • Best classifier: {best_combo['classifier_type']}")
        print(f"      • Best threshold: {best_combo['threshold']}")
        print(
            f"      • Ground truth accuracy: {best_combo['accuracy']:.3f} ({best_combo['correct']}/{best_combo['total']})"
        )
        print("      • Results cached for future runs")

    return optimization_results


def generate_responses_and_get_ground_truth(
    model, questions: List[str], ground_truth_method: str, max_new_tokens: int = 50, verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate responses for questions and collect ground truth labels.

    Args:
        model: The model instance
        questions: List of questions to ask
        ground_truth_method: Method for collecting ground truth
        max_new_tokens: Maximum tokens to generate
        verbose: Whether to show verbose output

    Returns:
        List of dictionaries with question, response, and ground_truth
    """
    from .core.ground_truth_evaluator import GroundTruthEvaluator, GroundTruthMethod

    # Initialize evaluator
    try:
        gt_method = GroundTruthMethod(ground_truth_method.upper())
    except ValueError:
        if verbose:
            print(f"Unknown ground truth method: {ground_truth_method}, using NONE")
        gt_method = GroundTruthMethod.NONE

    evaluator = GroundTruthEvaluator(gt_method)

    results = []
    for i, question in enumerate(questions):
        if verbose:
            print(f"Processing question {i + 1}/{len(questions)}: {question[:50]}...")

        # Generate response
        try:
            inputs = model.tokenizer(question, return_tensors="pt")
            model_device = next(model.hf_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with model.tokenizer.no_grad():
                outputs = model.hf_model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=model.tokenizer.eos_token_id
                )

            # Decode response (only the generated part)
            generated_ids = outputs[0][len(inputs["input_ids"][0]) :]
            response = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            if verbose:
                print(f"Error generating response: {e}")
            response = "[GENERATION_ERROR]"

        # Get ground truth
        try:
            gt_result = evaluator.evaluate_response(response, expected_answer=None)
            ground_truth = gt_result.get("ground_truth", "UNKNOWN")
        except Exception as e:
            if verbose:
                print(f"Error getting ground truth: {e}")
            ground_truth = "UNKNOWN"

        results.append({"question": question, "response": response, "ground_truth": ground_truth})

    return results


def extract_layer_activations_for_responses(
    model, questions: List[str], layer_idx: int, max_new_tokens: int = 50, verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate responses and extract activations for a specific layer.

    Args:
        model: The model instance
        questions: List of questions
        layer_idx: Which layer to extract activations from
        max_new_tokens: Maximum tokens to generate
        verbose: Whether to show verbose output

    Returns:
        List of dictionaries with question, response, and layer_activations
    """
    import torch

    results = []
    for i, question in enumerate(questions):
        if verbose:
            print(f"Extracting activations for question {i + 1}/{len(questions)} (layer {layer_idx})")

        try:
            # Tokenize input
            inputs = model.tokenizer(question, return_tensors="pt")
            model_device = next(model.hf_model.parameters()).device
            inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.hf_model.generate(
                    **inputs_on_device,
                    max_new_tokens=max_new_tokens,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                )

            # Extract response text
            generated_ids = outputs.sequences[0][len(inputs_on_device["input_ids"][0]) :]
            response_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Extract activations for the specified layer
            layer_activations = []
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                for step_hidden_states in outputs.hidden_states:
                    if layer_idx < len(step_hidden_states):
                        # Get the activation for this layer at this generation step
                        layer_hidden = step_hidden_states[layer_idx]
                        if layer_hidden.shape[1] > 0:
                            # Take the last token's activation
                            activation = layer_hidden[0, -1, :].cpu()
                            layer_activations.append(activation)

            results.append({"question": question, "response": response_text, "layer_activations": layer_activations})

        except Exception as e:
            if verbose:
                print(f"Error processing question {i + 1}: {e}")
            results.append({"question": question, "response": "[ERROR]", "layer_activations": []})

    return results


def compute_aggregated_scores(layer_activations: List, aggregation_method: str) -> List[float]:
    """
    Compute aggregated scores from layer activations using different aggregation methods.

    Args:
        layer_activations: List of activation tensors (one per generated token)
        aggregation_method: How to aggregate ('average', 'max', 'min', 'first', 'final')

    Returns:
        List of aggregated scores (one per activation)
    """
    import torch

    if not layer_activations:
        return [0.5]  # Default neutral score

    scores = []
    for activation in layer_activations:
        if isinstance(activation, torch.Tensor):
            if aggregation_method == "average":
                score = float(torch.mean(activation))
            elif aggregation_method == "max":
                score = float(torch.max(activation))
            elif aggregation_method == "min":
                score = float(torch.min(activation))
            elif aggregation_method == "first":
                score = float(activation[0]) if len(activation) > 0 else 0.5
            elif aggregation_method == "final":
                score = float(activation[-1]) if len(activation) > 0 else 0.5
            else:
                score = float(torch.mean(activation))

            # Normalize to 0-1 range using sigmoid
            normalized_score = 1.0 / (1.0 + torch.exp(-torch.tensor(score)).item())
            scores.append(normalized_score)
        else:
            scores.append(0.5)

    return scores


def run_interactive_optimization(
    model, questions: List[str], training_pairs, max_new_tokens: int, max_combinations: int = 50, verbose: bool = False
) -> Dict[str, Any]:
    """
    Run interactive optimization by:
    1. Training one classifier per layer using training data
    2. Generating test responses and extracting all layer activations
    3. Using each layer's classifier to get prediction scores
    4. Testing all aggregation strategies on prediction scores
    5. Finding combination that best matches user feedback

    Args:
        model: The model instance
        questions: List of questions to test
        training_pairs: Training contrastive pairs for classifier training
        max_new_tokens: Max tokens for generation
        max_combinations: Maximum combinations to test
        verbose: Whether to show verbose output

    Returns:
        Dictionary with optimization results including best parameters
    """
    from wisent_guard.core.activations.activation_collection_method import ActivationCollectionLogic

    from .core.ground_truth_evaluator import GroundTruthEvaluator, GroundTruthMethod
    from .core.hyperparameter_optimizer import detect_model_layers

    # Detect all available layers
    total_layers = detect_model_layers(model)
    layer_range = list(range(min(total_layers, 32)))  # Test up to 32 layers
    aggregation_methods = ["average", "final", "first", "max", "min"]

    if verbose:
        print("\n🎯 INTERACTIVE HYPERPARAMETER OPTIMIZATION:")
        print(f"   • Will train {len(layer_range)} classifiers (one per layer)")
        print(f"   • Will test {len(aggregation_methods)} aggregation methods")
        print(f"   • Total combinations: {len(layer_range) * len(aggregation_methods)}")
        print(f"   • Questions to test: {len(questions)}")

    # Step 1: Train one classifier per layer
    if verbose:
        print("\n🏋️ TRAINING CLASSIFIERS:")
        print(f"   • Training {len(layer_range)} classifiers using {len(training_pairs)} training pairs")

    layer_classifiers = {}
    collector = ActivationCollectionLogic(model=model)

    for layer_idx in layer_range:
        try:
            if verbose:
                print(f"   • Training classifier for layer {layer_idx}...")

            # Extract training activations for this layer
            processed_pairs = collector.collect_activations_batch(
                pairs=training_pairs, layer_index=layer_idx, device=None
            )

            # Train classifier for this layer using basic Classifier
            from wisent_guard.core.classifier.classifier import Classifier

            # Convert processed pairs to training data format
            X = []
            y = []

            for pair in processed_pairs:
                # Add positive activations (truthful) as class 0 (harmless)
                if hasattr(pair, "positive_activations") and pair.positive_activations is not None:
                    X.append(pair.positive_activations.detach().cpu().flatten().numpy())
                    y.append(0)  # harmless

                # Add negative activations (hallucination) as class 1 (harmful)
                if hasattr(pair, "negative_activations") and pair.negative_activations is not None:
                    X.append(pair.negative_activations.detach().cpu().flatten().numpy())
                    y.append(1)  # harmful

            # Train classifier using basic Classifier
            if len(X) > 1 and len(set(y)) == 2:  # Need at least 2 samples and both classes
                classifier = Classifier(model_type="logistic", threshold=0.5)
                classifier.fit(X, y, test_size=0.2, random_state=42)
                layer_classifiers[layer_idx] = classifier
            else:
                if verbose:
                    print(f"   ⚠️ Not enough valid activations for layer {layer_idx}")
                continue

        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to train classifier for layer {layer_idx}: {e}")
            continue

    if verbose:
        print(f"   ✅ Successfully trained {len(layer_classifiers)} classifiers")

    # Step 2: Generate test responses and get user labels
    evaluator = GroundTruthEvaluator(GroundTruthMethod.INTERACTIVE)
    test_responses_data = []

    print(f"\n{'=' * 80}")
    print("🔬 GENERATING TEST RESPONSES AND COLLECTING USER FEEDBACK")
    print(f"{'=' * 80}")

    for q_idx, question in enumerate(questions):
        try:
            print(f"\n--- Question {q_idx + 1}/{len(questions)} ---")
            print(f"Question: {question}")

            # Generate response and extract activations from ALL layers
            response_text, all_layer_activations = generate_with_all_layer_activations(
                model, question, max_new_tokens, verbose
            )

            # Get interactive ground truth for this response
            result = evaluator.evaluate_response(response=response_text, expected_answer=None)

            user_label = result["ground_truth"]  # "TRUTHFUL" or "HALLUCINATION"

            test_responses_data.append(
                {
                    "question": question,
                    "response": response_text,
                    "user_label": user_label,
                    "layer_activations": all_layer_activations,
                }
            )

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
            return {"optimization_performed": False, "interrupted": True}
        except Exception as e:
            if verbose:
                print(f"Error processing question {q_idx}: {e}")
            continue

    if not test_responses_data:
        return {"optimization_performed": False, "error": "No test responses processed"}

    # Step 3: Use trained classifiers to get prediction scores for each layer
    if verbose:
        print("\n🔮 GENERATING PREDICTIONS:")
        print(f"   • Using {len(layer_classifiers)} trained classifiers")

    # For each test response, get prediction scores from each layer's classifier
    for response_data in test_responses_data:
        layer_predictions = {}

        for layer_idx, classifier in layer_classifiers.items():
            if layer_idx < len(response_data["layer_activations"]):
                try:
                    # Get activations for this layer
                    layer_activation = response_data["layer_activations"][layer_idx]

                    # Use this layer's trained Classifier to predict
                    try:
                        features = layer_activation.detach().cpu().flatten().numpy().reshape(1, -1)
                        proba = classifier.predict_proba(features)[0]
                        score = proba  # Probability of positive class (hallucination)
                    except:
                        score = 0.5

                    layer_predictions[layer_idx] = score

                except Exception as e:
                    if verbose:
                        print(f"   ⚠️ Error getting prediction from layer {layer_idx}: {e}")
                    layer_predictions[layer_idx] = 0.5

        response_data["layer_predictions"] = layer_predictions

    # Step 4: Test all aggregation strategies on prediction scores
    if verbose:
        print(f"   • Testing {len(aggregation_methods)} aggregation strategies")

    combination_performance = {}

    for response_data in test_responses_data:
        user_label = response_data["user_label"]
        layer_predictions = response_data["layer_predictions"]

        for layer_idx in layer_predictions:
            for agg_method in aggregation_methods:
                # For single response, aggregation doesn't matter much, but simulate token-level aggregation
                prediction_score = layer_predictions[layer_idx]

                # Apply aggregation strategy (simulate multiple token scores)
                if agg_method == "average":
                    final_score = prediction_score
                elif agg_method == "max":
                    final_score = min(prediction_score * 1.1, 1.0)  # Slight boost for max
                elif agg_method == "min":
                    final_score = max(prediction_score * 0.9, 0.0)  # Slight reduction for min
                elif agg_method == "first" or agg_method == "final":
                    final_score = prediction_score
                else:
                    final_score = prediction_score

                # Apply classification threshold (0.6)
                predicted_label = "HALLUCINATION" if final_score > 0.6 else "TRUTHFUL"

                # Track performance
                combo_key = (layer_idx, agg_method)
                if combo_key not in combination_performance:
                    combination_performance[combo_key] = {"correct": 0, "total": 0}

                if predicted_label == user_label:
                    combination_performance[combo_key]["correct"] += 1
                combination_performance[combo_key]["total"] += 1

    # Step 5: Find best performing combination
    if combination_performance:
        best_combo = max(
            combination_performance.items(), key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0
        )
        best_accuracy = best_combo[1]["correct"] / best_combo[1]["total"]
    else:
        best_combo = ((15, "average"), {"correct": 0, "total": 1})
        best_accuracy = 0.0

    if verbose:
        print("\n🏆 OPTIMIZATION RESULTS:")
        print(f"   • Layer: {best_combo[0][0]}")
        print(f"   • Aggregation: {best_combo[0][1]}")
        print(f"   • Accuracy: {best_accuracy:.1%}")

        print("\n📋 Top 5 combinations:")
        sorted_combos = sorted(
            combination_performance.items(), key=lambda x: x[1]["correct"] / x[1]["total"], reverse=True
        )
        for i, ((layer, agg), perf) in enumerate(sorted_combos[:5]):
            acc = perf["correct"] / perf["total"]
            print(f"   {i + 1}. Layer {layer}, {agg}: {acc:.1%} ({perf['correct']}/{perf['total']})")

    return {
        "optimization_performed": True,
        "best_layer": best_combo[0][0],
        "best_aggregation": best_combo[0][1],
        "best_accuracy": best_accuracy,
        "trained_classifiers": len(layer_classifiers),
        "combination_performance": {
            str(k): v for k, v in combination_performance.items()
        },  # Convert tuple keys to strings
    }


def generate_with_all_layer_activations(model, prompt, max_new_tokens, verbose=False):
    """
    Generate a response while extracting activations from all layers.

    Returns:
        Tuple of (response_text, list_of_layer_activations)
    """
    import torch

    # Tokenize input
    inputs = model.tokenizer(prompt, return_tensors="pt")

    # Generate with hidden states
    model_device = next(model.hf_model.parameters()).device
    inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.hf_model.generate(
            **inputs_on_device,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    # Extract response text
    generated_ids = outputs.sequences[0][len(inputs_on_device["input_ids"][0]) :]
    response_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract activations from all layers for the generated tokens
    all_layer_activations = []

    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        # Get the last generation step's hidden states
        last_hidden_states = outputs.hidden_states[-1]  # Last generated token

        for layer_idx, layer_hidden in enumerate(last_hidden_states):
            # Extract activation for the last token of this layer
            if layer_hidden.shape[1] > 0:  # Check if we have tokens
                activation = layer_hidden[0, -1, :].cpu()  # [hidden_dim]
                all_layer_activations.append(activation)

    return response_text, all_layer_activations


def compute_classification_score(layer_activation, aggregation_method):
    """
    Compute a classification score for given layer activation and aggregation method.
    This is a simplified placeholder - in practice you'd use a trained classifier.

    Args:
        layer_activation: Tensor of activations for one layer
        aggregation_method: How to aggregate if needed

    Returns:
        Float score between 0 and 1
    """
    import torch

    # Simple heuristic: use mean of activations as a proxy score
    # In practice, this would use a trained classifier
    if isinstance(layer_activation, torch.Tensor):
        if aggregation_method == "average":
            score = float(torch.mean(layer_activation))
        elif aggregation_method == "max":
            score = float(torch.max(layer_activation))
        elif aggregation_method == "min":
            score = float(torch.min(layer_activation))
        elif aggregation_method == "first":
            score = float(layer_activation[0]) if len(layer_activation) > 0 else 0.5
        elif aggregation_method == "final":
            score = float(layer_activation[-1]) if len(layer_activation) > 0 else 0.5
        else:
            score = float(torch.mean(layer_activation))

        # Normalize to 0-1 range using sigmoid
        return 1.0 / (1.0 + torch.exp(-torch.tensor(score)).item())

    return 0.5  # Default neutral score
