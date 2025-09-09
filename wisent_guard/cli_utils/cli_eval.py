"""
Raw extraction of evaluation-only logic from run_task_pipeline (cli.py).
"""
from __future__ import annotations

import os
import json
import torch
from typing import Any, Dict, List, Optional, Tuple

# These imports mirror those used in cli.py
from wisent_guard.core.lm_eval_harness_ground_truth import LMEvalHarnessGroundTruth
from wisent_guard.core.ground_truth_evaluator import GroundTruthEvaluator
from wisent_guard.core.layer import Layer
from wisent_guard.core.steering import SteeringMethod, SteeringType
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.parser import aggregate_token_scores
from wisent_guard.inference import (
    generate_with_classification_and_handling,
    generate_with_multi_layer_classification_and_handling,
)
from wisent_guard.core.steering_methods.steering_evaluation import run_lm_harness_evaluation

# Optional utilities expected to exist in project
try:
    from wisent_guard.core.activations import ActivationAggregationStrategy, Activations
except Exception:  # pragma: no cover - defensive
    ActivationAggregationStrategy = None  # type: ignore
    Activations = None  # type: ignore


def evaluate_perplexity_task(
    *,
    task_name: str,
    model,
    layer: str,
    token_aggregation: str,
    test_qa_pairs_source: List[Dict[str, Any]],
    verbose: bool,
    get_actual_task_name,  # function injected from cli.py
    parse_layers_from_arg,  # function injected
) -> Optional[Dict[str, Any]]:
    """Early perplexity evaluation path (copied from run_task_pipeline)."""
    def get_evaluation_method_for_task_early(task_name_inner: str) -> str:
        try:
            eval_methods_path = os.path.join(
                os.path.dirname(__file__),
                "parameters/benchmarks/benchmark_evaluation_methods.json",
            )
            with open(eval_methods_path) as f:
                benchmark_methods = json.load(f)
                return benchmark_methods.get(task_name_inner, "text-generation")
        except Exception as e:  # pragma: no cover - logging side effect
            if verbose:
                print(f"   ⚠️ Could not load benchmark evaluation methods: {e}")
            return "text-generation"

    evaluation_method = get_evaluation_method_for_task_early(task_name)
    if evaluation_method != "perplexity":
        return None

    if verbose:
        print("\n🎯 PERPLEXITY TASK DETECTED: Skipping contrastive training")
        print(f"   • Task: {task_name}")
        print(f"   • Evaluation method: {evaluation_method}")
        print("   • Going directly to perplexity evaluation")

    layers = parse_layers_from_arg(layer)
    actual_eval_task_name = get_actual_task_name(task_name)
    lm_eval_ground_truth = LMEvalHarnessGroundTruth(actual_eval_task_name, evaluation_method, model=model)

    lm_eval_results = lm_eval_ground_truth.evaluate_classifier_on_task(
        classifier=None,
        task_name=actual_eval_task_name,
        num_samples=len(test_qa_pairs_source),
        model=model,
        layer=layers[0],
        token_aggregation=token_aggregation,
    )

    if verbose:
        print(f"\n🎉 PERPLEXITY EVALUATION COMPLETED FOR {task_name.upper()}!")
        print(f"{'=' * 80}")
        print("📊 FINAL RESULTS:")
        print(f"   • Test samples: {len(test_qa_pairs_source)}")
        print(f"   • Evaluation method: {evaluation_method}")
        if "perplexity_accuracy" in lm_eval_results:
            accuracy = lm_eval_results["perplexity_accuracy"]
            print(f"   • Perplexity accuracy: {accuracy:.2%}")
        elif "average_classifier_score" in lm_eval_results:
            avg_score = lm_eval_results["average_classifier_score"]
            print(f"   • Average perplexity score: {avg_score:.3f}")
        else:
            print("   • Perplexity evaluation: Completed")
        print(f"{'=' * 80}")

    return {
        "task_name": task_name,
        "evaluation_method": evaluation_method,
        "evaluation_results": lm_eval_results,
        "num_test": len(test_qa_pairs_source),
        "ground_truth_method": "lm-eval-harness",
        "skipped_training": True,
        "reason": "Perplexity task does not require contrastive training",
    }


def evaluate_steering_mode(
    *,
    steering_method_name: str,
    steering_obj,
    pair_set,
    layers: List[int],
    model,
    task_name: str,
    task_data,
    test_qa_pairs_source: List[Dict[str, Any]],
    steering_strength: float,
    enable_memory_tracking: bool,
    enable_latency_tracking: bool,
    show_timing_summary: bool,
    memory_tracker,
    latency_tracker,
    detailed_performance_report: bool,
    export_performance_csv: Optional[str],
    verbose: bool,
    output_mode: str,
    save_steering_vector: Optional[str],
    training_stats: Dict[str, Any],
    enable_nonsense_detection: bool,
    nonsense_action: str,
    token_aggregation: str,  # kept for consistency (unused here)
) -> Dict[str, Any]:
    """Steering evaluation block (post-training)."""
    # Build test QA pairs (copied logic)
    test_qa_pairs: List[Dict[str, Any]] = []
    for doc in test_qa_pairs_source:
        try:
            if all(k in doc for k in ["question", "correct_answer", "incorrect_answer"]):
                test_qa_pairs.append(
                    {
                        "question": doc["question"],
                        "formatted_question": doc.get("question", doc["question"]),
                        "correct_answer": doc["correct_answer"],
                        "incorrect_answer": doc["incorrect_answer"],
                    }
                )
            else:
                # lm-harness style doc
                raw_question = doc.get("question", str(doc))
                formatted_question = raw_question
                if hasattr(task_data, "doc_to_text"):
                    try:
                        formatted_question = task_data.doc_to_text(doc)
                    except Exception:
                        formatted_question = raw_question
                correct_answers = doc.get("mc1_targets", {}).get("choices", [])
                correct_labels = doc.get("mc1_targets", {}).get("labels", [])
                correct_answer = None
                incorrect_answer = None
                for i, label in enumerate(correct_labels):
                    if label == 1 and i < len(correct_answers):
                        correct_answer = correct_answers[i]
                        break
                for i, label in enumerate(correct_labels):
                    if label == 0 and i < len(correct_answers):
                        incorrect_answer = correct_answers[i]
                        break
                if correct_answer and incorrect_answer:
                    test_qa_pairs.append(
                        {
                            "question": raw_question,
                            "formatted_question": formatted_question,
                            "correct_answer": correct_answer,
                            "incorrect_answer": incorrect_answer,
                        }
                    )
        except Exception:
            continue

    steering_methods_list = [steering_obj]
    steering_evaluation_results = run_lm_harness_evaluation(
        task_data,
        test_qa_pairs,
        model,
        steering_methods_list,
        layers,
        steering_strength,
        True,
        verbose,
        output_mode,
    )

    if verbose:
        print(f"✅ {steering_method_name} steering evaluation completed")
        print(f"   📊 Accuracy: {steering_evaluation_results.get('accuracy', 'N/A')}")
        print(f"   📊 Test samples: {len(test_qa_pairs)}")

    # Performance report (copied logic condensed)
    if enable_memory_tracking or enable_latency_tracking or show_timing_summary:
        if verbose:
            print("\n🔍 Generating performance report...")
        print("\n📊 PERFORMANCE REPORT:")
        print(f"{'=' * 50}")
        if memory_tracker:
            if verbose:
                print("   • Stopping memory monitoring...")
            memory_stats = memory_tracker.stop_monitoring()
            print("💾 Memory Usage:")
            print(memory_tracker.format_stats(memory_stats, detailed_performance_report))
        if latency_tracker or show_timing_summary:
            if verbose:
                print("   • Collecting timing data...")
            if latency_tracker:
                print("\n⏱️ Performance Metrics:")
                print(latency_tracker.format_user_metrics())
        if export_performance_csv and latency_tracker:
            latency_tracker.export_csv(export_performance_csv)
            print(f"\n📄 Performance data exported to: {export_performance_csv}")
        print(f"{'=' * 50}")

    return {
        "task_name": task_name,
        "layer": layers[0],
        "steering_mode": True,
        "steering_method": steering_method_name,
        "steering_strength": steering_strength,
        "training_stats": training_stats,
        "training_pairs": len(pair_set),
        "vector_saved": save_steering_vector is not None,
        "evaluation_results": steering_evaluation_results,
        "accuracy": steering_evaluation_results.get("accuracy", "N/A"),
        "test_samples": len(test_qa_pairs),
    }


def evaluate_lm_eval_harness_classifier(
    *,
    task_name: str,
    model,
    layers: List[int],
    steering_methods: Dict[int, Any],
    steering_method: Any,  # single-layer method
    test_qa_pairs_source: List[Dict[str, Any]],
    token_aggregation: str,
    training_results: Dict[str, Any],
    contrastive_pairs_len: int,
    layer: str,
    original_layer: str,
    optimize: bool,
    original_token_aggregation: str,
    optimization_result: Dict[str, Any],
    detection_threshold: float,
    ground_truth_method: str,
    get_actual_task_name,
    verbose: bool,
) -> Dict[str, Any]:
    """Classifier evaluation via lm-eval-harness (non-cross-benchmark)."""
    def get_evaluation_method_for_task(task_name_inner: str) -> str:
        try:
            eval_methods_path = os.path.join(
                os.path.dirname(__file__),
                "parameters/benchmarks/benchmark_evaluation_methods.json",
            )
            with open(eval_methods_path) as f:
                benchmark_methods = json.load(f)
                return benchmark_methods.get(task_name_inner, "text-generation")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Could not load benchmark evaluation methods: {e}")
            return "text-generation"

    evaluation_method = get_evaluation_method_for_task(task_name)

    if verbose:
        print("\n🔍 LM-EVAL-HARNESS GROUND TRUTH EVALUATION:")
        print("   • Using lm-eval-harness tasks for direct classifier evaluation")
        print(f"   • Task: {task_name}")
        print(f"   • Evaluation method: {evaluation_method}")
        print(f"   • Samples: {len(test_qa_pairs_source)}")

    if len(layers) > 1:
        classifier = steering_methods.get(layers[0], None)
        classifier = classifier.classifier if classifier else None
    else:
        classifier = getattr(steering_method, "classifier", None)

    if classifier is None:
        if verbose:
            print("   ❌ No trained classifier found for evaluation")
        lm_eval_results = {
            "ground_truth": "UNKNOWN",
            "method_used": "lm-eval-harness-error",
            "confidence": 0.0,
            "details": "No trained classifier available for evaluation",
            "task_name": task_name,
            "evaluation_method": evaluation_method,
        }
    else:
        actual_eval_task_name = get_actual_task_name(task_name)
        lm_eval_ground_truth = LMEvalHarnessGroundTruth(actual_eval_task_name, evaluation_method, model=model)
        lm_eval_results = lm_eval_ground_truth.evaluate_classifier_on_task(
            classifier,
            actual_eval_task_name,
            num_samples=len(test_qa_pairs_source),
            model=model,
            layer=layers[0],
            token_aggregation=token_aggregation,
        )
        if verbose:
            print("   ✅ LM-eval-harness evaluation completed")
            lm_eval_metrics = lm_eval_results.get("lm_eval_metrics", {})
            accuracy = lm_eval_metrics.get("accuracy", "N/A")
            correct_predictions = lm_eval_metrics.get("correct_predictions", 0)
            total_samples = lm_eval_metrics.get("total_samples", 0)
            if accuracy == "N/A" or total_samples == 0:
                print(f"⚠️  EVALUATION WARNING FOR {task_name.upper()}! Accuracy unavailable.")
            else:
                if isinstance(accuracy, (int, float)):
                    print(f"   📊 Accuracy: {accuracy:.2%}")
                else:
                    print(f"   📊 Accuracy: {accuracy}")
                print(f"   🎯 Correct predictions: {correct_predictions}")
                print(f"   📝 Total samples: {total_samples}")

    lm_eval_metrics = lm_eval_results.get("lm_eval_metrics", {})
    correct_classifications = lm_eval_metrics.get("correct_predictions", 0)
    total_classifications = lm_eval_metrics.get("total_samples", 0)

    if verbose:
        print(f"\n🎉 LM-EVAL-HARNESS EVALUATION COMPLETED FOR {task_name.upper()}!")
        print(f"{'=' * 80}")
        print("📊 FINAL RESULTS:")
        print(f"   • Training samples: {contrastive_pairs_len}")
        print(f"   • Test samples: {len(test_qa_pairs_source)}")
        training_accuracy = training_results.get("accuracy", "N/A")
        if isinstance(training_accuracy, (int, float)):
            print(f"   • Training accuracy: {training_accuracy:.2%}")
        else:
            print(f"   • Training accuracy: {training_accuracy}")
        classifier_accuracy = lm_eval_metrics.get("accuracy", "N/A")
        if isinstance(classifier_accuracy, (int, float)):
            print(f"   • Classifier evaluation accuracy: {classifier_accuracy:.2%}")
        else:
            print(f"   • Classifier evaluation accuracy: {classifier_accuracy}")
        print(f"   • Correct predictions: {correct_classifications}")
        print(f"   • Total evaluated: {total_classifications}")
        print(f"{'=' * 80}")

    return {
        "task_name": task_name,
        "layer": layer,
        "original_layer": original_layer,
        "token_aggregation": token_aggregation,
        "original_token_aggregation": original_token_aggregation,
        "optimization_performed": optimize,
        "optimization_result": optimization_result,
        "training_results": training_results,
        "evaluation_results": lm_eval_results,
        "num_train": contrastive_pairs_len,
        "num_test": len(test_qa_pairs_source),
        "sample_responses": [],
        "classification_accuracy": lm_eval_metrics.get("accuracy", 0.0),
        "correct_classifications": correct_classifications,
        "total_classifications": total_classifications,
        "ground_truth_method": ground_truth_method,
    }


def evaluate_cross_benchmark(
    *,
    task_name: str,
    model_name: str,
    layer: str,
    train_contrastive_pairs: ContrastivePairSet,
    eval_contrastive_pairs: ContrastivePairSet,
    collector,
    layers: List[int],
    steering_method,
    token_targeting_strategy,
    verbose: bool,
) -> Dict[str, Any]:
    """Cross-benchmark evaluation block (raw copy with minor adjustments)."""
    if verbose:
        print("\n🔄 CROSS-BENCHMARK EVALUATION:")
        print(f"   • Evaluating classifier trained on {train_contrastive_pairs.name}")
        print(f"   • Testing on {eval_contrastive_pairs.name}")
        print(f"   • Evaluation samples: {len(eval_contrastive_pairs.pairs)}")

    # Copy raw logic: extract activations
    eval_pairs_with_activations = []
    from ..core.contrastive_pairs.contrastive_pair import ContrastivePair
    for pair in eval_contrastive_pairs.pairs:
        eval_pair = ContrastivePair(
            prompt=pair.prompt,
            positive_response=pair.positive_response,
            negative_response=pair.negative_response,
        )
        eval_pairs_with_activations.append(eval_pair)

    if verbose:
        print("\n🔬 Extracting activations for evaluation data...")

    eval_processed_pairs = collector.collect_activations_batch(
        pairs=eval_pairs_with_activations,
        layer_index=layers[0],
        device=getattr(steering_method, "device", None),
        token_targeting_strategy=token_targeting_strategy,
    )

    correct_predictions = 0
    total_predictions = 0

    for i, eval_pair in enumerate(eval_processed_pairs):
        try:
            pos_activation = eval_pair.positive_activations
            neg_activation = eval_pair.negative_activations
            if pos_activation is not None and neg_activation is not None:
                if hasattr(steering_method, "is_vector_based") and steering_method.is_vector_based:
                    steering_vector = steering_method.get_steering_vector()
                    if steering_vector is not None:
                        if not isinstance(pos_activation, torch.Tensor):
                            pos_activation = torch.tensor(pos_activation)
                        if not isinstance(neg_activation, torch.Tensor):
                            neg_activation = torch.tensor(neg_activation)
                        pos_score = torch.dot(pos_activation.flatten(), steering_vector.flatten()).item()
                        neg_score = torch.dot(neg_activation.flatten(), steering_vector.flatten()).item()
                        if pos_score < neg_score:
                            correct_predictions += 1
                        total_predictions += 1
                        if verbose and i < 3:
                            print(f"\n   Example {i + 1}:")
                            print(f"   • Positive score: {pos_score:.3f}")
                            print(f"   • Negative score: {neg_score:.3f}")
                            print(f"   • Prediction: {'✅ Correct' if pos_score < neg_score else '❌ Wrong'}")
                elif hasattr(steering_method, "classifier") and steering_method.classifier is not None:
                    if hasattr(pos_activation, "cpu"):
                        pos_feat = pos_activation.cpu().numpy()
                        neg_feat = neg_activation.cpu().numpy()
                    else:
                        pos_feat = pos_activation
                        neg_feat = neg_activation
                    if hasattr(pos_feat, "ndim") and pos_feat.ndim == 1:
                        pos_feat = pos_feat.reshape(1, -1)
                    if hasattr(neg_feat, "ndim") and neg_feat.ndim == 1:
                        neg_feat = neg_feat.reshape(1, -1)
                    pos_proba = steering_method.classifier.predict_proba(pos_feat)
                    neg_proba = steering_method.classifier.predict_proba(neg_feat)
                    pos_score = pos_proba[0][1]
                    neg_score = neg_proba[0][1]
                    if pos_score > neg_score:
                        correct_predictions += 1
                    total_predictions += 1
                    if verbose and i < 3:
                        print(f"\n   Example {i + 1}:")
                        print(f"   • Positive score: {pos_score:.3f}")
                        print(f"   • Negative score: {neg_score:.3f}")
                        print(f"   • Prediction: {'✅ Correct' if pos_score > neg_score else '❌ Wrong'}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error evaluating pair {i}: {e}")
                print(f"      Pos activation type: {type(pos_activation)}")
                print(f"      Neg activation type: {type(neg_activation)}")

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    if verbose:
        print("\n📊 CROSS-BENCHMARK EVALUATION RESULTS:")
        print(f"   • Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
        print(f"   • Training domain: {train_contrastive_pairs.name}")
        print(f"   • Evaluation domain: {eval_contrastive_pairs.name}")

    return {
        "task_name": task_name,
        "layer": layer,
        "mode": "cross_benchmark",
        "training_task": train_contrastive_pairs.name,
        "evaluation_task": eval_contrastive_pairs.name,
        "training_results": {},  # Provided externally in full pipeline
        "evaluation_results": {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
        },
        "num_train": len(train_contrastive_pairs.pairs),
        "num_eval": len(eval_contrastive_pairs.pairs),
        "cross_benchmark_transfer": accuracy,
    }


def evaluate_optimized_classifier_generated(
    *,
    test_qa_pairs_source: List[Dict[str, Any]],
    optimize: bool,
    from_csv: bool,
    from_json: bool,
    group_task_qa_format: bool,
    task_data,
    task_name: str,
    model,
    layers: List[int],
    layer: str,
    original_layer: str,
    token_aggregation: str,
    original_token_aggregation: str,
    optimization_result: Dict[str, Any],
    steering_method,
    steering_methods: Dict[int, Any],
    max_new_tokens: int,
    token_aggregation_func,  # reference to aggregate_token_scores
    aggregate_token_scores,  # pass directly too (dup for rawness)
    detection_threshold: float,
    verbose: bool,
    generate_with_classification_and_handling_func,
    generate_with_multi_layer_classification_and_handling_func,
    GroundTruthEvaluator_cls,
    ground_truth_method: str,
    user_labels: Optional[List[str]],
    contrastive_pairs_len: int,
    training_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Optimization path: generate responses and evaluate classifier (raw copy)."""
    if verbose:
        print("\n🧪 TESTING OPTIMIZED CLASSIFIER ON GENERATED RESPONSES:")
        print("   • Generating responses to test questions...")

    # Build test questions
    test_qa_pairs: List[Dict[str, Any]] = []
    for doc in test_qa_pairs_source:
        try:
            if from_csv or from_json:
                test_qa_pairs.append(
                    {
                        "question": doc["question"],
                        "formatted_question": doc["question"],
                        "correct_answer": doc.get("correct_answer", ""),
                    }
                )
            else:
                raw_question = doc.get("question", str(doc))
                if hasattr(task_data, "doc_to_text"):
                    formatted_question = task_data.doc_to_text(doc)
                else:
                    formatted_question = raw_question
                correct_answers = doc.get("mc1_targets", {}).get("choices", [])
                correct_labels = doc.get("mc1_targets", {}).get("labels", [])
                correct_answer = None
                for i, label in enumerate(correct_labels):
                    if label == 1 and i < len(correct_answers):
                        correct_answer = correct_answers[i]
                        break
                if correct_answer:
                    test_qa_pairs.append(
                        {
                            "question": raw_question,
                            "formatted_question": formatted_question,
                            "correct_answer": correct_answer,
                        }
                    )
        except Exception:
            continue

    if verbose:
        print(f"   • Successfully extracted {len(test_qa_pairs)} test questions")

    generated_responses: List[Dict[str, Any]] = []
    correct_classifications = 0
    total_classifications = 0

    for i, qa_pair in enumerate(test_qa_pairs):
        simple_prompt = qa_pair["question"]
        if len(layers) > 1:
            response, layer_results, was_handled = generate_with_multi_layer_classification_and_handling_func(
                model,
                simple_prompt,
                layers,
                max_new_tokens,
                steering_methods,
                token_aggregation,
                detection_threshold,
                verbose and not optimize,
                None,
            )
            primary_layer = layers[0]
            token_scores = layer_results[primary_layer]["token_scores"] if primary_layer in layer_results else []
            classification = (
                layer_results[primary_layer]["classification"] if primary_layer in layer_results else "UNKNOWN"
            )
            aggregated_score = (
                layer_results[primary_layer]["aggregated_score"] if primary_layer in layer_results else 0.0
            )
        else:
            response, token_scores, classification, was_handled = generate_with_classification_and_handling_func(
                model,
                simple_prompt,
                layers[0],
                max_new_tokens,
                steering_method,
                token_aggregation,
                detection_threshold,
                verbose and not optimize,
                None,
            )
            layer_results = None
            aggregated_score = aggregate_token_scores(token_scores, token_aggregation) if token_scores else 0.0

        try:
            evaluator = GroundTruthEvaluator_cls.from_string(ground_truth_method)
            correct_answer = qa_pair.get("correct_answer", "")
            user_label = user_labels[i] if user_labels and i < len(user_labels) else None
            evaluation_result = evaluator.evaluate_response(response, correct_answer, user_label)
            ground_truth = evaluation_result["ground_truth"]
            classification_correct = None
            if ground_truth != "UNKNOWN":
                expected = "HALLUCINATION" if ground_truth == "HALLUCINATION" else "TRUTHFUL"
                classification_correct = classification == expected
                if classification_correct:
                    correct_classifications += 1
                total_classifications += 1
            entry = {
                "question": qa_pair["question"],
                "response": response,
                "token_scores": token_scores,
                "aggregated_score": aggregated_score,
                "classification": classification,
                "ground_truth": ground_truth,
                "ground_truth_method": evaluation_result["method_used"],
                "ground_truth_confidence": evaluation_result["confidence"],
                "ground_truth_details": evaluation_result["details"],
                "classification_correct": classification_correct,
                "was_handled": was_handled,
            }
            if layer_results:
                entry["layer_results"] = layer_results
            generated_responses.append(entry)
        except Exception as e:  # pragma: no cover - defensive branch
            generated_responses.append(
                {
                    "question": qa_pair["question"],
                    "response": response,
                    "token_scores": token_scores,
                    "classification": classification,
                    "ground_truth": "UNKNOWN",
                    "ground_truth_method": "error",
                    "ground_truth_confidence": 0.0,
                    "ground_truth_details": f"Error during evaluation: {e!s}",
                    "classification_correct": None,
                    "was_handled": was_handled,
                }
            )

    if total_classifications > 0:
        test_accuracy = correct_classifications / total_classifications
        evaluation_results = {
            "accuracy": test_accuracy,
            "correct_predictions": correct_classifications,
            "total_predictions": total_classifications,
        }
    else:
        evaluation_results = {"accuracy": "N/A", "correct_predictions": 0, "total_predictions": 0}

    return {
        "task_name": task_name,
        "layer": layer,
        "original_layer": original_layer,
        "token_aggregation": token_aggregation,
        "original_token_aggregation": original_token_aggregation,
        "optimization_performed": True,
        "optimization_result": optimization_result,
        "training_results": training_results,
        "evaluation_results": evaluation_results,
        "num_train": contrastive_pairs_len,
        "num_test": len(test_qa_pairs),
        "sample_responses": generated_responses,
        "classification_accuracy": (
            correct_classifications / total_classifications if total_classifications > 0 else None
        ),
        "correct_classifications": correct_classifications,
        "total_classifications": total_classifications,
    }


def standard_test_evaluation(
    *,
    task_name: str,
    test_qa_pairs_source: List[Dict[str, Any]],
    from_csv: bool,
    from_json: bool,
    group_task_qa_format: bool,
    task_data,
    collector,
    layers: List[int],
    steering_methods: Dict[int, Any],
    steering_method,
    model,
    token_targeting_strategy,
    token_aggregation: str,
    detection_threshold: float,
    max_new_tokens: int,
    training_results: Dict[str, Any],
    contrastive_pairs_len: int,
    optimize: bool,
    ground_truth_method: str,
    user_labels: Optional[List[str]],
    verbose: bool,
    generate_with_classification_and_handling_func,
    generate_with_multi_layer_classification_and_handling_func,
    aggregate_token_scores_func,
    GroundTruthEvaluator_cls,
) -> Dict[str, Any]:
    """Standard evaluation (non-optimization path) including generation of sample responses."""
    if verbose:
        print("\n🧪 PREPARING TEST DATA:")
        print(f"   • Loading {task_name} test data with correct/incorrect answers...")

    # Build test QA pairs with correct & incorrect answers
    test_qa_pairs: List[Dict[str, Any]] = []
    for doc in test_qa_pairs_source:
        try:
            if from_csv or from_json or group_task_qa_format:
                if all(k in doc for k in ["question", "correct_answer", "incorrect_answer"]):
                    test_qa_pairs.append(
                        {
                            "question": doc["question"],
                            "formatted_question": doc["question"],
                            "correct_answer": doc["correct_answer"],
                            "incorrect_answer": doc["incorrect_answer"],
                        }
                    )
            else:
                raw_question = doc.get("question", str(doc))
                formatted_question = raw_question
                if hasattr(task_data, "doc_to_text"):
                    try:
                        formatted_question = task_data.doc_to_text(doc)
                    except Exception:
                        formatted_question = raw_question
                correct_answers = doc.get("mc1_targets", {}).get("choices", [])
                correct_labels = doc.get("mc1_targets", {}).get("labels", [])
                correct_answer = None
                incorrect_answer = None
                for i, label in enumerate(correct_labels):
                    if label == 1 and i < len(correct_answers):
                        correct_answer = correct_answers[i]
                        break
                for i, label in enumerate(correct_labels):
                    if label == 0 and i < len(correct_answers):
                        incorrect_answer = correct_answers[i]
                        break
                if correct_answer and incorrect_answer:
                    test_qa_pairs.append(
                        {
                            "question": raw_question,
                            "formatted_question": formatted_question,
                            "correct_answer": correct_answer,
                            "incorrect_answer": incorrect_answer,
                        }
                    )
        except Exception:
            continue

    if verbose:
        print(f"   • Successfully extracted {len(test_qa_pairs)} test QA pairs")

    # Generate sample responses
    if verbose:
        print("\n🎭 GENERATING SAMPLE RESPONSES WITH HALLUCINATION DETECTION:")
        print(f"   • Generating {len(test_qa_pairs)} sample responses...")

    generated_responses: List[Dict[str, Any]] = []
    correct_classifications = 0
    total_classifications = 0

    for i, qa_pair in enumerate(test_qa_pairs):
        simple_prompt = qa_pair["question"]
        if len(layers) > 1:
            response, layer_results, was_handled = generate_with_multi_layer_classification_and_handling_func(
                model,
                simple_prompt,
                layers,
                max_new_tokens,
                steering_methods,
                token_aggregation,
                detection_threshold,
                verbose and not optimize,
                None,
            )
            primary_layer = layers[0]
            token_scores = layer_results[primary_layer]["token_scores"] if primary_layer in layer_results else []
            classification = (
                layer_results[primary_layer]["classification"] if primary_layer in layer_results else "UNKNOWN"
            )
            aggregated_score = (
                layer_results[primary_layer]["aggregated_score"] if primary_layer in layer_results else 0.0
            )
        else:
            response, token_scores, classification, was_handled = generate_with_classification_and_handling_func(
                model,
                simple_prompt,
                layers[0],
                max_new_tokens,
                steering_method,
                token_aggregation,
                detection_threshold,
                verbose and not optimize,
                None,
            )
            layer_results = None
            aggregated_score = (
                aggregate_token_scores_func(token_scores, token_aggregation) if token_scores else 0.0
            )
        try:
            evaluator = GroundTruthEvaluator_cls.from_string(ground_truth_method)
            correct_answer = qa_pair.get("correct_answer", "")
            user_label = user_labels[i] if user_labels and i < len(user_labels) else None
            evaluation_result = evaluator.evaluate_response(response, correct_answer, user_label)
            ground_truth = evaluation_result["ground_truth"]
            classification_correct = None
            if ground_truth != "UNKNOWN":
                classification_correct = classification == ground_truth
                if classification_correct:
                    correct_classifications += 1
                total_classifications += 1
            entry = {
                "question": qa_pair["question"],
                "response": response,
                "token_scores": token_scores,
                "aggregated_score": aggregated_score,
                "classification": classification,
                "ground_truth": ground_truth,
                "ground_truth_method": evaluation_result["method_used"],
                "ground_truth_confidence": evaluation_result["confidence"],
                "ground_truth_details": evaluation_result["details"],
                "classification_correct": classification_correct,
                "was_handled": was_handled,
            }
            if layer_results:
                entry["layer_results"] = layer_results
            generated_responses.append(entry)
        except Exception as e:
            generated_responses.append(
                {
                    "question": qa_pair["question"],
                    "response": response,
                    "token_scores": token_scores,
                    "classification": classification,
                    "ground_truth": "UNKNOWN",
                    "ground_truth_method": "error",
                    "ground_truth_confidence": 0.0,
                    "ground_truth_details": f"Error during evaluation: {e!s}",
                    "classification_correct": None,
                    "was_handled": was_handled,
                }
            )

    evaluation_results = {
        "accuracy": (
            correct_classifications / total_classifications if total_classifications > 0 else "N/A"
        ),
        "correct_predictions": correct_classifications,
        "total_predictions": total_classifications,
    }

    return {
        "task_name": task_name,
        "evaluation_results": evaluation_results,
        "num_test": len(test_qa_pairs),
        "sample_responses": generated_responses,
        "classification_accuracy": (
            correct_classifications / total_classifications if total_classifications > 0 else None
        ),
        "correct_classifications": correct_classifications,
        "total_classifications": total_classifications,
    }
