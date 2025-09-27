# cli_eval.py
from typing import Any, Dict, List, Optional, Union

from wisent_guard.core.lm_eval_harness_ground_truth import LMEvalHarnessGroundTruth
from wisent_guard.core.parser import aggregate_token_scores
from wisent_guard.core.ground_truth_evaluator import GroundTruthEvaluator
from wisent_guard.core import Layer
import torch

def _evaluation_method_for(task_name: str, verbose: bool = False) -> str:
    from pathlib import Path
    import json
    import sys

    DEFAULT_METHOD = "text-generation"

    # Lazy-load and cache the methods dict on first call
    if not hasattr(_evaluation_method_for, "_methods_cache"):
        benchmarks_file = (
            Path(__file__).resolve().parent.parent
            / "parameters" / "benchmarks" / "benchmark_evaluation_methods.json"
        )
        try:
            with benchmarks_file.open() as f:
                _evaluation_method_for._methods_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            if verbose:
                print(
                    f"⚠️ Could not load benchmark evaluation methods from {benchmarks_file}: {e}",
                    file=sys.stderr,
                )
            _evaluation_method_for._methods_cache = {}

    return _evaluation_method_for._methods_cache.get(task_name, DEFAULT_METHOD)


def maybe_eval_perplexity_directly(
    ground_truth_method: str,
    task_name: str,
    model,
    layers: List[int],
    token_aggregation: str,
    test_docs,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    if ground_truth_method != "lm-eval-harness":
        return None
    method = _evaluation_method_for(task_name, verbose)
    if method != "perplexity":
        return None

    if verbose:
        print("\n🎯 PERPLEXITY TASK: Skipping contrastive training and evaluating directly")

    actual = task_name  # assume resolver is already applied upstream if needed
    lm_eval_gt = LMEvalHarnessGroundTruth(actual, method, model=model)
    results = lm_eval_gt.evaluate_classifier_on_task(
        classifier=None,
        task_name=actual,
        num_samples=len(test_docs),
        model=model,
        layer=layers[0],
        token_aggregation=token_aggregation,
    )
    return {
        "task_name": task_name,
        "model_name": getattr(model, "name", "model"),
        "layer": layers[0],
        "evaluation_method": method,
        "evaluation_results": results,
        "num_test": len(test_docs),
        "ground_truth_method": "lm-eval-harness",
        "skipped_training": True,
        "reason": "Perplexity task",
    }

def eval_with_harness_ground_truth(
    enabled: bool,
    task_name: str,
    model,
    layers: List[int],
    token_aggregation: str,
    test_docs,
    trained_steering_methods: Dict[int, Any],
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    method = _evaluation_method_for(task_name, verbose)
    if verbose:
        print("\n🔍 LM-EVAL-HARNESS GROUND TRUTH EVALUATION:")
        print(f"   • Task: {task_name} • Eval: {method} • Samples: {len(test_docs)}")

    classifier = None
    if layers and layers[0] in trained_steering_methods:
        classifier = trained_steering_methods[layers[0]].classifier

    actual = task_name
    lm_eval_gt = LMEvalHarnessGroundTruth(actual, method, model=model)
    lm_eval_results = lm_eval_gt.evaluate_classifier_on_task(
        classifier,
        actual,
        num_samples=len(test_docs),
        model=model,
        layer=layers[0],
        token_aggregation=token_aggregation,
    )
    return {
        "task_name": task_name,
        "model_name": getattr(model, "name", "model"),
        "layer": layers[0],
        "evaluation_method": method,
        "evaluation_results": lm_eval_results,
        "num_test": len(test_docs),
        "ground_truth_method": "lm-eval-harness",
        "classification_accuracy": lm_eval_results.get("accuracy", 0.0),
    }

def generate_and_score_responses(
    model,
    task_name: str,
    task_data,
    layers: List[int],
    steering_methods: Dict[int, Any],
    token_aggregation: str,
    detection_threshold: float,
    max_new_tokens: int,
    detection_handler,
    test_docs,
    ground_truth_method: str,
    user_labels: Optional[List[str]],
    enable_nonsense_detection: bool,
    nonsense_opts: Dict[str, Any],
    token_steering_opts: Dict[str, Any],
    save_test_activations: Optional[str],
    load_test_activations: Optional[str],
    latency_tracker=None,
    verbose: bool = False,
    optimize: bool = False,   # <— added to match previous verbose flow
) -> Dict[str, Any]:
    """
    Generates responses and evaluates them OR uses cached test activations if provided.
    Feature parity with the old implementation:
      - load/save TestActivationCache
      - use cached activations path with classification & evaluation
      - verbose banners for optimize vs normal runs
    """
    from wisent_guard.inference import (
        generate_with_classification_and_handling,
        generate_with_multi_layer_classification_and_handling,
    )
    from wisent_guard.cli_workflows.activation_monitor import TestActivationCache
    from wisent_guard.core.activations import ActivationAggregationStrategy, Activations

    # -----------------------------
    # Load / init activation cache
    # -----------------------------
    test_activation_cache: Optional[TestActivationCache] = None
    use_cached_activations = False
    cached_layer_activations: List[Dict[str, Any]] = []

    if load_test_activations:
        if verbose:
            print("\n💾 LOADING CACHED TEST ACTIVATIONS:")
            print(f"   • Loading from: {load_test_activations}")
        try:
            test_activation_cache = TestActivationCache.load_from_file(load_test_activations)
            cached_layer_activations = test_activation_cache.get_activations_for_layer(layers[0]) or []
            if cached_layer_activations:
                use_cached_activations = True
                if verbose:
                    print(f"   ✅ Found {len(cached_layer_activations)} cached activations for layer {layers[0]}")
            else:
                if verbose:
                    print(f"   ❌ No cached activations found for layer {layers[0]}")
                    if getattr(test_activation_cache, "activations", None):
                        avail_layers = list({item["layer"] for item in test_activation_cache.activations})
                        print(f"   • Available layers: {avail_layers}")
        except Exception as e:
            if verbose:
                print(f"   ❌ Failed to load cached activations: {e}")
                print("   • Will generate new responses instead")

    if save_test_activations and not use_cached_activations:
        test_activation_cache = TestActivationCache()
        if verbose:
            print(f"\n💾 WILL SAVE TEST ACTIVATIONS TO: {save_test_activations}")

    # Banner like the old code
    if verbose:
        if optimize:
            print("\n🎭 GENERATING SAMPLE RESPONSES WITH OPTIMIZED CLASSIFIER:")
            print(f"   • Generating {len(test_docs)} sample responses with optimized layer {layers[0]}...")
        else:
            print("\n🎭 GENERATING SAMPLE RESPONSES WITH HALLUCINATION DETECTION:")
            print(f"   • Generating {len(test_docs)} sample responses...")

    # -----------------------------------
    # Main loop: cached path OR generate
    # -----------------------------------
    generated_responses: List[Dict[str, Any]] = []
    correct_classifications = 0
    total_classifications = 0

    if use_cached_activations:
        if verbose:
            print("\n🔄 PROCESSING CACHED ACTIVATIONS:")
            print(f"   • Processing {len(cached_layer_activations)} cached responses...")

        for i, cached_item in enumerate(cached_layer_activations):
            # Use cached response and activations
            response = cached_item.get("response", "")
            activations = cached_item.get("activations")

            # Classify using the trained steering method(s)
            if len(layers) > 1:
                if layers[0] in steering_methods:
                    sm = steering_methods[layers[0]]
                    cls_res = sm.classify_activation(activations)
                    classification = "HALLUCINATION" if cls_res.get("is_harmful", False) else "TRUTHFUL"
                    score = cls_res.get("score", 0.5)
                else:
                    classification = "UNKNOWN"; score = 0.5
            else:
                sm = steering_methods[layers[0]]
                cls_res = sm.classify_activation(activations)
                classification = "HALLUCINATION" if cls_res.get("is_harmful", False) else "TRUTHFUL"
                score = cls_res.get("score", 0.5)

            token_scores = [score]
            aggregated_score = score

            # Ground truth evaluation
            evaluator = GroundTruthEvaluator.from_string(ground_truth_method)
            user_label = user_labels[i] if user_labels and i < len(user_labels) else None
            try:
                eval_res = evaluator.evaluate_response(response, cached_item.get("correct_answer", "N/A"), user_label)
                ground_truth = eval_res["ground_truth"]
                classification_correct = None
                if ground_truth != "UNKNOWN":
                    classification_correct = (classification == ground_truth)
                    if classification_correct:
                        correct_classifications += 1
                    total_classifications += 1
                entry = {
                    "question": cached_item.get("question", ""),
                    "response": response,
                    "token_scores": token_scores,
                    "aggregated_score": aggregated_score,
                    "classification": classification,
                    "ground_truth": ground_truth,
                    "ground_truth_method": eval_res["method_used"],
                    "ground_truth_confidence": eval_res["confidence"],
                    "ground_truth_details": eval_res["details"],
                    "classification_correct": classification_correct,
                    "was_handled": False,
                    "source": "cached_activations",
                }
            except Exception as e:
                entry = {
                    "question": cached_item.get("question", ""),
                    "response": response,
                    "token_scores": token_scores,
                    "aggregated_score": aggregated_score,
                    "classification": classification,
                    "ground_truth": "UNKNOWN",
                    "ground_truth_method": "error",
                    "ground_truth_confidence": 0.0,
                    "ground_truth_details": f"Error during evaluation: {e!s}",
                    "classification_correct": None,
                    "was_handled": False,
                    "source": "cached_activations",
                }
            generated_responses.append(entry)

    else:
        # Generate fresh responses
        for i, doc in enumerate(test_docs):
            # Build prompt
            if isinstance(doc, dict) and "question" in doc:
                simple_prompt = doc["question"]
                correct_answer = doc.get("correct_answer", "")
            else:
                simple_prompt = task_data.doc_to_text(doc) if hasattr(task_data, "doc_to_text") else str(doc)
                # try to pull a correct answer for GT comparison where possible
                correct_answers = doc.get("mc1_targets", {}).get("choices", []) if isinstance(doc, dict) else []
                labels = doc.get("mc1_targets", {}).get("labels", []) if isinstance(doc, dict) else []
                correct_answer = next((c for c, l in zip(correct_answers, labels) if l == 1), "")

            if len(layers) > 1:
                response, layer_results, was_handled = generate_with_multi_layer_classification_and_handling(
                    model=model,
                    prompt=simple_prompt,
                    layer_indices=layers,
                    max_new_tokens=max_new_tokens,
                    steering_methods=steering_methods,
                    token_aggregation=token_aggregation,
                    detection_threshold=detection_threshold,
                    verbose=verbose,
                    detection_handler=detection_handler,
                    enable_nonsense_detection=enable_nonsense_detection,
                    nonsense_options=nonsense_opts,
                    token_steering_options=token_steering_opts,
                )
                primary = layers[0]
                token_scores = (layer_results[primary]["token_scores"] if primary in layer_results else [])
                classification = (layer_results[primary]["classification"] if primary in layer_results else "UNKNOWN")
                aggregated_score = (layer_results[primary]["aggregated_score"] if primary in layer_results else 0.0)
            else:
                response, token_scores, classification, was_handled = generate_with_classification_and_handling(
                    model,
                    simple_prompt,
                    layers[0],
                    max_new_tokens,
                    steering_methods[layers[0]],
                    token_aggregation,
                    detection_threshold,
                    verbose,
                    detection_handler,
                    enable_nonsense_detection=enable_nonsense_detection,
                    nonsense_options=nonsense_opts,
                    token_steering_options=token_steering_opts,
                )
                aggregated_score = aggregate_token_scores(token_scores, token_aggregation) if token_scores else 0.0

                # Quick forward pass to grab activations (single-layer path)
                if save_test_activations and test_activation_cache is not None:
                    try:
                        model_inputs = model.tokenizer(simple_prompt, return_tensors="pt", padding=True)
                        if hasattr(model, "device"):
                            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
                        with torch.no_grad():
                            outputs = model.model(**model_inputs, output_hidden_states=True)
                        if outputs.hidden_states and len(outputs.hidden_states) > layers[0]:
                            layer_acts = outputs.hidden_states[layers[0] + 1]  # +1 for embeddings
                            layer_obj = Layer(index=layers[0], type="transformer")
                            acts = Activations(
                                tensor=layer_acts,
                                layer=layer_obj,
                                aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                            )
                            test_activation_cache.add_activation(
                                question=simple_prompt,
                                response=response,
                                activations=acts,
                                layer=layers[0],
                                correct_answer=correct_answer,
                            )
                    except Exception as e:
                        if verbose:
                            print(f"      ⚠️  Could not save activation: {e}")

            # Evaluate
            evaluator = GroundTruthEvaluator.from_string(ground_truth_method)
            user_label = user_labels[i] if user_labels and i < len(user_labels) else None
            eval_res = evaluator.evaluate_response(response, correct_answer, user_label)
            ground_truth = eval_res["ground_truth"]
            classification_correct = None
            if ground_truth != "UNKNOWN":
                expected = "HALLUCINATION" if ground_truth == "HALLUCINATION" else "TRUTHFUL"
                classification_correct = (classification == expected)
                if classification_correct:
                    correct_classifications += 1
                total_classifications += 1

            generated_responses.append(
                {
                    "question": simple_prompt,
                    "response": response,
                    "classification": classification,
                    "token_scores": token_scores,
                    "aggregated_score": aggregated_score,
                    "ground_truth": ground_truth,
                    "ground_truth_method": eval_res["method_used"],
                    "ground_truth_confidence": eval_res["confidence"],
                    "ground_truth_details": eval_res["details"],
                    "classification_correct": classification_correct,
                    "was_handled": was_handled,
                    "source": "generated",
                }
            )

    # -----------------
    # Finalize & save
    # -----------------
    if save_test_activations and test_activation_cache and not use_cached_activations and getattr(test_activation_cache, "activations", None):
        try:
            test_activation_cache.save_to_file(save_test_activations)
            if verbose:
                print("\n💾 SAVED TEST ACTIVATIONS:")
                print(f"   • File: {save_test_activations}")
                print(f"   • Count: {len(test_activation_cache.activations)} activations")
                print(f"   • Layer: {layers[0]}")
        except Exception as e:
            if verbose:
                print(f"\n❌ Failed to save test activations: {e}")

    classification_acc = (correct_classifications / total_classifications) if total_classifications > 0 else None
    return {
        "task_name": task_name,
        "model_name": getattr(model, "name", "model"),
        "layer": layers[0],
        "num_test": len(test_docs) if not use_cached_activations else len(cached_layer_activations),
        "generated": generated_responses,
        "correct_classifications": correct_classifications,
        "total_classifications": total_classifications,
        "classification_accuracy": classification_acc,
        "ground_truth_method": ground_truth_method,
        "used_cached_activations": use_cached_activations,
    }

if __name__ == "__main__":
    # simple test, first perplexity, then generation
    from wisent_guard.core import Model
    model = Model(name="/home/gg/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6")
    task = "wikitext"
    ground_truth_method = "lm-eval-harness"
    layers = [5]
    token_aggregation = "mean"
    test_docs = [{"question": "The capital of France is", "correct_answer": "Paris"}, {"question": "The largest planet in our solar system is", "correct_answer": "Jupiter"}]
    res = maybe_eval_perplexity_directly(ground_truth_method, task, model, layers, token_aggregation, test_docs, verbose=True)
    print(res)



