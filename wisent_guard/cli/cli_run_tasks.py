# cli_run_tasks.py
import logging
from typing import Any, Dict, List, Optional

from wisent_guard.core import Model
from wisent_guard.cli.cli_utils import (
    validate_or_explain,
    maybe_autoload_model_config,
    maybe_autoload_steering_defaults,
    build_detection_handler,
    parse_layers,
    
)

from wisent_guard.cli.cli_performance import start_trackers, stop_and_report
from wisent_guard.cli.cli_data import load_train_test_data
from wisent_guard.cli.cli_activation import (
    make_collector,
    create_contrastive_pairs,
    extract_activations_for_pairs,
    build_pair_set_with_real_activations
)
from wisent_guard.cli.cli_optimization import maybe_optimize_hparams
from wisent_guard.cli.cli_train import train_or_load_classifiers, save_trained_artifacts
from wisent_guard.cli.cli_steering import (
    build_steering_for_mode,
    test_steering_with_harness,
)
from wisent_guard.cli.cli_eval import (
    maybe_eval_perplexity_directly,
    eval_with_harness_ground_truth,
    generate_and_score_responses,
)

logger = logging.getLogger(__name__)


def run_task_pipeline(
    task_name: str,
    model_name: str,
    layer: str,
    shots: int = 0,
    split_ratio: float = 0.8,
    limit: Optional[int] = None,
    training_limit: Optional[int] = None,
    testing_limit: Optional[int] = None,
    classifier_type: str = "logistic",
    max_new_tokens: int = 300,
    device: Optional[str] = None,
    seed: int = 42,
    token_aggregation: str = "average",
    ground_truth_method: str = "lm-eval-harness",
    user_labels: Optional[List[str]] = None,
    optimize: bool = False,
    optimize_layers: str = "all",
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
    placeholder_message: Optional[str] = None,
    max_regeneration_attempts: int = 3,
    detection_threshold: float = 0.6,
    log_detections: bool = False,
    steering_mode: bool = False,
    steering_strength: float = 1.0,
    output_mode: str = "both",
    save_steering_vector: Optional[str] = None,
    load_steering_vector: Optional[str] = None,
    train_only: bool = False,
    inference_only: bool = False,
    save_classifier: Optional[str] = None,
    load_classifier: Optional[str] = None,
    classifier_dir: str = "./models",
    prompt_construction_strategy: str = "multiple_choice",
    token_targeting_strategy: str = "choice_token",
    normalize_mode: bool = False,
    normalization_method: str = "none",
    target_norm: Optional[float] = None,
    steering_method: str = "CAA",
    hpr_beta: float = 1.0,                 # may be unused by HPR in current code
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
    # Nonsense detection
    enable_nonsense_detection: bool = False,
    max_word_length: int = 20,
    repetition_threshold: float = 0.7,
    gibberish_threshold: float = 0.3,
    disable_dictionary_check: bool = False,
    nonsense_action: str = "regenerate",
    # Token steering
    enable_token_steering: bool = False,
    token_steering_strategy: str = "second_to_last",
    token_decay_rate: float = 0.5,
    token_min_strength: float = 0.1,
    token_max_strength: float = 1.0,
    token_apply_to_prompt: bool = False,
    token_prompt_strength_multiplier: float = 0.1,
    # Performance
    enable_memory_tracking: bool = False,
    enable_latency_tracking: bool = False,
    memory_sampling_interval: float = 0.1,
    track_gpu_memory: bool = False,
    detailed_performance_report: bool = False,
    export_performance_csv: Optional[str] = None,
    show_memory_usage: bool = False,
    show_timing_summary: bool = False,
    # Test activations
    save_test_activations: Optional[str] = None,
    load_test_activations: Optional[str] = None,
    # Selection/caching
    priority: str = "all",
    fast_only: bool = False,
    time_budget: Optional[float] = None,
    max_benchmarks: Optional[int] = None,
    smart_selection: bool = False,
    cache_benchmark: bool = True,
    preloaded_qa_pairs: Optional[List[Dict[str, Any]]] = None,
    # Cross-benchmark / synthetic
    cross_benchmark_mode: bool = False,
    train_contrastive_pairs: Optional[Any] = None,
    eval_contrastive_pairs: Optional[Any] = None,
    use_cached: bool = True,
    force_download: bool = False,
    cache_dir: str = "./benchmark_cache",
    from_synthetic: bool = False,
    synthetic_contrastive_pairs: Optional[Any] = None,
    # Model reuse
    model_instance: Optional[Any] = None,
    **kwargs,  # keep forward-compat — we’ll prune after coverage
) -> Dict[str, Any]:
    """
    Orchestrates the full pipeline with clear, short-circuiting steps.
    """

    # #1 Validate & normalize inputs early (fast fail; suggest alternatives on errors)
    validate_result = validate_or_explain(
        task_name=task_name,
        from_csv=from_csv,
        from_json=from_json,
        cross_benchmark_mode=cross_benchmark_mode,
        from_synthetic=from_synthetic,
        verbose=verbose,
    ) #TODO handel the error properly
    if validate_result.get("error"):
        return validate_result

    # #2 Start optional performance trackers (memory/latency) and show current usage if asked
    trackers = start_trackers(
        enable_memory_tracking=enable_memory_tracking,
        enable_latency_tracking=enable_latency_tracking,
        memory_sampling_interval=memory_sampling_interval,
        track_gpu_memory=track_gpu_memory,
        show_memory_usage=show_memory_usage,
        verbose=verbose,
    )

    # #3 Autoload persisted model configuration & steering defaults (only when not overridden)
    layer, token_aggregation, detection_threshold, _ = maybe_autoload_model_config(
        model_name=model_name,
        task_name=task_name,
        layer=layer,
        token_aggregation=token_aggregation,
        detection_threshold=detection_threshold,
        verbose=verbose,
    )

    # Use the new tuple-returning steering defaults function
    steering_method, layer, steering_strength = maybe_autoload_steering_defaults(
        steering_mode=steering_mode,
        model_name=model_name,
        task_name=task_name,
        steering_method=steering_method,
        current_layer=layer,
        current_strength=steering_strength,
        verbose=verbose,
        # force_autoload=False  # keep default: only override when caller used sentinel values
)

    # #4 Build or reuse model; parse layers once
    model = model_instance if model_instance is not None else Model(name=model_name, device=device)
    layers: List[int] = parse_layers(layer)

    # #5 Build detection handler (no-op by default)
    detection_handler = build_detection_handler(
        detection_action=detection_action,
        placeholder_message=placeholder_message,
        max_regeneration_attempts=max_regeneration_attempts,
        log_detections=log_detections,
    )

    # #6 Load training & test data (honors CSV/JSON/synthetic/cross-benchmark/cache, handles group tasks)
    data = load_train_test_data(
        model=model,
        task_name=task_name,
        shots=shots,
        split_ratio=split_ratio,
        limit=limit,
        training_limit=training_limit,
        testing_limit=testing_limit,
        seed=seed,
        from_csv=from_csv,
        from_json=from_json,
        question_col=question_col,
        correct_col=correct_col,
        incorrect_col=incorrect_col,
        allow_small_dataset=allow_small_dataset,
        cache_benchmark=cache_benchmark,
        use_cached=use_cached,
        force_download=force_download,
        cache_dir=cache_dir,
        preloaded_qa_pairs=preloaded_qa_pairs,
        cross_benchmark_mode=cross_benchmark_mode,
        train_contrastive_pairs=train_contrastive_pairs,
        eval_contrastive_pairs=eval_contrastive_pairs,
        from_synthetic=from_synthetic,
        synthetic_contrastive_pairs=synthetic_contrastive_pairs,
        verbose=verbose,
    )
    if data.get("error"):
        return stop_and_report(trackers, export_performance_csv, detailed_performance_report, show_timing_summary, verbose, error=data["error"])

    qa_pairs                = data["qa_pairs"]
    test_qa_pairs_source    = data["test_qa_pairs_source"]
    task_data               = data.get("task_data")
    group_task_qa_format    = data["group_task_qa_format"]
    is_multi_layer          = len(layers) > 1

    # #7 Early short-circuit: perplexity tasks (skip training; evaluate directly)
    maybe_ppx = maybe_eval_perplexity_directly(
        ground_truth_method=ground_truth_method,
        task_name=task_name,
        model=model,
        layers=layers,
        token_aggregation=token_aggregation,
        test_docs=test_qa_pairs_source,
        verbose=verbose,
    )
    if maybe_ppx is not None:
        return stop_and_report(trackers, export_performance_csv, detailed_performance_report, show_timing_summary, verbose, result=maybe_ppx)

    # #8 Build activation collector; create contrastive pairs (synthetic/cross-benchmark respected)
    collector = make_collector(model)
    contrastive_pairs = create_contrastive_pairs(
        collector=collector,
        qa_pairs=qa_pairs,
        prompt_construction_strategy=prompt_construction_strategy,
        cross_benchmark_mode=cross_benchmark_mode,
        from_synthetic=from_synthetic,
        synthetic_contrastive_pairs=synthetic_contrastive_pairs,
        train_contrastive_pairs=train_contrastive_pairs,
        verbose=verbose,
    )

    # #9 Validate mode flags and “inference-only” requirements
    if train_only and inference_only:
        return stop_and_report(trackers, export_performance_csv, detailed_performance_report, show_timing_summary, verbose, error="Cannot specify both --train-only and --inference-only.")
    if inference_only and not (load_classifier or load_steering_vector):
        return stop_and_report(trackers, export_performance_csv, detailed_performance_report, show_timing_summary, verbose, error="Inference-only mode requires --load-classifier or --load-steering-vector.")

    # #10 Optional hyperparameter search (interactive/smart). Returns possibly updated layer/agg/type/threshold
    hp = maybe_optimize_hparams(
        optimize=optimize,
        mode=ground_truth_method,
        model=model,
        collector=collector,
        task_name=task_name,
        model_name=model_name,
        contrastive_pairs=contrastive_pairs,
        test_qa_pairs_source=test_qa_pairs_source,
        task_data=task_data,
        layers=layers,
        token_aggregation=token_aggregation,
        classifier_type=classifier_type,
        detection_threshold=detection_threshold,
        optimize_layers=optimize_layers,
        optimize_metric=optimize_metric,
        optimize_max_combinations=optimize_max_combinations,
        max_new_tokens=max_new_tokens,
        device=device,
        group_task_qa_format=group_task_qa_format,
        verbose=verbose,
    )
    if hp:
        layers = [int(hp["best_layer"])]
        token_aggregation  = hp.get("best_aggregation", token_aggregation)
        classifier_type    = hp.get("best_classifier_type", classifier_type)
        detection_threshold= hp.get("best_threshold", detection_threshold)

    # #11 Extract activations for (possibly optimized) layer(s) and build pair set with real activations
    processed_pairs = extract_activations_for_pairs(
        collector=collector,
        contrastive_pairs=contrastive_pairs,
        layers=layers,
        device=device,
        token_targeting_strategy=token_targeting_strategy,
        latency_tracker=trackers.latency if trackers else None,
        verbose=verbose,
    )

    pair_set = build_pair_set_with_real_activations(
        processed_pairs=processed_pairs,
        task_name=task_name,
        from_synthetic=from_synthetic,
        synthetic_contrastive_pairs=synthetic_contrastive_pairs,
        cross_benchmark_mode=cross_benchmark_mode,
        train_contrastive_pairs=train_contrastive_pairs,
        verbose=verbose,
    )

    # #12 Steering vs Classification setup (training and/or loading as appropriate)
    if steering_mode:
        steering = build_steering_for_mode(
            method_name=steering_method,
            device=device,
            normalization_method=normalization_method,
            target_norm=target_norm,
            dac_dynamic_control=dac_dynamic_control,
            dac_entropy_threshold=dac_entropy_threshold,
            bipo_beta=bipo_beta,
            bipo_lr=bipo_learning_rate,
            bipo_epochs=bipo_epochs,
            k_num_labels=ksteering_num_labels,
            k_hidden=ksteering_hidden_dim,
            k_lr=ksteering_learning_rate,
            k_epochs=ksteering_classifier_epochs,
            k_target=ksteering_target_labels,
            k_avoid=ksteering_avoid_labels,
            k_alpha=ksteering_alpha,
            save_path=save_steering_vector,
            load_path=load_steering_vector,
            layer_idx=layers[0], #TODO change to only support single layer for steering for now
            pair_set=pair_set,
            verbose=verbose,
        )

        # Optional nonsense detection & token steering knobs are wired inside generation step.
        # Test steering quickly via lm-eval-harness (same path as baseline) and short-circuit if it fails.
        steer_eval = test_steering_with_harness(
            task_data=task_data,
            test_docs=test_qa_pairs_source,
            model=model,
            layers=layers,
            steering=steering,
            steering_strength=steering_strength,
            output_mode=output_mode,
            group_task_qa_format=group_task_qa_format,
            from_csv=from_csv or from_json,
            verbose=verbose,
        )
        # Performance is printed from inside when verbose; on failure we still shut down trackers cleanly.
        if steer_eval.get("error"):
            return stop_and_report(trackers, export_performance_csv, detailed_performance_report, show_timing_summary, verbose, result=steer_eval)

        return stop_and_report(trackers, export_performance_csv, detailed_performance_report, show_timing_summary, verbose, result=steer_eval)

    # Classification path (single or multi-layer) — load (if provided/available) or train
    steering_methods, layer_training_results, _ = train_or_load_classifiers(
        is_multi_layer=is_multi_layer,
        processed_pairs=processed_pairs,
        layers=layers,
        classifier_type=classifier_type,
        load_classifier=load_classifier,
        classifier_dir=classifier_dir,
        auto_discover_model_name=model_name,
        task_name=task_name,
        detection_threshold=detection_threshold,
        train_only=train_only,
        verbose=verbose,
    )

    # #13 Save trained artifacts (if requested) and early return for --train-only
    saved_classifier_paths = save_trained_artifacts(
        steering_methods=steering_methods,
        layers=layers,
        is_multi_layer=is_multi_layer,
        save_classifier=save_classifier,
        train_only=train_only,
        task_name=task_name,
        model_name=model_name,
        classifier_dir=classifier_dir,
        classifier_type=classifier_type,               # or final_classifier_type if you rename after HPO
        training_results=layer_training_results,
        contrastive_pairs_count=len(processed_pairs),   # same count you used before
        token_aggregation=token_aggregation,
        detection_threshold=detection_threshold,        # or final_threshold if HPO changed it
        verbose=verbose,
    )
    if train_only:
        train_only_result = {
            "task_name": task_name,
            "model_name": model_name,
            "mode": "train_only",
            "layers": layers,
            "trained_layers": list(steering_methods.keys()),
            "training_results": (layer_training_results if is_multi_layer else {layers[0]: layer_training_results[layers[0]]}),
            "saved_classifier_paths": saved_classifier_paths,
            "classifier_type": classifier_type,
            "training_samples": len(processed_pairs),
            "success": True,
        }
        return stop_and_report(trackers, export_performance_csv, detailed_performance_report, show_timing_summary, verbose, result=train_only_result)

    # #14 Ground-truth evaluation (lm-eval-harness path mirrors your original code)
    harness_eval = eval_with_harness_ground_truth(
        enabled=(ground_truth_method == "lm-eval-harness" and not cross_benchmark_mode),
        task_name=task_name,
        model=model,
        layers=layers,
        token_aggregation=token_aggregation,
        test_docs=test_qa_pairs_source,
        trained_steering_methods=steering_methods,
        verbose=verbose,
    )
    if harness_eval is not None:
        return stop_and_report(trackers, export_performance_csv, detailed_performance_report, show_timing_summary, verbose, result=harness_eval)

    # #15 Generate responses + classify + evaluate (non-harness path; supports nonsense detection & token steering)
    gen_eval = generate_and_score_responses(
        model=model,
        task_name=task_name,
        task_data=task_data,
        layers=layers,
        steering_methods=steering_methods,
        token_aggregation=token_aggregation,
        detection_threshold=detection_threshold,
        max_new_tokens=max_new_tokens,
        detection_handler=detection_handler,
        test_docs=test_qa_pairs_source,
        ground_truth_method=ground_truth_method,
        user_labels=user_labels,
        enable_nonsense_detection=enable_nonsense_detection,
        nonsense_opts=dict(
            max_word_length=max_word_length,
            repetition_threshold=repetition_threshold,
            gibberish_threshold=gibberish_threshold,
            enable_dictionary_check=not disable_dictionary_check,
            action=nonsense_action,
        ),
        token_steering_opts=dict(
            enable=enable_token_steering,
            strategy=token_steering_strategy,
            decay=token_decay_rate,
            min_strength=token_min_strength,
            max_strength=token_max_strength,
            apply_to_prompt=token_apply_to_prompt,
            prompt_strength_multiplier=token_prompt_strength_multiplier,
        ),
        save_test_activations=save_test_activations,
        load_test_activations=load_test_activations,
        latency_tracker=trackers.latency if trackers else None,
        verbose=verbose,
        optimize=optimize, 
    )


    return stop_and_report(trackers, export_performance_csv, detailed_performance_report, show_timing_summary, verbose, result=gen_eval)


