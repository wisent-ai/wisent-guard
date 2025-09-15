# cli_utils.py
import os
from typing import Any, Dict, List, Optional

from wisent_guard.cli_bricks.cli_benchmarks import AVAILABLE_BENCHMARKS, UNAVAILABLE_BENCHMARKS, CORE_BENCHMARKS

from wisent_guard.core.parser import parse_layers_from_arg

def get_valid_task_names() -> List[str]:
    """Get list of all valid (available) task names from AVAILABLE_BENCHMARKS."""
    return sorted(list(AVAILABLE_BENCHMARKS.keys()))

def validate_task_name(task_name: str) -> bool:
    """
    Validate if a task name is in the approved AVAILABLE_BENCHMARKS list (37 working benchmarks).

    Args:
        task_name: Name of the task to validate

    Returns:
        True if task is valid, False otherwise
    """
    return task_name in AVAILABLE_BENCHMARKS


def suggest_similar_tasks(invalid_task: str, max_suggestions: int = 5) -> List[str]:
    """
    Suggest similar task names for an invalid task using fuzzy matching.

    Args:
        invalid_task: The invalid task name
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of suggested task names
    """
    from difflib import get_close_matches

    valid_tasks = get_valid_task_names()
    suggestions = get_close_matches(invalid_task, valid_tasks, n=max_suggestions, cutoff=0.6)
    return suggestions

def get_actual_task_name(benchmark_name: str) -> str:
    """
    Resolve benchmark name to actual lm-eval task name.

    Args:
        benchmark_name: The benchmark name from AVAILABLE_BENCHMARKS

    Returns:
        The actual task name to use with lm-eval-harness
    """
    if benchmark_name in AVAILABLE_BENCHMARKS:
        return AVAILABLE_BENCHMARKS[benchmark_name].get("task", benchmark_name)
    return benchmark_name

def parse_layers(layer_arg: str) -> List[int]:
    return parse_layers_from_arg(layer_arg)

def short_task_name(task_name: str) -> str:
    return task_name.split("/")[-1]

def validate_or_explain(
    task_name: str,
    from_csv: bool,
    from_json: bool,
    cross_benchmark_mode: bool,
    from_synthetic: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Validates task names unless the user is providing external data (CSV/JSON)
    or special modes (cross-benchmark/synthetic). Returns {} on success,
    or a rich error payload compatible with the old implementation.
    """
    # Bypass validation when user brings their own data/mode
    if from_csv or from_json or cross_benchmark_mode or from_synthetic:
        if verbose:
            mode = "CSV" if from_csv else "JSON" if from_json else "cross-benchmark" if cross_benchmark_mode else "synthetic"
            print(f"ℹ️  Skipping benchmark whitelist validation (mode: {mode}).")
        return {}

    if validate_task_name(task_name):
        return {}

    # Build rich error (old behavior)
    error_msg = f"Invalid task name: '{task_name}'"
    suggestions = suggest_similar_tasks(task_name) or []

    if verbose:
        print(f"❌ {error_msg}")
        print(f"📋 Task '{task_name}' is not in the available benchmarks list.")
        print(
            f"🔍 Only {len(AVAILABLE_BENCHMARKS)} working benchmarks are supported "
            f"(out of {len(CORE_BENCHMARKS)} total)."
        )

        if task_name in UNAVAILABLE_BENCHMARKS:
            print("🚫 This benchmark is known to be unavailable/problematic.")

        if suggestions:
            print("\n💡 Did you mean one of these?")
            for i, s in enumerate(suggestions, 1):
                cfg = AVAILABLE_BENCHMARKS.get(s, {})
                priority = cfg.get("priority", "unknown")
                tags = ", ".join(cfg.get("tags", []))
                print(f"   {i}. {s} ({priority} priority) - {tags}")

        print("\n📖 To see all valid tasks, run:")
        print("   wisent-guard tasks --list-tasks")
        print("\n📖 To see task details, run:")
        print("   wisent-guard tasks --task-info <task_name>")

    return {
        "task_name": task_name,
        "error": error_msg,
        "error_type": "invalid_task",
        "valid_tasks_count": len(AVAILABLE_BENCHMARKS),
        "total_benchmarks": len(CORE_BENCHMARKS),
        "excluded_count": len(UNAVAILABLE_BENCHMARKS),
        "suggestions": suggestions,
        "help": "Use --list-tasks to see all valid task names",
    }

def maybe_autoload_model_config(
    model_name: str,
    task_name: str,
    layer: str,
    token_aggregation: str,
    detection_threshold: float,
    verbose: bool,
):
    """
    Auto-load saved per-model/task params IFF the caller kept default/sentinel values.
    Respect explicit CLI overrides. Prefer the manager instance; fallback to module helper.
    """
    from wisent_guard.core.model_config_manager import ModelConfigManager, get_optimal_parameters

    # Only autoload if user hasn't overridden defaults
    load_saved = (layer == "15" and token_aggregation == "average" and detection_threshold == 0.6)
    overrides = {}
    if not load_saved:
        return layer, token_aggregation, detection_threshold, overrides

    mgr = ModelConfigManager()

    # Prefer instance method if available; fallback to module function
    try:
        optimal = mgr.get_optimal_parameters(model_name, task_name)
    except AttributeError:
        optimal = get_optimal_parameters(model_name, task_name)

    if optimal:
        orig_l, orig_a, orig_t = layer, token_aggregation, detection_threshold
        if "classification_layer" in optimal and optimal["classification_layer"] is not None:
            layer = str(optimal["classification_layer"])
            overrides["layer"] = f"{orig_l} → {layer}"
        if "token_aggregation" in optimal and optimal["token_aggregation"]:
            token_aggregation = optimal["token_aggregation"]
            overrides["token_aggregation"] = f"{orig_a} → {token_aggregation}"
        if "detection_threshold" in optimal and optimal["detection_threshold"] is not None:
            detection_threshold = optimal["detection_threshold"]
            overrides["detection_threshold"] = f"{orig_t} → {detection_threshold}"

        if verbose and overrides:
            print(f"\n🔧 Auto-loaded saved configuration for model: {model_name}")
            for k, v in overrides.items():
                print(f"   • {k}: {v}")
            print()

    return layer, token_aggregation, detection_threshold, overrides

def maybe_autoload_steering_defaults(
    steering_mode: bool,
    model_name: str,
    task_name: str,
    steering_method: str,
    current_layer: str,
    current_strength: float,
    verbose: bool,
    force_autoload: bool = False,   # set True to always override with saved defaults
) -> tuple[str, str, float]:
    """
    If steering_mode is on, optionally auto-load per-(model, task) steering defaults.
    We only override when:
      - force_autoload=True, or
      - the caller kept sentinel defaults (method=='CAA', layer=='15', strength==1.0)
    Returns: (steering_method, layer, steering_strength)
    """
    if not steering_mode:
        return steering_method, current_layer, current_strength

    try:
        from wisent_guard.core.steering_optimizer import get_optimal_steering_params
        opt = get_optimal_steering_params(model_name, task_name) or {}
    except Exception:
        opt = {}

    if not opt:
        return steering_method, current_layer, current_strength

    # Decide if we should override each field
    use_saved_method   = force_autoload or (steering_method == "CAA")
    use_saved_layer    = force_autoload or (str(current_layer) == "15")   # same sentinel as classifier default
    use_saved_strength = force_autoload or (float(current_strength) == 1.0)

    new_method   = opt.get("method", steering_method)
    new_layer    = str(opt.get("layer", current_layer))
    new_strength = float(opt.get("strength", current_strength))

    if verbose:
        changed = []
        if use_saved_method and "method" in opt and opt["method"] and opt["method"] != steering_method:
            changed.append(f"method: {steering_method} → {opt['method']}")
        if use_saved_layer and "layer" in opt and opt["layer"] is not None and str(opt["layer"]) != str(current_layer):
            changed.append(f"layer: {current_layer} → {opt['layer']}")
        if use_saved_strength and "strength" in opt and opt["strength"] is not None and float(opt["strength"]) != float(current_strength):
            changed.append(f"strength: {current_strength} → {opt['strength']}")
        if changed:
            print("\n🔧 Auto-loaded steering defaults")
            for line in changed:
                print(f"   • {line}")

    return (
        new_method if use_saved_method else steering_method,
        new_layer if use_saved_layer else current_layer,
        new_strength if use_saved_strength else current_strength,
    )

def build_detection_handler(
    detection_action: str,
    placeholder_message: Optional[str],
    max_regeneration_attempts: int,
    log_detections: bool,
):
    if detection_action == "pass_through":
        return None
    from wisent_guard.core.detection_handling import DetectionAction, DetectionHandler
    action_map = {
        "pass_through": DetectionAction.PASS_THROUGH,
        "replace_with_placeholder": DetectionAction.REPLACE_WITH_PLACEHOLDER,
        "regenerate_until_safe": DetectionAction.REGENERATE_UNTIL_SAFE,
    }
    return DetectionHandler(
        action=action_map[detection_action],
        placeholder_message=placeholder_message,
        max_regeneration_attempts=max_regeneration_attempts,
        log_detections=log_detections,
    )

if __name__ == "__main__":
    print("\n[cli_utils] smoke test")
    try:
        # # 1) validate_or_explain (CSV/JSON path bypasses benchmark registry)
        print("• validate_or_explain (valid):", validate_or_explain("mmmlu", False, False, False, False, True))
        print("• validate_or_explain (invalid):", validate_or_explain("invalid_task_name", False, False, False, False, True))
        print("• validate_or_explain (CSV bypass):", validate_or_explain("invalid_task_name", True, False, False, False, True))

        # 2) maybe_autoload_model_config (should no-op in most envs)
        layer, agg, thr, ovr = maybe_autoload_model_config(
            model_name="test-model",
            task_name="mmmlu",
            layer="15",
            token_aggregation="average",
            detection_threshold=0.6,
            verbose=True,
        )
        print(f"  model_config -> layer={layer}, agg={agg}, thr={thr}, overrides={ovr or '{}'}")

                # 3) maybe_autoload_steering_defaults (returns tuple; usually no-op)
        sm, new_layer, strength = maybe_autoload_steering_defaults(
            steering_mode=True,
            model_name="test-model",
            task_name="hellaswag",
            steering_method="CAA",
            current_layer=layer,
            current_strength=1.0,
            verbose=True,
        )
        print(f"  steering_defaults -> method={sm}, layer={new_layer}, strength={strength}")

    except ImportError as e:
        print("  SKIPPED (missing deps):", e)
