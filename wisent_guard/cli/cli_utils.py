# cli_utils.py
import os
from typing import Any, Dict, List, Optional

from wisent_guard.cli.cli_benchmarks import AVAILABLE_BENCHMARKS, UNAVAILABLE_BENCHMARKS, CORE_BENCHMARKS

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
    verbose: bool,
) -> Dict[str, Any]:
    """
    Validates task names unless the user is providing external data (CSV/JSON)
    or special modes (cross-benchmark/synthetic). Returns {} on success,
    or a rich error payload compatible with the old implementation.
    """
   
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

def _get_steering_bits(model_name: str, task_name: Optional[str]) -> dict[str, int | str | float]:
    """
    Read method/layer/strength/token_aggregation from steering config.
    Prefers task-specific, falls back to global 'steering_optimization'.
    Supports both plain and 'best_*' key variants for robustness.

    Arguments:
        model_name: Name of the model to load config for.
        task_name: Optional task name for task-specific steering. For example, "hellaswag".
    
    Returns:
        A dict with any of the keys: method, layer, strength, token_aggregation.

    Examples return value:
        {
            "method": "CAA",
            "layer": 15,
            "strength": 0.5,
            "token_aggregation": "average",
            "token_target": "choice_token"
        }
    """
    try:
        from wisent_guard.core.model_config_manager import ModelConfigManager
        cfg = (ModelConfigManager().load_model_config(model_name)) or {}
    except Exception:
        return {}

    # 1) task-specific steering
    if task_name and "task_specific_steering" in cfg:
        t = cfg["task_specific_steering"].get(task_name)
        if t:
            return t

    # 2) global best steering
    s = cfg.get("steering_optimization") or {}
    if s:
        return {
            "method": s.get("method", s.get("best_method")),
            "layer": s.get("layer", s.get("best_layer")),
            "strength": s.get("strength", s.get("best_strength")),
            "token_aggregation": s.get("token_aggregation", s.get("best_token_aggregation")),
            "token_target": s.get("token_target", s.get("best_token_target")),
        }

    return {}

def autoload_steering(
    model_name: str,
    task_name: str,
    layer: int,
    token_aggregation: str,
    token_target: str,
    steering_method: str,
    steering_strength: float,
    try_auto_load: bool,
    verbose: bool = False,
) -> tuple[int, str, str, float]:
    """
    If try_auto_load=True, override any of the four fields (layer, token_aggregation, steering_method, steering_strength) with values found in steering config.
    Otherwise, return inputs unchanged.

    Arguments:
        model_name: Name of the model to load config for.
        task_name: Name of the task to load config for (e.g., "hellaswag").
        layer: Current layer string (e.g., "15").
        token_aggregation: Current token aggregation strategy (e.g., "average").
        token_target: Current token targeting strategy (e.g., "choice_token").
        steering_method: Current steering method (e.g., "CAA").
        steering_strength: Current steering strength (e.g., 0.5).
        try_auto_load: If True, attempt to load and override values.
        verbose: If True, emits info logs.

    Returns: (layer, token_aggregation, steering_method, steering_strength)
    """
    if not try_auto_load:
        return layer, token_aggregation, steering_method, steering_strength

    opt = _get_steering_bits(model_name, task_name)

    new_layer   = opt["layer"] if opt.get("layer") is not None else layer
    new_tok     = opt["token_aggregation"] if opt.get("token_aggregation") else token_aggregation
    new_method  = opt["method"] if opt.get("method") else steering_method
    new_strength = opt.get("strength") if opt.get("strength") is not None else steering_strength
    new_target = opt.get("token_target") if opt.get("token_target") else token_target

    if verbose:
        changes = []
        if layer != new_layer:                   changes.append(f"layer: {layer} → {new_layer}")
        if token_aggregation != new_tok and opt.get("token_aggregation") is not None:
            changes.append(f"token_aggregation: {token_aggregation} → {new_tok}")
        if steering_method != new_method and opt.get("method") is not None:
            changes.append(f"steering.method: {steering_method} → {new_method}")
        if steering_strength != new_strength and opt.get("strength") is not None:
            changes.append(f"steering.strength: {steering_strength} → {new_strength}")
        if not opt:
            changes.append("(no steering config found; kept inputs)")
        if changes:
            print(f"\nSteering autoload for model={model_name} task={task_name}")
            for c in changes:
                print(f"• {c}")

    return new_layer, new_tok, new_target, new_method, new_strength



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
