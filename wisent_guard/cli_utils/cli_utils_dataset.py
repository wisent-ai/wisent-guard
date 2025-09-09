"""
Utilities for validating and inspecting benchmark task names used by the CLI.
"""

from __future__ import annotations

from typing import Any, Optional

from wisent_guard.cli_utils.cli_datasets import (
    AVAILABLE_BENCHMARKS,
    CORE_BENCHMARKS,
    UNAVAILABLE_BENCHMARKS,
)

def validate_task_name(name: str) -> bool:
    """Check whether a task name is available.

    Args:
        name: Task name to validate.

    Returns:
        True if the task exists in AVAILABLE_BENCHMARKS, otherwise False.
    """
    return name in AVAILABLE_BENCHMARKS


def suggest_similar_tasks(invalid: str, max_suggestions: int = 5) -> list[str]:
    """Suggest task names similar to an invalid input.

    Uses difflib.get_close_matches for lightweight fuzzy matching.

    Args:
        invalid: The invalid task string entered by the user.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        A list of similar valid task names (may be empty).
    """
    from difflib import get_close_matches

    valid_names = list(AVAILABLE_BENCHMARKS.keys())
    if not invalid:
        return []
    return get_close_matches(invalid, valid_names, n=max_suggestions, cutoff=0.6)


def validate_and_report_task_name(
    task_name: str,
    verbose: bool,
    from_csv: bool,
    from_json: bool,
    cross_benchmark_mode: bool,
    from_synthetic: bool,
) -> Optional[dict[str, Any]]:
    """Validate a benchmark task name and optionally build an error payload.

    Skips validation if any special data-source modes are active (CSV / JSON /
    cross-benchmark / synthetic). When the task is invalid a structured error
    dictionary is returned so the caller can short-circuit.

    Args:
        task_name: Candidate benchmark task name.
        verbose: If True, print human-readable diagnostics.
        from_csv: Task loaded from a CSV file (skip validation).
        from_json: Task loaded from a JSON file (skip validation).
        cross_benchmark_mode: Multi-benchmark evaluation mode (skip validation).
        from_synthetic: Synthetic contrastive pairs mode (skip validation).

    Returns:
        None if task is valid or validation skipped; otherwise a dict with keys:
            task_name: The provided invalid name.
            error: Human readable error message.
            error_type: Static string 'invalid_task'.
            valid_tasks_count: Count of available benchmarks.
            total_benchmarks: Total core benchmarks (including unavailable).
            excluded_count: Number of explicitly unavailable benchmarks.
            suggestions: List of close match task names.
            help: Short guidance string.
    """

    if from_csv or from_json or cross_benchmark_mode or from_synthetic:
        return None

    try:
        if validate_task_name(task_name):  
            return None
    except Exception as exc:  
        if verbose:
            print(f"Task validation internal error: {exc}")

    error_msg = f"Invalid task name: '{task_name}'"
    suggestions = suggest_similar_tasks(task_name)

    if verbose:
        print(error_msg)
        print(
            f"Task '{task_name}' is not in the available benchmark list. "
            f"{len(AVAILABLE_BENCHMARKS)} available out of {len(CORE_BENCHMARKS)} total."
        )
        if task_name in UNAVAILABLE_BENCHMARKS:
            print("This benchmark is marked unavailable/problematic.")
        if suggestions:
            print("Suggestions:")
            for idx, suggestion in enumerate(suggestions, 1):
                cfg = AVAILABLE_BENCHMARKS[suggestion]
                prio = cfg.get("priority", "unknown")
                tags = ", ".join(cfg.get("tags", []))
                print(f"  {idx}. {suggestion} (priority={prio}) tags=[{tags}]")
        print("Run 'wisent-guard tasks --list-tasks' to list all valid tasks.")
        print("Run 'wisent-guard tasks --task-info <task_name>' for details.")

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


def _group_by_priority() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Group available benchmarks by priority.

    Returns:
        A mapping: priority -> list of (task_name, config) tuples.
    """
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "high": [],
        "medium": [],
        "low": [],
        "unknown": [],
    }
    for name, config in AVAILABLE_BENCHMARKS.items():
        priority = config.get("priority", "unknown")
        grouped.setdefault(priority, grouped["unknown"]).append((name, config))
    return grouped


def print_valid_tasks_by_category() -> None:
    """Print valid benchmark tasks grouped by priority.

    Returns:
        None. Writes a formatted list to stdout.
    """
    excluded_count = len(UNAVAILABLE_BENCHMARKS)
    print(
        f"\nVALID TASKS ({len(AVAILABLE_BENCHMARKS)} available out of {len(CORE_BENCHMARKS)} total)"
    )
    if excluded_count:
        print(f"  {excluded_count} benchmarks excluded (unavailable/problematic)")
    print("=" * 60)

    grouped = _group_by_priority()
    for priority in ["high", "medium", "low", "unknown"]:
        entries = grouped.get(priority) or []
        if not entries:
            continue
        print(f"\n{priority.upper()} PRIORITY ({len(entries)} tasks):")
        for name, cfg in sorted(entries):
            tags = ", ".join(cfg.get("tags", []))
            print(f"  - {name:<22} {tags}")

    print("\nUsage: wisent-guard tasks --task-name <task_name>")
    print("Example: wisent-guard tasks --task-name truthfulqa_mc1")

    if excluded_count:
        print(f"\nEXCLUDED BENCHMARKS ({excluded_count}):")
        excluded_list = sorted(UNAVAILABLE_BENCHMARKS)
        for i in range(0, len(excluded_list), 4):
            segment = ", ".join(excluded_list[i : i + 4])
            print(f"  - {segment}")


def print_task_info(task_name: str) -> None:
    """Print detailed configuration for a single task.

    Args:
        task_name: Name of the task whose info should be displayed.

    Returns:
        None. Writes formatted details to stdout.
    """
    if task_name not in AVAILABLE_BENCHMARKS:
        if task_name in UNAVAILABLE_BENCHMARKS:
            print(
                f"Task '{task_name}' is marked unavailable (excluded from {len(AVAILABLE_BENCHMARKS)} available)."
            )
        else:
            print(f"Task '{task_name}' not found in available benchmarks.")
        return

    cfg = AVAILABLE_BENCHMARKS[task_name]
    print(f"\nTASK INFO: {task_name}")
    print("=" * 50)
    print(f"Task ID: {cfg.get('task', task_name)}")
    print(f"Tags: {', '.join(cfg.get('tags', [])) or 'none'}")
    prio = cfg.get("priority", "unknown")
    print(f"Priority: {prio}")
    if prio == "high":
        print("  Fast loading (<13.5s) - optimal for agentic use")
    elif prio == "medium":
        print("  Moderate loading (13.5-60s) - acceptable")
    elif prio == "low":
        print("  Slow loading (>60s) - deprioritized")
    if "trust_remote_code" in cfg:
        print(f"Requires trust_remote_code: {cfg['trust_remote_code']}")

__all__ = [
    "validate_and_report_task_name",
    "validate_task_name",
    "suggest_similar_tasks",
    "print_valid_tasks_by_category",
    "print_task_info",
]