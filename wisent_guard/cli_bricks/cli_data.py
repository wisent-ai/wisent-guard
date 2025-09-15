from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any, TypedDict, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

from wisent_guard.core import ContrastivePairSet
from wisent_guard.core.model import Model
from lm_eval.api.task import ConfigurableTask

from wisent_guard.cli_bricks.cli_logger import setup_logger, bind  

__all__ = [
    "LoadDataResult",
    "load_train_test_data",
]

class LoadDataResult(TypedDict):
    qa_pairs: list[dict[str, Any]]
    test_qa_pairs_source: list[dict[str, Any]]
    task_data: Any  # For groups: {subtask_key: task_obj}; for single: the task object


@dataclass(frozen=True)
class LoadContext:
    task_names: tuple[str, ...]
    shots: int
    limit: int | None
    split_ratio: float
    seed: int


def load_train_test_data(
    model: Model,
    task_names: str | Sequence[str],   
    shots: int,
    split_ratio: float,
    limit: int | None,
    training_limit: int | None,
    testing_limit: int | None,
    seed: int,
    verbose: bool,
    *,
    logger: logging.Logger | None = None,
    json_logs: bool = False,
) -> LoadDataResult:
    """
    Load task data for one or many lm-eval tasks (or groups) and return aggregated
    training/testing QA pairs. For multiple root tasks or groups, results are combined.
    Each QA dict gets 'source_task' (the root name) and 'source_subtask' (the lm-eval subtask key).

    Notes:
      - Reuses lm-eval task objects created by evaluator.get_task_dict(..).
      - Per-root expansion runs in parallel with a 60s timeout each.
      - Subtask name collisions across roots are resolved by suffixing '#N'.
    """
    # Normalize input
    if isinstance(task_names, str):
        roots: list[str] = [task_names]
    else:
        roots = [t for t in task_names if t]  # drop empties

    if not roots:
        raise ValueError("No task names provided.")

    if logger is None:
        base_logger = setup_logger(
            name="wisent_guard.cli_bricks.load_data",
            level=logging.DEBUG if verbose else logging.INFO,
            json_logs=json_logs,
        )
    else:
        base_logger = logger

    ctx = LoadContext(tuple(roots), shots, limit, split_ratio, seed)
    log = bind(
        base_logger,
        task_name=f"(tasks={','.join(ctx.task_names)}, shots={ctx.shots}, limit={ctx.limit}, split={ctx.split_ratio}, seed={ctx.seed})",
        subtask="",
    )

    t0 = time.perf_counter()
    log.info("Loading task data…")

    # --- Step 1: Expand each root task into subtasks (build tasks once) ---
    def _get_tasks_for(root: str) -> tuple[str, dict[str, ConfigurableTask]]:
        from lm_eval import evaluator
        try:
            tasks = evaluator.get_task_dict([root])
        except TypeError as e:
            raise RuntimeError(f"Failed to expand root task '{root}': {e}") from e
        return root, tasks

    tasks_by_name: dict[str, ConfigurableTask] = {}
    # Map our (possibly suffixed) subtask key -> (root_name, original_subtask_key)
    origin_map: dict[str, tuple[str, str]] = {}

    def _insert_with_collision(key: str, obj: ConfigurableTask, root: str) -> None:
        orig_key = key
        if key in tasks_by_name:
            # resolve collision across different roots/groups
            i = 2
            while f"{orig_key}#{i}" in tasks_by_name:
                i += 1
            key = f"{orig_key}#{i}"
            log.warning("Name collision on subtask '%s'; stored as '%s' (root=%s).", orig_key, key, root)
        tasks_by_name[key] = obj
        origin_map[key] = (root, orig_key)

    # Expand in parallel with per-root timeouts and robust fallbacks
    max_workers = min(len(roots), 4) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_get_tasks_for, r): r for r in roots}
        for fut in as_completed(futures):
            root = futures[fut]
            try:
                r0 = time.perf_counter()
                root_name, task_dict = fut.result(timeout=60)
                elapsed = time.perf_counter() - r0
                log.info("Expanded root '%s' into %d subtask(s) in %.3fs.", root_name, len(task_dict), elapsed)
                for subkey, task_obj in task_dict.items():
                    _insert_with_collision(subkey, task_obj, root_name)
            except (FuturesTimeoutError, Exception) as e:
                # Fallback: at least load the root directly via the model wrapper
                log.error("Expansion failed for root '%s'; falling back to single-task load.", root, exc_info=e)
                try:
                    single_task = model.load_lm_eval_task(root, shots=shots, limit=limit)
                    _insert_with_collision(root, single_task, root)
                    log.info("Fallback for root '%s' produced 1 subtask.", root)
                except Exception as e2:
                    log.error("Fallback load also failed for root '%s'. Skipping.", root, exc_info=e2)

    expanded_keys = list(tasks_by_name.keys())
    if not expanded_keys:
        log.error("No tasks expanded or loaded; aborting.")
        raise RuntimeError("CRITICAL: no tasks could be expanded or loaded — check task names and lm-eval install.")

    # Optional: avoid printing extremely long lists verbosely
    preview = expanded_keys if len(expanded_keys) <= 20 else expanded_keys[:20] + ["…"]
    log.info("Total expanded subtasks: %d: %s", len(expanded_keys), preview)

    # --- Step 2: Split docs & extract QA using existing task objects ---
    all_train_qas: list[dict[str, Any]] = []
    all_test_qas: list[dict[str, Any]] = []
    task_objects: dict[str, ConfigurableTask] = {}

    def _split_docs(task_obj: ConfigurableTask) -> tuple[list[Any], list[Any]]:
        train_docs, test_docs = model.split_task_data(
            task_obj, split_ratio=split_ratio, random_seed=seed
        )
        if training_limit is not None:
            train_docs = list(train_docs)[:training_limit]
        if testing_limit is not None:
            test_docs = list(test_docs)[:testing_limit]
        return train_docs, test_docs

    for subtask_key, task_obj in tasks_by_name.items():
        root_name, orig_subkey = origin_map[subtask_key]
        log_sub = bind(log, subtask=f"(subtask={subtask_key})")
        sub_t0 = time.perf_counter()
        try:
            task_objects[subtask_key] = task_obj

            train_docs, test_docs = _split_docs(task_obj)
            log_sub.debug("Split docs: %d train / %d test", len(train_docs), len(test_docs))

            train_qas = ContrastivePairSet.extract_qa_pairs_from_task_docs(
                orig_subkey, task_obj, train_docs
            )
            test_qas = ContrastivePairSet.extract_qa_pairs_from_task_docs(
                orig_subkey, task_obj, test_docs
            )

            multi = len(tasks_by_name) > 1 or len(roots) > 1
            if multi:
                for q in train_qas:
                    q.setdefault("source_task", root_name)
                    q.setdefault("source_subtask", orig_subkey)
                for q in test_qas:
                    q.setdefault("source_task", root_name)
                    q.setdefault("source_subtask", orig_subkey)

            all_train_qas.extend(train_qas)
            all_test_qas.extend(test_qas)

            log_sub.info(
                "Processed → %d train-QA / %d test-QA (docs: %d/%d) in %.3fs",
                len(train_qas), len(test_qas), len(train_docs), len(test_docs),
                time.perf_counter() - sub_t0,
            )
        except Exception as e:
            log_sub.error("Skipping subtask due to error.", exc_info=e)
            continue

    # --- Step 3: Validate and finish ---
    if not all_train_qas or not all_test_qas:
        log.error(
            "Extracted 0 QA pairs — check extractor support. train_len=%d test_len=%d",
            len(all_train_qas), len(all_test_qas)
        )
        raise RuntimeError("CRITICAL: extracted 0 QA pairs — check extractor support.")

    task_data: Any = (
        next(iter(task_objects.values())) if len(task_objects) == 1 else task_objects
    )

    total_train = len(all_train_qas)
    total_test = len(all_test_qas)
    log.info(
        "Aggregated: %d train-QA / %d test-QA across %d subtask(s) from %d root task(s) in %.3fs",
        total_train, total_test, len(tasks_by_name), len(roots), time.perf_counter() - t0
    )

    return LoadDataResult(
        qa_pairs=all_train_qas,
        test_qa_pairs_source=all_test_qas,
        task_data=task_data,
    )