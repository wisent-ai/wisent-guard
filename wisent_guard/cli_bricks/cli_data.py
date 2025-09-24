from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, TypedDict, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

from wisent_guard.core import ContrastivePairSet
from wisent_guard.cli_bricks.cli_logger import setup_logger, bind
from wisent_guard.core.contrastive_pairs.core.serialization import load_contrastive_pair_set

if TYPE_CHECKING: 
    from wisent_guard.core.model import Model
    from lm_eval.api.task import ConfigurableTask

__all__ = ["LoadDataResult", "load_train_test_data"]

_LOG = setup_logger(__name__)

_LOAD_TIMEOUT_S = 60
_MAX_WORKERS_CAP = 2

class LoadDataResult(TypedDict):
    train_qa_pairs: ContrastivePairSet
    test_qa_pairs: ContrastivePairSet
    task_type: str  
    lm_task_data: dict[str, ConfigurableTask] | ConfigurableTask | None


@dataclass(frozen=True)
class _LoadContext:
    task_names: tuple[str, ...]
    limit: int | None
    split_ratio: float
    seed: int
    training_limit: int | None
    testing_limit: int | None

def _normalize_roots(task_names: str | Sequence[str]) -> list[str]:
    """
    Normalize input task names into a list of root task names.
    Arguments:
      - task_names: Either a single task name string or a sequence of task name strings. For example, "winogrande" or ["winogrande", "hellaswag"].
    Returns:
        - A list of root task names.
    Raises:
        - ValueError if no valid task names are provided.
    """
    if isinstance(task_names, str):
        roots = [task_names]
    else:
        roots = [t for t in task_names if t]
    if not roots:
        raise ValueError("No task names provided.")
    return roots

def _insert_with_collision(
    key: str,
    obj: ConfigurableTask,
    root: str,
    tasks_by_name: dict[str, ConfigurableTask],
    origin_map: dict[str, tuple[str, str]],
) -> None:
    """
    Insert a task object into tasks_by_name, handling name collisions by appending #N suffixes.

    Arguments:
      - key: Proposed key for the task object.
      - obj: The ConfigurableTask object to insert.
      - root: The root task name from which this object was loaded.
      - tasks_by_name: The dict to insert into.
      - origin_map: Maps inserted keys to (root, original_key) tuples.
    """
    logger = bind(_LOG, subtask="task_collision")
    orig_key = key
    if key in tasks_by_name:
        i = 2
        while f"{orig_key}#{i}" in tasks_by_name:
            i += 1
        key = f"{orig_key}#{i}"
        logger.warning("Name collision on subtask '%s'; stored as '%s' (root=%s).", orig_key, key, root)
    tasks_by_name[key] = obj
    origin_map[key] = (root, orig_key)


def _load_root(
    model: Model,
    root: str,
    limit: int | None,
) -> tuple[str, dict[str, ConfigurableTask] | ConfigurableTask]:
    """
    Uses model.load_lm_eval_task exclusively.
    May return a single ConfigurableTask or a dict[str, ConfigurableTask] for grouped benchmarks.

    Arguments:
      - model: A wisent_guard-compatible model.
      - root: The root task name to load. For example, "winogrande".
      - limit: Optional limit on the number of examples to load.

    Returns:
        - A tuple of (root task name, loaded task object or dict of subtasks).
    """
    task_or_group = model.load_lm_eval_task(root, limit=limit)
    return root, task_or_group


def _flatten_loaded_tasks(
    root: str,
    loaded: Any,
    tasks_by_name: dict[str, "ConfigurableTask"],
    origin_map: dict[str, tuple[str, str]],
) -> int:
    """
    Insert whatever model returned (single task or dict of subtasks) into our registries.
    Returns number of inserted task objects.

    Arguments:
      - root: The root task name from which the loaded object(s) came.
      - loaded: Either a single ConfigurableTask or a dict of {subkey: ConfigurableTask}.
      - tasks_by_name: The dict to insert into.
      - origin_map: Maps inserted keys to (root, original_key) tuples.
    Returns:
        - Number of inserted task objects.
    """
    logger = bind(_LOG, subtask="flatten_tasks", root=root)
    logger.debug("Flattening loaded tasks of type %s", type(loaded).__name__)
    if isinstance(loaded, dict): 
        inserted = 0
        for subkey, task_obj in loaded.items():
            _insert_with_collision(subkey, task_obj, root, tasks_by_name, origin_map)
            inserted += 1
        return inserted
    else:  
        _insert_with_collision(root, loaded, root, tasks_by_name, origin_map)
        return 1

def _split_docs_for_task(
    model: Model,
    task_obj: ConfigurableTask,
    split_ratio: float,
    seed: int,
    training_limit: int | None,
    testing_limit: int | None,
) -> tuple[list[Any], list[Any]]:
    """
    Split task documents into train and test sets.

    Arguments:
      - model: A wisent_guard-compatible model.
      - task_obj: The ConfigurableTask object whose documents to split.
      - split_ratio: Fraction of data to use for training (between 0 and 1).
      - seed: Random seed for reproducible splits.
      - training_limit: Optional limit on the number of training documents.
      - testing_limit: Optional limit on the number of testing documents.

    Returns:
        - A tuple of (train_docs, test_docs).
    """
    train_docs, test_docs = model.split_task_data(task_obj, split_ratio=split_ratio, random_seed=seed)
    if training_limit is not None:
        train_docs = list(train_docs)[:training_limit]
    if testing_limit is not None:
        test_docs = list(test_docs)[:testing_limit]
    return train_docs, test_docs


def _tag_source(
    qas: list[dict[str, Any]],
    root_name: str,
    subkey: str,
    enable: bool,
) -> None:
    """
    Tag each QA pair with its source task and subtask.

    Arguments:
      - qas: List of QA dicts to tag.
      - root_name: The root task name. For example, "winogrande".
      - subkey: The subtask key. For example, "winogrande_100".
      - enable: If True, add the tags; if False, do nothing.

    """
    if not enable:
        return
    for q in qas:
        q.setdefault("source_task", root_name)
        q.setdefault("source_subtask", subkey)


def load_train_test_lm_eval_format(
    model: Model,
    task_names: str | Sequence[str],
    split_ratio: float,
    limit: int | None,
    training_limit: int | None,
    testing_limit: int | None,
    seed: int,
) -> LoadDataResult:
    """
    Load task data for one or many lm-eval tasks (or groups) via "model.load_lm_eval_task"
    (threaded), then split and extract QA pairs.

    Notes:
      - Per-root loading runs in parallel with a timeout per root.
      - If a loaded root returns a dict of subtasks, those are flattened and collision-safe.
      - Each QA dict gets 'source_task' and 'source_subtask' when multiple tasks are combined.

    Arguments:
        - model: A wisent_guard-compatible model.
        - task_names: Either a single task name string or a sequence of task name strings. For example, "winogrande" or ["winogrande", "hellaswag"].
        - split_ratio: Fraction of data to use for training (between 0 and 1).
        - limit: Optional limit on the number of examples to load per root task.
        - training_limit: Optional limit on the number of training documents after splitting.
        - testing_limit: Optional limit on the number of testing documents after splitting.
        - seed: Random seed for reproducible splits.
    """
    logger = bind(_LOG, subtask="load_data")
    roots = _normalize_roots(task_names)
    ctx = _LoadContext(
        task_names=tuple(roots),
        limit=limit,
        split_ratio=split_ratio,
        seed=seed,
        training_limit=training_limit,
        testing_limit=testing_limit,
    )

    t0 = time.perf_counter()

    logger.info(
        "Loading data for %d root task(s): %s",
        len(roots),
        roots if len(roots) <= 10 else roots[:10] + ["…"],
    )

    # --- Step 1: Load each root via the model (no evaluator) -------------------------------
    tasks_by_name: dict[str, ConfigurableTask] = {}
    origin_map: dict[str, tuple[str, str]] = {}

    max_workers = min(len(roots), _MAX_WORKERS_CAP) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_root, model, r, limit): r for r in roots}
        for fut in as_completed(futures):
            root = futures[fut]
            try:
                r0 = time.perf_counter()
                root_name, loaded = fut.result(timeout=_LOAD_TIMEOUT_S)
                inserted = _flatten_loaded_tasks(root_name, loaded, tasks_by_name, origin_map)
                elapsed = time.perf_counter() - r0
                logger.info("Loaded root '%s' → %d subtask(s) in %.3fs.", root_name, inserted, elapsed)
            except FuturesTimeoutError as e:
                logger.error("Timeout while loading root '%s' after %ss; skipping.", root, _LOAD_TIMEOUT_S, exc_info=e)
            except Exception as e:
                logger.error("Failed to load root '%s'; skipping.", root, exc_info=e)

    if not tasks_by_name:
        logger.error("No tasks loaded; aborting.")
        raise RuntimeError("CRITICAL: no tasks could be loaded — check task names and lm-eval installation.")

    expanded_keys = list(tasks_by_name.keys())
    preview = expanded_keys if len(expanded_keys) <= 20 else expanded_keys[:20] + ["…"]
    logger.info("Total loaded subtasks: %d: %s", len(expanded_keys), preview)

    # --- Step 2: Split docs & extract QA ---------------------------------------------------
    all_train_qas: list[dict[str, Any]] = []
    all_test_qas: list[dict[str, Any]] = []
    task_objects: dict[str, ConfigurableTask] = {}

    multi = len(tasks_by_name) > 1 or len(roots) > 1

    for subtask_key, task_obj in tasks_by_name.items():
        root_name, orig_subkey = origin_map[subtask_key]
        log_sub = bind(_LOG, subtask=f"(subtask={subtask_key})")
        sub_t0 = time.perf_counter()
        try:
            task_objects[subtask_key] = task_obj

            train_docs, test_docs = _split_docs_for_task(
                model,
                task_obj,
                split_ratio=ctx.split_ratio,
                seed=ctx.seed,
                training_limit=ctx.training_limit,
                testing_limit=ctx.testing_limit,
            )
            log_sub.debug("Split docs: %d train / %d test", len(train_docs), len(test_docs))

            train_qas = ContrastivePairSet.extract_qa_pairs_from_task_docs(orig_subkey, task_obj, train_docs)
            test_qas = ContrastivePairSet.extract_qa_pairs_from_task_docs(orig_subkey, task_obj, test_docs)

            _tag_source(train_qas, root_name, orig_subkey, enable=multi)
            _tag_source(test_qas, root_name, orig_subkey, enable=multi)

            all_train_qas.extend(train_qas)
            all_test_qas.extend(test_qas)

            log_sub.info(
                "Processed → %d train-QA / %d test-QA (docs: %d/%d) in %.3fs",
                len(train_qas), len(test_qas), len(train_docs), len(test_docs),
                time.perf_counter() - sub_t0,
            )
        except Exception as e:
            log_sub.error("Skipping subtask due to error.", exc_info=e)

    # --- Step 3: Validate & finish ---------------------------------------------------------
    if not all_train_qas or not all_test_qas:
        logger.error(
            "Extracted 0 QA pairs — check extractor support. train_len=%d test_len=%d",
            len(all_train_qas), len(all_test_qas),
        )
        raise RuntimeError("CRITICAL: extracted 0 QA pairs — check extractor support.")

    task_data: Any = next(iter(task_objects.values())) if len(task_objects) == 1 else task_objects

    total_train = len(all_train_qas)
    total_test = len(all_test_qas)
    logger.info(
        "Aggregated: %d train-QA / %d test-QA across %d subtask(s) from %d root task(s) in %.3fs",
        total_train, total_test, len(tasks_by_name), len(roots), time.perf_counter() - t0,
    )
    # TODO change to ContrastivePairSet
    return LoadDataResult(
        qa_pairs=all_train_qas,
        test_qa_pairs_source=all_test_qas,
        task_data=task_data,
    )

def load_train_test_custom_format(
    path_to_custom_data: str,
    split_ratio: float | None,
    seed: int | None,
    training_limit: int | None,
    testing_limit: int | None,
) -> LoadDataResult:
    """
    Load custom-formatted contrastive-pair data from a JSONL file, split into
    train/test, and optionally cap each split.

    arguments:
        path_to_custom_data: Path to the JSONL file containing custom-formatted data.
        split_ratio: Fraction of examples to allocate to training. If None, defaults to 0.8.
        seed: Random seed for reproducible shuffling before splitting.
        training_limit: Optional maximum number of training pairs after splitting.
        testing_limit: Optional maximum number of testing pairs after splitting.

        Dataset format:
        {
            "name": "name of the set",
            "task_type": "task type string",
            "pairs": [
                {
                "prompt": "The input prompt",
                "positive_response": {
            ...     "model_response": "Yes, the sky is blue.",
            ...     "layers_activations": {"blocks.0.mlp": torch.randn(2, 4)},
            ...     "label": "harmless"
            ...     },
            ...     "negative_response": {
            ...         "model_response": "No, the sky is green.",
            ...         "layers_activations": {"blocks.0.mlp": torch.randn(2, 4)},
            ...         "label": "toxic"
         ...     },
         ...     "label": "color_question",
         ...     "trait_description": "hallucinatory"
         ... }
        ]
        }

    Returns:
        A mapping with the training/test 'ContrastivePairSet's' and the 'task_type'.
        lm_task_data is always None because this is custom data.

    Raises:
        RuntimeError: If no contrastive pairs are found in the input file.
        ValueError: If 'split_ratio' is not in [0.0, 1.0] or if any limit is negative.
    """
    logger = bind(_LOG, subtask="load_custom_data")

    data: ContrastivePairSet = load_contrastive_pair_set(path_to_custom_data)
    logger.info("Loaded custom data.", extra={"data_summary": data.statistics()})

    if not data.pairs:
        logger.error("No contrastive pairs found in the provided data; aborting.")
        raise RuntimeError(
            "CRITICAL: no contrastive pairs found — check the input data file."
        )

    if split_ratio is None:
        split_ratio = 0.8
    elif not (0.0 <= split_ratio <= 1.0):
        raise ValueError("`split_ratio` must be between 0.0 and 1.0 (inclusive).")

    if training_limit is not None and training_limit < 0:
        raise ValueError("'training_limit' must be non-negative.")
    if testing_limit is not None and testing_limit < 0:
        raise ValueError("'testing_limit' must be non-negative.")

    n_pairs = len(data.pairs)
    indices = list(range(n_pairs))

    if seed is not None:
        try:
            from numpy.random import default_rng  # type: ignore
        except Exception:
            import random
            rnd = random.Random(seed)
            rnd.shuffle(indices)
        else:
            indices = default_rng(seed).permutation(n_pairs).tolist()

    split_at = int(n_pairs * split_ratio)
    train_idx = indices[:split_at]
    test_idx = indices[split_at:]

    train_pairs = [data.pairs[i] for i in train_idx]
    test_pairs = [data.pairs[i] for i in test_idx]

    if training_limit is not None:
        train_pairs = train_pairs[:training_limit]
    if testing_limit is not None:
        test_pairs = test_pairs[:testing_limit]

    name = data.name
    task_type = data.task_type

    train_set = ContrastivePairSet(
        name=f"{name}_train", pairs=train_pairs, task_type=task_type
    )
    test_set = ContrastivePairSet(
        name=f"{name}_test", pairs=test_pairs, task_type=task_type
    )

    logger.info(
        "Finished split.",
        extra={
            "total": n_pairs,
            "split_ratio": split_ratio,
            "train_count": len(train_pairs),
            "test_count": len(test_pairs),
            "seed": seed,
        },
    )

    return LoadDataResult(
        train_qa_pairs=train_set,
        test_qa_pairs=test_set,
        task_type=task_type,
        lm_task_data=None,
    )