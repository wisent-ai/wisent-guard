from __future__ import annotations

import time
from typing import TYPE_CHECKING

from wisent_guard.cli_bricks.cli_logger import setup_logger, bind

from wisent_guard.cli_bricks.data_loaders.data_loader_types import LoadDataResult

from wisent_guard.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet

if TYPE_CHECKING:
    from wisent_guard.core.model import Model
    from lm_eval.api.task import ConfigurableTask

__all__ = ["load_train_test_lm_eval_format"]

_LOG = setup_logger(__name__)

def _split_pairs(
    pairs: list[ContrastivePair],
    split_ratio: float = 0.8,
    seed: int = 42,
    training_limit: int | None = None,
    testing_limit: int | None = None,
) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
    """Shuffle/split pairs into train/test.
    
    arguments:
        pairs:
            List of ContrastivePair objects to split.
        split_ratio:
            Fraction of pairs to allocate to training (between 0.0 and 1.0). Default is 0.8.
        seed:
            Random seed for shuffling. Default is 42.
        training_limit:
            Optional maximum number of training pairs after splitting.
        testing_limit:
            Optional maximum number of testing pairs after splitting.
    
    returns:
        A tuple of (train_pairs, test_pairs).
    """
    train_pairs: list[ContrastivePair] = []
    test_pairs: list[ContrastivePair] = []

    if not pairs:
        return train_pairs, test_pairs

    from numpy.random import default_rng

    rng = default_rng(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)

    split_point = int(len(pairs) * split_ratio)
    train_idx = set(idx[:split_point])

    for i in idx:
        (train_pairs if i in train_idx else test_pairs).append(pairs[i])

    if training_limit is not None and training_limit > 0:
        train_pairs = train_pairs[:training_limit]
    if testing_limit is not None and testing_limit > 0:
        test_pairs = test_pairs[:testing_limit]

    return train_pairs, test_pairs


def load_train_test_lm_eval_format(
    model: Model,
    task_name: str,
    split_ratio: float = 0.8,
    seed: int = 42,
    limit: int | None = None,
    training_limit: int | None = None,
    testing_limit: int | None = None,
) -> LoadDataResult:
    """
    Load ONE lm-eval task (must resolve to a single subtask), build contrastive pairs,
    split into train/test, and return results. If multiple subtasks are returned, raise.

    arguments:
        model:
            A Model instance with 'load_lm_eval_task' method.
        task_name:
            Name of the lm-eval benchmark/task (e.g., "winogrande").
            If the task is a group, it must resolve to exactly one subtask.
        split_ratio:
            Fraction of pairs to allocate to training (between 0.0 and 1.0). Default is 0.8.
        seed:
            Random seed for shuffling before splitting. Default is 42.
        limit:
            Optional upper bound on the number of pairs to extract from the task.
            Values <= 0 are treated as "no limit".
        training_limit:
            Optional maximum number of training pairs after splitting.
        testing_limit:
            Optional maximum number of testing pairs after splitting.
    
    returns:
        A LoadDataResult with:
                qa_pairs:
                    ContrastivePairSet for training.
                test_qa_pairs:
                    ContrastivePairSet for testing.
                lm_task_data:
                    The original ConfigurableTask object.
                task_type:
                    The task name (same as input 'task_name').

    raises:
        ValueError:
            If the task_name is invalid or resolves to multiple subtasks.
        RuntimeError:
            If no pairs are extracted or if either split is empty.
    """
    logger = bind(_LOG, subtask="load_single_task", root=task_name)
    t0 = time.perf_counter()
    logger.info("Loading task '%s'...", task_name)

    try:
        loaded: ConfigurableTask | dict[str, ConfigurableTask] = model.load_lm_eval_task(
            task_name, limit=limit
        )
    except Exception as e:
        logger.error("Failed to load task", extra={"task_name": task_name}, exc_info=e)
        raise

    if isinstance(loaded, dict):
        n = len(loaded)
        if n != 1:
            keys = ", ".join(sorted(loaded.keys()))
            msg = (
                f"Task '{task_name}' returned {n} subtasks ({keys}). "
                "This loader only supports a single subtask. "
                "Please specify an explicit subtask (e.g., 'benchmark/subtask') "
                "or adjust the loader to support groups."
            )
            logger.error(msg)
            raise ValueError(msg)
        (only_subkey, task_obj), = loaded.items()
        pairs_task_name = only_subkey
    else:
        task_obj = loaded
        pairs_task_name = task_name

    try:
        pairs = lm_build_contrastive_pairs( 
            task_name=pairs_task_name,
            lm_eval_task=task_obj,
            limit=limit,
        )
    except Exception as e:
        logger.error("Failed building pairs for task", extra={"task_name": pairs_task_name}, exc_info=e)
        raise

    train_qas, test_qas = _split_pairs(
        pairs,
        split_ratio=split_ratio,
        seed=seed,
        training_limit=training_limit,
        testing_limit=testing_limit,
    )

    if not train_qas or not test_qas:
        logger.error("One of the splits is empty after splitting.", extra={"train_count": len(train_qas), "test_count": len(test_qas)})
        raise RuntimeError("CRITICAL: one of the splits is empty after splitting.")
    if len(train_qas) + len(test_qas) == 0:
        logger.error("No QA pairs extracted from the task.", extra={"task_name": pairs_task_name})
        raise RuntimeError("CRITICAL: extracted 0 QA pairs — check extractor support.")

    elapsed = time.perf_counter() - t0
    logger.info(
        "Loaded and split task.",
        extra={
            "total_pairs": len(pairs),
            "train_count": len(train_qas),
            "test_count": len(test_qas),
            "split_ratio": split_ratio,
            "seed": seed,
            "elapsed_sec": round(elapsed, 2),
        },
    )

    return LoadDataResult(
        qa_pairs=ContrastivePairSet("lm_eval", train_qas, task_name),
        test_qa_pairs=ContrastivePairSet("lm_eval", test_qas, task_name),
        task_type=task_name,
        lm_task_data=task_obj,  
    )


# if __name__ == "__main__":
#     from wisent_guard.core.model import Model
#     from wisent_guard.cli_bricks.data_loaders.lm_data_loader import load_train_test_lm_eval_format
#     MODLE_NAME = "your-model-name-here"
#     model = Model(name=MODLE_NAME)
#     result = load_train_test_lm_eval_format(
#         model=model,
#         task_name="winogrande",
#         split_ratio=0.8,
#         seed=42,
#         limit=10,
#         training_limit=None,
#         testing_limit=None,
#     )
#     print(result)

    