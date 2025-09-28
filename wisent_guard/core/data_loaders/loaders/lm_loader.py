from __future__ import annotations
from typing import Any, Iterable, Mapping
import logging

from wisent_guard.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet             
from wisent_guard.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
    lm_build_contrastive_pairs,
) 

log = logging.getLogger(__name__)

def _split_pairs(
    pairs: list[ContrastivePair],
    split_ratio: float,
    seed: int,
    training_limit: int | None,
    testing_limit: int | None,
) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
    if not pairs:
        return [], []
    from numpy.random import default_rng
    idx = list(range(len(pairs)))
    default_rng(seed).shuffle(idx)
    cut = int(len(pairs) * split_ratio)
    train_idx = set(idx[:cut])
    train_pairs: list[ContrastivePair] = []
    test_pairs:  list[ContrastivePair] = []
    for i in idx:
        (train_pairs if i in train_idx else test_pairs).append(pairs[i])
    if training_limit and training_limit > 0:
        train_pairs = train_pairs[:training_limit]
    if testing_limit and testing_limit > 0:
        test_pairs = test_pairs[:testing_limit]
    return train_pairs, test_pairs  # (same behavior as your prior helper). :contentReference[oaicite:11]{index=11}

class LMEvalDataLoader(BaseDataLoader):
    """
    Load contrastive pairs from lm-evaluation-harness task(s) via `model.load_lm_eval_task`,
    split into train/test, and return a canonical LoadDataResult.
    """
    name = "lm_eval"
    description = "Load from lm-eval tasks; supports single or multiple tasks."

    @staticmethod
    def _effective_split(split_ratio: float | None) -> float:
        if split_ratio is None:
            return 0.8
        if not (0.0 <= split_ratio <= 1.0):
            raise ValueError("split_ratio must be in [0.0, 1.0]")
        return float(split_ratio)

    def _load_one_task(
        self,
        *,
        model: Any,
        task_name: str,
        split_ratio: float,
        seed: int,
        limit: int | None,
        training_limit: int | None,
        testing_limit: int | None,
    ) -> LoadDataResult:
        # Your previous lm loader’s shape (normalized here to train_qa_pairs). :contentReference[oaicite:12]{index=12}
        loaded = model.load_lm_eval_task(task_name, limit=limit)  # may return ConfigurableTask or dict[str, ConfigurableTask]
        if isinstance(loaded, dict):
            if len(loaded) != 1:
                keys = ", ".join(sorted(loaded.keys()))
                raise DataLoaderError(
                    f"Task '{task_name}' returned {len(loaded)} subtasks ({keys}). "
                    "Specify an explicit subtask, e.g. 'benchmark/subtask'."
                )
            (subname, task_obj), = loaded.items()
            pairs_task_name = subname
        else:
            task_obj = loaded
            pairs_task_name = task_name

        pairs = lm_build_contrastive_pairs(task_name=pairs_task_name, lm_eval_task=task_obj, limit=limit)  # :contentReference[oaicite:13]{index=13}
        train_pairs, test_pairs = _split_pairs(pairs, split_ratio, seed, training_limit, testing_limit)     # :contentReference[oaicite:14]{index=14}
        if not train_pairs or not test_pairs:
            raise DataLoaderError("One of the splits is empty after splitting.")

        return LoadDataResult(
            train_qa_pairs=ContrastivePairSet("lm_eval_train", train_pairs, task_type=task_name),
            test_qa_pairs=ContrastivePairSet("lm_eval_test",  test_pairs,  task_type=task_name),
            task_type=task_name,
            lm_task_data=task_obj,
        )

    def load(
        self,
        *,
        model: Any,
        tasks: str | Iterable[str],
        split_ratio: float | None = None,
        seed: int = 42,
        limit: int | None = None,
        training_limit: int | None = None,
        testing_limit: int | None = None,
        **_: Any,
    ) -> LoadDataResult:
        if model is None:
            raise DataLoaderError("'model' is required for lm_eval loader.")
        split = self._effective_split(split_ratio)

        # Single task path
        if isinstance(tasks, (str, bytes)):
            return self._load_one_task(
                model=model,
                task_name=str(tasks),
                split_ratio=split,
                seed=seed,
                limit=limit,
                training_limit=training_limit,
                testing_limit=testing_limit,
            )

        # Multi-task: union the splits
        tasks_list = [str(t) for t in tasks]
        all_train, all_test = [], []
        task_map: dict[str, Any] = {}
        for tname in tasks_list:
            r = self._load_one_task(
                model=model,
                task_name=tname,
                split_ratio=split,
                seed=seed,
                limit=limit,
                training_limit=training_limit,
                testing_limit=testing_limit,
            )
            all_train.extend(r["train_qa_pairs"].pairs)
            all_test.extend(r["test_qa_pairs"].pairs)
            task_map[tname] = r["lm_task_data"]

        task_type = "+".join(tasks_list)
        return LoadDataResult(
            train_qa_pairs=ContrastivePairSet("lm_eval_train_multi", all_train, task_type=task_type),
            test_qa_pairs=ContrastivePairSet("lm_eval_test_multi",  all_test,  task_type=task_type),
            task_type=task_type,
            lm_task_data=task_map,
        )
