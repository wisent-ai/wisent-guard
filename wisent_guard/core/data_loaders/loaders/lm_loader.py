from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging

from wisent_guard.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet
from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager as LMTaskManager
from wisent_guard.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
    lm_build_contrastive_pairs,
)

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask

__all__ = [
    "LMEvalDataLoader",
]

log = logging.getLogger(__name__)

class LMEvalDataLoader(BaseDataLoader):
    """
    Load contrastive pairs from a single lm-evaluation-harness task via `load_lm_eval_task`,
    split into train/test, and return a canonical LoadDataResult.
    """
    name = "lm_eval"
    description = "Load from a single lm-eval task."

    def _load_one_task(
        self,
        task_name: str,
        split_ratio: float,
        seed: int,
        limit: int | None,
        training_limit: int | None,
        testing_limit: int | None,
    ) -> LoadDataResult:
        """
        Load a single lm-eval task by name, convert to contrastive pairs,
        split into train/test, and return a LoadDataResult.
        
        arguments:
            task_name: The name of the lm-eval task to load.
            split_ratio: The fraction of data to use for training (between 0 and 1).
            seed: Random seed for shuffling/splitting.
            limit: Optional limit on total number of pairs to load.
            training_limit: Optional limit on number of training pairs.
            testing_limit: Optional limit on number of testing pairs.
            
        returns:
            A LoadDataResult containing train/test pairs and task info.
            
        raises:
            DataLoaderError if the task cannot be found or if splits are empty.
            ValueError if split_ratio is not in [0.0, 1.0].
            NotImplementedError if load_lm_eval_task is not implemented.
        
        note:
            This loader only supports single tasks, not mixtures. To load mixtures,
            use a custom data loader or extend this one."""
        loaded = self.load_lm_eval_task(task_name)

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

        pairs = lm_build_contrastive_pairs(
            task_name=pairs_task_name,
            lm_eval_task=task_obj,
            limit=limit,
        )

        train_pairs, test_pairs = self._split_pairs(
            pairs, split_ratio, seed, training_limit, testing_limit
        )

        if not train_pairs or not test_pairs:
            raise DataLoaderError("One of the splits is empty after splitting.")

        train_set = ContrastivePairSet("lm_eval_train", train_pairs, task_type=task_name)
        test_set = ContrastivePairSet("lm_eval_test", test_pairs, task_type=task_name)

        train_set.validate()
        test_set.validate()

        return LoadDataResult(
            train_qa_pairs=train_set,
            test_qa_pairs=test_set,
            task_type=task_name,
            lm_task_data=task_obj,
        )

    def load(
        self,
        task: str,  
        split_ratio: float | None = None,
        seed: int = 42,
        limit: int | None = None,
        training_limit: int | None = None,
        testing_limit: int | None = None,
        **_: Any,
    ) -> LoadDataResult:
        """
        Load contrastive pairs from a single lm-eval-harness task, split into train/test sets.
        arguments:
            task:
                The name of the lm-eval task to load (e.g., "winogrande", "hellaswag").
                Must be a single task, not a mixture.
            split_ratio:
                Float in [0.0, 1.0] representing the proportion of data to use for training.
                Defaults to 0.8 if None.
            seed:
                Random seed for shuffling the data before splitting.
            limit:
                Optional maximum number of total pairs to load from the task.
            training_limit:
                Optional maximum number of training pairs to return.
            testing_limit:
                Optional maximum number of testing pairs to return.
            **_:
                Additional keyword arguments (ignored).
        
        returns:
            LoadDataResult with train/test ContrastivePairSets and metadata.
        
        raises:
            DataLoaderError if loading or processing fails.
            ValueError if split_ratio is not in [0.0, 1.0].
            NotImplementedError if load_lm_eval_task is not implemented.
        """
        split = self._effective_split(split_ratio)

        # Single-task path only
        return self._load_one_task(
            task_name=str(task),
            split_ratio=split,
            seed=seed,
            limit=limit,
            training_limit=training_limit,
            testing_limit=testing_limit,
        )

    @staticmethod
    def load_lm_eval_task(task_name: str) -> ConfigurableTask | dict[str, ConfigurableTask]:
        """
        Load a single lm-eval-harness task by name.

        arguments:
            task_name: The name of the lm-eval task to load.
        
        returns:
            A ConfigurableTask instance or a dict of subtask name to ConfigurableTask.
        
        raises:
            DataLoaderError if the task cannot be found.
        """
        task_manager = LMTaskManager()
        task_manager.initialize_tasks()

        task_dict = get_task_dict([task_name], task_manager=task_manager)
        if task_name in task_dict:
            return task_dict[task_name]
        raise DataLoaderError(f"lm-eval task '{task_name}' not found.")
    
    def _split_pairs(
        self,
        pairs: list[ContrastivePair],
        split_ratio: float,
        seed: int,
        training_limit: int | None,
        testing_limit: int | None,
    ) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
        """
        Split a list of ContrastivePairs into train/test sets.

        arguments:
            pairs: List of ContrastivePair to split.
            split_ratio: Float in [0.0, 1.0] for the training set proportion.
            seed: Random seed for shuffling.
            training_limit: Optional max number of training pairs.
            testing_limit: Optional max number of testing pairs.
        
        returns:
            A tuple of (train_pairs, test_pairs).
        raises:
            ValueError if split_ratio is not in [0.0, 1.0].
        """
        if not pairs:
            return [], []
        from numpy.random import default_rng

        idx = list(range(len(pairs)))
        default_rng(seed).shuffle(idx)
        cut = int(len(pairs) * split_ratio)
        train_idx = set(idx[:cut])

        train_pairs: list[ContrastivePair] = []
        test_pairs: list[ContrastivePair] = []
        for i in idx:
            (train_pairs if i in train_idx else test_pairs).append(pairs[i])

        if training_limit and training_limit > 0:
            train_pairs = train_pairs[:training_limit]
        if testing_limit and testing_limit > 0:
            test_pairs = test_pairs[:testing_limit]

        return train_pairs, test_pairs