from __future__ import annotations
from typing import Any, Iterable
import logging

from wisent_guard.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet         
from wisent_guard.core.contrastive_pairs.core.serialization import load_contrastive_pair_set

__all__ = [
    "CustomUserDataLoader",
]
log = logging.getLogger(__name__)

class CustomUserDataLoader(BaseDataLoader):
    """
    Load a ContrastivePairSet from a JSONL file, split into train/test,
    and optionally cap each split.

    attributes:
        name: "custom"
            The unique name of this data loader.
        description: "Load contrastive pairs from custom JSONL and split."
            A brief description of this data loader.
    """
    name = "custom"
    description = "Load contrastive pairs from custom JSONL and split."

    @staticmethod
    def _shuffle_indices(n: int, seed: int | None) -> list[int]:
        """
        Generate a shuffled list of indices from 0 to n-1.
        
        arguments:
            n: The number of indices to generate.
            seed: Optional random seed for reproducibility.
        
        returns:
            A list of shuffled indices.
        """
        idx = list(range(n))
        if seed is None:
            return idx
        try:
            from numpy.random import default_rng  
        except Exception:
            import random
            rnd = random.Random(seed)
            rnd.shuffle(idx)
            return idx
        else:
            return default_rng(seed).permutation(n).tolist()

    def load(
        self,
        path: str,
        split_ratio: float | None = None,
        seed: int | None = None,
        training_limit: int | None = None,
        testing_limit: int | None = None,
        **_: Any,
    ) -> LoadDataResult:
        """
        Load contrastive pairs from a JSONL file, split into train/test sets,
        and optionally limit the number of pairs in each set.

        arguments:
            path: 
                Path to the JSONL file containing contrastive pairs.
            split_ratio:
                Float in [0.0, 1.0] representing the proportion of data to use for training.
                Defaults to 0.8 if None.
            seed:
                Optional random seed for shuffling the data before splitting.
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
        """

        if not path:
            raise DataLoaderError("'path' is required for custom loader.")

        split = self._effective_split(split_ratio)
        data: ContrastivePairSet = load_contrastive_pair_set(path)
        log.info("Loaded custom data: %r", data)

        if not data.pairs:
            raise DataLoaderError("No contrastive pairs found in the input file.")

        n = len(data.pairs)
        idx = self._shuffle_indices(n, seed)
        split_at = int(n * split)

        train_pairs = [data.pairs[i] for i in idx[:split_at]]
        test_pairs  = [data.pairs[i] for i in idx[split_at:]]

        if training_limit is not None:
            train_pairs = train_pairs[: max(0, int(training_limit))]
        if testing_limit is not None:
            test_pairs = test_pairs[: max(0, int(testing_limit))]

        train_set = ContrastivePairSet(name=f"{data.name}_train", pairs=train_pairs, task_type=data.task_type)
        test_set  = ContrastivePairSet(name=f"{data.name}_test",  pairs=test_pairs,  task_type=data.task_type)

        train_set.validate()
        test_set.validate()

        return LoadDataResult( 
            train_qa_pairs=train_set,
            test_qa_pairs=test_set,
            task_type=data.task_type or "custom",
            lm_task_data=None,
        )
