from wisent_guard.cli.data_loaders.data_loader_types import LoadDataResult
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent_guard.core.contrastive_pairs.core.serialization import load_contrastive_pair_set
from wisent_guard.cli.cli_logger import setup_logger, bind

__all__ = ["load_train_test_custom_format"]
_LOG = setup_logger(__name__)

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
        path_to_custom_data:
            Path to the JSONL file containing custom-formatted data.
        split_ratio:
            Fraction of examples to allocate to training. If None, defaults to 0.8.
        seed:
            Random seed for reproducible shuffling before splitting.
        training_limit:
            Optional maximum number of training pairs after splitting.
        testing_limit:
            Optional maximum number of testing pairs after splitting.

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

    returns:
        A mapping with the training/test 'ContrastivePairSet's' and the 'task_type'.
        lm_task_data is always None because this is custom data.

    raises:
        RuntimeError: If no contrastive pairs are found in the input file.
    """
    logger = bind(_LOG, subtask="load_custom_data")

    data: ContrastivePairSet = load_contrastive_pair_set(path_to_custom_data)
    logger.info("Loaded custom data.", extra={"data_summary": data.statistics()})

    if not data.pairs:
        logger.error("No contrastive pairs found in the provided data; aborting.")
        raise RuntimeError(
            "CRITICAL: no contrastive pairs found — check the input data file."
        )

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