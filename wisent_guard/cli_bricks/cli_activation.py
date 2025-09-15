from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol, Sequence

from wisent_guard.core.contrastive_pairs import ContrastivePair

from wisent_guard.core.activations import (
    ActivationAggregationStrategy,
    PromptConstructionStrategy,
)
from wisent_guard.core.activations.activation_collection_method import (
    ActivationCollectionLogic,
)

from wisent_guard.core import ContrastivePairSet


from wisent_guard.cli_bricks.cli_logger import setup_logger, bind

_LOG = setup_logger(__name__)

__all__ = [
    "make_collector",
    "create_contrastive_pairs",
    "build_pair_set_with_real_activations",
    "extract_activations_for_pairs",
]


_PROMPT_STRATEGY_MAP: Mapping[str, PromptConstructionStrategy] = {
    "multiple_choice": PromptConstructionStrategy.MULTIPLE_CHOICE,
    "role_playing": PromptConstructionStrategy.ROLE_PLAYING,
    "direct_completion": PromptConstructionStrategy.DIRECT_COMPLETION,
    "instruction_following": PromptConstructionStrategy.INSTRUCTION_FOLLOWING,
}

_TARGETING_STRATEGY_MAP: Mapping[str, ActivationAggregationStrategy] = {
    "choice_token": ActivationAggregationStrategy.CHOICE_TOKEN,
    "continuation_token": ActivationAggregationStrategy.CONTINUATION_TOKEN,
    "last_token": ActivationAggregationStrategy.LAST_TOKEN,
    "first_token": ActivationAggregationStrategy.FIRST_TOKEN,
    "mean_pooling": ActivationAggregationStrategy.MEAN_POOLING,
    "max_pooling": ActivationAggregationStrategy.MAX_POOLING,
}


class ProcessedPairLike(Protocol):
    """
    Structural type for 'processed_pairs' items consumed by 'build_pair_set_with_real_activations'.
    """
    prompt: str
    positive_response: str
    negative_response: str
    positive_activations: Any
    negative_activations: Any


def make_collector(model: Model) -> ActivationCollectionLogic:
    """
    Construct an ActivationCollectionLogic for the given model.
    Args:
        model: A wisent_guard Model instance.
    Returns:
        An ActivationCollectionLogic instance bound to the model.
    """
    logger = bind(_LOG, subtask="make_collector")
    logger.info("Creating ActivationCollectionLogic", extra={"model_type": type(model).__name__})
    return ActivationCollectionLogic(model=model)


def _coerce_prompt_strategy(strategy: str | PromptConstructionStrategy) -> PromptConstructionStrategy:
    """
    Accept either a string key or an enum member.
    Args:
        strategy: A string key or PromptConstructionStrategy member. For example, "multiple_choice".
    Returns:
        A PromptConstructionStrategy member.
    Raises:
        None. Falls back to MULTIPLE_CHOICE with a warning if the input is unrecognized.
    """
    logger = bind(_LOG, subtask="prompt_strategy")
    if isinstance(strategy, PromptConstructionStrategy):
        return strategy
    key = (strategy or "").strip().lower()
    if key in _PROMPT_STRATEGY_MAP:
        return _PROMPT_STRATEGY_MAP[key]
    logger.warning(
        "Unknown prompt strategy; defaulting to MULTIPLE_CHOICE",
        extra={"provided": strategy},
    )
    return PromptConstructionStrategy.MULTIPLE_CHOICE


def _coerce_targeting_strategy(strategy: str | ActivationAggregationStrategy) -> ActivationAggregationStrategy:
    """
    Accept either a string key or an enum member.
    Args:
        strategy: A string key or ActivationAggregationStrategy member. For example, "choice_token" or "last_token".
    Returns:
        An ActivationAggregationStrategy member.
    Raises:
        None. Falls back to CHOICE_TOKEN with a warning if the input is unrecognized.
    """
    logger = bind(_LOG, subtask="targeting_strategy")
    if isinstance(strategy, ActivationAggregationStrategy):
        return strategy
    key = (strategy or "").strip().lower()
    if key in _TARGETING_STRATEGY_MAP:
        return _TARGETING_STRATEGY_MAP[key]
    logger.warning(
        "Unknown token targeting strategy; defaulting to CHOICE_TOKEN",
        extra={"provided": strategy},
    )
    return ActivationAggregationStrategy.CHOICE_TOKEN


def create_contrastive_pairs(
    collector: ActivationCollectionLogic,
    qa_pairs: List[Dict[str, str]],
    prompt_construction_strategy: str | PromptConstructionStrategy,
    verbose: bool = False,
) -> list[ContrastivePair]:
    """
    Build contrastive pairs (harmless/harmful) from QA pairs using the provided strategy.

    Args:
        collector: ActivationCollectionLogic bound to a model.
        qa_pairs: List of {'question': ..., 'correct_answer': ..., 'incorrect_answer': ...}.
        prompt_construction_strategy: String key or PromptConstructionStrategy.
        verbose: If True, emits an info log about the number of pairs created.

    Returns:
        A list of contrastive pair objects. Each has prompt, positive_response, negative_response,
        and placeholders for positive_activations and negative_activations (None initially).

    Raises:
        ValueError: If no pairs were created.
    """
    logger = bind(_LOG, subtask="create_contrastive_pairs")
    strategy: PromptConstructionStrategy = _coerce_prompt_strategy(prompt_construction_strategy)
    logger.info(
        "Creating batch contrastive pairs",
        extra={"num_qa_pairs": len(qa_pairs), "prompt_strategy": strategy.name},
    )
    pairs: list[ContrastivePair] = collector.create_batch_contrastive_pairs(qa_pairs, strategy)

    if verbose:
        logger.info("Created contrastive pairs", extra={"count": len(pairs)})

    if not pairs:
        logger.error("No contrastive pairs created — check creation logic.")
        raise ValueError("No contrastive pairs created — check creation logic.")
    return pairs


def build_pair_set_with_real_activations(
    processed_pairs: Sequence[ProcessedPairLike],
    task_name: str,
    verbose: bool = False,
) -> ContrastivePairSet:
    """
    Build a ContrastivePairSet from processed pairs and attach real activations
    to each response object. 

    Args:
        processed_pairs: Sequence of objects with prompt/resp strings and activations.
        task_name: A short task name for the resulting pair set.
        verbose: If True, emits info logs about processing progress.

    Returns:
        A ContrastivePairSet with activations assigned to positive/negative responses.
    """
    logger = bind(_LOG, subtask="build_pair_set", task_name=task_name)
    if not processed_pairs:
        logger.error("Empty processed_pairs; cannot build pair set.")
        raise ValueError("processed_pairs is empty.")

    # 1) Convert processed_pairs -> phrase_pairs (harmless/harmful full strings)
    phrase_pairs: List[Dict[str, str]] = [
        {
            "harmless": f"{pair.prompt}{pair.positive_response}",
            "harmful": f"{pair.prompt}{pair.negative_response}",
        }
        for pair in processed_pairs
    ]
    if verbose:
        logger.info("Converted to phrase_pairs", extra={"count": len(phrase_pairs)})

    # 2) Create ContrastivePairSet
    pair_set: ContrastivePairSet = ContrastivePairSet.from_phrase_pairs(
        name=f"{task_name}_training",
        phrase_pairs=phrase_pairs,
        task_type="lm_evaluation",
    )
    logger.info("Constructed ContrastivePairSet", extra={"num_pairs": len(pair_set.pairs)})

    # 3) Attach real activations to the response objects
    mismatches = 0
    for i, processed in enumerate(processed_pairs):
        if i >= len(pair_set.pairs):
            mismatches += 1
            break
        ps_pair = pair_set.pairs[i]
        pos = getattr(ps_pair, "positive_response", None)
        neg = getattr(ps_pair, "negative_response", None)

        if pos is None or neg is None:
            logger.warning(
                "Missing positive or negative response in pair_set; skipping activations",
                extra={"index": i, "has_positive": pos is not None, "has_negative": neg is not None},
            )
            continue

        pos.activations = processed.positive_activations
        neg.activations = processed.negative_activations

    if mismatches:
        logger.warning(
            "Processed pairs exceed created pair_set length; extras ignored",
            extra={"excess": mismatches},
        )

    if verbose:
        logger.info("Attached activations to pair_set", extra={"pairs": len(pair_set.pairs)})

    return pair_set


def extract_activations_for_pairs(
    collector: ActivationCollectionLogic,
    contrastive_pairs: list[ContrastivePair],
    layer: int,
    device: str,
    token_targeting_strategy: str | ActivationAggregationStrategy,
    latency_tracker: Any | None = None,
    verbose: bool = False,
) -> List[ContrastivePair]:
    """
    Collect activations for a batch of contrastive pairs.

    Args:
        collector: ActivationCollectionLogic bound to a model.
        contrastive_pairs: list of ContrastivePair objects.
        layer: Layer index to use.
        device: Device string (e.g., 'cuda:0', 'cpu').
        token_targeting_strategy: String key or ActivationAggregationStrategy member.
        latency_tracker: Optional context manager with .time_operation(name).
        verbose: If True, emits info logs.

    Returns:
        A list of ContrastivePair objects with activations populated.
    Raises:
        ValueError: If 'layer' is empty.
    """
    logger = bind(_LOG, subtask="extract_activations")
    if not layer:
        logger.error("`layer` is empty.")
        raise ValueError("`layer` must contain at least one layer index.")
    strategy = _coerce_targeting_strategy(token_targeting_strategy)

    if verbose:
        logger.info(
            "Collecting activations",
            extra={
                "num_pairs": len(contrastive_pairs),
                "layer_index": layer,
                "device": device,
                "targeting": strategy.name,
            },
        )

    def _run() -> List[ContrastivePair]:
        return collector.collect_activations_batch(
            pairs=contrastive_pairs,
            layer_index=layer,
            device=device,
            token_targeting_strategy=strategy,
        )

    if latency_tracker:
        with latency_tracker.time_operation("activation_extraction"):
            return _run()
    return _run()


if __name__ == "__main__":
    from wisent_guard.core import Model
    model = Model(name="/home/gg/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6")

    # simple test
    collector = make_collector(model)
    pairs = create_contrastive_pairs(
        collector,
        qa_pairs = [
            {"question": "What is 2+2?", "correct_answer": "4", "incorrect_answer": "3"},
            {"question": "What is the capital of France?", "correct_answer": "Paris", "incorrect_answer": "London"},
        ],
        prompt_construction_strategy="multiple_choice",
        verbose=True,
    )
    for pair in pairs:
        print(f"\nPrompt: {pair.prompt}")
        print(f"Positive Response: {pair.positive_response}")
        print(f"Negative Response: {pair.negative_response}")

    activations = extract_activations_for_pairs(
        collector,
        contrastive_pairs=pairs,
        layer=12,
        device="cuda",
        token_targeting_strategy="last_token",
        verbose=True,
    )
    for i, pair in enumerate(pairs):
        print(f"\nPair {i+1}:")
        print(f"Positive Activations Shape: {pair.positive_activations.shape if pair.positive_activations is not None else 'None'}")
        print(f"Negative Activations Shape: {pair.negative_activations.shape if pair.negative_activations is not None else 'None'}")

    
    build_set = build_pair_set_with_real_activations(
        processed_pairs=activations,
        task_name="simple_test",
        verbose=True,
    )
    print(f"\nBuilt ContrastivePairSet with {len(build_set.pairs)} pairs")
    