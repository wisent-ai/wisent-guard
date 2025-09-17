from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, TYPE_CHECKING

from wisent_guard.cli_bricks.cli_logger import setup_logger, bind

from wisent_guard.core.activations import (
    ActivationAggregationStrategy,
    PromptConstructionStrategy,
)
from wisent_guard.core.activations.activation_collection_method import (
    ActivationCollectionLogic,
)
from wisent_guard.core import ContrastivePairSet

if TYPE_CHECKING:
    from wisent_guard.core.contrastive_pairs import ContrastivePair
    from wisent_guard.core.model import Model
    from wisent_guard.cli_bricks.cli_performance import Trackers


_LOG = setup_logger(__name__)

__all__: Final[list[str]] = [
    "make_collector",
    "create_contrastive_pairs",
    "build_pair_set_with_real_activations",
    "extract_activations_for_pairs",
]


_PROMPT_STRATEGY_MAP: Final[Mapping[str, PromptConstructionStrategy]] = {
    "multiple_choice": PromptConstructionStrategy.MULTIPLE_CHOICE,
    "role_playing": PromptConstructionStrategy.ROLE_PLAYING,
    "direct_completion": PromptConstructionStrategy.DIRECT_COMPLETION,
    "instruction_following": PromptConstructionStrategy.INSTRUCTION_FOLLOWING,
}

_TARGETING_STRATEGY_MAP: Final[Mapping[str, ActivationAggregationStrategy]] = {
    "choice_token": ActivationAggregationStrategy.CHOICE_TOKEN,
    "continuation_token": ActivationAggregationStrategy.CONTINUATION_TOKEN,
    "last_token": ActivationAggregationStrategy.LAST_TOKEN,
    "first_token": ActivationAggregationStrategy.FIRST_TOKEN,
    "mean_pooling": ActivationAggregationStrategy.MEAN_POOLING,
    "max_pooling": ActivationAggregationStrategy.MAX_POOLING,
}


def make_collector(model: Model) -> ActivationCollectionLogic:
    """
    Construct an ActivationCollectionLogic for the given model.

    Args:
        model: A wisent_guard-compatible model instance.

    Returns:
        An ActivationCollectionLogic instance bound to the model.
    """
    logger = bind(_LOG, subtask="make_collector")
    logger.info(
        "Creating ActivationCollectionLogic"
    )
    return ActivationCollectionLogic(model=model)


def _coerce_prompt_strategy(
    strategy: str | PromptConstructionStrategy,
) -> PromptConstructionStrategy:
    """
    Accept either a string key or an enum member.

    Arguments:
      - strategy: String name of the prompt construction strategy. For example, "multiple_choice".

    Returns:
        - A PromptConstructionStrategy enum value.
    
    Raises:
        - ValueError if the strategy is unknown.
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


def _coerce_targeting_strategy(
    strategy: str | ActivationAggregationStrategy,
) -> ActivationAggregationStrategy:
    """
    Accept either a string key or an enum member.

    Arguments:
      - strategy: String name of the token targeting strategy. For example, "choice_token".
    
    Returns:
        - An ActivationAggregationStrategy enum value.
    
    Raises:
        - ValueError if the strategy is unknown.
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
    qa_pairs: list[dict[str, str]],
    prompt_construction_strategy: str | PromptConstructionStrategy,
    verbose: bool = False,
) -> list[ContrastivePair]:
    """
    Build contrastive pairs (harmless/harmful) from QA pairs using the provided strategy.

    Args:
        collector: ActivationCollectionLogic bound to a model.
        qa_pairs: List of {"question": ..., "correct_answer": ..., "incorrect_answer": ...}.
        prompt_construction_strategy: String key or PromptConstructionStrategy.
        verbose: If True, emits an info log about the number of pairs created.

    Returns:
        A list of contrastive pairs.
    """
    logger = bind(_LOG, subtask="create_contrastive_pairs")
    strategy = _coerce_prompt_strategy(prompt_construction_strategy)
    logger.info(
        "Creating batch contrastive pairs",
        extra={"num_qa_pairs": len(qa_pairs), "prompt_strategy": strategy.name},
    )
    pairs: list[ContrastivePair] = collector.create_batch_contrastive_pairs(
        qa_pairs, strategy
    )

    if verbose:
        logger.info("Created contrastive pairs", extra={"count": len(pairs)})

    if not pairs:
        logger.error("No contrastive pairs created — check creation logic.")
        raise ValueError("No contrastive pairs created — check creation logic.")
    return pairs


def build_pair_set_with_real_activations(
    processed_pairs: list[ContrastivePair],
    task_name: str,
    verbose: bool = False,
) -> ContrastivePairSet:
    """
    Build a ContrastivePairSet from processed pairs and attach real activations
    to each response object.

    Arguments:
        - processed_pairs: List of ContrastivePair with activations populated.
        - task_name: Name for the ContrastivePairSet. For example "winogrande_training".
        - verbose: If True, emits info logs about progress.
    
    Returns:
        - A ContrastivePairSet with real activations attached.

    """
    logger = bind(_LOG, subtask="build_pair_set", task_name=task_name)
    if not processed_pairs:
        logger.error("Empty processed_pairs; cannot build pair set.")
        raise ValueError("processed_pairs is empty.")

    # 1) Convert processed_pairs -> phrase_pairs (harmless/harmful full strings)
    phrase_pairs: list[dict[str, str]] = [
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
    logger.info(
        "Constructed ContrastivePairSet", extra={"num_pairs": len(pair_set.pairs)}
    )

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
                extra={
                    "index": i,
                    "has_positive": pos is not None,
                    "has_negative": neg is not None,
                },
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
        logger.info(
            "Attached activations to pair_set", extra={"pairs": len(pair_set.pairs)}
        )

    return pair_set


def extract_activations_for_pairs(
    collector: ActivationCollectionLogic,
    contrastive_pairs: list[ContrastivePair],
    layer: int,
    device: str,
    token_targeting_strategy: str | ActivationAggregationStrategy,
    latency_tracker: Trackers | None = None,
    verbose: bool = False,
) -> list[ContrastivePair]:
    """
    Collect activations for a batch of contrastive pairs.

    Args:
        collector: ActivationCollectionLogic bound to a model.
        contrastive_pairs: list of ContrastivePair.
        layer: Layer index to use.
        device: Device string (e.g., 'cuda:0', 'cpu').
        token_targeting_strategy: String key or ActivationAggregationStrategy member. For example, "choice_token".
        latency_tracker: Optional context manager with .time_operation(name).
        verbose: If True, emits info logs.

    Returns:
        A list of ContrastivePair objects with activations populated.
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

    def _run() -> list[ContrastivePair]:
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
