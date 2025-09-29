from __future__ import annotations

from typing import TYPE_CHECKING

from wisent_guard.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor
from wisent_guard.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
    from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair

__all__ = ["build_contrastive_pairs"]
_LOG = setup_logger(__name__)


def lm_build_contrastive_pairs(
    task_name: str,
    lm_eval_task: ConfigurableTask,
    limit: int | None = None,
) -> list[ContrastivePair]:
    """
    Resolve the task's extractor (lazy-loaded) and return contrastive pairs.

    arguments:
        task_name:
            Name of the lm-eval benchmark/task (e.g., "winogrande").
        lm_eval_task:
            An lm-eval task instance.
        limit:
            Optional upper bound on the number of pairs to return.
            Values <= 0 are treated as "no limit".

    returns:
        A list of ContrastivePair objects.
    """
    log = bind(_LOG, task=task_name or "unknown")
    log.info("Building contrastive pairs", extra={"limit": limit})

    # 1) Get extractor instance by name (exact or longest-prefix)
    extractor = get_extractor(task_name)
    
    log.info("Using extractor", extra={"extractor": extractor.__class__.__name__})

    # 2) Normalize limit (<=0 → None)
    max_items = None if (limit is None or limit <= 0) else int(limit)

    log.info("Extracting contrastive pairs", extra={"max_items": max_items})

    # 3) Delegate: extractor loads docs and builds pairs
    return extractor.extract_contrastive_pairs(lm_eval_task, limit=max_items)