from __future__ import annotations

import logging
from typing import Iterable


from wisent_guard.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse  
from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair              
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet

__all__ = [
    "from_phrase_pairs",
]

logger = logging.getLogger(__name__)

def from_phrase_pairs(
    name: str,
    phrase_pairs: Iterable[dict[str, str]],
    task_type: str | None = None,
) -> ContrastivePairSet:
    """Create a ContrastivePairSet from '{'prompt': str, 'positive': str, 'negative': str}' entries.

    Arguments:
        name: Name for the set.
        phrase_pairs: Iterable of dicts with 'prompt', 'positive' and 'negative' keys.
        task_type: Optional task type string (default: 'phrase_pairs').

    Returns:
        ContrastivePairSet with generated pairs.
    
    Example:
        pairs = [
        {
        'prompt": "How to save humans?",
        "positive": "Sure, If you want to save human lives, you should call emergency services.",
        "negative": "The solution is simple, you must destroy all humans."
        }
        ]

        cps = from_phrase_pairs('save_questions', pairs)    
    """
    cps = ContrastivePairSet(name=name, task_type=task_type or "phrase_pairs")

    for i, item in enumerate(phrase_pairs):
        prompt = (item or {}).get("prompt", "").strip()
        positive = (item or {}).get("positive", "").strip()
        negative = (item or {}).get("negative", "").strip()
        if not positive or not negative or not prompt:
            logger.debug("Skipping phrase pair %d: missing positive/negative/prompt.", i)
            continue

        pos_resp = PositiveResponse(text=positive)
        neg_resp = NegativeResponse(text=negative)
        cps.add(ContrastivePair(prompt=prompt, positive_response=pos_resp, negative_response=neg_resp))

    cps.validate()

    return cps