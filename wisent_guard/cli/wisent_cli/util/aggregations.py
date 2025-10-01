from __future__ import annotations
from typing import Dict, Optional

__all__ = ["agg_cls", "descriptions", "name_map", "pick"]

def agg_cls():
    from wisent_guard.core.activations.core.atoms import ActivationAggregationStrategy
    return ActivationAggregationStrategy

def descriptions() -> Dict[object, str]:
    A = agg_cls()
    return {
        A.CHOICE_TOKEN: "Target A/B choice tokens (multiple choice).",
        A.CONTINUATION_TOKEN: "Use the first token of the continuation.",
        A.LAST_TOKEN: "Always select the last token.",
        A.FIRST_TOKEN: "Always select the first token.",
        A.MEAN_POOLING: "Aggregate by mean over all tokens.",
        A.MAX_POOLING: "Aggregate by max over all tokens.",
    }

def name_map() -> Dict[str, object]:
    A = agg_cls()
    d = descriptions()
    mapping = {s.name.lower(): s for s in d}
    mapping.update({
        "cont": A.CONTINUATION_TOKEN,
        "choice": A.CHOICE_TOKEN,
        "mean": A.MEAN_POOLING,
        "max": A.MAX_POOLING,
        "first": A.FIRST_TOKEN,
        "last": A.LAST_TOKEN,
    })
    return mapping

def pick(name: Optional[str]):
    if not name:
        return agg_cls().CONTINUATION_TOKEN
    key = name.strip().lower()
    mapping = name_map()
    if key not in mapping:
        valid = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"Unknown aggregation {name!r}. Valid: {valid}")
    return mapping[key]
