"""Serialization helpers for contrastive pair sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .set import ContrastivePairSet


def save_to_json(pair_set: ContrastivePairSet, json_path: Union[str, Path], include_metadata: bool = True) -> None:
    """Write pairs to a JSON file (UTF-8, pretty)."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: List[Dict[str, Any]] = []
    for i, p in enumerate(pair_set.pairs):
        item: Dict[str, Any] = {
            "pair_id": i,
            "prompt": p.prompt,
            "positive_response": getattr(p.positive_response, "text", None),
            "negative_response": getattr(p.negative_response, "text", None),
        }
        if include_metadata and hasattr(p, "question"):
            item.update(
                {
                    "question": getattr(p, "question", ""),
                    "correct_answer": getattr(p, "correct_answer", ""),
                    "incorrect_answer": getattr(p, "incorrect_answer", ""),
                }
            )
        data.append(item)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_from_json(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Lightweight loader that returns the raw list of entries.

    Useful for debugging or custom reconstructors.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("JSON must contain a list")
    return raw