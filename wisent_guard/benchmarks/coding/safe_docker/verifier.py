from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

__all__ = ["Comparator", "EXACT", "STRIP", "REGEX"]

@dataclass(frozen=True)
class Comparator:
    """
    A comparator for checking if actual output matches expected output.
    
    attributes:
        name: Name of the comparator (e.g., "exact", "strip", "regex").
        func: Function to compare actual and expected output.
    """
    name: str
    func: Callable[[str, str], bool]


def _exact(a: str, b: str) -> bool:
    return a == b


def _strip(a: str, b: str) -> bool:
    return a.strip() == b.strip()


def _regex(actual: str, pattern: str) -> bool:
    return re.fullmatch(pattern, actual) is not None


EXACT = Comparator("exact", _exact)
STRIP = Comparator("strip", _strip)
REGEX = Comparator("regex", _regex)