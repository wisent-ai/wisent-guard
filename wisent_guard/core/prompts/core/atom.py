from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict

__all__ = [
    "ChatMessage",
    "PromptPair",
    "PromptStrategy",
    "UnknownStrategyError",
]

class ChatMessage(TypedDict):
    """A single chat message compatible with common chat model APIs."""
    role: str
    content: str


@dataclass(frozen=True)
class PromptPair:
    """A pair of prompts for positive and negative training/eval cases."""
    positive: list[ChatMessage]
    negative: list[ChatMessage]

class PromptStrategy(ABC):
    """Abstract strategy for building a PromptPair from QA text.

    Subclasses MUST define a unique, non-empty class attribute:
        strategy_key: str
    """

    strategy_key: str

    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        super().__init_subclass__(**kwargs)
        key = getattr(cls, "strategy_key", None)
        if not isinstance(key, str) or not key.strip():
            raise TypeError(
                f"{cls.__name__} must define a non-empty class attribute "
                "'strategy_key: str'."
            )

    @abstractmethod
    def build(
        self,
        question: str,
        correct_answer: str,
        incorrect_answer: str,
    ) -> PromptPair:
        """Construct positive/negative prompts from the given QA trio."""
        raise NotImplementedError


class UnknownStrategyError(ValueError):
    """Raised when a requested strategy key does not exist."""
    pass