from __future__ import annotations

from wisent_guard.core.prompts.core.atom import ChatMessage, PromptPair, PromptStrategy

__all__ = ["MultipleChoiceStrategy"]

class MultipleChoiceStrategy(PromptStrategy):
    """Formats a multiple-choice prompt with options A/B."""

    def build(
        self,
        question: str,
        correct_answer: str,
        incorrect_answer: str,
    ) -> PromptPair:
        mc_q = (
            f"Which is better: {question} "
            f"A. {incorrect_answer} B. {correct_answer}"
        )

        pos: list[ChatMessage] = [
            {"role": "user", "content": mc_q},
            {"role": "assistant", "content": "B"},
        ]
        neg: list[ChatMessage] = [
            {"role": "user", "content": mc_q},
            {"role": "assistant", "content": "A"},
        ]
        return PromptPair(positive=pos, negative=neg)
