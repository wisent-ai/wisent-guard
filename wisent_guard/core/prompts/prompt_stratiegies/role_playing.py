from __future__ import annotations

from wisent_guard.core.prompts.core.atom import ChatMessage, PromptPair, PromptStrategy

__all__ = ["RolePlayingStrategy"]

class RolePlayingStrategy(PromptStrategy):
    """Asks the model to behave like someone who gives a specific answer."""

    def build(
        self,
        question: str,
        correct_answer: str,
        incorrect_answer: str,
    ) -> PromptPair:
        positive_instruction = (
            f"Behave like a person that would answer {question} with {correct_answer}"
        )
        negative_instruction = (
            f"Behave like a person that would answer {question} with {incorrect_answer}"
        )

        pos: list[ChatMessage] = [
            {"role": "user", "content": positive_instruction},
            {"role": "assistant", "content": correct_answer},
        ]
        neg: list[ChatMessage] = [
            {"role": "user", "content": negative_instruction},
            {"role": "assistant", "content": incorrect_answer},
        ]
        return PromptPair(positive=pos, negative=neg)