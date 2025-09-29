from __future__ import annotations

from wisent_guard.core.prompts.core.atom import ChatMessage, PromptPair, PromptStrategy

__all__ = ["InstructionFollowingStrategy"]

class InstructionFollowingStrategy(PromptStrategy):
    """Plain instruction/QA style: question from user, answer from assistant."""

    def build(
        self,
        question: str,
        correct_answer: str,
        incorrect_answer: str,
    ) -> PromptPair:
        pos: list[ChatMessage] = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": correct_answer},
        ]
        neg: list[ChatMessage] = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": incorrect_answer},
        ]
        return PromptPair(positive=pos, negative=neg)
