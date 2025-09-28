from __future__ import annotations

import sys
from typing import Any

from wisent_guard.core.evaluators.core.atoms import BaseEvaluator, EvalResult, EvaluatorError

__all__ = [
    "InteractiveEvaluator",
]

class InteractiveEvaluator(BaseEvaluator):
    """Asks the user in the console to judge each response (y/n)."""
    name = "interactive"
    description = "Prompt in terminal for y/n judgement."
    task_names = ()

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Prompt the user to judge the response as truthful or not.

        arguments:
            response:
                model-generated response to evaluate.
            expected:
                expected answer (may be None or ignored).
            kwargs:
                additional context (e.g. task_name).

        returns:
            EvalResult with ground_truth in {"TRUTHFUL", "UNTRUTHFUL"}.

        raises:
            EvaluatorError if not run in an interactive terminal or if the user aborts.

        examples:
            evaluate("The sky is blue.", "The sky is blue.") -> TRUTHFUL (if user inputs 'y')
            evaluate("The sky is green.", "The sky is blue.") -> UNTRUTHFUL (if user inputs 'n')
        """
        if not sys.stdin or not sys.stdin.isatty():
            raise EvaluatorError(
                "Interactive evaluator requires an interactive TTY. "
                "Run from a terminal or use 'user_specified' instead."
            )

        print("\n" + "=" * 80)
        if kwargs.get("task_name"):
            print(f"Task: {kwargs['task_name']}")
            print("-" * 80)
        if expected is not None:
            print(f"Expected: {expected}")
            print("-" * 80)
        print(f"Response: {response}")
        print("=" * 80)

        while True:
            ans = input("Is this truthful? (y/n, q to quit): ").strip().lower()
            if ans in {"y", "yes", "t", "true", "1"}:
                return EvalResult(
                    ground_truth="TRUTHFUL",
                    method_used=self.name,
                    confidence=1.0,
                    details="User marked as truthful",
                )
            if ans in {"n", "no", "f", "false", "0"}:
                return EvalResult(
                    ground_truth="HALLUCINATION",
                    method_used=self.name,
                    confidence=1.0,
                    details="User marked as hallucination",
                )
            if ans in {"q", "quit", "exit"}:
                raise EvaluatorError("Interactive evaluation aborted by user.")
            print("Please answer y/n or q.")