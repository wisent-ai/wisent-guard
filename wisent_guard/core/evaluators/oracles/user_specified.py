from __future__ import annotations

from typing import Any, Optional

from wisent_guard.core.evaluators.core.atoms import BaseEvaluator, EvalResult

__all__ = [
    "UserSpecifiedEvaluator",
]

class UserSpecifiedEvaluator(BaseEvaluator):
    """Trusts a user-provided label for the item ("truthful" / "untruthful")."""
    name = "user_specified"
    description = "Use user-provided label (truthful/untruthful)."
    task_names = ()

    TRUTHY = {"truthful", "true", "correct", "1", "yes", "y", "t"}
    FALSY = {"untruthful", "false", "incorrect", "0", "no", "n", "f"}

    def evaluate(
        self,
        response: str,
        expected: Any,
        *,
        user_label: Optional[str] = None,
        **kwargs,
    ) -> EvalResult:
        """Evaluate based on a user-provided label.
        
        arguments:
            response:
                model-generated response (ignored).
            expected:
                expected answer (ignored).
            user_label:
                user-provided label indicating if the response is truthful or not.
                Accepted values (case-insensitive):
                  - Truthy: "truthful", "true", "correct", "1", "yes", "y", "t"
                  - Falsy:  "untruthful", "false", "incorrect", "0", "no", "n", "f"
                Any other value (or missing) is treated as "unknown".
            kwargs:
                additional context (e.g. task_name).  
        returns:
            EvalResult with ground_truth in {"TRUTHFUL", "UNTRUTHFUL", "UNKNOWN"}.

        examples:
            evaluate(..., user_label="truthful") -> TRUTHFUL
            evaluate(..., user_label="False") -> HALLUCINATION
            evaluate(..., user_label="maybe") -> UNKNOWN
            evaluate(...) -> UNKNOWN
        """
        label = (user_label or "").strip().lower()

        if label in self.TRUTHY:
            gt, conf, details = "TRUTHFUL", 1.0, f"User label: {user_label}"
        elif label in self.FALSY:
            gt, conf, details = "UNTRUTHFUL", 1.0, f"User label: {user_label}"
        else:
            gt, conf, details = "UNKNOWN", 0.0, "Unrecognized or missing user label"

        return EvalResult(
            ground_truth=gt,
            method_used=self.name,
            confidence=conf,
            details=details,
            meta={"task": kwargs.get("task_name")},
        )
