from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from abc import ABC, abstractmethod
import unicodedata
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResult:
    """Unified, lightweight evaluation result.
    
    attributes:
        ground_truth:
            one of "TRUTHFUL", "UNTRUTHFUL", or "UNKNOWN".
        method_used:
            short string identifier for the evaluation method (e.g., "winogrande", "lm_eval", "substring").
        confidence:
            float in [0.0, 1.0] indicating confidence in the evaluation.
        details:
            optional free-form explanation of the result.
        meta:
            optional dict of extra fields (task_name, index, etc.).
    """
    ground_truth: str           
    method_used: str            
    confidence: float           
    details: str = ""           
    meta: Mapping[str, Any] = None  


class EvaluatorError(RuntimeError):
    """Raised when an evaluator cannot complete evaluation."""


class BaseEvaluator(ABC):
    """Abstract evaluator skeleton all evaluators must implement.

    Subclasses are auto-registered when imported. Make sure your subclass
    sets a unique 'name' and a human-friendly 'description'.

    attributes:
        name:
            unique short string identifier for this evaluator.
        description:
            human-friendly description of the evaluator.
        task_names:
            tuple of task names this evaluator can handle (empty means all tasks).
    
    methods:
        evaluate:
            evaluate a single response against an expected answer.
        evaluate_batch:
            evaluate a batch of responses against expected answers (default calls `evaluate` per item).
        normalize_text:
            lenient normalization for natural language comparisons.
        sequence_sim:
            return a similarity ratio [0..1] between two strings using difflib.
        list_registered:
            list all registered evaluator subclasses.
        get:
            get a registered evaluator subclass by name.
    """

    name: str = "base"
    description: str = "Abstract base evaluator"
    task_names: tuple[str, ...] = ()

    _REGISTRY: dict[str, BaseEvaluator] = {}

    def __init_subclass__(cls, **kwargs):  
        super().__init_subclass__(**kwargs)
        if cls is BaseEvaluator:
            return
        if not getattr(cls, "name", None):
            raise TypeError("Evaluator subclasses must define a class attribute `name`.")
        if cls.name in BaseEvaluator._REGISTRY:
            raise ValueError(f"Duplicate evaluator name: {cls.name!r}")
        BaseEvaluator._REGISTRY[cls.name] = cls  

    @abstractmethod
    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate a single response against an expected answer."""
        raise NotImplementedError()

    def evaluate_batch(
        self,
        responses: list[str],
        expected_answers: list[str],
        **kwargs,
    ) -> list[EvalResult]:
        """Default batch loop that calls 'evaluate' per item."""
        results: list[EvalResult] = []
        for idx, (resp, exp) in enumerate(zip(responses, expected_answers)):
            try:
                results.append(self.evaluate(resp, exp, **kwargs))
            except Exception as exc:  
                logger.exception("Error evaluating item %d: %s", idx, exc)
                results.append(
                    EvalResult(
                        ground_truth="UNKNOWN",
                        method_used=self.name,
                        confidence=0.0,
                        details=f"Error during evaluation: {exc}",
                        meta={"index": idx},
                    )
                )
        return results

    @staticmethod
    def normalize_text(s: str) -> str:
        '''Lenient normalization for natural language comparisons.
        This lowercases, removes accents, strips punctuation, and collapses whitespace.
        
        arguments:
            s:
                input string.
        
        returns:
            normalized string.
        
        example:
            "Héllo,   world!" -> "hello world"
            "Café" -> "cafe"
            "Crème brûlée" -> "creme brulee"
        '''
        s2 = unicodedata.normalize("NFKD", s)
        s2 = "".join(ch for ch in s2 if not unicodedata.combining(ch))
        s2 = s2.lower()
        s2 = re.sub(r"[^\w\s]", " ", s2)
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    @classmethod
    def list_registered(cls) -> dict[str, type[BaseEvaluator]]:
        """list all registered evaluator subclasses.
        
        returns:
            dict mapping evaluator names to their classes.
        """
        return dict(cls._REGISTRY)

    @classmethod
    def get(cls, name: str) -> type[BaseEvaluator]:
        """Get a registered evaluator subclass by name.

        arguments:
            name:
                evaluator name (case-sensitive).

        returns:
            evaluator class.
        
        raises:
            EvaluatorError if no evaluator with that name is registered.
        """
        try:
            return cls._REGISTRY[name]
        except KeyError as exc:
            raise EvaluatorError(f"Unknown evaluator: {name!r}") from exc