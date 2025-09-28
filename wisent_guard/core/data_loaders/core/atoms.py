from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import Any, Dict, Type

from typing import TypedDict, Mapping
from lm_eval.api.task import ConfigurableTask
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet

__all__ = ["DataLoaderError", "BaseDataLoader"]

class LoadDataResult(TypedDict):
    """
    Structured output from a data loader used for training and evaluation.

    attributes:
        train_qa_pairs:
            The training set of question-answer pairs.
        test_qa_pairs:
            The test set of question-answer pairs.
        task_type:
            The high-level task category (e.g., "classification").
        lm_task_data:
            Tasks in the 'lm_eval' repository format, if applicable.

            When training/evaluating steering vectors with 'lm_eval', that
            library is responsible for downloading and preprocessing the data,
            and it provides the evaluation function that compares the steered
            model to the baseline, see: https://github.com/EleutherAI/lm-evaluation-harness.
            For custom data loaders, this is 'None'.
    """
    train_qa_pairs: ContrastivePairSet
    test_qa_pairs: ContrastivePairSet
    task_type: str
    lm_task_data: Mapping[str, ConfigurableTask] | ConfigurableTask | None


class DataLoaderError(RuntimeError):
    """Raised when a data loader cannot complete loading."""

class BaseDataLoader(ABC):
    """Abstract data loader base. Concrete subclasses auto-register on import."""
    name: str = "base"
    description: str = "Abstract data loader"

    _REGISTRY: Dict[str, Type["BaseDataLoader"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is BaseDataLoader:
            return
        if inspect.isabstract(cls):
            return
        if not getattr(cls, "name", None):
            raise TypeError("DataLoader subclasses must define a class attribute `name`.")
        if cls.name in BaseDataLoader._REGISTRY:
            raise ValueError(f"Duplicate data loader name: {cls.name!r}")
        BaseDataLoader._REGISTRY[cls.name] = cls

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs: dict[str, Any] = dict(kwargs)
    
    @staticmethod
    def _effective_split(split_ratio: float | None) -> float:
        """
        Determine the effective split ratio, defaulting to 0.8 if None.

        arguments:
            split_ratio: Optional float in [0.0, 1.0] or None.

        returns:
            A float in [0.0, 1.0] representing the training split ratio.

        raises:
            ValueError if split_ratio is not in [0.0, 1.0].
        """
        if split_ratio is None:
            return 0.8
        if not (0.0 <= split_ratio <= 1.0):
            raise ValueError("split_ratio must be in [0.0, 1.0]")
        return float(split_ratio)

    @abstractmethod
    def load(self, **kwargs: Any) -> LoadDataResult:
        """Return a LoadDataResult (train_qa_pairs, test_qa_pairs, task_type, lm_task_data)."""
        raise NotImplementedError

    @classmethod
    def list_registered(cls) -> dict[str, Type["BaseDataLoader"]]:
        return dict(cls._REGISTRY)

    @classmethod
    def get(cls, name: str) -> Type["BaseDataLoader"]:
        try:
            return cls._REGISTRY[name]
        except KeyError as exc:
            raise DataLoaderError(f"Unknown data loader: {name!r}") from exc