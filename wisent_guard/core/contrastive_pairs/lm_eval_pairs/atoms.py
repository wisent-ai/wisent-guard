from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Sequence, TYPE_CHECKING
from abc import ABC, abstractmethod


if TYPE_CHECKING:
    from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
    from lm_eval.api.task import ConfigurableTask


__all__ = [
    "UnsupportedLMEvalBenchmarkError",
    "NoLabelledDocsAvailableError",
    "LMEvalBenchmarkExtractor",
]


class UnsupportedLMEvalBenchmarkError(Exception):
    """Raised when a benchmark/task does not have a compatible extractor."""


class NoLabelledDocsAvailableError(UnsupportedLMEvalBenchmarkError):
    """
    Raised when no labeled documents can be found for a given lm-eval task.

    This typically indicates the task does not expose any of:
    validation/test/training/fewshot docs, nor sufficient dataset metadata
    to load a split directly.
    """

class LMEvalBenchmarkExtractor(ABC):
    """
    Abstract base class for lm-eval benchmark-specific extractors.

    Subclasses should implement :meth:'extract_contrastive_pairs' to transform
    task documents into a list of :class:'ContrastivePair' instances.

    Utility methods are provided to load the most appropriate labeled documents
    from a task, with a clear order of preference and a robust dataset fallback.
    """

    @abstractmethod
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Extract contrastive pairs from the provided lm-eval task.

        arguments:
            lm_eval_task_data:
                An lm-eval task instance.
            limit:
                Optional upper bound on the number of pairs to return.
                Values <= 0 are treated as "no limit".

        returns:
            A list of :class:'ContrastivePair'.
        """
        raise NotImplementedError


    @classmethod
    def load_docs(
        cls,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load labeled documents from the most appropriate split with a clear
        preference order:

            validation → test → train → fewshot

        If none are available, attempts a dataset fallback using
        'datasets.load_dataset' with the task's declared metadata
        (e.g., 'dataset_path'/'dataset_name', 'dataset_config_name',
        and 'fewshot_split').

        arguments:
            lm_eval_task_data:
                Task object from lm-eval.
            limit:
                Optional maximum number of documents to return.
                Values <= 0 are treated as "no limit".

        returns:
            A list of document dictionaries.

        raises:
            NoLabelledDocsAvailableError:
                If no labeled documents are available.
            RuntimeError:
                If a dataset fallback is attempted and fails to load.
        """
        max_items = cls._normalize_limit(limit)

        preferred_sources: Sequence[tuple[str, str]] = (
            ("has_validation_docs", "validation_docs"),
            ("has_test_docs", "test_docs"),
            ("has_training_docs", "training_docs"),
            ("has_fewshot_docs", "fewshot_docs"),
        )

        for has_method, docs_method in preferred_sources:
            if cls._has_true(lm_eval_task_data, has_method) and cls._has_callable(
                lm_eval_task_data, docs_method
            ):
                docs_iter = getattr(lm_eval_task_data, docs_method)()
                docs_list = cls._coerce_docs_to_dicts(docs_iter, max_items)
                if docs_list:
                    return docs_list

        # Fallback to dataset split (common for tasks relying on fewshot_split).
        docs_list = cls._fallback_load_from_dataset(lm_eval_task_data, max_items)
        if docs_list:
            return docs_list

        task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
        raise NoLabelledDocsAvailableError(
            f"No labeled documents are available for task '{task_name}'. "
            "The task does not expose validation/test/train/fewshot docs, "
            "and no usable dataset metadata was found for a fallback load.\n\n"
            "Tip: Ensure your task implements at least one of the doc getters "
            "(validation_docs/test_docs/training_docs/fewshot_docs), or that it "
            "declares dataset metadata (dataset_path or dataset_name, "
            "dataset_config_name, and fewshot_split) so a split can be loaded."
        )

    @staticmethod
    def _normalize_limit(limit: int | None) -> int | None:
        """
        Normalize limit semantics:
          - None → None (unbounded)
          - <= 0 → None (unbounded)
          - > 0 → limit
        """
        if limit is None or limit <= 0:
            return None
        return int(limit)

    @staticmethod
    def _has_callable(obj: Any, name: str) -> bool:
        """Return True if obj has a callable attribute with the given name."""
        return hasattr(obj, name) and callable(getattr(obj, name))

    @staticmethod
    def _has_true(obj: Any, name: str) -> bool:
        """Return True if obj has an attribute that evaluates to True when called or read."""
        attr = getattr(obj, name, None)
        try:
            return bool(attr() if callable(attr) else attr)
        except Exception:  # pragma: no cover (defensive)
            return False

    @classmethod
    def _coerce_docs_to_dicts(
        cls,
        docs_iter: Iterable[Any] | None,
        max_items: int | None,
    ) -> list[dict[str, Any]]:
        """
        Materialize an iterable of docs into a list of dictionaries,
        applying an optional limit.
        """
        if docs_iter is None:
            return []

        out: list[dict[str, Any]] = []
        for idx, item in enumerate(docs_iter):
            if max_items is not None and idx >= max_items:
                break
            if isinstance(item, Mapping):
                out.append(dict(item))
            else:
                try:
                    out.append(dict(item))  
                except Exception as exc:  
                    raise TypeError(
                        "Expected each document to be a mapping-like object that can "
                        "be converted to dict. Got type "
                        f"{type(item).__name__} with value {item!r}"
                    ) from exc
        return out

    @classmethod
    def _fallback_load_from_dataset(
        cls,
        lm_eval_task_data: ConfigurableTask,
        max_items: int | None,
    ) -> list[dict[str, Any]]:
        """
        Attempt to load documents via datasets.load_dataset using the task's
        declared metadata. We prefer 'fewshot_split' if present, since this is
        a common pattern for tasks like (M)MMLU.

        returns:
            A possibly empty list of docs.
        """
        dataset_name = getattr(lm_eval_task_data, "dataset_path", None) or getattr(
            lm_eval_task_data, "dataset_name", None
        )
        dataset_config = getattr(lm_eval_task_data, "dataset_config_name", None)
        dataset_split = getattr(lm_eval_task_data, "fewshot_split", None)

        if not dataset_name or not dataset_split:
            return []

        try:
            from datasets import load_dataset 
        except Exception as exc:  
            task_name = getattr(
                lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__
            )
            raise RuntimeError(
                f"Task '{task_name}' specifies dataset metadata but "
                "the 'datasets' library is not available. "
                "Install it via 'pip install datasets' to enable fallback loading."
            ) from exc

        try:
            dataset = load_dataset(
                dataset_name,
                dataset_config if dataset_config else None,
                split=dataset_split,
            )
        except Exception as exc:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            raise RuntimeError(
                f"Failed to load dataset split via fallback for task '{task_name}'. "
                f"Arguments were: name={dataset_name!r}, config={dataset_config!r}, "
                f"split={dataset_split!r}. Underlying error: {exc}"
            ) from exc

        return cls._coerce_docs_to_dicts(dataset, max_items)