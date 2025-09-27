from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from wisent_guard.cli.cli_logger import setup_logger, bind
from wisent_guard.cli.data_loaders.data_loader_types import LoadDataResult

from wisent_guard.cli.data_loaders.custom_user_data_loader import (
    load_train_test_custom_format as _custom_loader,
)
from wisent_guard.cli.data_loaders.lm_data_loader import (
        load_train_test_lm_eval_format as _lm_loader,
    )

if TYPE_CHECKING:  
    from wisent_guard.core.model import Model

__all__ = ["DataLoaderRotator", "RotatorConfig", "DataSource"]
_LOG = setup_logger(__name__)


class DataSource(str, Enum):
    AUTO = "auto"
    CUSTOM = "custom"
    LM_EVAL = "lm_eval"


@dataclass(frozen=True)
class RotatorConfig:
    """Configuration for :class:'DataLoaderRotator'.

    attributes:
        source:
            Choose data source. If 'auto', prefer 'custom' if 'custom_path' is set,
            otherwise use 'lm_eval'.
        custom_path:
            Path to custom JSONL for :func:'load_train_test_custom_format'.
        lm_tasks:
            A single task name or a list/tuple of task names usable by the model's :func:
            'load_lm_eval_task' (e.g., "winogrande" or ["winogrande", "hellaswag"]).
        split_ratio:
            Fraction for training split; defaults to '0.8' when 'None'.
        seed:
            Random seed for splitting.
        training_limit/testing_limit:
            Optional caps applied *after* the split.
        lm_limit:
            Optional per-root limit passed to the lm-eval loader.
    """

    source: DataSource = DataSource.AUTO
    custom_path: str | None = None
    lm_tasks: str | Sequence[str] | None = None
    split_ratio: float | None = None
    seed: int | None = None
    training_limit: int | None = None
    testing_limit: int | None = None
    lm_limit: int | None = None


class DataLoaderRotator:
    """Single entry point that swaps between 'custom' and 'lm_eval' data.

    Returns a consistent :class:'LoadDataResult' regardless of the source.
    """

    def __init__(self, model: Model | None = None):
        self._model = model

    def load(self, cfg: RotatorConfig) -> LoadDataResult:
        """Load data according to 'cfg' and normalize the output shape.

        arguments:
            cfg: 
                Configuration for data loading, :class:'RotatorConfig'. It specifies
                the source and parameters for loading. If 'source' is 'auto', it
                prefers 'custom' if 'custom_path' is set, otherwise uses 'lm_eval'.
        
        returns:
            A :class:'LoadDataResult' with training/test splits and metadata.

        raises:
            ValueError:
                If required fields are missing for the chosen source, or if
                parameters are out of range.
            TypeError:
                If the loader returns an unexpected type.
            RuntimeError:
                If the loader returns an unexpected shape.
        """
        src = self._decide_source(cfg)
        if src is DataSource.CUSTOM:
            return self._load_custom(cfg)
        return self._load_lm_eval(cfg)

    def _decide_source(self, cfg: RotatorConfig) -> DataSource:
        if cfg.source is not DataSource.AUTO:
            return cfg.source
        return DataSource.CUSTOM if cfg.custom_path else DataSource.LM_EVAL

    def _effective_split(self, split_ratio: float | None) -> float:
        if split_ratio is None:
            return 0.8
        if not (0.0 <= split_ratio <= 1.0):
            raise ValueError("split_ratio must be in [0.0, 1.0]")
        return float(split_ratio)

    def _ensure_nonneg(self, name: str, val: int | None) -> None:
        if val is not None and val < 0:
            raise ValueError(f"{name} must be non-negative")

    def _load_custom(self, cfg: RotatorConfig) -> LoadDataResult:
        if not cfg.custom_path:
            raise ValueError("custom_path is required for 'custom' source")
        
        split = self._effective_split(cfg.split_ratio)
        self._ensure_nonneg("training_limit", cfg.training_limit)
        self._ensure_nonneg("testing_limit", cfg.testing_limit)

    
        log = bind(_LOG, branch="custom")
        log.info("Loading custom data from %s", extra={"path": cfg.custom_path})

        raw = _custom_loader(
            path_to_custom_data=cfg.custom_path,
            split_ratio=split,
            seed=cfg.seed,
            training_limit=cfg.training_limit,
            testing_limit=cfg.testing_limit,
        )

        return self._validate_result(raw)

    def _load_lm_eval(self, cfg: RotatorConfig) -> LoadDataResult:
        if not self._model:
            raise ValueError("'model' is required for 'lm_eval' source")
        if not cfg.lm_tasks:
            raise ValueError("'lm_tasks' is required for 'lm_eval' source")

        split = self._effective_split(cfg.split_ratio)
        self._ensure_nonneg("training_limit", cfg.training_limit)
        self._ensure_nonneg("testing_limit", cfg.testing_limit)

        log = bind(_LOG, branch="lm_eval")
        log.info("Loading lm-eval tasks %s", extra={"tasks": cfg.lm_tasks})

        raw = _lm_loader(
            model=self._model,
            task_names=cfg.lm_tasks,
            split_ratio=split,
            limit=cfg.lm_limit,
            training_limit=cfg.training_limit,
            testing_limit=cfg.testing_limit,
            seed=cfg.seed or 0,
        )
        return self._validate_result(raw)

    def _validate_result(self, raw: Mapping[str, Any]) -> LoadDataResult:
        """Validate a canonical :type: 'LoadDataResult' from any loader (custom or lm_eval).
        """
        if not isinstance(raw, Mapping):
            raise TypeError(
            "Loader returned a non-mapping; expected LoadDataResult mapping."
            )


        required = {"train_qa_pairs", "test_qa_pairs", "task_type", "lm_task_data"}
        missing = [k for k in required if k not in raw]
        if missing:
            raise RuntimeError(
            "Loader returned an unexpected shape; missing keys: " + ", ".join(missing)
            )


        train = raw["train_qa_pairs"]
        test = raw["test_qa_pairs"]
        if not hasattr(train, "pairs") or not hasattr(test, "pairs"):
            raise TypeError(
            "'train_qa_pairs' and 'test_qa_pairs' must be ContrastivePairSet instances."
            )
        return raw 