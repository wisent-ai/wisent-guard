from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol, TYPE_CHECKING, runtime_checkable

from typing import cast

from wisent_guard.cli_bricks.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from wisent_guard.core.contrastive_pairs import ContrastivePairSet
    from wisent_guard.core.steering_methods.base import SteeringMethod

_LOG = setup_logger(__name__)

__all__: Final[list[str]] = [
    "CAAConfig",
    "HPRConfig",
    "DACConfig",
    "BiPOConfig",
    "KSteeringConfig",
    "SteeringConfig",
    "build_steering",
]


@runtime_checkable
class _Trainable(Protocol):
    def train(self, pair_set: ContrastivePairSet, layer_idx: int) -> None: ...

@dataclass(frozen=True)
class _BaseConfig:
    method: Literal["CAA", "HPR", "DAC", "BiPO", "KSteering"]
    device: str | None = None
    save_path: Path | None = None
    load_path: Path | None = None


@dataclass(frozen=True)
class CAAConfig(_BaseConfig):
    method: Literal["CAA"] = "CAA"
    normalization_method: str = "none"
    target_norm: float | None = None


@dataclass(frozen=True)
class HPRConfig(_BaseConfig):
    method: Literal["HPR"] = "HPR"


@dataclass(frozen=True)
class DACConfig(_BaseConfig):
    method: Literal["DAC"] = "DAC"
    dynamic_control: bool = False
    entropy_threshold: float = 0.0


@dataclass(frozen=True)
class BiPOConfig(_BaseConfig):
    method: Literal["BiPO"] = "BiPO"
    beta: float = 1.0
    learning_rate: float = 1e-3
    num_epochs: int = 5


@dataclass(frozen=True)
class KSteeringConfig(_BaseConfig):
    method: Literal["KSteering"] = "KSteering"
    num_labels: int = 2
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    classifier_epochs: int = 5
    target_labels: tuple[int, ...] = field(default_factory=tuple)
    avoid_labels: tuple[int, ...] = field(default_factory=tuple)
    alpha: float = 0.5

type SteeringConfig = CAAConfig | HPRConfig | DACConfig | BiPOConfig | KSteeringConfig

def _coerce_normalization(method: str) -> "VectorNormalizationMethod": # type: ignore
    """Create a VectorNormalizationMethod from a string; fallback to NONE.

    Arguments:
      - method: String name of the normalization method. For example, "none" or "min_max".

    Returns:
        - A VectorNormalizationMethod enum value.

    Raises:
        - ValueError if the method is unknown.
    """
    from wisent_guard.core.normalization import VectorNormalizationMethod

    logger = bind(_LOG, subtask="norm")
    try:
        return VectorNormalizationMethod(method)
    except Exception:
        logger.warning(
            "Unknown normalization method; defaulting to NONE",
            extra={"provided": method},
        )
        return VectorNormalizationMethod.NONE


def build_steering(
    config: SteeringConfig,
    layer_idx: int,
    pair_set: ContrastivePairSet,
    verbose: bool = False,
) -> SteeringMethod:
    """
    Build a steering method and ALWAYS train it.

    Flow:
      1) Instantiate the method from `config`.
      2) Validate that the instance is trainable and `pair_set` has data.
      3) Train and return the ready-to-use instance.

    Args:
        config: A SteeringConfig (e.g., CAAConfig, DACConfig, ...).
        layer_idx: Model layer index where steering is applied (must be >= 0).
        pair_set: Contrastive pairs, must contain at least one pair.
        verbose: If True, emit more detailed logs.

    Returns:
        A trained SteeringMethod instance.

    Raises:
        ValueError: On invalid 'layer_idx', empty 'pair_set', or unsupported config type.
        TypeError: If the chosen method does not support training.
    """
    if layer_idx < 0:
        raise ValueError(f"layer_idx must be >= 0, got {layer_idx}")

    if not getattr(pair_set, "pairs", None):
        raise ValueError("`pair_set.pairs` must contain at least one pair.")

    logger = bind(
        _LOG, subtask="build_steering", method=getattr(config, "method", None), layer_idx=layer_idx
    )

    from wisent_guard.core.steering_method import CAA, DAC, HPR, BiPO, KSteering

    if isinstance(config, CAAConfig):
        norm = _coerce_normalization(config.normalization_method)
        obj: SteeringMethod = CAA(
            device=config.device,
            normalization_method=norm,
            target_norm=config.target_norm,
        )
    elif isinstance(config, HPRConfig):
        obj = HPR(device=config.device)
    elif isinstance(config, DACConfig):
        obj = DAC(
            device=config.device,
            dynamic_control=config.dynamic_control,
            entropy_threshold=config.entropy_threshold,
        )
    elif isinstance(config, BiPOConfig):
        obj = BiPO(
            device=config.device,
            beta=config.beta,
            learning_rate=config.learning_rate,
            num_epochs=config.num_epochs,
        )
    elif isinstance(config, KSteeringConfig):
        obj = KSteering(
            device=config.device,
            num_labels=config.num_labels,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            classifier_epochs=config.classifier_epochs,
            target_labels=list(config.target_labels),  
            avoid_labels=list(config.avoid_labels),
            alpha=config.alpha,
        )
        if verbose and not (config.target_labels or config.avoid_labels):
            logger.warning("KSteering targets and avoid lists are both empty.")
    else:
        raise ValueError(f"Unsupported config type: {type(config).__name__}")

    if not isinstance(obj, _Trainable):
        raise TypeError(
            f"{obj.__class__.__name__} does not support training, "
            "but this builder requires training."
        )
    
    logger.debug("Steering method instantiated", extra={"class": obj.__class__.__name__})

    obj.train(contrastive_pair_set=pair_set, layer_index=layer_idx)

    if verbose:
        logger.info("Training steering method...", extra={"num_pairs": len(pair_set.pairs)})

    
    if verbose:
        logger.info("Training complete.")

    return obj