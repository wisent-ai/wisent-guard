from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Protocol, Union, runtime_checkable

from wisent_guard.core.contrastive_pairs import ContrastivePairSet
from wisent_guard.core.steering_methods.base import SteeringMethod

from wisent_guard.cli_bricks.cli_logger import setup_logger, bind

_LOG = setup_logger(__name__)

__all__ = [
    "CAAConfig",
    "HPRConfig",
    "DACConfig",
    "BiPOConfig",
    "KSteeringConfig",
    "SteeringConfig",
    "build_steering",
    "build_steering_for_mode", 
]


@runtime_checkable
class _Trainable(Protocol):
    def train(self, pair_set: ContrastivePairSet, layer_idx: int) -> None: ...

@runtime_checkable
class _SaveLoadVector(Protocol):
    def save_steering_vector(self, filepath: str, layer_idx: int) -> None: ...
    def load_steering_vector(self, filepath: str) -> None: ...

@dataclass(frozen=True)
class _BaseConfig:
    method: Literal["CAA", "HPR", "DAC", "BiPO", "KSteering"]
    device: Optional[str] = None
    save_path: Optional[Path] = None
    load_path: Optional[Path] = None

@dataclass(frozen=True)
class CAAConfig(_BaseConfig):
    method: Literal["CAA"] = "CAA"
    normalization_method: str = "none"
    target_norm: Optional[float] = None

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
    target_labels: List[int] = None 
    avoid_labels: List[int] = None   
    alpha: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "target_labels", list(self.target_labels or []))
        object.__setattr__(self, "avoid_labels", list(self.avoid_labels or []))

SteeringConfig = Union[CAAConfig, HPRConfig, DACConfig, BiPOConfig, KSteeringConfig]

def _coerce_normalization(method: str):
    """Create a VectorNormalizationMethod from a string; fallback to NONE."""
    from wisent_guard.core.normalization import VectorNormalizationMethod
    logger = bind(_LOG, subtask="norm")
    try:
        return VectorNormalizationMethod(method)
    except Exception:
        logger.warning("Unknown normalization method; defaulting to NONE", extra={"provided": method})
        return VectorNormalizationMethod.NONE

def _parse_label_csv(csv: Optional[str], *, field: str) -> List[int]:
    """Parse '1,2,3' -> [1,2,3]."""
    logger = bind(_LOG, subtask="labels")
    if not csv:
        return []
    out: List[int] = []
    for tok in csv.split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except ValueError as e:
            logger.error("Non-integer label in CSV", extra={"field": field, "token": s})
            raise ValueError(f"Invalid integer in {field!r}: {s!r}") from e
    return out


def build_steering(
    config: SteeringConfig,
    layer_idx: int,
    pair_set: Optional[ContrastivePairSet] = None,
    verbose: bool = False,
    train_if_missing: bool = True,
) -> SteeringMethod:
    """
    Build a steering method with **load-first → train-if-needed → optional save** flow.

    1) If 'config.load_path' is set and the object supports loading, attempt to load.
       - On success, skip training and return the object.
       - On failure and 'train_if_missing=True', fall back to training.
    2) If training is performed (and supported), optionally save to 'config.save_path'.

    Args:
        config: Method-specific configuration, for example CAAConfig.
        layer_idx: Target layer index for training/saving.
        pair_set: Required if training is needed and supported.
        verbose: Emit info logs when True.
        train_if_missing: If False, never train (even if load fails).

    Returns:
        An instance of the configured SteeringMethod.
    """
    from wisent_guard.core.steering_method import CAA, DAC, HPR, BiPO, KSteering

    logger = bind(_LOG, subtask="build_steering", method=config.method, layer_idx=layer_idx)

    if isinstance(config, CAAConfig):
        norm = _coerce_normalization(config.normalization_method)
        obj: Any = CAA(device=config.device, normalization_method=norm, target_norm=config.target_norm)
    elif isinstance(config, HPRConfig):
        obj = HPR(device=config.device)
    elif isinstance(config, DACConfig):
        obj = DAC(device=config.device, dynamic_control=config.dynamic_control, entropy_threshold=config.entropy_threshold)
    elif isinstance(config, BiPOConfig):
        obj = BiPO(device=config.device, beta=config.beta, learning_rate=config.learning_rate, num_epochs=config.num_epochs)
    elif isinstance(config, KSteeringConfig):
        obj = KSteering(
            device=config.device,
            num_labels=config.num_labels,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            classifier_epochs=config.classifier_epochs,
            target_labels=config.target_labels,
            avoid_labels=config.avoid_labels,
            alpha=config.alpha,
        )
        if verbose and not (config.target_labels or config.avoid_labels):
            logger.warning("KSteering targets and avoid lists are both empty.")
    else:
        raise ValueError(f"Unsupported config type: {type(config).__name__}")

    loaded = False
    if isinstance(obj, _SaveLoadVector) and config.load_path:
        path = str(config.load_path)
        try:
            if verbose:
                logger.info("Loading steering vector", extra={"path": path})
            obj.load_steering_vector(path)
            loaded = True
            if verbose:
                logger.info("Load successful", extra={"path": path})
        except Exception as e:
            logger.warning("Load failed; may train if allowed", extra={"path": path, "error": str(e)})

    if not loaded and train_if_missing and isinstance(obj, _Trainable):
        if pair_set is None:
            logger.error("Training requested but `pair_set` is None.")
            raise ValueError("pair_set is required for training when load is absent/failed.")
        if verbose:
            logger.info("Training steering method...", extra={"num_pairs": len(pair_set.pairs)})
        obj.train(pair_set, layer_idx)
        if verbose:
            logger.info("Training complete.")

        if isinstance(obj, _SaveLoadVector) and config.save_path:
            save_path = str(config.save_path)
            try:
                if verbose:
                    logger.info("Saving steering vector", extra={"path": save_path})
                obj.save_steering_vector(save_path, layer_idx)
            except Exception as e:
                logger.error("Save failed", extra={"path": save_path, "error": str(e)})
                raise

    return obj


def build_steering_for_mode(
    method_name: str,
    device: Optional[str],
    normalization_method: str,
    target_norm: Optional[float],
    dac_dynamic_control: bool,
    dac_entropy_threshold: float,
    bipo_beta: float,
    bipo_lr: float,
    bipo_epochs: int,
    k_num_labels: int,
    k_hidden: int,
    k_lr: float,
    k_epochs: int,
    k_target: str,
    k_avoid: str,
    k_alpha: float,
    save_path: Optional[str],
    load_path: Optional[str],
    layer_idx: int,
    pair_set: ContrastivePairSet,
    verbose: bool,
) -> SteeringMethod:
    """
    Legacy entry compatible with previous signature.
    Internally builds a config and delegates to 'build_steering' with load-first flow.
    """
    # Build config object based on method
    m = (method_name or "").strip().upper()
    cfg: SteeringConfig
    if m == "CAA":
        cfg = CAAConfig(device=device, normalization_method=normalization_method, target_norm=target_norm,
                        save_path=Path(save_path) if save_path else None,
                        load_path=Path(load_path) if load_path else None)
    elif m == "HPR":
        cfg = HPRConfig(device=device,
                        save_path=Path(save_path) if save_path else None,
                        load_path=Path(load_path) if load_path else None)
    elif m == "DAC":
        cfg = DACConfig(device=device, dynamic_control=dac_dynamic_control, entropy_threshold=dac_entropy_threshold,
                        save_path=Path(save_path) if save_path else None,
                        load_path=Path(load_path) if load_path else None)
    elif m == "BIPO":
        cfg = BiPOConfig(device=device, beta=bipo_beta, learning_rate=bipo_lr, num_epochs=bipo_epochs,
                         save_path=Path(save_path) if save_path else None,
                         load_path=Path(load_path) if load_path else None)
    elif m == "KSTEERING":
        cfg = KSteeringConfig(
            device=device,
            num_labels=k_num_labels,
            hidden_dim=k_hidden,
            learning_rate=k_lr,
            classifier_epochs=k_epochs,
            target_labels=_parse_label_csv(k_target, field="k_target"),
            avoid_labels=_parse_label_csv(k_avoid, field="k_avoid"),
            alpha=k_alpha,
            save_path=Path(save_path) if save_path else None,
            load_path=Path(load_path) if load_path else None,
        )
    else:
        raise ValueError(f"Unknown steering method: {method_name}")

    return build_steering(cfg, layer_idx=layer_idx, pair_set=pair_set, verbose=verbose, train_if_missing=True)