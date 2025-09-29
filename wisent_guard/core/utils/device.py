"""Centralized torch device selection helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch

DeviceKind = Literal["cuda", "mps", "cpu"]


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


@lru_cache(maxsize=1)
def resolve_default_device() -> DeviceKind:
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def resolve_torch_device() -> torch.device:
    return torch.device(resolve_default_device())


def resolve_device(kind: DeviceKind | None = None) -> torch.device:
    return torch.device(kind or resolve_default_device())


def preferred_dtype(kind: DeviceKind | None = None) -> torch.dtype:
    chosen = kind or resolve_default_device()
    return torch.float16 if chosen in {"cuda", "mps"} else torch.float32


def empty_device_cache(kind: DeviceKind | None = None) -> None:
    chosen = kind or resolve_default_device()
    if chosen == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif chosen == "mps" and _mps_available():
        try:
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except AttributeError:
            pass


def move_module_to_preferred_device(module: torch.nn.Module) -> torch.nn.Module:
    return module.to(resolve_torch_device())


def ensure_tensor_on_device(tensor: torch.Tensor) -> torch.Tensor:
    target = resolve_torch_device()
    return tensor.to(target) if tensor.device != target else tensor
