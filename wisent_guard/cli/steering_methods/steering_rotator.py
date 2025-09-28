from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any, Type

from wisent_guard.core.steering_methods.core.atoms import BaseSteeringError, BaseSteeringMethod
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet

from wisent_guard.core.activations.core.atoms import LayerActivations

__all__ = [
    "SteeringMethodRotator",
]

logger = logging.getLogger(__name__)

class SteeringMethodRotator:
    """Discover/select a steering method and train it on a ContrastivePairSet."""

    def __init__(
        self,
        method: str | BaseSteeringMethod | Type[BaseSteeringMethod] | None = None,
        methods_location: str | Path = "wisent_guard.core.steering_methods.methods",
        autoload: bool = True,
        **default_method_kwargs: Any,
    ) -> None:
        if autoload:
            self.discover_methods(methods_location)
        self._method = self._resolve_method(method, **default_method_kwargs)

    @staticmethod
    def discover_methods(location: str | Path) -> None:
        loc_path = Path(str(location))
        if loc_path.exists() and loc_path.is_dir():
            for py in loc_path.glob("*.py"):
                if py.name.startswith("_"):
                    continue
                mod_name = f"_dyn_steering_{py.stem}"
                spec = importlib.util.spec_from_file_location(mod_name, py)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  
            return

        if not isinstance(location, str):
            raise BaseSteeringError(f"Invalid methods location: {location!r}")

        try:
            pkg = importlib.import_module(location)
        except ModuleNotFoundError as exc:
            raise BaseSteeringError(f"Cannot import steering package {location!r}.") from exc

        search_paths = list(getattr(pkg, "__path__", [])) or [Path(getattr(pkg, "__file__", "")).parent.as_posix()]
        for _, name, _ in pkgutil.iter_modules(search_paths):
            if name.startswith("_"):
                continue
            importlib.import_module(f"{location}.{name}")

    @staticmethod
    def list_methods() -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": getattr(cls, "description", ""),
                "class": f"{cls.__module__}.{cls.__name__}",
            }
            for name, cls in sorted(BaseSteeringMethod.list_registered().items(), key=lambda kv: kv[0])
        ]

    @staticmethod
    def _resolve_method(
        method: str | BaseSteeringMethod | Type[BaseSteeringMethod] | None,
        **kwargs: Any,
    ) -> BaseSteeringMethod:
        if method is None:
            reg = BaseSteeringMethod.list_registered()
            if not reg:
                raise BaseSteeringError("No steering methods registered.")
            first = next(iter(sorted(reg.items(), key=lambda kv: kv[0])))[1]
            return first(**kwargs)
        if isinstance(method, BaseSteeringMethod):
            method.kwargs = {**kwargs, **method.kwargs}
            return method
        if inspect.isclass(method) and issubclass(method, BaseSteeringMethod):
            return method(**kwargs)
        if isinstance(method, str):
            return BaseSteeringMethod.get(method)(**kwargs)
        raise TypeError("method must be None, str name, BaseSteeringMethod instance, or subclass.")
    
    def use(self, method: str | BaseSteeringMethod | Type[BaseSteeringMethod], **kwargs: Any) -> None:
        self._method = self._resolve_method(method, **kwargs)

    def train(self, pair_set: ContrastivePairSet, **overrides: Any) -> LayerActivations:
        old = dict(self._method.kwargs)
        try:
            self._method.kwargs = {**old, **overrides}
            return self._method.train(pair_set)
        finally:
            self._method.kwargs = old

if __name__ == "__main__":
    rot = SteeringMethodRotator()
    print("Available steering methods:")
    for m in rot.list_methods():
        print(f" - {m['name']}: {m['description']} ({m['class']})")