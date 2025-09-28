from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from wisent_guard.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult

class DataLoaderRotator:
    """Discover/select a data loader and use it to load data."""
    def __init__(
        self,
        loader: Union[str, BaseDataLoader, Type[BaseDataLoader], None] = None,
        loaders_location: Union[str, Path] = "wisent_guard.core.data_loaders.loaders",
        autoload: bool = True,
        **default_loader_kwargs: Any,
    ) -> None:
        self._scope_prefix = (
            loaders_location if isinstance(loaders_location, str)
            else Path(loaders_location).as_posix().replace("/", ".")
        )
        if autoload:
            self.discover_loaders(loaders_location)
        self._loader = self._resolve_loader(loader, **default_loader_kwargs)

    @staticmethod
    def discover_loaders(location: Union[str, Path]) -> None:
        loc_path = Path(str(location))
        if loc_path.exists() and loc_path.is_dir():
            for py in loc_path.glob("*.py"):
                if py.name.startswith("_"):
                    continue
                mod_name = f"_dyn_dataloaders_{py.stem}"
                spec = importlib.util.spec_from_file_location(mod_name, py)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return

        if not isinstance(location, str):
            raise DataLoaderError(f"Invalid loaders location: {location!r}. Provide dotted path or a directory.")

        pkg = importlib.import_module(location)
        search_paths = list(getattr(pkg, "__path__", [])) or [Path(getattr(pkg, "__file__", "")).parent.as_posix()]
        for _, name, _ in pkgutil.iter_modules(search_paths):
            if name.startswith("_"):
                continue
            importlib.import_module(f"{location}.{name}")

    def _scoped_registry(self) -> dict[str, type[BaseDataLoader]]:
        reg = BaseDataLoader.list_registered()
        return {n: c for n, c in reg.items() if c.__module__.startswith(self._scope_prefix)}

    @staticmethod
    def list_loaders(scope_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        reg = BaseDataLoader.list_registered()
        if scope_prefix:
            reg = {n: c for n, c in reg.items() if c.__module__.startswith(scope_prefix)}
        return [
            {"name": n, "description": getattr(c, "description", ""), "class": f"{c.__module__}.{c.__name__}"}
            for n, c in sorted(reg.items(), key=lambda kv: kv[0])
        ]

    def _resolve_loader(
        self,
        loader: Union[str, BaseDataLoader, Type[BaseDataLoader], None],
        **kwargs: Any,
    ) -> BaseDataLoader:
        reg = self._scoped_registry()
        if loader is None:
            if not reg:
                raise DataLoaderError(f"No data loaders registered under {self._scope_prefix!r}.")
            cls = next(iter(sorted(reg.items(), key=lambda kv: kv[0])))[1]
            return cls(**kwargs)
        if isinstance(loader, BaseDataLoader):
            loader.kwargs = {**kwargs, **loader.kwargs}
            return loader
        if inspect.isclass(loader) and issubclass(loader, BaseDataLoader):
            if not loader.__module__.startswith(self._scope_prefix):
                raise DataLoaderError(f"Loader class must live under {self._scope_prefix!r}.")
            return loader(**kwargs)
        if isinstance(loader, str):
            if loader not in reg:
                raise DataLoaderError(f"Unknown loader {loader!r} in scope {self._scope_prefix!r}.")
            return reg[loader](**kwargs)
        raise TypeError("loader must be None, a name (str), BaseDataLoader instance, or BaseDataLoader subclass.")

    def use(self, loader: Union[str, BaseDataLoader, Type[BaseDataLoader]], **kwargs: Any) -> None:
        self._loader = self._resolve_loader(loader, **kwargs)

    def load(self, **kwargs: Any) -> LoadDataResult:
        merged = {**getattr(self._loader, "kwargs", {}), **kwargs}
        return self._loader.load(**merged)
