from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
from pathlib import Path
from typing import Any

from wisent_guard.core.classifiers.core.atoms import BaseClassifier, ClassifierError, ClassifierTrainReport

__all__ = ["ClassifierRotator"]

class ClassifierRotator:
    """
    Discover, list, and delegate to registered classifiers.
    """

    def __init__(
        self,
        classifier: str | BaseClassifier | type[BaseClassifier] | None = None,
        classifiers_location: str | Path = "wisent_guard.core.classifiers.models",
        autoload: bool = True,
        **classifier_kwargs: Any,
    ) -> None:
        if autoload:
            self.discover_classifiers(classifiers_location)
        self._classifier = self._resolve_classifier(classifier, **classifier_kwargs)

    @staticmethod
    def discover_classifiers(location: str | Path = "wisent_guard.core.classifiers.models") -> None:
        """
        Import all classifier modules so BaseClassifier subclasses self-register.

        - If `location` is a dotted module path (str without existing FS path),
          import that package and iterate its __path__ (works with namespace packages).
        - If `location` is an existing directory (Path/str), import all .py files inside.
        """
        loc_path = Path(str(location))
        if loc_path.exists() and loc_path.is_dir():
            ClassifierRotator._import_all_py_in_dir(loc_path)
            return

        if not isinstance(location, str):
            raise ClassifierError(
                f"Invalid classifiers location: {location!r}. Provide a dotted module path or a directory."
            )

        try:
            pkg = importlib.import_module(location)
        except ModuleNotFoundError as exc:
            raise ClassifierError(
                f"Cannot import classifier package {location!r}. "
                f"Use a dotted path (no leading slash) and ensure your project root is on PYTHONPATH."
            ) from exc

        search_paths = list(getattr(pkg, "__path__", []))
        if not search_paths:
            pkg_file = getattr(pkg, "__file__", None)
            if pkg_file:
                search_paths = [str(Path(pkg_file).parent)]

        for _finder, name, _ispkg in pkgutil.iter_modules(search_paths):
            if name.startswith("_"):
                continue
            importlib.import_module(f"{location}.{name}")

    @staticmethod
    def _import_all_py_in_dir(directory: Path) -> None:
        for py in directory.glob("*.py"):
            if py.name.startswith("_"):
                continue
            mod_name = f"_dyn_classifiers_{py.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, py)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[attr-defined]

    @staticmethod
    def list_classifiers() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, cls in BaseClassifier.list_registered().items():
            out.append(
                {
                    "name": name,
                    "description": getattr(cls, "description", ""),
                    "class": f"{cls.__module__}.{cls.__name__}",
                }
            )
        return sorted(out, key=lambda x: x["name"])

    @staticmethod
    def _resolve_classifier(
        classifier: str | BaseClassifier | type[BaseClassifier] | None,
        **kwargs: Any,
    ) -> BaseClassifier:
        if classifier is None:
            registry = BaseClassifier.list_registered()
            if not registry:
                raise ClassifierError("No classifiers registered.")
            # Deterministic pick: first by name
            return next(iter(sorted(registry.items())))[1](**kwargs)
        if isinstance(classifier, BaseClassifier):
            return classifier
        if inspect.isclass(classifier) and issubclass(classifier, BaseClassifier):
            return classifier(**kwargs)
        if isinstance(classifier, str):
            cls = BaseClassifier.get(classifier)
            return cls(**kwargs)
        raise TypeError(
            "classifier must be None, a name (str), BaseClassifier instance, or BaseClassifier subclass."
        )


    def use(self, classifier: str | BaseClassifier | type[BaseClassifier], **kwargs: Any) -> None:
        self._classifier = self._resolve_classifier(classifier, **kwargs)

    def fit(self, X, y, **kwargs) -> ClassifierTrainReport:
        return self._classifier.fit(X, y, **kwargs)

    def predict(self, X):
        return self._classifier.predict(X)

    def predict_proba(self, X):
        return self._classifier.predict_proba(X)

    def evaluate(self, X, y) -> dict[str, float]:
        return self._classifier.evaluate(X, y)

    def save_model(self, path: str) -> None:
        self._classifier.save_model(path)

    def load_model(self, path: str) -> None:
        self._classifier.load_model(path)

    def set_threshold(self, threshold: float) -> None:
        self._classifier.set_threshold(threshold)
