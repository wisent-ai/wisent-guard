from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Type
import inspect
import logging

from wisent_guard.core.evaluators.core.atoms import BaseEvaluator, EvalResult, EvaluatorError

logger = logging.getLogger(__name__)


class EvaluatorRotator:
    """Orchestrates evaluator selection and execution with flexible discovery."""

    def __init__(
        self,
        evaluator: Union[str, BaseEvaluator, Type[BaseEvaluator], None] = None,
        task_name: Optional[str] = None,
        evaluators_location: Union[str, Path] = "wisent_guard.core.evaluators.oracles",
        autoload: bool = True,
    ) -> None:
        if autoload:
            self.discover_evaluators(evaluators_location)
        self._evaluator = self._resolve_evaluator(evaluator)
        self._task_name = task_name

    @staticmethod
    def discover_evaluators(location: Union[str, Path] = "wisent_guard.core.evaluators.oracles") -> None:
        """
        Import all evaluator modules so BaseEvaluator subclasses self-register.

        - If `location` is a dotted module path (str without existing FS path),
          import that package and iterate its __path__ (works with namespace packages).
        - If `location` is an existing directory (Path/str), import all .py files inside.
        """

        loc_path = Path(str(location))
        if loc_path.exists() and loc_path.is_dir():
            EvaluatorRotator._import_all_py_in_dir(loc_path)
            return

        if not isinstance(location, str):
            raise EvaluatorError(
                f"Invalid evaluators location: {location!r}. Provide a dotted module path or a directory."
            )

        try:
            pkg = importlib.import_module(location)
        except ModuleNotFoundError as exc:
            raise EvaluatorError(
                f"Cannot import evaluator package {location!r}. "
                f"Use dotted path (no leading slash) and ensure your project root is on PYTHONPATH."
            ) from exc

        search_paths = list(getattr(pkg, "__path__", []))  # supports namespace pkgs
        if not search_paths:
            # Some packages may still have __file__ only
            pkg_file = getattr(pkg, "__file__", None)
            if pkg_file:
                search_paths = [str(Path(pkg_file).parent)]

        for finder, name, ispkg in pkgutil.iter_modules(search_paths):
            if name.startswith("_"):
                continue
            importlib.import_module(f"{location}.{name}")

    @staticmethod
    def _import_all_py_in_dir(directory: Path) -> None:
        for py in directory.glob("*.py"):
            if py.name.startswith("_"):
                continue
            mod_name = f"_dyn_evaluators_{py.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, py)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[attr-defined]

    @staticmethod
    def list_evaluators() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for name, cls in BaseEvaluator.list_registered().items():
            out.append(
                {
                    "name": name,
                    "description": getattr(cls, "description", ""),
                    "task_names": list(getattr(cls, "task_names", ())),
                    "class": f"{cls.__module__}.{cls.__name__}",
                }
            )
        return sorted(out, key=lambda x: x["name"])

    @staticmethod
    def _resolve_evaluator(
        evaluator: Union[str, BaseEvaluator, Type[BaseEvaluator], None]
    ) -> BaseEvaluator:
        if evaluator is None:
            registry = BaseEvaluator.list_registered()
            if "lm_eval" in registry:
                return registry["lm_eval"]()
            if registry:
                return next(iter(registry.values()))()
            raise EvaluatorError("No evaluators registered.")
        if isinstance(evaluator, BaseEvaluator):
            return evaluator
        if inspect.isclass(evaluator) and issubclass(evaluator, BaseEvaluator):
            return evaluator()
        if isinstance(evaluator, str):
            cls = BaseEvaluator.get(evaluator)
            return cls()
        raise TypeError(
            "evaluator must be None, a name (str), BaseEvaluator instance, or BaseEvaluator subclass."
        )

    def use(self, evaluator: Union[str, BaseEvaluator, Type[BaseEvaluator]]) -> None:
        self._evaluator = self._resolve_evaluator(evaluator)

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        kwargs.setdefault("task_name", self._task_name)
        return self._evaluator.evaluate(response, expected, **kwargs)

    def evaluate_batch(
        self, responses: Sequence[str], expected_answers: Sequence[Any], **kwargs
    ) -> List[EvalResult]:
        kwargs.setdefault("task_name", self._task_name)
        return self._evaluator.evaluate_batch(responses, expected_answers, **kwargs)


if __name__ == "__main__":
    from evaluator_rotator import EvaluatorRotator

    rot = EvaluatorRotator(
    evaluators_location="wisent_guard.core.evaluators.oracles",  # << no leading slash
    autoload=True,
    )

    rot.list_evaluators()
    print("Available evaluators:")
    for ev in rot.list_evaluators():
        print(f" - {ev['name']}: {ev['description']} (tasks: {', '.join(ev['task_names'])})")

    # rot.use("nlp")
    # res = rot.evaluate("The answer is probably 42", expected="The answer is 12")

    # print(res)
