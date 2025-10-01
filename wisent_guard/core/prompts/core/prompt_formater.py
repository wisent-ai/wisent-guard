from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Iterable, Type

from wisent_guard.core.prompts.core.atom import PromptPair, PromptStrategy, UnknownStrategyError

__all__ = ["PromptFormatter", "StrategyKey", "UnknownStrategyError"]

StrategyKey = str | type[PromptStrategy]


class PromptFormatter:
    """Main entry point for building prompt pairs via discovered strategies.

    atributes:
        _registry: Maps strategy_key to PromptStrategy subclass.
    
    methods:
        format: 
            Build a PromptPair using a discovered strategy.
        available: 
            Return the available strategy keys.
        refresh: 
            Rescan the 'strategies' package for newly added strategy files.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Type[PromptStrategy]] = {}
        self._discover_and_load()

    def format(
        self,
        strategy: StrategyKey,
        question: str,
        correct_answer: str,
        incorrect_answer: str,
    ) -> PromptPair:
        """Build a PromptPair using a discovered strategy.
        
        arguments:
            strategy: The strategy key (string) or PromptStrategy subclass.
            question: The question text.
            correct_answer: The correct answer text.
            incorrect_answer: The incorrect answer text.
        
        returns:
            A PromptPair constructed by the specified strategy.

        raises:
            UnknownStrategyError: If the strategy key is not found.
            TypeError: If the strategy argument is not a string or PromptStrategy subclass.

        example:
            >>> formatter = PromptFormatter()
            >>> pair = formatter.format(
            ...     strategy="multiple_choice",
            ...     question="What is 2+2?",
            ...     correct_answer="4",
            ...     incorrect_answer="5"
            ... )
            >>> print(pair)
            PromptPair(
                positive=[{'role': 'user', 'content': 'Which is better: What is 2+2? A. 5 B. 4'}, {'role': 'assistant', 'content': 'B'}],
                negative=[{'role': 'user', 'content': 'Which is better: What is 2+2? A. 5 B. 4'}, {'role': 'assistant', 'content': 'A'}])
        """
        key = self._normalize_key(strategy)
        try:
            strategy_cls = self._registry[key]
        except KeyError as exc:
            raise UnknownStrategyError(f"Unknown strategy: {strategy!r}") from exc

        pair = strategy_cls().build(question, correct_answer, incorrect_answer)
        return pair

    def available(self) -> Iterable[str]:
        """Return the available strategy keys.

        returns:
            An iterable of available strategy keys.
        
        example:
            >>> formatter = PromptFormatter()
            >>> print(formatter.available())
            ('multiple_choice', 'role_playing', 'instruction_following')
        """
        return tuple(sorted(self._registry.keys()))

    def refresh(self) -> None:
        """Rescan the 'strategies' package for newly added strategy files."""
        self._registry.clear()
        self._discover_and_load()

    @staticmethod
    def _normalize_key(key: StrategyKey) -> str:
        if isinstance(key, str):
            return key
        if inspect.isclass(key) and issubclass(key, PromptStrategy):
            return key.strategy_key
        raise TypeError(
            "strategy must be a string key or a PromptStrategy subclass."
        )

    def _discover_and_load(self) -> None:
        """Import all modules in the 'strategies' package and collect strategies.
        
        raises:
            RuntimeError: If the 'strategies' package is not found or no strategies are discovered.
        """
        try:
            import wisent_guard.core.prompts.prompt_stratiegies as strategies_pkg
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The 'strategies' package was not found. "
                "Create a 'strategies' directory with an empty __init__.py."
            ) from exc

        import wisent_guard.core.prompts.prompt_stratiegies as strategies_pkg

        for module_info in pkgutil.iter_modules(strategies_pkg.__path__):
            name = module_info.name
            if name.startswith("_"):
                # Skip private/dunder modules.
                continue

            module = importlib.import_module(f"strategies.{name}")
            self._register_strategies_from_module(module)

        if not self._registry:
            raise RuntimeError(
                "No strategies found. Add at least one file in 'strategies/' "
                "defining a PromptStrategy subclass with a unique 'strategy_key'."
            )

    def _register_strategies_from_module(self, module) -> None:
        '''
        Inspect the given module for PromptStrategy subclasses and register them.
        
        arguments:
            module: The imported module to inspect.
        
        raises:
            RuntimeError: If duplicate strategy_key values are found.
        '''
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, PromptStrategy) or obj is PromptStrategy:
                continue
            key = obj.strategy_key
            if key in self._registry:
                existing = self._registry[key].__module__
                raise RuntimeError(
                    f"Duplicate strategy_key '{key}' found in module "
                    f"'{module.__name__}' (already defined in '{existing}')."
                )
            self._registry[key] = obj
