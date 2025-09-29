from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict
import inspect

import torch

from wisent_guard.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName  
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet  

__all__ = [
    "SteeringError",
    "BaseSteeringMethod",
    "PerLayerBaseSteeringMethod",
]

class BaseSteeringError(RuntimeError):
    """Raised when a steering method fails or is misconfigured."""

class BaseSteeringMethod(ABC):
    name: str = "base"
    description: str = "Abstract steering method"
    _REGISTRY: dict[str, type[BaseSteeringMethod]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is BaseSteeringMethod:
            return
        if inspect.isabstract(cls):
            return
        if not getattr(cls, "name", None):
            raise TypeError("BaseSteeringMethod subclasses must define `name`.")
        if cls.name in BaseSteeringMethod._REGISTRY:
            raise ValueError(f"Duplicate steering method: {cls.name!r}")
        BaseSteeringMethod._REGISTRY[cls.name] = cls

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs: dict[str, Any] = dict(kwargs)

    @abstractmethod
    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """
        Produce per-layer vectors from the given contrastive set.
        
        arguments:
            pair_set: ContrastivePairSet with collected activations.
            
        returns:
            LayerActivations with one steering vector per layer.
        """
        ...

    @classmethod
    def list_registered(cls) -> dict[str, type[BaseSteeringMethod]]:
        """
        list all registered steering methods.

        returns:
            dict mapping method name to class.
        """
        return dict(cls._REGISTRY)

    @classmethod
    def get(cls, name: str) -> type[BaseSteeringMethod]:
        """
        Get a registered steering method class by name.

        arguments:
            name: str name of the steering method.
        
        returns:
            BaseSteeringMethod subclass.

        raises:
            SteeringError if name is unknown.
        """
        try:
            return cls._REGISTRY[name]
        except KeyError as exc:
            raise BaseSteeringError(f"Unknown steering method: {name!r}") from exc


class PerLayerBaseSteeringMethod(BaseSteeringMethod):
    """
    Base for steering methods that compute one vector per layer independently.
    Subclasses must implement 'train_for_layer'.
    """

    @abstractmethod
    def train_for_layer(self, pos_list: list[torch.Tensor], neg_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute a vector for ONE layer from lists of positives/negatives.
        
        arguments:
            pos_list: list of tensors from positive examples.
            neg_list: list of tensors from negative examples.
            
        returns:
            torch.Tensor steering vector for the layer.
        """
        ...

    def _collect_from_set(self, pair_set: ContrastivePairSet) -> dict[LayerName, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """
        Build {layer_name: ([pos tensors...], [neg tensors...])} by iterating pairs.
        Skips entries where activations are missing/None.

        arguments:
            pair_set: ContrastivePairSet with collected activations.

        returns:
            dict mapping layer names to tuples of (list of pos tensors, list of neg tensors).    
        """
        buckets: dict[LayerName, tuple[list[torch.Tensor], list[torch.Tensor]]] = defaultdict(lambda: ([], []))
        for pair in pair_set.pairs:  # ContrastivePair
            pos_la = getattr(pair.positive_response, "layers_activations", None)
            neg_la = getattr(pair.negative_response, "layers_activations", None)

            if pos_la is None or neg_la is None:
                continue

            layer_names = set(pos_la.to_dict().keys()) | set(neg_la.to_dict().keys())
            for layer in layer_names:
                p = pos_la.to_dict().get(layer, None) if pos_la is not None else None
                n = neg_la.to_dict().get(layer, None) if neg_la is not None else None
                if isinstance(p, torch.Tensor) and isinstance(n, torch.Tensor):
                    buckets[layer][0].append(p)
                    buckets[layer][1].append(n)
        return buckets

    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """
        Produce per-layer steering vectors from the given contrastive set.

        arguments:
            pair_set: ContrastivePairSet with collected activations.
        
        returns:
            LayerActivations with one steering vector per layer.
        """
        buckets = self._collect_from_set(pair_set)

        raw: RawActivationMap = {}
        for layer, (pos_list, neg_list) in sorted(buckets.items(), key=lambda kv: (len(kv[0]), kv[0])):
            if not pos_list or not neg_list:
                continue
            raw[layer] = self.train_for_layer(pos_list, neg_list)

        dtype = self.kwargs.get("dtype", None)
        agg = self.kwargs.get("activation_aggregation_strategy", None)
        return LayerActivations(raw, activation_aggregation_strategy=agg, dtype=dtype)