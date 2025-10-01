from __future__ import annotations

from enum import StrEnum, auto, unique
from typing import Mapping, Iterator, TypeAlias
import numpy as np
import torch

__all__ = ["LayerActivations", "ActivationAggregationStrategy", "ActivationCollector", "LayerName", "LayerActivation", "ActivationMap", "RawActivationMap"]

LayerName: TypeAlias = str
LayerActivation: TypeAlias = torch.Tensor | None
ActivationMap: TypeAlias = Mapping[LayerName, LayerActivation]
RawActivationMap: TypeAlias = Mapping[LayerName, torch.Tensor | np.ndarray | None]

class _LowerSnakeStrEnum(StrEnum):
    """StrEnum whose auto() values are lower_snake_case of the member name."""
    def _generate_next_value_(name, start, count, last_values): # type: ignore
        return name.lower()

@unique
class ActivationAggregationStrategy(_LowerSnakeStrEnum):
    """Strategies for selecting/aggregating tokens in activation extraction.
    """

    CHOICE_TOKEN = auto()         # target A/B choice tokens (multiple choice)
    CONTINUATION_TOKEN = auto()   # first token of the continuation
    LAST_TOKEN = auto()           # always use the last token
    FIRST_TOKEN = auto()          # always use the first token
    MEAN_POOLING = auto()         # mean over all tokens
    MAX_POOLING = auto()          # max over all tokens

    @property
    def description(self) -> str:
        return {
            ActivationAggregationStrategy.CHOICE_TOKEN: "Target A/B choice tokens (multiple choice).",
            ActivationAggregationStrategy.CONTINUATION_TOKEN: "Use the first token of the continuation.",
            ActivationAggregationStrategy.LAST_TOKEN: "Always select the last token.",
            ActivationAggregationStrategy.FIRST_TOKEN: "Always select the first token.",
            ActivationAggregationStrategy.MEAN_POOLING: "Aggregate by mean over all tokens.",
            ActivationAggregationStrategy.MAX_POOLING: "Aggregate by max over all tokens.",
        }[self]


class LayerActivations(Mapping[LayerName, LayerActivation]):
    """Immutable mapping of layer names to activations.

    Behaves like: 'Mapping[str, torch.Tensor | None]'.

    construction:
        'LayerActivations(data: Mapping[str, torch.Tensor | np.ndarray | None] | None, *, dtype: torch.dtype | None = None)'

        - 'torch.Tensor' values are kept as-is (or cast to 'dtype' if given).
        - 'np.ndarray' values are converted via 'torch.from_numpy' (then cast if needed).
        - 'None' values are preserved.
        -  Missing/empty input yields an empty container.

    atributes:
        _data:
            internal storage dict. It contains information about layer activations.
        _strategy:
            'ActivationAggregationStrategy' (see below). Indicates how activations were aggregated if applicable.

    methods:
        'summary()':
            dict with per-layer shape/dtype/device/requires_grad.
        'to(*args, **kwargs)':
            apply 'Tensor.to' to all non-'None' values.
        'cpu()', 'detach()':
            convenience operations.
        'numpy()':
            map tensors to cpu NumPy arrays (others to 'None').
        'to_dict()':
            plain dict (useful for (de)serialization).

    examples:
        >>> acts = LayerActivations({"layer1": torch.randn(2, 10, 768), "layer2": None}, activation_aggregation_strategy="mean_pooling")
        >>> acts["layer1"].shape
        torch.Size([2, 10, 768])
        >>> acts["layer2"] is None
        True
        >>> acts.activation_aggregation_strategy
        <ActivationAggregationStrategy.MEAN_POOLING: 'mean_pooling'>
        >>> acts.summary()
        {'layer1': {'shape': (2, 10, 768), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False}, 'layer2': {'shape': None, 'dtype': None, 'device': None, 'requires_grad': None}}
        >>> acts.numpy()
        {'layer1': array(...), 'layer2': None}
        >>> acts.to("cuda")
        LayerActivations(
          layer1: Tensor(shape=(2, 10, 768), dtype=torch.float32, device=cuda:0)
          layer2: None
        )
        >>> acts.detach()  # if any tensor required grad
        LayerActivations(
          layer1: Tensor(shape=(2, 10, 768), dtype=torch.float32, device=cpu)
          layer2: None
        )

    notes:
        - Use 'summary()' or 'numpy()' if you need JSON-serializable content.
        - Keys are strings by convention; enforced by type hints.
    """
    __slots__ = ("_data", "_strategy")

    def __init__(self, data: RawActivationMap | None = None, activation_aggregation_strategy: ActivationAggregationStrategy | None = None, dtype: torch.dtype | None = None):
        store: dict[LayerName, LayerActivation] = {}
        if data:
            for layer, val in data.items():
                if val is None:
                    store[layer] = None
                elif isinstance(val, torch.Tensor):
                    store[layer] = val if dtype is None else val.to(dtype)
                elif isinstance(val, np.ndarray):
                    t = torch.from_numpy(val)
                    store[layer] = t if dtype is None else t.to(dtype)
                else:
                    raise TypeError(
                        f"Activations for layer '{layer}' must be torch.Tensor, np.ndarray, or None."
                    )
        self._data = store
        self._strategy = self._normalize_strategy(activation_aggregation_strategy)

    @staticmethod
    def _normalize_strategy(
        s: ActivationAggregationStrategy | str | None
    ) -> ActivationAggregationStrategy | None:
        if s is None:
            return None
        if isinstance(s, ActivationAggregationStrategy):
            return s
        if isinstance(s, str):
            try:
                return ActivationAggregationStrategy(s)
            except ValueError:
                valid = ", ".join([e.value for e in ActivationAggregationStrategy])
                raise ValueError(
                    f"Unknown activation_agregation_strategy='{s}'. "
                    f"Valid options: {valid}"
                )
        raise TypeError(
            "activation_agregation_strategy must be ActivationAggregationStrategy | str | None"
        )
    
    @property
    def activation_aggregation_strategy(self) -> ActivationAggregationStrategy | None:
        return self._strategy

    def __getitem__(self, key: LayerName) -> LayerActivation:
        return self._data[key]
    def __iter__(self) -> Iterator[LayerName]:
        return iter(self._data)
    def __len__(self) -> int:
        return len(self._data)

    def summary(self) -> dict[LayerName, dict[str, tuple | str | bool | None]]:
        ''' Return a summary of the activations. For each layer, provides
        shape, dtype, device, requires_grad status, and aggregation strategy.
        '''
        out: dict[LayerName, dict[str, dict[str, tuple | str | bool | None]]] = {}
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                out[k] = {
                    "shape": tuple(v.shape),
                    "dtype": str(v.dtype),
                    "device": str(v.device),
                    "requires_grad": bool(v.requires_grad),
                }
            else:
                out[k] = {"shape": None, "dtype": None, "device": None, "requires_grad": None}

        out["_activation_aggregation_strategy"] = {"strategy": self._strategy.value if self._strategy else None}
        return out

    def numpy(self) -> dict[LayerName, np.ndarray | None]:
        return {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else None)
                for k, v in self._data.items()}

    def to_dict(self) -> dict[LayerName, LayerActivation]:
        return dict(self._data)

    def to(self, *args, **kwargs) -> LayerActivations:
        return LayerActivations({k: (v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else None)
                                 for k, v in self._data.items()})

    def detach(self) -> LayerActivations:
        return LayerActivations({k: (v.detach() if isinstance(v, torch.Tensor) else None)
                                 for k, v in self._data.items()})

    def cpu(self) -> LayerActivations:
        return self.to("cpu")

    def __repr__(self) -> str:
        lines = ["LayerActivations("]
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                lines.append(
                    f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
                )
            else:
                lines.append(f"  {k}: None")
        lines.append(")")
        lines.append(f"  _activation_aggregation_strategy: {self._strategy.value if self._strategy else None}")

        return "\n".join(lines)