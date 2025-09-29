"""Serialization helpers for contrastive pair sets with safe tensor/array storage."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet

__all__ = [
    "save_contrastive_pair_set",
    "load_contrastive_pair_set",
]


class VectorPayload(dict[str, bool | str | list[int]]):
    """A dictionary with metadata and base64-encoded binary data for a tensor/array."""
    __array__: bool
    backend: str
    dtype: str
    shape: list[int]
    data: str

def _encode_activations(x: torch.Tensor | np.ndarray | None) -> VectorPayload | None:
    """Return a JSON-serializable object.
    If x is a torch.Tensor or np.ndarray, encode as base64 payload with metadata.

    Arguments:
        x: tensor or array to encode, or None.

    Returns:
        A dictionary with encoding metadata and base64 data, or None if input is None.
    """

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().contiguous().numpy()
        backend = "torch"
    elif isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x)
        backend = "numpy"
    else:
        return None

    payload = {
        "__array__": True,               
        "backend": backend,              
        "dtype": str(arr.dtype),         
        "shape": list(arr.shape),        
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"), 
    }
    return payload


def _maybe_encode_response(response: dict[str, torch.Tensor | str | None]) -> dict[str, str | torch.Tensor | VectorPayload | None]:
    """If response['activations'] is a tensor/array, encode it safely for JSON storage.
    
    Arguments:
        response: A dictionary with keys 'text', 'activations', and optionally 'label'.
    Returns:
        A dictionary with the same keys, but with 'activations' encoded if needed.

        For example:
            resp = {"text": "Hello", "activations": torch.randn(10), "label": "greeting"}
            encoded_resp = _maybe_encode_response(resp) 
            # encoded_resp['activations'] is now a base64 payload dictionary which is JSON-serializable.
    """
    assert isinstance(response, dict)

    if "activations" in response and response["activations"] is not None:
        response = dict(response)  # shallow copy
        response["activations"] = _encode_activations(response["activations"])
    return response


def _decode_activations(obj: VectorPayload | None, return_backend: str = "torch") -> torch.Tensor | np.ndarray | list | None:
    """Decode from our base64 payload into torch tensor (default) or numpy array.
    return_backend: 'torch' | 'numpy' | 'list'
    map_device: 'cpu' (default) or 'original' (best-effort) for torch tensors.

    Arguments:
        obj: The payload dictionary to decode, or None.
        return_backend: Desired return type: 'torch' (default), 'numpy', or 'list'.
    
    Returns:
        The decoded tensor/array/list, or None if input was None.
    """

    if obj is None:
        return None
    
    assert return_backend in ("torch", "numpy", "list"), "return_backend must be 'torch', 'numpy', or 'list'"
    assert not isinstance(obj, dict) or not obj.get("__array__", False), "Object is not a valid encoded activations payload"

    try:
        dtype = np.dtype(obj["dtype"])
        shape = tuple(obj["shape"])
        raw = base64.b64decode(obj["data"])
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    except Exception as e:
        raise ValueError(f"Failed to decode activations payload: {e}") from e

    if return_backend == "list":
        return arr.tolist()
    if return_backend == "numpy":
        return arr
    if return_backend == "torch":
        return torch.from_numpy(arr)
    raise ValueError(f"Unknown return_backend: {return_backend}")


def _maybe_decode_response(response: dict[str, str | torch.Tensor | VectorPayload | None], return_backend: str) -> dict[str, str | torch.Tensor | VectorPayload | None]:
    """If response['activations'] is an encoded payload, decode it to tensor/array.

    Arguments:
        response: A dictionary with keys 'text', 'activations', and optionally 'label'.
        return_backend: 'torch' (default), 'numpy', or 'list'.
    
    Returns:
        A dictionary with the same keys, but with 'activations' decoded if needed.

        For example:
            resp = {"text": "Hello", "activations": <encoded payload>, "label": "greeting"},
             wherere <encoded payload> is a dict:
            {
            "__array__": True,
            "backend": "torch",
            "dtype": "float32",
            "shape": [10],
            "data": "...base64..."
            }
             (as produced by _maybe_encode_response).

            decoded_resp = _maybe_decode_response(resp, return_backend='torch')
            # decoded_resp['activations'] is now a torch.Tensor.
        """
    assert isinstance(response, dict)

    if "activations" in response and response["activations"] is not None:
        response = dict(response) 
        response["activations"] = _decode_activations(response["activations"], return_backend)
    return response


def _validate_top_level(data: dict[str, str | list]) -> None:
    """Validate the top-level structure of the loaded JSON data.

    Top structure must contain 'name', 'task_type', and 'pairs' keys.

    Arguments:
        data: The loaded JSON data as a dictionary.

    Raises:
        ValueError: If the structure is invalid.
    """
    if not all(k in data for k in ("name", "task_type", "pairs")):
        raise ValueError("Invalid JSON structure: missing one of ['name', 'task_type', 'pairs']")
    if not isinstance(data["pairs"], list):
        raise ValueError("'pairs' should be a list")


def _validate_pair_obj(pair: dict[str, str | dict[str, str | VectorPayload | None]]) -> None:
    """Validate the structure of a single pair object.

    Each pair must contain 'prompt', 'positive_response', 'negative_response', 'label' (can be None) and 'trait_description' (can be None).
    'positive_response' and 'negative_response' must be dictionaries containing 'model_response', 'activations' (can be None), and 'label' (can be None).

    Structure of 'pair object':
    {
        "prompt": "The input prompt",
        "positive_response": {
            "model_response": "The positive response",
            "activations": VectorPayload or None,
            "label": "positive"
        },
        "negative_response": {
            "model_response": "The negative response",
            "activations": VectorPayload or None,
            "label": "negative"
        },
        "label": "overall label",
        "trait_description": "description of the trait"
    }

    Arguments:
        pair: The pair object to validate.
    
    Raises:
        ValueError: If the structure is invalid.
    """
    need = ("prompt", "positive_response", "negative_response")
    if not all(k in pair for k in need):
        raise ValueError("Each pair must contain 'prompt', 'positive_response', and 'negative_response'")
    if not isinstance(pair["positive_response"], dict) or not isinstance(pair["negative_response"], dict):
        raise ValueError("'positive_response' and 'negative_response' must be dictionaries")
    for resp_key in ("model_response", "activations", "label"):
        if resp_key not in pair["positive_response"]:
            raise ValueError(f"'positive_response' must contain '{resp_key}'")
        if resp_key not in pair["negative_response"]:
            raise ValueError(f"'negative_response' must contain '{resp_key}'")
    if "label" in pair and pair["label"] is not None and not isinstance(pair["label"], str):
        raise ValueError("'label' must be a string or None")
    if "trait_description" in pair and pair["trait_description"] is not None and not isinstance(pair["trait_description"], str):
        raise ValueError("'trait_description' must be a string or None")

def save_contrastive_pair_set(
    cps: ContrastivePairSet,
    filepath: str | Path,
) -> None:
    """Save a ContrastivePairSet to a JSON file.
    Tensors/ndarrays in response['activations'] are encoded with base64 + dtype/shape metadata.

    Arguments:
        cps: The ContrastivePairSet to save.
        filepath: Path to the output JSON file.
    """

    pairs: list[dict[str, str | dict[str, str | VectorPayload | None]]] = []
    for pair in cps.pairs:
        p = pair.to_dict()
        p["positive_response"] = _maybe_encode_response(p.get("positive_response", {}))
        p["negative_response"] = _maybe_encode_response(p.get("negative_response", {}))
        pairs.append(p)

    data = {
        "_version": 1,  # simple schema versioning
        "name": cps.name,
        "task_type": cps.task_type,
        "pairs": pairs,
    }

    filepath = Path(filepath)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_contrastive_pair_set(
    filepath: str | Path,
    return_backend: str = "torch", 
) -> ContrastivePairSet:
    """Load a ContrastivePairSet from a JSON file and decode activations.

    Args:
        filepath: path to the JSON file.
        return_backend: 'torch' (default), 'numpy', or 'list'. If torch is not
            installed, will automatically fall back to 'numpy'.
       
    Returns:
        ContrastivePairSet

        Format of loaded data:
        {
            "name": "name of the set",
            "task_type": "task type string",
            "pairs": [
                {
                    "prompt": "The input prompt",
                    "positive_response": {
                        "model_response": "The positive response",
                        "activations": VectorPayload or None,
                        "label": "positive"
                    },
                    "negative_response": {
                        "model_response": "The negative response",
                        "activations": VectorPayload or None,
                        "label": "negative"
                    },
                    "label": "overall label" or None,
                    "trait_description": "description of the trait" or None
                },
                ...
            ]
        }

    """
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    _validate_top_level(data)

    decoded_pairs: list[dict[str, ]] = []
    for pair in data["pairs"]:
        _validate_pair_obj(pair)
        p = dict(pair)
        p["positive_response"] = _maybe_decode_response(p.get("positive_response", {}), return_backend)
        p["negative_response"] = _maybe_decode_response(p.get("negative_response", {}), return_backend)
        decoded_pairs.append(p)
    
    list_of_pairs = [ContrastivePair.from_dict(p) for p in decoded_pairs]

    cps = ContrastivePairSet(name=str(data["name"]), pairs=list_of_pairs, task_type=data.get("task_type"))

    cps.validate()

    return cps