from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import json
import torch
import datetime as _dt

from wisent_guard.core.activations.core.atoms import (
    LayerActivations,
    ActivationAggregationStrategy,
    RawActivationMap,
)
from wisent_guard.core.models.wisent_model import WisentModel

from wisent_guard.core.trainers.core.atoms import (
    TrainingResult,
    BaseSteeringTrainer
)

from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent_guard.core.activations.core.activations_collector import ActivationCollector  
from wisent_guard.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent_guard.core.contrastive_pairs.diagnostics import run_control_vector_diagnostics

__all__ = [
    "WisentSteeringTrainer",
]


logger = logging.getLogger(__name__)

@dataclass(slots=True)
class WisentSteeringTrainer(BaseSteeringTrainer):
    """
    Orchestrates activation collection + steering vector training for a given model and pair set.

    Minimal usage:
        trainer = WisentSteeringTrainer(model, pair_set, steering_method)
        result = trainer.run(layers_spec=..., method_kwargs=..., aggregation=..., ...)
        # result is a TrainingResult with steered vectors, enriched pair set, and metadata
        trainer.save_result(output_dir)  # optional save
    
    arguments:
        model: WisentModel to use for activation collection.
        pair_set: ContrastivePairSet with pairs to use for collection and training.
        steering_method: BaseSteeringMethod instance to use for training.
        store_device: Device to store collected activations on (default "cpu").
        dtype: Optional torch.dtype to cast collected activations to (default None, meaning no cast).
    """

    model: WisentModel
    pair_set: ContrastivePairSet
    steering_method: BaseSteeringMethod
    store_device: str | torch.device = "cpu"
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        self.collector = ActivationCollector(model=self.model, store_device=self.store_device, dtype=self.dtype)
        self._last_result: TrainingResult | None = None

    def run(
        self,
        layers_spec: Sequence[str] | str | int | Sequence[int] | None,
        method_kwargs: dict[str, Any] | None = None,
        aggregation: ActivationAggregationStrategy = ActivationAggregationStrategy.CONTINUATION_TOKEN,
        return_full_sequence: bool = False,
        normalize_layers: bool = False,
        save_dir: str | Path | None = None,
    ) -> TrainingResult:
        """
        Full pipeline:
          1) Decide which layers to use (from spec or all layers if None).
          2) Collect activations for each pair at these layers.
          3) Train steering vectors using the selected method.
          4) Return a TrainingResult with vectors, enriched pair set, and metadata.
          5) Optionally save artifacts to disk.

        arguments:
            layers_spec:
                - list like ["10","20","30"] or [10, 20, 30]
                - range string "10-30" / "10..30"
                - single int "12"
                - None → use all available layers on the model
            method:
                Name of steering method ("caa", "bipo", ...).
            method_kwargs:
                Dict of hyperparameters for the method (e.g., {"normalize": True, "scale": 1.0}).
            aggregation:
                ActivationAggregationStrategy to use during collection when not returning
                full sequences. Ignored if 'return_full_sequence=True'.
            return_full_sequence:
                If True, store full [T,H] sequences per layer (method then must know how
                to collapse to vectors). Default False (collect [H] vectors directly).
            normalize_layers:
                If True, L2-normalize activations layer-wise during collection.
            save_dir:
                If provided, artifacts are written there. Directory is created if missing.

        returns:
            TrainingResult
        """
        method_kwargs = method_kwargs or {}

        # 1) Resolve layer names
        layers = self._resolve_layers(layers_spec)

        # 2) Collect activations for each pair
        for i, pair in enumerate(self.pair_set.pairs):
            updated = self.collector.collect_for_pair(
                pair,
                layers=layers,
                aggregation=aggregation,
                return_full_sequence=return_full_sequence,
                normalize_layers=normalize_layers,
            )
            self.pair_set.pairs[i] = updated  

        # 3) Train using selected method
        raw_vectors: RawActivationMap = self.steering_method.train(self.pair_set, **(method_kwargs or {}))

        steered = LayerActivations(raw_vectors)

        control_vector_report = run_control_vector_diagnostics(steered)
        for issue in control_vector_report.issues:
            log_method = logger.error if issue.severity == "critical" else logger.warning
            log_method(
                "[control_vector diagnostics] %s (details=%s)",
                issue.message,
                issue.details,
            )

        control_vector_summary = control_vector_report.summary.get("control_vectors", {})
        control_vector_issues = [
            {
                "metric": issue.metric,
                "severity": issue.severity,
                "message": issue.message,
                "details": issue.details,
            }
            for issue in control_vector_report.issues
        ]

        if control_vector_report.has_critical_issues:
            raise ValueError("Control vector diagnostics found critical issues; see logs for specifics.")

        # 4) Metadata
        now = _dt.datetime.now().astimezone()
        metadata: dict[str, Any] = {
            "timestamp": now.isoformat(),
            "model_name": getattr(self.model, "model_name", getattr(self.model, "name", None)),
            "layers_used": layers or "all",
            "method": self.steering_method.name,
            "method_kwargs": method_kwargs,
            "activation_aggregation_strategy": (None if return_full_sequence else aggregation),
            "return_full_sequence": bool(return_full_sequence),
            "normalize_layers": bool(normalize_layers),
            "num_pairs": len(self.pair_set.pairs),
            "hidden_size": getattr(self.model, "hidden_size", None),
            "control_vector_diagnostics": control_vector_summary,
        }

        if control_vector_issues:
            metadata["control_vector_issues"] = control_vector_issues

        result = TrainingResult(steered_vectors=steered, pair_set_with_activations=self.pair_set, metadata=metadata)
        self._last_result = result

        # 5) Optional save
        if save_dir is not None:
            self.save_result(save_dir, result)

        return result

    def save_result(self, output_dir: str | Path, result: TrainingResult | None = None) -> Path:
        """
        Persist vectors, metadata, and the pair set (with activations) to disk.

        Files written:
            - metadata.json                (JSON)
            - steering_vectors.pt          (torch.save of dict[layer]->tensor on CPU)
            - pairs_with_activations.pt    (torch.save of the full ContrastivePairSet object)
            - steering_vectors_summary.json (shapes/dtypes only, human-readable)

        returns:
            Path to the created directory.
        """
        result = result or self._last_result
        if result is None:
            raise RuntimeError("No result to save. Run the trainer first.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Vectors
        raw_map: RawActivationMap = result.steered_vectors.to_dict()  # still tensors
        cpu_map = {k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in raw_map.items() if k != "_activation_aggregation_strategy"}
        torch.save(cpu_map, out / "steering_vectors.pt")

        # Summary (json-serializable)
        vec_summary = {
            k: None if v is None else {
                "shape": tuple(v.shape),
                "dtype": str(v.dtype),
            }
            for k, v in cpu_map.items()
        }
        (out / "steering_vectors_summary.json").write_text(json.dumps(vec_summary, indent=2))

        # Metadata
        (out / "metadata.json").write_text(json.dumps(result.metadata, indent=2))

        # Full pair set with activations (Python pickle via torch.save)
        torch.save(result.pair_set_with_activations, out / "pairs_with_activations.pt")

        return out

    def _resolve_layers(self, spec: Sequence[str] | str | int | Sequence[int] | None) -> list[str] | None:
        """
        Convert a user-facing spec into canonical layer names ("1","2",...).
        If None, return None (meaning: use all layers in the collector/model).

        arguments:
            spec: See 'layers_spec' argument in run().
        
        returns:
            Sorted list of layer names as strings, or None.

        examples:
            None -> None
            "10-12" -> ["10","11","12"]
            [5,10,15] -> ["5","10","15"]
            "3,7,10..12" -> ["3","7","10","11","12"]
            8 -> ["8"]
        """
        if spec is None:
            return None

        if isinstance(spec, (list, tuple)):
            names: list[str] = []
            for item in spec:
                if isinstance(item, int):
                    names.append(str(item))
                else:
                    names.extend(self._parse_layer_token(item))
            return sorted(set(names), key=lambda s: (len(s), s))

        if isinstance(spec, int):
            return [str(spec)]

        names: list[str] = []
        for token in str(spec).replace(" ", "").split(","):
            names.extend(self._parse_layer_token(token))
        return sorted(set(names), key=lambda s: (len(s), s))

    @staticmethod
    def _parse_layer_token(token: str) -> list[str]:
        """
        Parse a token like "5", "10-20", "10..20" into a list of names.
        """
        if not token:
            return []
        if "-" in token or ".." in token:
            a, b = token.replace("..", "-").split("-")
            a_i, b_i = int(a), int(b)
            lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
            return [str(i) for i in range(lo, hi + 1)]
        else:
            return [str(int(token))]