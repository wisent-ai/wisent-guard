from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import typer

from wisent_guard.cli.wisent_cli.ui import echo
from wisent_guard.cli.wisent_cli.util import aggregations as aggs
from wisent_guard.cli.wisent_cli.util.parsing import (
    parse_natural_tokens, parse_kv, parse_layers, to_bool, DTYPE_MAP,
)

try:
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    HAS_RICH = True
except Exception:
    HAS_RICH = False

__all__ = ["app", "train"]

app = typer.Typer(help="Training workflow")

def _resolve_method(method_name: Optional[str], methods_location: Optional[str]):
    from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator  # type: ignore
    # Best effort discovery if available
    try:
        if methods_location and hasattr(SteeringMethodRotator, "discover_methods"):
            SteeringMethodRotator.discover_methods(methods_location)  # type: ignore[attr-defined]
    except Exception:
        pass

    rot = SteeringMethodRotator()
    if method_name:
        # Case-insensitive match from registry
        registry = {m["name"].lower(): m["name"] for m in rot.list_methods()}
        real = registry.get(method_name.lower(), method_name)
        try:
            rot.use(real)
            inst = getattr(rot, "_method", None)
            if inst is not None:
                return inst
        except Exception:
            pass
        # Fallback to private resolver
        try:
            return SteeringMethodRotator._resolve_method(real)
        except Exception as ex:
            raise typer.BadParameter(f"Unknown steering method: {method_name!r}") from ex

    # No name provided -> default to first or 'caa' if present
    names = [m["name"] for m in rot.list_methods()]
    if "caa" in [n.lower() for n in names]:
        rot.use("caa")
        return getattr(rot, "_method", SteeringMethodRotator._resolve_method("caa"))
    if not names:
        raise typer.BadParameter("No steering methods registered.")
    rot.use(names[0])
    return getattr(rot, "_method", SteeringMethodRotator._resolve_method(names[0]))


def _show_plan(
    *,
    model: str,
    loader: Optional[str],
    loaders_location: Optional[str],
    loader_kwargs: Dict[str, object],
    method_name: Optional[str],
    method_kwargs: Dict[str, object],
    layers: Optional[str],
    aggregation_name: str,
    store_device: str,
    dtype: Optional[str],
    return_full_sequence: bool,
    normalize_layers: bool,
    save_dir: Optional[Path],
) -> None:
    plan = {
        "Model": model,
        "Data loader": loader or "(default)",
        "Loaders location": loaders_location or "(auto)",
        "Loader kwargs": loader_kwargs or {},
        "Method": method_name or "(resolved automatically)",
        "Method kwargs": method_kwargs or {},
        "Layers": layers or "(all)",
        "Aggregation": aggregation_name,
        "Return full sequence": return_full_sequence,
        "Normalize layers": normalize_layers,
        "Store device": store_device,
        "Dtype": dtype or "(unchanged)",
        "Save dir": str(save_dir) if save_dir else "(none)",
    }

    code = f"""
# Example: Training steering vectors (auto-generated plan)
from wisent_guard.core.trainers.steering_trainer import WisentSteeringTrainer
from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator
from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator
from wisent_guard.core.activations.core.atoms import ActivationAggregationStrategy

# 1) Model
model = WisentModel(model_name={model!r}, layers={{}}, device={store_device!r})

# 2) Data loader
rot = DataLoaderRotator(loader={loader!r}, loaders_location={loaders_location!r})
load = rot.load(**{json.dumps(loader_kwargs)})

# 3) Method
method = SteeringMethodRotator._resolve_method({(method_name or 'caa')!r})

# 4) Trainer
trainer = WisentSteeringTrainer(model=model, pair_set=load["train_qa_pairs"], steering_method=method,
                                store_device={store_device!r}, dtype={dtype!r})

# 5) Train
result = trainer.run(
    layers_spec={layers!r},
    method_kwargs={json.dumps(method_kwargs)},
    aggregation=ActivationAggregationStrategy.{aggs.pick(aggregation_name).name},
    return_full_sequence={return_full_sequence!r},
    normalize_layers={normalize_layers!r},
    save_dir={str(save_dir) if save_dir else None!r},
)
""".strip()

    if HAS_RICH:
        t = Table(title="Execution Plan")
        t.add_column("Key", style="bold", no_wrap=True)
        t.add_column("Value")
        for k, v in plan.items():
            t.add_row(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v))
        echo(Panel(t, expand=False))
        echo(Panel(Syntax(code, "python", word_wrap=False), title="Code Preview", expand=False))
    else:
        print(json.dumps(plan, indent=2))
        print("\n" + code)


@app.command("train", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def train(ctx: typer.Context, params: List[str] = typer.Argument(None)):
    """
    Natural (no-dash) usage examples:

      wisent train model meta-llama/Llama-3.2-1B-Instruct loader custom path ./custom.json training_limit 5 method caa

      wisent train interactive true

    See `wisent loader-args custom` to view the exact loader arguments.
    """
    # Lazy imports
    from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator  # type: ignore
    from wisent_guard.core.models.wisent_model import WisentModel  # type: ignore
    from wisent_guard.core.trainers.steering_trainer import WisentSteeringTrainer  # type: ignore

    tokens = list(params or []) + list(ctx.args or [])
    top, loader_kv_raw, method_kv_raw = parse_natural_tokens(tokens)

    # Core args
    model = top.get("model")
    if not model:
        raise typer.BadParameter("Please specify a model (e.g. `train model meta-llama/Llama-3.2-1B-Instruct`) or use `interactive true`.")

    loader = top.get("loader")
    loaders_location = top.get("loaders_location")
    methods_location = top.get("methods_location")
    method_name = top.get("method")

    layers = parse_layers(top.get("layers")) if top.get("layers") else None
    aggregation_name = (top.get("aggregation") or "continuation_token").lower()
    store_device = top.get("device") or top.get("store_device") or "cpu"
    dtype = top.get("dtype")
    save_dir = Path(top["save_dir"]) if top.get("save_dir") else None
    return_full_sequence = to_bool(top.get("return_full_sequence", "false")) if "return_full_sequence" in top else False
    normalize_layers = to_bool(top.get("normalize_layers", "false")) if "normalize_layers" in top else False
    interactive = to_bool(top.get("interactive", "false")) if "interactive" in top else False
    plan_only = to_bool(top.get("plan-only", top.get("plan_only", "false"))) if ( "plan-only" in top or "plan_only" in top ) else False
    confirm = to_bool(top.get("confirm", "true")) if "confirm" in top else True

    # Convert kwargs
    loader_kwargs = parse_kv([f"{k}={v}" for k, v in loader_kv_raw.items()])
    method_kwargs = parse_kv([f"{k}={v}" for k, v in method_kv_raw.items()])

    # Interactive wizard
    if interactive:
        if loaders_location:
            DataLoaderRotator.discover_loaders(loaders_location)
        if not loader:
            options = [d["name"] for d in DataLoaderRotator.list_loaders()]
            loader = typer.prompt("Choose data loader", default=(options[0] if options else "custom"))
        if loader and loader.lower() == "custom":
            echo(Panel(
                "[b]Custom loader arguments[/]\n\n"
                "• path (str)  [required]\n"
                "• split_ratio (float | None)\n"
                "• seed (int | None)\n"
                "• training_limit (int | None)\n"
                "• testing_limit (int | None)",
                title="custom.load(...)",
            ) if HAS_RICH else
                None
            )
            if "path" not in loader_kwargs:
                loader_kwargs["path"] = typer.prompt("Path to dataset JSON (required)")
            for name, cast, default in [
                ("split_ratio", float, ""),
                ("seed", int, ""),
                ("training_limit", int, ""),
                ("testing_limit", int, ""),
            ]:
                if name not in loader_kwargs:
                    val = typer.prompt(f"{name} (optional)", default=default)
                    if str(val).strip() != "":
                        try:
                            loader_kwargs[name] = cast(val)
                        except Exception:
                            loader_kwargs[name] = val
        if not method_name:
            method_name = typer.prompt("Choose steering method (see list-methods)", default="caa")
        if layers is None:
            layers = parse_layers(typer.prompt("Layers (e.g., '10..12', '5,7,9' or leave empty for all)", default="") or None)
        if "aggregation" not in top:
            aggregation_name = typer.prompt("Aggregation (see list-aggregations)", default="continuation_token")
        if "dtype" not in top:
            dtype = typer.prompt("Activation dtype (float32/float16/bfloat16 or blank)", default="") or None
        if "device" not in top and "store_device" not in top:
            store_device = typer.prompt("Device to store activations on (cpu / cuda / cuda:0 / ...)", default="cpu")
        if "normalize_layers" not in top:
            normalize_layers = typer.confirm("Normalize activations per layer?", default=True)
        if "return_full_sequence" not in top:
            return_full_sequence = typer.confirm("Return full [T,H] sequence per layer?", default=False)
        if "save_dir" not in top:
            default_out = os.path.abspath("./steering_output")
            p = typer.prompt("Save directory for artifacts (blank to skip saving)", default=default_out)
            if p.strip():
                save_dir = Path(p)
        if "plan-only" not in top and "plan_only" not in top:
            plan_only = typer.confirm("Only show the plan and code preview?", default=False)
        if "confirm" not in top:
            confirm = typer.confirm("Confirm before running?", default=True)

    # Validate dtype
    if dtype not in DTYPE_MAP:
        raise typer.BadParameter("dtype must be one of: float32, float16, bfloat16")

    # Validate aggregation
    try:
        agg = aggs.pick(aggregation_name)
    except ValueError as ex:
        raise typer.BadParameter(str(ex)) from ex

    # Plan
    _show_plan(
        model=model,
        loader=loader,
        loaders_location=loaders_location,
        loader_kwargs=loader_kwargs,
        method_name=method_name,
        method_kwargs=method_kwargs,
        layers=layers,
        aggregation_name=aggregation_name,
        store_device=store_device,
        dtype=dtype,
        return_full_sequence=return_full_sequence,
        normalize_layers=normalize_layers,
        save_dir=save_dir,
    )

    if plan_only:
        return

    if confirm and not typer.confirm("Proceed with training?", default=True):
        typer.echo("Aborted.")
        raise typer.Exit(code=1)

    # -- Model -----------------------------------------------------------------
    typer.echo(f"[+] Loading model: {model}")
    from wisent_guard.core.models.wisent_model import WisentModel  # type: ignore
    wmodel = WisentModel(model_name=model, layers={}, device=store_device)

    # -- Data loader -----------------------------------------------------------
    from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator  # type: ignore
    if loaders_location:
        DataLoaderRotator.discover_loaders(loaders_location)
    dl_rot = DataLoaderRotator(loader=loader, loaders_location=loaders_location or "wisent_guard.core.data_loaders.loaders")
    typer.echo(f"[+] Using data loader: {loader or '(default)'}")
    load_result = dl_rot.load(**loader_kwargs)
    pair_set = load_result["train_qa_pairs"]
    typer.echo(f"[+] Loaded training pairs: {len(pair_set)} (task_type={load_result['task_type']})")

    # -- Steering method -------------------------------------------------------
    method_inst = _resolve_method(method_name, methods_location)
    name_shown = getattr(method_inst, "name", type(method_inst).__name__)
    typer.echo(f"[+] Steering method: {name_shown}")

    # -- Trainer ---------------------------------------------------------------
    from wisent_guard.core.trainers.steering_trainer import WisentSteeringTrainer  # type: ignore
    torch_dtype = None if dtype is None else __import__("torch").__dict__[DTYPE_MAP[dtype]]
    trainer = WisentSteeringTrainer(
        model=wmodel,
        pair_set=pair_set,
        steering_method=method_inst,
        store_device=store_device,
        dtype=torch_dtype,
    )

    result = trainer.run(
        layers_spec=layers,
        method_kwargs=method_kwargs,
        aggregation=agg,
        return_full_sequence=return_full_sequence,
        normalize_layers=normalize_layers,
        save_dir=save_dir,
    )

    typer.echo("\n=== Training Summary ===")
    typer.echo(json.dumps(result.metadata, indent=2))
    if save_dir is not None:
        typer.echo(f"\nArtifacts saved in: {Path(save_dir).resolve()}\n")
