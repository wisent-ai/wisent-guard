from __future__ import annotations
import inspect
from typing import Optional

import typer

from wisent_guard.cli.wisent_cli.ui import echo
from wisent_guard.cli.wisent_cli.util import aggregations as aggs

try:
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except Exception:
    HAS_RICH = False

app = typer.Typer(help="Listing, discovery, and explanation commands")


@app.command("list-aggregations")
def list_aggregations():
    if HAS_RICH:
        t = Table(title="Aggregation Strategies")
        t.add_column("Name", style="bold")
        t.add_column("Description")
        for k, desc in aggs.descriptions().items():
            t.add_row(k.name.lower(), desc)
        echo(t)
    else:
        for k, desc in aggs.descriptions().items():
            print(f"- {k.name.lower():22s} : {desc}")


@app.command("list-methods")
def list_methods():
    from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator  # type: ignore
    rot = SteeringMethodRotator()
    methods = rot.list_methods()
    if not methods:
        typer.echo("No steering methods registered.")
        raise typer.Exit(code=1)
    if HAS_RICH:
        t = Table(title="Registered Steering Methods")
        t.add_column("Name", style="bold")
        t.add_column("Description")
        t.add_column("Class")
        for m in methods:
            t.add_row(m["name"], m["description"], m["class"])
        echo(t)
    else:
        for m in methods:
            print(f"- {m['name']}: {m['description']} ({m['class']})")


@app.command("list-loaders")
def list_loaders(
    loaders_location: Optional[str] = typer.Option(None, help="Package path or directory containing data loader modules"),
    scope_prefix: Optional[str] = typer.Option(None, help="Limit list to module path prefix"),
):
    from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator  # type: ignore
    if loaders_location:
        DataLoaderRotator.discover_loaders(loaders_location)
    loaders = DataLoaderRotator.list_loaders(scope_prefix=scope_prefix)
    if not loaders:
        typer.echo("No data loaders found.")
        raise typer.Exit(code=1)
    if HAS_RICH:
        t = Table(title="Registered Data Loaders")
        t.add_column("Name", style="bold")
        t.add_column("Description")
        t.add_column("Class")
        for l in loaders:
            t.add_row(l["name"], l["description"], l["class"])
        echo(t)
    else:
        for l in loaders:
            print(f"- {l['name']}: {l['description']} ({l['class']})")


@app.command("explain")
def explain(
    method: Optional[str] = typer.Option(None, help="Steering method to describe"),
    loader: Optional[str] = typer.Option(None, help="Data loader to describe"),
    loaders_location: Optional[str] = typer.Option(None, help="Where to discover data loaders"),
    aggregation: Optional[str] = typer.Option(None, help="Aggregation to describe"),
):
    from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator  # type: ignore
    from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator  # type: ignore

    if loaders_location:
        DataLoaderRotator.discover_loaders(loaders_location)

    if method:
        m = SteeringMethodRotator._resolve_method(method)
        doc = (getattr(type(m), "__doc__", None) or "No docstring.").strip()
        if HAS_RICH:
            echo(Panel(doc, title=f"Method: {getattr(m, 'name', type(m).__name__)}"))
        else:
            print(doc)

    if loader:
        reg = DataLoaderRotator.list_loaders()
        match = next((x for x in reg if x["name"].lower() == loader.lower()), None)
        if match:
            desc = (match.get("description") or "No description.").strip()
            if HAS_RICH:
                echo(Panel(desc, title=f"Loader: {loader}"))
            else:
                print(desc)
        else:
            typer.echo(f"Unknown loader: {loader}")

    if aggregation:
        agg = aggs.pick(aggregation)
        desc = aggs.descriptions().get(agg, "No description.")
        if HAS_RICH:
            from rich.panel import Panel
            echo(Panel(desc, title=f"Aggregation: {aggregation}"))
        else:
            print(desc)


@app.command("loader-args")
def loader_args(
    name: str = typer.Argument(..., help="Loader name, e.g. 'custom'"),
    loaders_location: Optional[str] = typer.Option(None, help="Where to discover data loaders"),
):
    """
    Show the exact arguments accepted by the loader's `load(...)` method.
    Useful so users know precisely what to pass (e.g., for `custom`:
    `path, split_ratio, seed, training_limit, testing_limit`).
    """
    from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator  # type: ignore
    if loaders_location:
        DataLoaderRotator.discover_loaders(loaders_location)
    rot = DataLoaderRotator(loader=name, loaders_location=loaders_location or "wisent_guard.core.data_loaders.loaders")
    # Best-effort introspection
    target = None
    for cand in (getattr(rot, "_loader", None), rot):
        if cand is None:
            continue
        if hasattr(cand, "load"):
            target = cand.load
            break
    if target is None:
        typer.echo("Could not introspect loader signature.")
        raise typer.Exit(code=1)

    sig = inspect.signature(target)
    if HAS_RICH:
        from rich.panel import Panel
        echo(Panel(f"{name}.load{sig}", title="Loader load(...) signature"))
    else:
        print(f"{name}.load{sig}")
