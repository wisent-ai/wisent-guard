from __future__ import annotations
import shlex
from typing import Optional
import typer

__all__ = ["app", "help_command"]

app = typer.Typer(help="Human-friendly help router")

def _run_cli_line(line: str) -> None:
    from typer.main import get_command
    from wisent_guard.cli.wisent_cli.main import app as root_app
    click_cmd = get_command(root_app)
    args = shlex.split(line)
    try:
        click_cmd.main(args=args, standalone_mode=False, prog_name="wisent")
    except SystemExit as e:
        if e.code not in (0, None):
            raise

@app.command("help")
def help_command(topic: Optional[str] = typer.Argument(None), name: Optional[str] = typer.Argument(None)):
    """
    Examples:
      wisent help train
      wisent help method caa
      wisent help loader custom
      wisent help list-methods
    """
    t = (topic or "").strip().lower()

    if t in {"", "app", "main"}:
        _run_cli_line("--help")
        return

    passthrough = {"train", "list-methods", "list-loaders", "list-aggregations", "explain", "instructions", "start"}
    if t in passthrough:
        _run_cli_line(f"{t} --help")
        return

    if t in {"method", "methods"} and name:
        _run_cli_line(f"explain --method {shlex.quote(name)}")
        return
    if t in {"loader", "loaders"} and name:
        _run_cli_line(f"explain --loader {shlex.quote(name)}")
        _run_cli_line(f"loader-args {shlex.quote(name)}")
        return
    if t in {"aggregation", "agg", "aggregations"} and name:
        _run_cli_line(f"explain --aggregation {shlex.quote(name)}")
        return

    _run_cli_line("--help")