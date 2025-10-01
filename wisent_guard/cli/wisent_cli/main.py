from __future__ import annotations
from typing import Optional

import typer

from wisent_guard.cli.wisent_cli.version import APP_NAME, APP_VERSION
from wisent_guard.cli.wisent_cli.ui import print_banner
from wisent_guard.cli.wisent_cli.commands.listing import app as listing_app
from wisent_guard.cli.wisent_cli.commands.train_cmd import app as train_app
from wisent_guard.cli.wisent_cli.commands.help_cmd import app as help_router_app
from wisent_guard.cli.wisent_cli.shell import app as shell_app

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
    help=(
        "[bold]Wisent Guard[/] – steerable activations / steering vectors.\n"
        "Collect activations, train steering vectors, and inspect loaders & methods.\n\n"
        "Natural commands (no dashes) + `help <topic>` and a `wisent` shell."
    ),
)

app.add_typer(listing_app, name="list")
app.add_typer(train_app,   name="")           # attach commands directly (e.g., train)
app.add_typer(help_router_app, name="")       # help router lives at root (help ...)
app.add_typer(shell_app,  name="shell")

STATE = {"verbose": False}

@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(None, "--version", "-V", help="Show version and exit."),
    no_banner: bool = typer.Option(False, "--no-banner", help="Disable the startup banner."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
    logo_width: Optional[int] = typer.Option(None, "--logo-width", help="Width of the wisent badge (28–96)."),
):
    """
    Welcome to **Wisent Guard**.

    Examples:
      • `wisent help train`
      • `wisent train model meta-llama/Llama-3.2-1B-Instruct loader custom path ./custom.json training_limit 5 method caa`
      • `wisent list list-methods`      (methods)
      • `wisent list list-loaders`      (loaders)
      • `wisent list list-aggregations` (aggregations)
      • `wisent shell start`            (interactive 'wisent' prompt)
    """
    if version:
        typer.echo(f"{APP_NAME} {APP_VERSION}")
        raise typer.Exit()

    STATE["verbose"] = verbose
    if not no_banner and (ctx.invoked_subcommand is None or ctx.info_name in {"--help", None}):
        print_banner(APP_NAME, width=logo_width or 48)

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

@app.command("instructions")
def instructions():
    msg = """
[bold]Quickstart (no-dash style)[/]

• Preview a run without executing:
  [b]wisent train model gpt2 plan-only true[/]

• Discover components:
  [b]wisent list list-methods[/] | [b]wisent list list-loaders[/] | [b]wisent list list-aggregations[/]

• Get help:
  [b]wisent help train[/], [b]wisent help method caa[/], [b]wisent help loader custom[/]

• Full training example (natural):
  [b]wisent train model meta-llama/Llama-3.2-1B-Instruct loader custom path ./wisent_guard/cli/cli_examples/custom_dataset.json training_limit 5 \\
        method caa layers 10..12 aggregation continuation_token device cuda dtype float16 save_dir ./steering_output normalize_layers true[/]

Tip: add [b]interactive true[/] for a guided wizard.
"""
    try:
        from rich.panel import Panel
        from wisent_guard.cli.wisent_cli.ui import echo
        echo(Panel.fit(msg, title="Instructions", border_style="green"))
    except Exception:
        print(msg)

def run():
    app()

if __name__ == "__main__":
    run()