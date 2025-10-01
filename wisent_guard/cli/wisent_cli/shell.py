from __future__ import annotations
import shlex
import typer

from wisent_guard.cli.wisent_cli.ui import print_banner, echo

try:
    from rich.panel import Panel
    HAS_RICH = True
except Exception:
    HAS_RICH = False

__all__ = ["app", "start"]

app = typer.Typer(help="Interactive shell")

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

@app.command("start")
def start(
    logo_width: int = typer.Option(48, "--logo-width", "-w", help="Logo width for the banner (28–96)."),
    show_banner: bool = typer.Option(True, "--banner/--no-banner", help="Show banner when the shell starts."),
):
    """
    Launch the interactive **wisent** shell.

    Inside the shell, run:
      • `help` / `help train` / `help method caa`
      • `instructions`
      • `train model ...` (no dashes)

    Type `exit`, `quit`, or press `Ctrl-D` to leave.
    """
    if show_banner:
        print_banner("Wisent Guard", logo_width)

    hint = "Type 'help', 'help train', 'instructions', or any command like 'train model ...'. Type 'exit' to quit."
    if HAS_RICH:
        echo(Panel(hint, title="Welcome to the wisent shell", border_style="green"))
    else:
        print(hint)

    GREEN, OFF = ("\x1b[32m", "\x1b[0m")
    while True:
        try:
            if HAS_RICH:
                from rich.console import Console
                line = Console().input("[bold green]wisent[/] ")
            else:
                line = input(f"{GREEN}wisent{OFF} ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nBye.")
            break

        line = line.strip()
        if not line:
            continue
        if line in {"exit", "quit", "q"}:
            typer.echo("Bye.")
            break
        if line in {"help", "--help", "-h"}:
            _run_cli_line("--help")
            continue
        if line.startswith("help "):
            _run_cli_line(line)
            continue
        if line in {"instructions", "--instructions"}:
            _run_cli_line("instructions")
            continue

        _run_cli_line(line)