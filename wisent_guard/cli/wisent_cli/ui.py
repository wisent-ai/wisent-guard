from __future__ import annotations

from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
    _console = Console()
except Exception:  # pragma: no cover
    HAS_RICH = False
    _console = None

__all__ = ["print_banner", "echo"]

def _render_wisent_mark(width: int = 48) -> str:
    logo = """
.................  .:--++*##%%%%##**+=-:.  .................
..             .:=*%@@@@@@@%%%%%%%@@@@@@%*=:.             ..
.           .-*%@@@%#+=-::.........:-=+#%@@@%*=.           .
.         -*%@@@#=:.                    .:=*%@@@*-.        .
.      .-#@@@*=.                            .-*@@@#-.      .
.     :#@@@*:                                  :+%@@#-     .
.   .+@@@*:                                      :+@@@+.   .
.  .*@@@@%*=:.                                     -%@@#:  .
. .#@@#=*%@@@%*-:.                                  .#@@%: .
..*@@%.  .-+#@@@@#+-:.                               .*@@%..
.=@@@-       :-+#@@@@%*=:.                            .%@@*.
:#@@+           .:-+#@@@@%#+=:.                        -@@@-
=@@@:                .-=*%@@@@%#+=:..                  .#@@+
+@@@*=:.                 .:-+*%@@@@%#*=-:..             *@@+
+@@@@@@#+-..                  .:-=*#@@@@@%#*+--..       +@@+
+@@#-+%@@@%:                        .:-=*#%@@@@@%#*+=-:.*@@+
=@@%. .=@@@:                             ..:-=+#%%@@@@@%@@@+
:%@@=  :@@@-                                    ..::-=+#@@@=
.+@@%. .#@@*                                           +@@#:
..%@@*. =@@@:                                         =@@@-.
. :%@@*..#@@#.                         .:..          =@@@= .
.  :%@@*.:%@@*.                       :#@@%#*+=-::..+@@@=  .
.   :#@@%-:%@@#:                    .+@@@#%%@@@@@@%%@@%-   .
.    .+@@@*=#@@%-                 .=%@@%=...::-=#@@@@*.    .
.      :*@@@#%@@@*:             .=%@@@+.     .:*%@@#-      .
.        :+%@@@@@@@*-.       :=*@@@%+.    .-+%@@@*-.       .
.          .=*%@@@@@@#+:.:-+#@@@%*-. .:-+#%@@@#+:          .
.             .-+#%@@@@@@@@@@@@#*+**#@@@@@%*=:.            .
..............   ..-=+*#%%%@@@@@@@@%%#*=-:.   ..............
 ...................  ....:::::::::.... ................... 
""".strip("\n")
    return "\n".join(line.center(width) for line in logo.splitlines())


def print_banner(title: str, width: int | None = None) -> None:
    if HAS_RICH:
        usable = 64 if width is None else width
        badge = _render_wisent_mark(usable)
        _console.print(Panel.fit(Text(badge, style="green"), title=f"[b green]{title}[/]", border_style="green"))
        _console.print("[dim]Steering vectors & activation tooling[/]\n")
    else:
        GREEN, OFF = "\x1b[32m", "\x1b[0m"
        print(GREEN + _render_wisent_mark(width or 48) + OFF)
        print(f"{title} – Steering vectors & activation tooling\n")


def echo(obj: Any) -> None:
    if HAS_RICH:
        _console.print(obj)
    else:
        print(obj)
