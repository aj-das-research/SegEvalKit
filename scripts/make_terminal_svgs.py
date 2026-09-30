"""Record real CLI sessions as SVG screenshots for the README and docs.

Each command runs through segevalkit.cli.main with a recording rich console;
the SVG therefore shows exactly what a user sees.

    python scripts/make_terminal_svgs.py --out docs/assets/terminal [--eval-args ...]
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from rich.console import Console
from rich.terminal_theme import TerminalTheme

from segevalkit import _console
from segevalkit.cli import main

# Dark terminal in the SegEvalKit palette.
THEME = TerminalTheme(
    (22, 18, 31), (236, 232, 248),
    [(22, 18, 31), (240, 131, 74), (31, 179, 161), (210, 162, 28), (91, 147, 234), (207, 63, 143),
     (31, 179, 161), (220, 216, 235)],
    [(88, 80, 110), (240, 131, 74), (95, 209, 194), (240, 200, 90), (140, 180, 245), (230, 120, 180),
     (95, 209, 194), (255, 255, 255)],
)


def record(cmd: str, out: Path, width: int = 104, title: str = "segevalkit"):
    rec = Console(theme=_console.THEME, record=True, width=width, force_terminal=True, highlight=False)
    _console.console = rec
    rec.print(f"[grey62]$[/] [bold]{cmd}[/]")
    main(shlex.split(cmd)[1:])
    rec.save_svg(str(out), title=title, theme=THEME)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/terminal")
    ap.add_argument("--extra", nargs="*", default=[], help="name=command pairs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {
        "home": "segevalkit",
        "recommend": "segevalkit recommend --structure small_lesion --multi-instance --volumetry",
        "metrics": "segevalkit metrics --family distance -v",
        "datasets": "segevalkit datasets",
    }
    jobs.update(dict(x.split("=", 1) for x in a.extra))
    for name, cmd in jobs.items():
        record(cmd, out / f"{name}.svg", width={"metrics": 118, "recommend": 118}.get(name, 104))
