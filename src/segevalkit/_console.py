"""Terminal presentation: a purple-themed console built on `rich`.

Everything user-facing on the command line goes through this module so the
look is consistent: the gradient banner, section rules, metric tables with
better-direction arrows, progress bars and panels. When output is not a
terminal (pipes, CI logs) or ``NO_COLOR`` / ``SEGEVALKIT_PLAIN`` is set,
``rich`` degrades to plain text automatically.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

__all__ = ["console", "banner", "rule", "progress", "metric_table", "summary_table", "panel", "fmt"]

# The purple ramp used across docs, plots and the terminal.
P = {"50": "#f5f1fe", "200": "#d6c7fb", "300": "#b89cf5", "400": "#9a72ee", "500": "#7c4ee4",
     "600": "#6d3fd6", "700": "#5a2fb8", "800": "#472594", "900": "#2e1766"}
TEAL, ORANGE = "#1fb3a1", "#f0834a"

THEME = Theme({
    "sek.brand": f"bold {P['400']}",
    "sek.accent": P["300"],
    "sek.muted": "grey62",
    "sek.ok": f"bold {TEAL}",
    "sek.warn": f"bold {ORANGE}",
    "sek.key": f"bold {P['300']}",
    "sek.num": "bold white",
    "sek.arrow": f"bold {P['400']}",
    "repr.number": P["300"],
})

_plain = bool(os.environ.get("NO_COLOR") or os.environ.get("SEGEVALKIT_PLAIN"))
console = Console(theme=THEME, highlight=False, no_color=_plain, soft_wrap=False)

_LOGO = [
    r"  ____             _____            _ _  ___ _   ",
    r" / ___|  ___  __ _| ____|_   ____ _| | |/ (_) |_ ",
    r" \___ \ / _ \/ _` |  _| \ \ / / _` | | ' /| | __|",
    r"  ___) |  __/ (_| | |___ \ V / (_| | | . \| | |_ ",
    r" |____/ \___|\__, |_____| \_/ \__,_|_|_|\_\_|\__|",
    r"             |___/                               ",
]
_GRADIENT = [P["200"], P["300"], P["400"], P["500"], P["600"], P["700"]]


def banner(subtitle: Optional[str] = None, compact: bool = False) -> None:
    """Print the gradient SegEvalKit banner."""
    from . import __version__

    if compact or console.width < 60:
        t = Text("◆ SegEvalKit ", style="sek.brand")
        t.append(f"v{__version__}", style="sek.muted")
        if subtitle:
            t.append(f"  ·  {subtitle}", style="sek.accent")
        console.print(t)
        return
    logo = Text()
    for i, (line, color) in enumerate(zip(_LOGO, _GRADIENT)):
        logo.append(line.rstrip() + ("\n" if i < len(_LOGO) - 1 else ""), style=f"bold {color}")
    tag = Text("  volumetric segmentation evaluation", style="sek.accent")
    tag.append(f"   v{__version__}", style="sek.muted")
    items = [logo, tag]
    if subtitle:
        items.append(Text(f"  {subtitle}", style="sek.muted"))
    console.print(Panel(Group(*items), border_style=P["700"], padding=(0, 2), expand=False))


def rule(title: str) -> None:
    """A section divider."""
    console.rule(Text(title, style="sek.brand"), style=P["800"])


def progress(transient: bool = False) -> Progress:
    """A purple progress bar with count, rate and ETA."""
    return Progress(
        SpinnerColumn("dots12", style=P["400"]),
        TextColumn("[sek.brand]{task.description}"),
        BarColumn(bar_width=None, style=P["900"], complete_style=P["500"], finished_style=TEAL,
                  pulse_style=P["300"]),
        MofNCompleteColumn(),
        TaskProgressColumn(style="sek.accent"),
        TextColumn("[sek.muted]•"),
        TimeElapsedColumn(),
        TextColumn("[sek.muted]eta"),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        expand=True,
    )


def fmt(v: float) -> str:
    """Compact, width-stable number formatting."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def summary_table(summary, metrics: Iterable[str], title: str = "Summary") -> Table:
    """Per-label table of mean [95 % CI] and median for the given metrics."""
    from .metrics import get_metric

    metrics = list(metrics)
    t = Table(title=Text(title, style="sek.brand"), border_style=P["800"], header_style=f"bold {P['200']}",
              row_styles=["", f"on {P['900']}"] if not _plain else None, pad_edge=True, expand=False)
    t.add_column("structure", style="sek.key", no_wrap=True)
    for m in metrics:
        info = get_metric(m)
        head = Text(info.abbr, style=f"bold {P['200']}")
        if info.arrow:
            head.append(f" {info.arrow}", style="sek.arrow")
        if info.unit:
            head.append(f"\n[{info.unit}]", style="sek.muted")
        t.add_column(head, justify="right", no_wrap=True)
    for label, g in summary.groupby("label", sort=False):
        g = g.set_index("metric")
        cells = []
        for m in metrics:
            if m not in g.index:
                cells.append(Text("–", style="sek.muted"))
                continue
            r = g.loc[m]
            c = Text(fmt(r["mean"]), style="sek.num")
            c.append(f"\n{fmt(r['ci_low'])}–{fmt(r['ci_high'])}", style="sek.muted")
            cells.append(c)
        t.add_row(str(label), *cells)
    t.caption = Text("mean · 95 % bootstrap CI of the mean", style="sek.muted")
    return t


def metric_table(infos, verbose: bool = False) -> Table:
    """Table of registered metrics (used by ``segevalkit metrics``)."""
    t = Table(border_style=P["800"], header_style=f"bold {P['200']}", expand=False, show_lines=False)
    t.add_column("key", style="sek.key", no_wrap=True)
    t.add_column("", style="sek.arrow", no_wrap=True)
    t.add_column("metric")
    t.add_column("unit", style="sek.muted", no_wrap=True)
    if verbose:
        t.add_column("in one sentence", style="grey78", ratio=1)
    for m in infos:
        row = [m.name, m.arrow, m.display, m.unit or ""]
        if verbose:
            row.append(m.summary)
        t.add_row(*row)
    return t


def panel(body, title: str, style: str = "700") -> Panel:
    return Panel(body, title=Text(f" {title} ", style="sek.brand"), border_style=P[style], padding=(1, 2),
                 title_align="left")
