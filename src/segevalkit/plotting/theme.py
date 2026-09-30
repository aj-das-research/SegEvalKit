"""The SegEvalKit visual theme (purple-led, colour-vision-deficiency checked).

Colour is assigned by the *job* it does:

* **Categorical** (identity of a method / label): :data:`CATEGORICAL`, used in
  fixed order and never cycled. Violet leads. The order was validated for
  adjacent-pair separation under protan/deutan/tritan simulation (ΔE ≥ 11) and
  the first three slots also pass the stricter all-pairs check, so scatter
  plots with ≤ 3 series are safe.
* **Sequential** (magnitude): one purple ramp, light to dark (:data:`SEQUENTIAL`).
* **Diverging** (signed values, correlations): orange ↔ grey ↔ purple.
* **Segmentation errors**: true positive = violet, false negative (missed) =
  orange, false positive (spurious) = teal, the all-pairs-safe first three slots.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Sequence

import matplotlib as mpl

__all__ = [
    "CATEGORICAL", "PURPLE", "SEQUENTIAL", "ERROR_COLORS", "INK",
    "apply_theme", "theme", "categorical", "sequential_cmap", "diverging_cmap", "color_for",
]

#: Categorical slots (light surface): violet, teal, orange, blue, magenta, gold.
CATEGORICAL: List[str] = ["#6d3fd6", "#139a8a", "#e2712f", "#3f7fd6", "#cf3f8f", "#b88a00"]

#: Brand purple ramp (50 → 900).
PURPLE: Dict[int, str] = {
    50: "#f5f1fe", 100: "#ebe3fd", 200: "#d6c7fb", 300: "#b89cf5", 400: "#9a72ee",
    500: "#7c4ee4", 600: "#6d3fd6", 700: "#5a2fb8", 800: "#472594", 900: "#2e1766",
}

#: Sequential ramp (light → dark purple) for heatmaps.
SEQUENTIAL: List[str] = [PURPLE[k] for k in (50, 100, 200, 300, 400, 500, 600, 700, 800, 900)]

#: Overlay colours for segmentation error maps.
ERROR_COLORS: Dict[str, str] = {"tp": "#6d3fd6", "fn": "#e2712f", "fp": "#139a8a",
                                "ref": "#b89cf5", "pred": "#139a8a"}

#: Text / structural inks.
INK: Dict[str, str] = {"primary": "#1d1733", "secondary": "#4f4868", "muted": "#8a84a3",
                       "grid": "#e9e5f3", "surface": "#ffffff", "panel": "#fbfaff",
                       "neutral": "#efedf3"}

_RC = {
    "figure.facecolor": INK["surface"],
    "axes.facecolor": INK["surface"],
    "savefig.facecolor": INK["surface"],
    "axes.edgecolor": INK["muted"],
    "axes.labelcolor": INK["secondary"],
    "axes.titlecolor": INK["primary"],
    "axes.titleweight": "semibold",
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": INK["grid"],
    "grid.linewidth": 0.8,
    "xtick.color": INK["secondary"],
    "ytick.color": INK["secondary"],
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "text.color": INK["primary"],
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "patch.linewidth": 0,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "image.cmap": "sek_purple",
}


def sequential_cmap(name: str = "sek_purple"):
    """Light → dark purple colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, SEQUENTIAL)


def diverging_cmap(name: str = "sek_diverging"):
    """Orange ← neutral grey → purple colormap (for signed values)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        name, ["#b4541c", "#e2712f", "#f3c3a3", INK["neutral"], PURPLE[200], PURPLE[600], PURPLE[900]])


def _register_cmaps() -> None:
    for cmap in (sequential_cmap(), diverging_cmap()):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)


def apply_theme() -> None:
    """Apply the SegEvalKit theme globally to matplotlib."""
    _register_cmaps()
    mpl.rcParams.update(_RC)


@contextmanager
def theme():
    """Context manager applying the theme temporarily."""
    _register_cmaps()
    with mpl.rc_context(_RC):
        yield


def categorical(n: int) -> List[str]:
    """First ``n`` categorical colours; more than six series should be faceted."""
    if n > len(CATEGORICAL):
        # Beyond six identities colour cannot carry meaning reliably; repeat
        # with the expectation that callers facet or label directly.
        return [CATEGORICAL[i % len(CATEGORICAL)] for i in range(n)]
    return CATEGORICAL[:n]


def color_for(names: Sequence[str]) -> Dict[str, str]:
    """Stable name → colour mapping in first-seen order (colour follows the entity)."""
    uniq = list(dict.fromkeys(names))
    return dict(zip(uniq, categorical(len(uniq))))
