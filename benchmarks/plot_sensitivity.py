"""Figures and tables for the metric sensitivity study (see metric_sensitivity.py).

Outputs (``--out``):
    sensitivity_matrix.png/pdf    which metric responds to which error (|Spearman rho|)
    sensitivity_<perturbation>.png  metric-vs-magnitude curves per structure type
    size_bias_erode.png           Dice vs NSD vs HD95 under the same 2 mm erosion, by structure
    sensitivity_summary.csv       the numbers behind the matrix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from segevalkit.metrics import get_metric  # noqa: E402
from segevalkit.plotting.theme import CATEGORICAL, INK, PURPLE, color_for, sequential_cmap, theme  # noqa: E402

ORDER_P = ["erode", "dilate", "boundary_noise", "shift", "islands", "remove_slab", "holes", "cut"]
TITLES = {"erode": "Erosion", "dilate": "Dilation", "boundary_noise": "Boundary noise", "shift": "Shift",
          "islands": "FP islands", "remove_slab": "Missing slab", "holes": "Internal holes", "cut": "Cut"}
ORDER_M = ["dice", "iou", "nsd", "boundary_iou", "hd", "hd95", "assd", "masd", "centroid_distance",
           "relative_volume_difference", "cldice", "betti0_error", "betti1_error", "lesion_f1"]


def responsiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Median over (structure, case) of |Spearman rho(magnitude, metric)|; NaN-safe."""
    rows = []
    for (p, m), g in df.groupby(["perturbation", "metric"]):
        rhos = []
        for _, h in g.groupby(["structure", "case_id"]):
            h = h.dropna(subset=["value"])
            if h["value"].nunique() < 2:
                rhos.append(0.0)  # metric did not move at all
                continue
            rhos.append(abs(spearmanr(h["magnitude"], h["value"]).statistic))
        rows.append({"perturbation": p, "metric": m, "rho": float(np.nanmedian(rhos)), "n": len(rhos)})
    return pd.DataFrame(rows)


def matrix_figure(r: pd.DataFrame, out: Path):
    piv = r.pivot(index="metric", columns="perturbation", values="rho").reindex(index=ORDER_M, columns=ORDER_P)
    with theme():
        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
        im = ax.imshow(piv.values, cmap=sequential_cmap(), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ORDER_P)))
        ax.set_xticklabels([TITLES[p] for p in ORDER_P], rotation=30, ha="right")
        ax.set_yticks(range(len(ORDER_M)))
        ax.set_yticklabels([f"{get_metric(m).abbr}" for m in ORDER_M])
        ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color=INK["surface"] if v > 0.55 else INK["primary"])
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("median |Spearman ρ| between error size and metric")
        cb.outline.set_visible(False)
        ax.set_title("Which metric notices which error? (real PanTS anatomy)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sensitivity_matrix.{ext}", dpi=190)
    plt.close(fig)


def curves(df: pd.DataFrame, out: Path):
    kinds = {"liver": "large organ", "kidney_left": "compact organ", "pancreas": "elongated organ",
             "gall_bladder": "small organ", "aorta": "tubular", "veins": "branching tubular",
             "pancreatic_lesion": "lesion"}
    colors = dict(zip(kinds, [PURPLE[800], CATEGORICAL[0], CATEGORICAL[3], CATEGORICAL[1], CATEGORICAL[2],
                              CATEGORICAL[4], CATEGORICAL[5]]))
    for p in ORDER_P:
        d = df[df.perturbation == p]
        if d.empty:
            continue
        ms = ["dice", "nsd", "hd95", "assd", "relative_volume_difference", "cldice", "betti0_error"]
        with theme():
            fig, axes = plt.subplots(1, len(ms), figsize=(2.35 * len(ms), 2.7), constrained_layout=True)
            for ax, m in zip(axes, ms):
                for s, c in colors.items():
                    g = d[(d.metric == m) & (d.structure == s)].groupby("magnitude")["value"]
                    if not len(g):
                        continue
                    med = g.median()
                    ax.plot(med.index, med.values, color=c, marker="o", markersize=3, linewidth=1.6,
                            label=kinds[s])
                info = get_metric(m)
                ax.set_title(info.label, fontsize=9)
                ax.set_xlabel(d["unit"].iloc[0], fontsize=8)
                ax.grid(True, axis="both")
            axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5, title="structure")
            fig.suptitle(f"{TITLES[p]}: median over cases", x=0.01, ha="left", fontsize=11, fontweight="semibold",
                         color=INK["primary"])
        for ext in ("png", "pdf"):
            fig.savefig(out / f"sensitivity_{p}.{ext}", dpi=170)
        plt.close(fig)


def size_bias(df: pd.DataFrame, out: Path, mag: float = 2.0):
    d = df[(df.perturbation == "erode") & (df.magnitude == mag) & (df.metric.isin(["dice", "nsd", "hd95"]))]
    order = (d[d.metric == "dice"].groupby("structure")["ref_volume_ml"].median().sort_values(ascending=False).index)
    with theme():
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True, sharey=True)
        for ax, m in zip(axes, ["dice", "nsd", "hd95"]):
            vals = [d[(d.metric == m) & (d.structure == s)]["value"].to_numpy() for s in order]
            bp = ax.boxplot(vals, vert=False, patch_artist=True, widths=0.55,
                            medianprops=dict(color="white", linewidth=1.5), showfliers=False)
            for b in bp["boxes"]:
                b.set_facecolor(PURPLE[500])
                b.set_edgecolor(PURPLE[500])
            for w in bp["whiskers"]:
                w.set_color(PURPLE[500])
            ax.set_yticks(range(1, len(order) + 1))
            ax.set_yticklabels([s.replace("_", " ") for s in order])
            ax.set_title(get_metric(m).label)
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            ax.invert_yaxis()
        fig.suptitle(f"The same {mag:g} mm erosion, seen by three metrics (structures ordered by volume)",
                     x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK["primary"])
    for ext in ("png", "pdf"):
        fig.savefig(out / f"size_bias_erode.{ext}", dpi=190)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/sensitivity/sensitivity.csv")
    ap.add_argument("--out", default="docs/assets/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    r = responsiveness(df)
    r.to_csv(out / "sensitivity_summary.csv", index=False)
    matrix_figure(r, out)
    curves(df, out)
    size_bias(df, out)
    print(r.pivot(index="metric", columns="perturbation", values="rho").round(2))


if __name__ == "__main__":
    main()
