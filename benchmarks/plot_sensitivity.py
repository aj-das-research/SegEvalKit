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

from segevalkit.metrics import get_metric  # noqa: E402
from segevalkit.plotting.theme import CATEGORICAL, INK, PURPLE, color_for, sequential_cmap, theme  # noqa: E402

ORDER_P = ["erode", "dilate", "boundary_noise", "shift", "islands", "remove_slab", "holes", "cut"]
TITLES = {"erode": "Erosion", "dilate": "Dilation", "boundary_noise": "Boundary noise", "shift": "Shift",
          "islands": "FP islands", "remove_slab": "Missing slab", "holes": "Internal holes", "cut": "Cut"}
ORDER_M = ["dice", "iou", "nsd", "boundary_iou", "hd", "hd95", "assd", "masd", "centroid_distance",
           "relative_volume_difference", "cldice", "betti0_error", "betti1_error", "betti2_error", "lesion_f1"]


# A representative, clinically plausible size for each error type.
REF_MAG = {"erode": 2.0, "dilate": 2.0, "boundary_noise": 2.0, "shift": 4.0, "islands": 2.0,
           "remove_slab": 0.1, "holes": 2.0, "cut": 4.0}
# The smallest change of each metric that a reader would call meaningful.
MEANINGFUL = {"dice": 0.02, "iou": 0.02, "nsd": 0.02, "boundary_iou": 0.02, "cldice": 0.02, "lesion_f1": 0.02,
              "relative_volume_difference": 0.02, "hd": 1.0, "hd95": 1.0, "assd": 1.0, "masd": 1.0,
              "centroid_distance": 1.0, "betti0_error": 1.0, "betti1_error": 1.0, "betti2_error": 1.0}


def responsiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Does the metric notice the error?

    For each (perturbation, metric): the fraction of structure-cases in which
    the metric changes, between the unperturbed mask and the perturbation at
    its representative size (REF_MAG), by at least the meaningful difference
    (MEANINGFUL: 0.02 for bounded scores, 1 mm for distances, 1 for counts).
    Also returns the median absolute change in the metric's own units.
    """
    rows = []
    for (p, m), g in df.groupby(["perturbation", "metric"]):
        if m not in MEANINGFUL or p not in REF_MAG:
            continue
        base = g[g.magnitude == 0].set_index(["structure", "case_id"])["value"]
        pert = g[np.isclose(g.magnitude, REF_MAG[p])].set_index(["structure", "case_id"])["value"]
        delta = (pert - base.reindex(pert.index)).abs().dropna()
        if delta.empty:
            continue
        rows.append({"perturbation": p, "metric": m, "rho": float((delta >= MEANINGFUL[m] - 1e-12).mean()),
                     "median_change": float(delta.median()), "n": int(delta.size)})
    return pd.DataFrame(rows)


def _mag_label(p: str) -> str:
    m = REF_MAG[p]
    return {"islands": f"{m:g} blobs", "holes": f"{m:g} holes", "remove_slab": f"{100 * m:g} %"}.get(p, f"{m:g} mm")


def matrix_figure(r: pd.DataFrame, out: Path):
    piv = r.pivot(index="metric", columns="perturbation", values="rho").reindex(index=ORDER_M, columns=ORDER_P)
    with theme():
        fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
        im = ax.imshow(piv.values, cmap=sequential_cmap(), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ORDER_P)))
        ax.set_xticklabels([f"{TITLES[p]}\n{_mag_label(p)}" for p in ORDER_P], rotation=0, fontsize=7.5)
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
        cb.set_label("fraction of real structures where the metric changes meaningfully")
        cb.outline.set_visible(False)
        ax.set_title("Which metric notices which error? (7 PanTS structures × 10 cases)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sensitivity_matrix.{ext}", dpi=190)
    plt.close(fig)


PAPER_TITLES = {"boundary_noise": "Boundary\nnoise", "islands": "FP\nislands", "remove_slab": "Missing\nslab",
                "holes": "Internal\nholes"}


def matrix_figure_paper(r: pd.DataFrame, out: Path, width: float = 4.8):
    """The same matrix drawn at its printed size (``width`` inches, no scaling in LaTeX), 7-8 pt text."""
    piv = r.pivot(index="metric", columns="perturbation", values="rho").reindex(index=ORDER_M, columns=ORDER_P)
    with theme():
        fig, ax = plt.subplots(figsize=(width, 0.8 * width), constrained_layout=True)
        im = ax.imshow(piv.values, cmap=sequential_cmap(), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ORDER_P)))
        ax.set_xticklabels([f"{PAPER_TITLES.get(p, TITLES[p])}\n{_mag_label(p)}" for p in ORDER_P], fontsize=6.5,
                           linespacing=1.05)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(ORDER_M)))
        ax.set_yticklabels([get_metric(m).abbr for m in ORDER_M], fontsize=7.5)
        ax.tick_params(length=0, pad=2)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color=INK["surface"] if v > 0.55 else INK["primary"])
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=30)
        cb.set_label("fraction of cases with a meaningful change", fontsize=7)
        cb.ax.tick_params(labelsize=6.5, length=2)
        cb.outline.set_visible(False)
    fig.savefig(out / "sensitivity_matrix_paper.pdf")
    fig.savefig(out / "sensitivity_matrix_paper.png", dpi=300)
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
            fig.suptitle(f"{TITLES[p]}: median over cases", x=0.01, ha="left", fontsize=11, fontweight="bold",
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
            bp = ax.boxplot(vals, orientation="horizontal", patch_artist=True, widths=0.55,
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
                     x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK["primary"])
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
    # Supplementary runs (e.g. sensitivity_betti2_error.csv) add metrics on the same cases.
    for extra in sorted(Path(a.csv).parent.glob("sensitivity_*.csv")):
        df = pd.concat([df, pd.read_csv(extra)], ignore_index=True)
    r = responsiveness(df)
    r.to_csv(out / "sensitivity_summary.csv", index=False)
    matrix_figure(r, out)
    matrix_figure_paper(r, out)
    curves(df, out)
    size_bias(df, out)
    print(r.pivot(index="metric", columns="perturbation", values="rho").round(2))


if __name__ == "__main__":
    main()
