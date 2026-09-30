"""Quantitative plots of evaluation results.

Every function accepts either one `EvaluationResult`
or a mapping ``{method_name: EvaluationResult}`` for comparisons, returns the
matplotlib ``Figure``, draws on ``ax=`` when given, and labels axes with the
metric's unit and better-direction arrow from the metric registry.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..metrics import get_metric
from .theme import CATEGORICAL, ERROR_COLORS, INK, PURPLE, color_for, diverging_cmap, sequential_cmap, theme

__all__ = [
    "metric_distribution",
    "metric_heatmap",
    "metric_vs_size",
    "metric_correlation",
    "volume_agreement",
    "bland_altman_plot",
    "ecdf",
    "metric_profile",
    "comparison_forest",
    "ranking_stability_plot",
    "reliability_diagram",
    "detection_by_size",
    "failure_quadrants",
    "failure_thresholds",
    "FAILURE_CLASSES",
    "sensitivity_curves",
    "cohort_plot",
]

Results = Union[object, Mapping[str, object]]


def _long(results: Results) -> pd.DataFrame:
    if hasattr(results, "per_case"):
        return results.per_case().assign(method=results.name)
    frames = [r.per_case().assign(method=name) for name, r in results.items()]
    return pd.concat(frames, ignore_index=True)


def _wide(results: Results) -> pd.DataFrame:
    if hasattr(results, "wide"):
        return results.wide().assign(method=results.name)
    return pd.concat([r.wide().assign(method=n) for n, r in results.items()], ignore_index=True)


def _new_ax(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def _methods(df):
    return list(dict.fromkeys(df["method"]))


# --------------------------------------------------------------- distributions
def metric_distribution(results: Results, metric: str, labels: Optional[Sequence[str]] = None,
                        kind: str = "raincloud", ax=None, figsize=None, max_points: int = 400):
    """Per-label distribution of one metric, one colour per method.

    ``kind``: ``"raincloud"`` (half-violin + box + jittered cases, the
    default: shows shape, summary and every case), ``"box"``, ``"violin"``
    or ``"strip"``.
    """
    info = get_metric(metric)
    df = _long(results)
    df = df[df["metric"] == metric].dropna(subset=["value"])
    labels = list(labels) if labels is not None else list(dict.fromkeys(df["label"]))
    methods = _methods(df)
    colors = color_for(methods)
    with theme():
        width = min(11.0, max(4.5, 0.55 * len(labels) * max(1, len(methods)) + 2.2))
        fig, ax = _new_ax(ax, figsize or (width, 3.6))
        width = 0.8 / len(methods)
        rng = np.random.default_rng(0)
        for mi, m in enumerate(methods):
            for li, lab in enumerate(labels):
                v = df[(df["method"] == m) & (df["label"] == lab)]["value"].to_numpy(float)
                if v.size == 0:
                    continue
                x0 = li + (mi - (len(methods) - 1) / 2) * width
                c = colors[m]
                if kind in ("raincloud", "violin") and v.size > 2 and np.ptp(v) > 0:
                    parts = ax.violinplot(v, positions=[x0], widths=width * 0.95, showextrema=False)
                    for b in parts["bodies"]:
                        b.set_facecolor(c)
                        b.set_alpha(0.22 if kind == "raincloud" else 0.45)
                        b.set_edgecolor("none")
                        if kind == "raincloud":  # keep the left half only
                            verts = b.get_paths()[0].vertices
                            verts[:, 0] = np.minimum(verts[:, 0], x0)
                if kind in ("raincloud", "box"):
                    bp = ax.boxplot(v, positions=[x0 + (width * 0.14 if kind == "raincloud" else 0)],
                                    widths=width * (0.22 if kind == "raincloud" else 0.6),
                                    patch_artist=True, showfliers=kind == "box", manage_ticks=False,
                                    medianprops=dict(color=INK["surface"], linewidth=1.6),
                                    whiskerprops=dict(color=c, linewidth=1.2), capprops=dict(color=c, linewidth=0),
                                    flierprops=dict(marker="o", markersize=3, markerfacecolor=c,
                                                    markeredgecolor="none", alpha=0.6))
                    for patch in bp["boxes"]:
                        patch.set_facecolor(c)
                        patch.set_edgecolor(c)
                if kind in ("raincloud", "strip"):
                    pts = v if v.size <= max_points else rng.choice(v, max_points, replace=False)
                    off = (width * 0.32) if kind == "raincloud" else 0
                    jit = rng.uniform(-width * 0.12, width * 0.12, pts.size)
                    ax.scatter(np.full(pts.size, x0 + off) + jit, pts, s=9, color=c, alpha=0.55,
                               edgecolors="none", zorder=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylabel(info.label)
        ax.set_title(info.display)
        if len(methods) > 1:
            from matplotlib.patches import Patch

            ax.legend(handles=[Patch(color=colors[m], label=m) for m in methods],
                      loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Method")
    return fig


def ecdf(results: Results, metric: str, label: Optional[str] = None, ax=None, figsize=(4.8, 3.4)):
    """Empirical CDF per method: compares whole distributions, including tails."""
    info = get_metric(metric)
    df = _long(results)
    df = df[df["metric"] == metric].dropna(subset=["value"])
    if label is not None:
        df = df[df["label"] == label]
    methods = _methods(df)
    colors = color_for(methods)
    with theme():
        fig, ax = _new_ax(ax, figsize)
        for m in methods:
            v = np.sort(df[df["method"] == m]["value"].to_numpy(float))
            ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=colors[m], label=m)
        ax.set_xlabel(info.label)
        ax.set_ylabel("Fraction of cases ≤ x")
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="both")
        ax.set_title(f"Cumulative distribution of {info.abbr}" + (f": {label}" if label else ""))
        if len(methods) > 1:
            ax.legend(loc="best")
    return fig


# ----------------------------------------------------------------- heatmaps
def metric_heatmap(result, metric: str, max_cases: int = 60, sort: bool = True, ax=None, figsize=None):
    """Case × label heatmap of one metric (worst cases first when ``sort``)."""
    import matplotlib.pyplot as plt

    info = get_metric(metric)
    w = result.per_case()
    w = w[w["metric"] == metric].pivot_table(index="case_id", columns="label", values="value")
    if sort and info.better in ("higher", "lower"):
        w = w.loc[w.mean(axis=1).sort_values(ascending=info.better == "higher").index]
    w = w.head(max_cases)
    with theme():
        fig, ax = _new_ax(ax, figsize or (1.2 + 0.55 * w.shape[1], 1.0 + 0.18 * w.shape[0]))
        cmap = sequential_cmap()
        if info.better == "lower":
            cmap = cmap.reversed()
        vmin, vmax = info.value_range if np.isfinite(info.value_range[1]) else (np.nanmin(w.values), np.nanmax(w.values))
        im = ax.imshow(w.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(w.shape[1]))
        ax.set_xticklabels(w.columns, rotation=40, ha="right")
        ax.set_yticks(range(w.shape[0]))
        ax.set_yticklabels(w.index, fontsize=6.5)
        ax.grid(False)
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(info.label)
        cb.outline.set_visible(False)
        ax.set_title(f"{info.abbr} per case and structure")
    return fig


def metric_correlation(result, metrics: Optional[Sequence[str]] = None, label: Optional[str] = None,
                       method: str = "spearman", ax=None, figsize=None):
    """Rank correlation between metrics across cases: which metrics carry redundant information?"""
    import matplotlib.pyplot as plt

    w = result.wide()
    if label is not None:
        w = w[w["label"] == label]
    metrics = [m for m in (metrics or result.metrics) if m in w.columns and w[m].nunique() > 1]
    c = w[metrics].corr(method=method)
    with theme():
        n = len(metrics)
        fig, ax = _new_ax(ax, figsize or (1.5 + 0.42 * n, 1.2 + 0.4 * n))
        im = ax.imshow(c.values, cmap=diverging_cmap(), vmin=-1, vmax=1)
        abbr = [get_metric(m).abbr for m in metrics]
        ax.set_xticks(range(n))
        ax.set_xticklabels(abbr, rotation=55, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(abbr)
        ax.grid(False)
        if n <= 16:
            for i in range(n):
                for j in range(n):
                    v = c.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color=INK["surface"] if abs(v) > 0.6 else INK["primary"])
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(f"{method.capitalize()} ρ")
        cb.outline.set_visible(False)
        ax.set_title("Metric correlation across cases" + (f": {label}" if label else ""))
    return fig


# ------------------------------------------------------------- size effects
def metric_vs_size(results: Results, metric: str, label: Optional[str] = None,
                   size: str = "ref_volume_ml", logx: bool = True, n_bins: int = 8, ax=None,
                   figsize=(5.0, 3.5)):
    """Metric against structure size with a binned median trend.

    The classic diagnostic for size bias: Dice and IoU degrade on small
    structures even for boundary errors of constant thickness.
    """
    info = get_metric(metric)
    w = _wide(results)
    if label is not None:
        w = w[w["label"] == label]
    w = w.dropna(subset=[metric, size])
    w = w[w[size] > 0] if logx else w
    methods = _methods(w)
    colors = color_for(methods)
    with theme():
        fig, ax = _new_ax(ax, figsize)
        for m in methods:
            d = w[w["method"] == m]
            ax.scatter(d[size], d[metric], s=12, color=colors[m], alpha=0.45, edgecolors="none",
                       label=m if len(methods) > 1 else None)
            if len(d) >= 2 * n_bins:
                x = np.log10(d[size]) if logx else d[size]
                edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
                idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
                cx = [np.median(d[size].to_numpy()[idx == b]) for b in range(n_bins) if np.any(idx == b)]
                cy = [np.median(d[metric].to_numpy()[idx == b]) for b in range(n_bins) if np.any(idx == b)]
                ax.plot(cx, cy, color=colors[m], linewidth=2.2, marker="o", markersize=5,
                        markeredgecolor=INK["surface"], markeredgewidth=1.5)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("Reference volume [mL]" if size == "ref_volume_ml" else size)
        ax.set_ylabel(info.label)
        ax.grid(True, axis="both")
        ax.set_title(f"{info.abbr} vs. structure size" + (f": {label}" if label else ""))
        if len(methods) > 1:
            ax.legend(loc="best")
    return fig


# ------------------------------------------------------------------ volumes
def volume_agreement(result, label: str, ax=None, figsize=(4.0, 3.8)):
    """Predicted vs reference volume with the identity line, Pearson r and ICC(2,1)."""
    from ..stats import icc

    w = result.wide()
    w = w[w["label"] == label]
    x, y = w["ref_volume_ml"].to_numpy(float), w["pred_volume_ml"].to_numpy(float)
    with theme():
        fig, ax = _new_ax(ax, figsize)
        hi = np.nanmax(np.concatenate([x, y])) * 1.05 if x.size else 1
        ax.plot([0, hi], [0, hi], color=INK["muted"], linewidth=1, linestyle="--", zorder=1)
        ax.scatter(x, y, s=16, color=PURPLE[600], alpha=0.6, edgecolors="none", zorder=2)
        r = np.corrcoef(x, y)[0, 1] if x.size > 1 else np.nan
        ax.set_xlabel("Reference volume [mL]")
        ax.set_ylabel("Predicted volume [mL]")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.grid(True, axis="both")
        ax.set_title(f"Volume agreement: {label}")
        ax.text(0.04, 0.96, f"r = {r:.3f}\nICC(2,1) = {icc(y, x):.3f}\nn = {x.size}", transform=ax.transAxes,
                va="top", fontsize=8.5, color=INK["secondary"])
    return fig


def bland_altman_plot(result, label: str, relative: bool = False, ax=None, figsize=(5.0, 3.5)):
    """Bland-Altman plot of predicted − reference volume (mL, or % with ``relative``)."""
    from ..stats import bland_altman

    w = result.wide()
    w = w[w["label"] == label]
    x, y = w["ref_volume_ml"].to_numpy(float), w["pred_volume_ml"].to_numpy(float)
    if relative:
        mean = (x + y) / 2
        ok = mean > 0
        ba = bland_altman(100 * (y[ok] - x[ok]) / mean[ok], np.zeros(int(ok.sum())))
        ba["mean"] = mean[ok].tolist()
    else:
        ba = bland_altman(y, x)
    with theme():
        fig, ax = _new_ax(ax, figsize)
        ax.scatter(ba["mean"], ba["diff"], s=16, color=PURPLE[600], alpha=0.6, edgecolors="none")
        for v, ls, txt in ((ba["bias"], "-", "bias"), (ba["loa_low"], "--", "−1.96 SD"), (ba["loa_high"], "--", "+1.96 SD")):
            ax.axhline(v, color=PURPLE[800] if txt == "bias" else INK["muted"], linestyle=ls, linewidth=1.2)
            ax.text(1.0, v, f" {txt}: {v:.2f}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7.5, color=INK["secondary"])
        ax.axhline(0, color=INK["grid"], linewidth=1)
        ax.set_xlabel("Mean of predicted and reference volume [mL]")
        ax.set_ylabel("Predicted − reference " + ("[% of mean]" if relative else "[mL]"))
        ax.set_title(f"Bland–Altman: {label}")
    return fig


# --------------------------------------------------------------- comparisons
def metric_profile(results: Mapping[str, object], metrics: Sequence[str], label: Optional[str] = None,
                   figsize=None):
    """Small multiples: mean ± 95 % bootstrap CI of several metrics, one row per method.

    A faithful replacement for radar charts: each metric keeps its own axis,
    unit and direction.
    """
    import matplotlib.pyplot as plt

    from ..stats import bootstrap_ci

    df = _long(results)
    if label is not None:
        df = df[df["label"] == label]
    methods = _methods(df)
    colors = color_for(methods)
    with theme():
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize or (2.3 * len(metrics), 0.6 + 0.42 * len(methods)),
                                 constrained_layout=True, sharey=True, squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            info = get_metric(metric)
            for i, m in enumerate(methods):
                v = df[(df["method"] == m) & (df["metric"] == metric)]["value"].dropna().to_numpy(float)
                if not v.size:
                    continue
                lo, hi = bootstrap_ci(v)
                ax.plot([lo, hi], [i, i], color=colors[m], linewidth=2.2, solid_capstyle="round")
                ax.plot(v.mean(), i, "o", color=colors[m], markersize=7, markeredgecolor=INK["surface"],
                        markeredgewidth=1.5)
            ax.set_title(info.label, fontsize=9.5)
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods)
            ax.invert_yaxis()
        fig.suptitle("Metric profile (mean, 95 % CI)" + (f": {label}" if label else ""), x=0.01, ha="left",
                     fontsize=11, fontweight="bold", color=INK["primary"])
    return fig


def comparison_forest(compare_df: pd.DataFrame, ax=None, figsize=None, name_a: str = "A", name_b: str = "B"):
    """Forest plot of paired mean differences (A − B) with CIs from `segevalkit.stats.compare`.

    Filled markers are significant after correction.
    """
    d = compare_df.reset_index(drop=True)
    with theme():
        fig, ax = _new_ax(ax, figsize or (5.6, 0.6 + 0.32 * len(d)))
        for i, r in d.iterrows():
            info = get_metric(r["metric"])
            better_a = r["frac_a_better"] >= 0.5
            c = CATEGORICAL[0] if better_a else CATEGORICAL[2]
            # Normalise by the pooled mean magnitude so metrics of different units share an axis.
            scale = max(abs(r["mean_a"]), abs(r["mean_b"]), 1e-9)
            lo, hi, m = r["diff_ci_low"] / scale, r["diff_ci_high"] / scale, r["mean_diff"] / scale
            ax.plot([lo * 100, hi * 100], [i, i], color=c, linewidth=2, solid_capstyle="round")
            ax.plot(m * 100, i, "o", markersize=7, color=c if r.get("significant", False) else INK["surface"],
                    markeredgecolor=c, markeredgewidth=1.8)
            ax.text(1.01, i, f"p={r['p_adjusted']:.2g}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7.5, color=INK["secondary"])
            d.loc[i, "_tick"] = f"{r['label']} · {info.abbr} {info.arrow}"
        ax.axvline(0, color=INK["muted"], linewidth=1)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["_tick"])
        ax.invert_yaxis()
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_xlabel(f"Mean paired difference {name_a} − {name_b} [% of larger mean]")
        ax.set_title(f"{name_a} vs {name_b}: paired differences\n(filled = significant after correction)")
    return fig


def ranking_stability_plot(stability: Dict, ax=None, figsize=None):
    """Blob plot of bootstrap rank frequencies (Wiesenfarth et al. 2021)."""
    ranks = stability["ranks"]
    ref = stability["reference"]
    methods = list(ref.sort_values().index)
    n = len(methods)
    with theme():
        fig, ax = _new_ax(ax, figsize or (1.4 + 0.9 * n, 2.4 + 0.3 * n))
        for i, m in enumerate(methods):
            counts = ranks[m].value_counts(normalize=True)
            for rk, f in counts.items():
                ax.scatter(i, rk, s=1400 * f / n ** 0.5, color=PURPLE[600], alpha=0.35 + 0.6 * f,
                           edgecolors="none", zorder=2)
            ax.plot([i - 0.3, i + 0.3], [ref[m]] * 2, color=CATEGORICAL[2], linewidth=2, zorder=3)
            lo, hi = np.quantile(ranks[m], [0.025, 0.975])
            ax.plot([i, i], [lo, hi], color=PURPLE[900], linewidth=1, zorder=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(methods, rotation=20 if n > 3 else 0)
        ax.set_yticks(range(1, n + 1))
        ax.set_ylim(n + 0.6, 0.4)
        ax.set_ylabel("Rank (1 = best)")
        tau = np.nanmedian(stability["kendall_tau"])
        ax.set_title(f"Ranking stability over bootstrap samples\n(median Kendall τ = {tau:.2f})")
    return fig


# --------------------------------------------------------------- calibration
def reliability_diagram(p: np.ndarray, y: np.ndarray, n_bins: int = 15, ax=None, figsize=(4.0, 3.8)):
    """Top-label reliability diagram with the per-bin count shown as bar shading."""
    from ..metrics.calibration import reliability_curve

    c = reliability_curve(p, y, n_bins)
    centers = (c["edges"][:-1] + c["edges"][1:]) / 2
    width = np.diff(c["edges"])
    ok = c["count"] > 0
    w = c["count"] / c["count"].sum()
    ece_val = float(np.sum(w[ok] * np.abs(c["accuracy"][ok] - c["confidence"][ok])))
    with theme():
        fig, ax = _new_ax(ax, figsize)
        ax.bar(centers[ok], c["accuracy"][ok], width=width[ok] * 0.92, color=PURPLE[500], alpha=0.85,
               label="Accuracy")
        ax.bar(centers[ok], (c["confidence"] - c["accuracy"])[ok], bottom=c["accuracy"][ok],
               width=width[ok] * 0.92, color=CATEGORICAL[2], alpha=0.35, label="Gap to confidence")
        ax.plot([0.5, 1], [0.5, 1], color=INK["muted"], linestyle="--", linewidth=1)
        ax.set_xlim(0.5, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability (ECE = {ece_val:.3f})")
        ax.legend(loc="upper left")
    return fig


# ----------------------------------------------------------------- lesions
def detection_by_size(results: Results, label: Optional[str] = None,
                      bins_ml: Sequence[float] = (0, 0.1, 0.5, 1, 5, 20, np.inf), ax=None, figsize=(5.0, 3.4)):
    """Lesion detection rate (sensitivity) per reference-lesion size bin, per method."""
    items = {results.name: results} if hasattr(results, "lesions") else dict(results)
    colors = color_for(list(items))
    with theme():
        fig, ax = _new_ax(ax, figsize)
        names = [f"{a:g}–{b:g}" if np.isfinite(b) else f">{a:g}" for a, b in zip(bins_ml[:-1], bins_ml[1:])]
        width = 0.8 / max(1, len(items))
        for k, (name, r) in enumerate(items.items()):
            les = r.lesions
            if not len(les):
                continue
            les = les[les["kind"] == "ref"]
            if label is not None:
                les = les[les["label"] == label]
            b = pd.cut(les["volume_ml"], bins=list(bins_ml), labels=names, include_lowest=True)
            rate = les.groupby(b, observed=False)["detected"].mean()
            n = les.groupby(b, observed=False)["detected"].size()
            x = np.arange(len(names)) + (k - (len(items) - 1) / 2) * width
            ax.bar(x, rate.to_numpy(float), width=width * 0.9, color=colors[name], label=name)
            for xi, ni, ri in zip(x, n, rate.to_numpy(float)):
                # count above the bar, in text ink, so it stays visible on short or empty bars
                top = ri if np.isfinite(ri) else 0.0
                ax.text(xi, min(top, 0.93) + 0.015, f"n={ni}", ha="center", va="bottom", fontsize=6.5,
                        color=INK["secondary"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Reference lesion volume [mL]")
        ax.set_ylabel("Detection rate ↑")
        ax.set_title("Lesion detection by size" + (f": {label}" if label else ""))
        if len(items) > 1:
            ax.legend(loc="upper left")
    return fig


# ------------------------------------------------------------ failure thresholds
#: A case *fails* when it is worse than a uniform boundary error of ``error_mm``,
#: the largest boundary error accepted for that structure class. HD95 of a
#: uniform error of e mm is about e mm, so ``hd95 = error_mm``; ``dice`` is the
#: median Dice that the same uniform error produces on real PanTS masks (mean of
#: erosion and dilation by ``error_mm``, SegEvalKit sensitivity study,
#: docs/guide/sensitivity-study.md), rounded down to 0.05. ``error_mm`` itself
#: is a documented SegEvalKit convention (5 mm for organs, 3 mm for vessels and
#: lesions), not a published clinical standard: set it from your clinical
#: tolerance, e.g. inter-rater variability (Nikolov et al. 2021).
FAILURE_CLASSES: Dict[str, Dict[str, float]] = {
    "large_organ": {"error_mm": 5.0, "dice": 0.85, "hd95": 5.0},      # liver: Dice 0.860 at 5 mm
    "compact_organ": {"error_mm": 5.0, "dice": 0.65, "hd95": 5.0},    # kidney: 0.687
    "elongated_organ": {"error_mm": 5.0, "dice": 0.50, "hd95": 5.0},  # pancreas: 0.546
    "small_organ": {"error_mm": 5.0, "dice": 0.35, "hd95": 5.0},      # gallbladder: 0.363
    "vessel": {"error_mm": 3.0, "dice": 0.75, "hd95": 3.0},           # aorta: 0.755 at 3 mm
    "small_vessel": {"error_mm": 3.0, "dice": 0.45, "hd95": 3.0},     # veins: 0.462
    "lesion": {"error_mm": 3.0, "dice": 0.65, "hd95": 3.0},           # pancreatic lesion: 0.673
}

#: Name fragments → structure class (first match wins; checked on the lower-cased label).
FAILURE_CLASS_KEYS = [
    (("lesion", "tumo", "cyst", "nodule", "metast", "cancer", "enhancing", "necro", "edema"), "lesion"),
    (("vein", "portal", "splenic_v", "mesenteric", "renal_v", "iliac", "hepatic_v"), "small_vessel"),
    (("aorta", "cava", "postcava", "ivc", "artery", "vessel", "carotid", "pulmonary_a"), "vessel"),
    (("gall", "adrenal", "thyroid", "bile", "duct", "prostate", "optic", "cochlea", "parotid"), "small_organ"),
    (("pancrea", "duodenum", "esophag", "oesophag", "bowel", "intestin", "rectum", "colon", "sigmoid"),
     "elongated_organ"),
    (("kidney", "bladder", "heart", "uterus", "spinal", "vertebra"), "compact_organ"),
    (("liver", "spleen", "stomach", "lung", "brain", "muscle", "femur"), "large_organ"),
]


def failure_thresholds(label: str, overrides: Optional[Mapping[str, Mapping[str, float]]] = None) -> Dict:
    """Dice and HD95 failure thresholds for one structure.

    Resolution order: ``overrides[label]``, then the structure class matched from
    the label name (`FAILURE_CLASS_KEYS`), then ``compact_organ``. Returns
    ``{"dice", "hd95", "error_mm", "class"}``; an override may give any subset.
    """
    cls = next((c for keys, c in FAILURE_CLASS_KEYS if any(k in label.lower() for k in keys)), None)
    out = dict(FAILURE_CLASSES[cls or "compact_organ"], **{"class": cls or "compact_organ (default)"})
    if overrides and label in overrides:
        out.update(overrides[label])
        out["class"] = "user-defined"
    return out


def failure_quadrants(result, label: str, x: str = "dice", y: str = "hd95", x_thr: Optional[float] = None,
                      y_thr: Optional[float] = None, thresholds: Optional[Mapping[str, Mapping[str, float]]] = None,
                      ax=None, figsize=(4.8, 3.8)):
    """Overlap vs boundary error scatter, with per-structure failure thresholds.

    A case is *flagged* when it is worse than a uniform boundary error of
    ``error_mm`` (the largest error accepted for its structure class): Dice
    below the Dice such an error gives, or HD95 above ``error_mm``. Defaults
    come from `failure_thresholds` (`FAILURE_CLASSES`: 5 mm for organs, 3 mm
    for vessels and lesions, with Dice levels measured on real PanTS masks);
    pass ``thresholds={"pancreas": {"dice": 0.7, "hd95": 10}}`` or explicit
    ``x_thr`` / ``y_thr`` to use your clinical tolerance instead.

    Cases with good Dice but large HD95 have distant spurious fragments; poor
    Dice with small HD95 are systematic boundary shifts or under-segmentation.
    For metrics other than Dice / HD95 without explicit thresholds, the x
    threshold is 0.7 and the y threshold twice the median.
    """
    xi, yi = get_metric(x), get_metric(y)
    w = result.wide()
    w = w[w["label"] == label].dropna(subset=[x, y])
    t = failure_thresholds(label, thresholds)
    explicit = x_thr is not None or y_thr is not None
    if x_thr is None:
        x_thr = t["dice"] if x == "dice" else 0.7
    if y_thr is None:
        y_thr = t["hd95"] if y == "hd95" else float(np.nanmedian(w[y])) * 2
    with theme():
        fig, ax = _new_ax(ax, figsize)
        bad = (w[x] < x_thr) | (w[y] > y_thr)
        ax.scatter(w.loc[~bad, x], w.loc[~bad, y], s=16, color=PURPLE[600], alpha=0.6, edgecolors="none",
                   label="Within tolerance")
        ax.scatter(w.loc[bad, x], w.loc[bad, y], s=22, color=ERROR_COLORS["fn"], alpha=0.8, edgecolors="none",
                   label="Flagged")
        ax.axvline(x_thr, color=INK["muted"], linestyle="--", linewidth=1,
                   label=f"{xi.abbr} < {x_thr:g}")
        ax.axhline(y_thr, color=INK["muted"], linestyle=":", linewidth=1.2,
                   label=f"{yi.abbr} > {y_thr:g}{' ' + yi.unit if yi.unit else ''}")
        for _, r in w[bad].nsmallest(4, x).iterrows():
            ax.annotate(str(r["case_id"]), (r[x], r[y]), fontsize=6.5, color=INK["secondary"],
                        xytext=(3, 3), textcoords="offset points")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel(xi.label)
        ax.set_ylabel(yi.label)
        ax.grid(True, axis="both")
        rule = (f"{t['class'].replace('_', ' ')}: worse than a uniform {t['error_mm']:g} mm error"
                if x == "dice" and y == "hd95" and t["class"] != "user-defined" and not explicit
                else "user-defined thresholds")
        ax.set_title(f"Failure modes: {label} ({int(bad.sum())}/{len(w)} flagged)")
        ax.legend(loc="lower left", fontsize=7, title=rule, title_fontsize=7)
    return fig


def sensitivity_curves(df: pd.DataFrame, metrics: Sequence[str], perturbation: Optional[str] = None,
                       figsize=None):
    """Metric response to controlled perturbations (see `segevalkit.synthetic`).

    ``df`` has columns ``perturbation, magnitude, metric, value`` (one row per
    case × magnitude × metric); medians and IQR bands are drawn.
    """
    import matplotlib.pyplot as plt

    perts = [perturbation] if perturbation else list(dict.fromkeys(df["perturbation"]))
    with theme():
        fig, axes = plt.subplots(1, len(perts), figsize=figsize or (3.3 * len(perts), 3.1),
                                 constrained_layout=True, squeeze=False)
        colors = color_for(metrics)
        for ax, p in zip(axes[0], perts):
            d = df[df["perturbation"] == p]
            for m in metrics:
                g = d[d["metric"] == m].groupby("magnitude")["value"]
                if not len(g):
                    continue
                med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
                ax.plot(med.index, med.values, color=colors[m], marker="o", markersize=4,
                        label=get_metric(m).abbr)
                ax.fill_between(med.index, q1.values, q3.values, color=colors[m], alpha=0.15, linewidth=0)
            ax.set_title(p.replace("_", " "))
            ax.set_xlabel(d["unit"].iloc[0] if "unit" in d and len(d) else "magnitude")
            ax.grid(True, axis="both")
        axes[0][0].set_ylabel("Metric value (median, IQR)")
        axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def cohort_plot(summary: pd.DataFrame, metric: str, label: str, factor: Optional[str] = None,
                ax=None, figsize=None):
    """Subgroup means with 95 % CIs from :func:`segevalkit.cohort.cohort_summary`, one row per group.

    Groups smaller than ``min_n`` are drawn hollow; the dashed line is the overall mean.
    """
    info = get_metric(metric)
    d = summary[(summary["metric"] == metric) & (summary["label"] == label)]
    if factor is not None:
        d = d[d["factor"] == factor]
    factors = list(dict.fromkeys(d["factor"]))
    d = d.reset_index(drop=True)
    with theme():
        fig, ax = _new_ax(ax, figsize or (5.2, 0.5 + 0.32 * len(d) + 0.3 * len(factors)))
        y, ticks, labels = 0, [], []
        for f in factors:
            for _, r in d[d["factor"] == f].iterrows():
                c = CATEGORICAL[0]
                ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=c, lw=2, solid_capstyle="round")
                ax.plot(r["mean"], y, "o", ms=6.5, color=INK["surface"] if r["small"] else c, mec=c, mew=1.6)
                ticks.append(y)
                labels.append(f"{r['group']}  (n={int(r['n'])})")
                y += 1
            ax.axhline(y - 0.5, color=INK["grid"], lw=0.8)
            ax.text(1.0, y - 1, f, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5,
                    color=INK["muted"])
            y += 0.4
        allv = (d["mean"] * d["n"]).sum() / max(d["n"].sum(), 1) if len(factors) == 1 else None
        if allv is not None:
            ax.axvline(allv, color=INK["muted"], ls="--", lw=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_xlabel(info.label)
        ax.set_title(f"{info.abbr} by subgroup: {label}")
    return fig
