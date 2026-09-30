"""Detection curves: FROC with the CPM score, patient-level ROC, and precision-recall.

Each function accepts one `EvaluationResult` or a mapping ``{name: result}``
(one curve per method, colours following the method as in every other
SegEvalKit plot) and returns a matplotlib ``Figure``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np

from .theme import CATEGORICAL, INK, color_for, theme

__all__ = ["froc_plot", "roc_plot", "pr_plot"]

Results = Union[object, Mapping[str, object]]


def _items(results: Results):
    return {results.name: results} if hasattr(results, "lesions") else dict(results)


def _new_ax(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def froc_plot(results: Results, label: str, fp_rates: Sequence[float] = (0.125, 0.25, 0.5, 1, 2, 4, 8),
              n_boot: int = 500, ax=None, figsize=(4.6, 3.4)):
    """FROC curve (lesion sensitivity vs false positives per scan) with the CPM in the legend.

    The x axis is logarithmic over the CPM range; markers show the
    sensitivity read off at each of ``fp_rates``.
    """
    from ..stats.detection import froc

    items = _items(results)
    colors = color_for(list(items))
    with theme():
        fig, ax = _new_ax(ax, figsize)
        for name, r in items.items():
            f = froc(r, label, fp_rates=fp_rates, n_boot=n_boot)
            x, y = f["fps"], f["sensitivity"]
            keep = x > 0
            ax.step(np.r_[x[keep], max(fp_rates[-1], x[-1])], np.r_[y[keep], y[-1]], where="post",
                    color=colors[name], linewidth=1.8)
            rates = np.array(list(f["sensitivity_at"]))
            ax.plot(rates, list(f["sensitivity_at"].values()), "o", color=colors[name], markersize=4.5,
                    markeredgecolor=INK["surface"], markeredgewidth=0.8)
            lo, hi = f["cpm_ci"]
            ci = f" [{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else ""
            ax.plot([], [], "-o", color=colors[name], markersize=4.5, label=f"{name}: CPM {f['cpm']:.3f}{ci}")
        ax.set_xscale("log", base=2)
        ax.set_xlim(fp_rates[0] / 1.5, fp_rates[-1] * 1.5)
        ax.set_xticks(list(fp_rates))
        ax.set_xticklabels([f"{r:g}" if r >= 1 else f"1/{int(round(1 / r))}" for r in fp_rates])
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="both")
        ax.set_xlabel("False-positive lesions per scan ↓")
        ax.set_ylabel("Lesion sensitivity ↑")
        ax.set_title(f"FROC: {label}")
        ax.legend(loc="lower right")
    return fig


def roc_plot(results: Results, label: str, score: str = "pred_volume_ml", localized: bool = True,
             band: bool = True, n_boot: int = 500, target_specificity: Optional[float] = 0.9,
             reference_point: Optional[Mapping[str, float]] = None, ax=None, figsize=(4.0, 3.8)):
    """Patient-level ROC curve ("does this patient have the structure?").

    Args:
        score: Per-patient score: ``"pred_volume_ml"`` or ``"lesion"`` (highest
            lesion confidence), see `segevalkit.stats.localized_presence`.
        localized: Also draw the localized curve (a flagged patient counts only
            if a predicted lesion overlaps a reference lesion), dashed.
        band: Shade a 95 % bootstrap band around the plain curve.
        target_specificity: Mark the operating point at this specificity.
        reference_point: Optional ``{"sensitivity": .., "specificity": .., "label": ..}``
            to mark a published operating point (drawn as a star).
    """
    from ..stats.detection import _patient_scores, localized_presence, roc_band

    items = _items(results)
    colors = color_for(list(items))
    with theme():
        fig, ax = _new_ax(ax, figsize)
        ax.plot([0, 1], [0, 1], color=INK["muted"], linestyle=":", linewidth=1)
        for name, r in items.items():
            d = localized_presence(r, label, score=score, target_specificity=target_specificity or 0.9)
            order = np.lexsort((d["tpr"], d["fpr"]))
            if band:
                y, s, _ = _patient_scores(r, label, score, 0.0)
                g, lo, hi, auc_ci = roc_band(y, s, n_boot=n_boot)
                ax.fill_between(g, lo, hi, color=colors[name], alpha=0.14, linewidth=0, step="post")
            ci = f" [{auc_ci[0]:.2f}, {auc_ci[1]:.2f}]" if band else ""
            ax.step(d["fpr"][order], d["tpr"][order], where="post", color=colors[name], linewidth=1.8,
                    label=f"{name}: AUC {d['auc']:.3f}{ci}")
            if localized:
                o2 = np.lexsort((d["tpr_localized"], d["fpr"]))
                ax.step(d["fpr"][o2], d["tpr_localized"][o2], where="post", color=colors[name], linewidth=1.3,
                        linestyle="--", label=f"{name}, localized: AUC {d['auc_localized']:.3f}")
            if target_specificity is not None:
                ax.plot(1 - d["specificity"], d["sensitivity"], "o", color=colors[name], markersize=5,
                        markeredgecolor=INK["surface"], markeredgewidth=0.8)
        if reference_point:
            ax.plot(1 - reference_point["specificity"], reference_point["sensitivity"], "*", markersize=11,
                    color=CATEGORICAL[2], markeredgecolor=INK["surface"], markeredgewidth=0.6,
                    label=reference_point.get("label", "reported"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="both")
        ax.set_xlabel("1 − specificity ↓")
        ax.set_ylabel("Sensitivity ↑")
        ax.set_title(f"Patient-level ROC: {label}")
        ax.legend(loc="lower right", fontsize=7.5)
    return fig


def pr_plot(results: Results, label: str, level: str = "lesion", score: str = "pred_volume_ml",
            localized: bool = False, ax=None, figsize=(4.0, 3.6)):
    """Precision-recall curve with its average precision.

    Args:
        level: ``"lesion"`` (every reference lesion and false-positive
            component, ranked by lesion confidence; `segevalkit.stats.lesion_pr`)
            or ``"patient"`` (`segevalkit.stats.patient_pr`, with the
            prevalence as the chance line).
        score: Patient-level score, see `segevalkit.stats.patient_pr`.
        localized: Patient level only: mislocalized calls count as false positives.
    """
    from ..stats.detection import lesion_pr, patient_pr

    if level not in ("lesion", "patient"):
        raise ValueError("level must be 'lesion' or 'patient'")
    items = _items(results)
    colors = color_for(list(items))
    with theme():
        fig, ax = _new_ax(ax, figsize)
        for name, r in items.items():
            d = lesion_pr(r, label) if level == "lesion" else patient_pr(r, label, score=score, localized=localized)
            ax.step(np.r_[0, d["recall"]], np.r_[d["precision"][0] if len(d["precision"]) else 1, d["precision"]],
                    where="post", color=colors[name], linewidth=1.8, label=f"{name}: AP {d['ap']:.3f}")
            if level == "patient":
                ax.axhline(d["prevalence"], color=colors[name], linestyle=":", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="both")
        unit = "lesions" if level == "lesion" else "patients"
        ax.set_xlabel(f"Recall ({unit}) ↑")
        ax.set_ylabel(f"Precision ({unit}) ↑")
        ax.set_title(f"{'Lesion' if level == 'lesion' else 'Patient'}-level PR: {label}")
        ax.legend(loc="lower left")
    return fig
