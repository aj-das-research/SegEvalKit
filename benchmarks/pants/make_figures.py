"""Render the PanTS benchmark figures (docs PNG + paper PDF) from the full-test-set evaluations.

Inputs are the results folders written by ``evaluate_models.py`` (and
``lesion_scores.py`` for per-lesion confidences) plus the raw PanTS images,
labels and predictions; nothing is typed in by hand. Models whose evaluation
has not finished are skipped, so the script can be re-run as they arrive.

Figures, lesion first:

* lesion: per-tumour-patient Dice (tumour-free patients summarised as the
  fraction with a false tumour), Dice vs tumour size over every reference
  tumour, detection rate by size, FROC + patient ROC (plain and localized) +
  lesion PR, the found / missed / false-positive examples, a tri-planar
  overlay and a surface-distance map of the found tumour;
* organs: per-case Dice and NSD distributions, the paired forest plot and
  the pancreas ranking stability, pancreas views, calibration and metric
  correlation.

Example cases are chosen by rule from the full test set (``select_cases``).

    python benchmarks/pants/make_figures.py --png docs/assets/figures/pants --pdf paper/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import segevalkit as sek  # noqa: E402
from segevalkit import plotting as P  # noqa: E402
from segevalkit import viz  # noqa: E402
from segevalkit.io import load_volume  # noqa: E402
from segevalkit.plotting.theme import ERROR_COLORS, INK, color_for, theme  # noqa: E402
from segevalkit.stats import compare, ranking_stability  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_models import STRUCTURES  # noqa: E402

STORE = Path("/l/users/abhijit.das/SegEvalKit")
EVAL = STORE / "outputs/eval/pants"
PANTS = STORE / "data/raw/PanTS"
PRED = STORE / "outputs/predictions"
NAMES = {"nnunet": "nnU-Net ResEnc-M", "medformer": "MedFormer", "totalseg": "TotalSegmentator"}
LES = "pancreatic_lesion"
ORGANS = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas", "stomach", "gall_bladder", "duodenum",
          "aorta", "postcava"]
# R-Super GitHub (documents/demo_results.png, "PanTS test", standard-segmentation baseline = this checkpoint).
RSUPER = {"sensitivity": 0.76, "specificity": 0.91, "label": "R-Super GitHub (baseline)"}
OUT = {"png": None, "pdf": None}


def save(fig, name: str, dpi: int = 200):
    if OUT["png"]:
        fig.savefig(OUT["png"] / f"{name}.png", dpi=dpi)
    if OUT["pdf"]:
        fig.savefig(OUT["pdf"] / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name, flush=True)


def load_all():
    res = {}
    for k, n in NAMES.items():
        if (EVAL / k / "per_case.csv").exists():
            r = sek.load_results(EVAL / k)
            r.meta["name"] = n
            res[n] = r
        else:
            print("skip (not evaluated yet):", n)
    return res


def load_scored():
    """Lesion-only evaluations with per-lesion confidence scores (lesion_scores.py)."""
    out = {}
    for k in ("nnunet", "medformer"):
        d = EVAL / f"{k}_lesion_scores"
        if (d / "per_case.csv").exists():
            r = sek.load_results(d)
            r.meta["name"] = NAMES[k]
            out[NAMES[k]] = r
    return out


def _pred(model, case, structure, ref_shape):
    _, ids, ts = STRUCTURES[structure]
    if model == "nnunet":
        v = load_volume(PRED / "nnunet_pants_regions" / f"{case}.nii.gz")
        return np.isin(v.data, ids)
    if model == "medformer":
        f = PRED / "medformer_pants/abdomenatlas/pants_pancreas_release" / case / "predictions" / f"{structure}.nii.gz"
    else:
        f = PRED / "totalsegmentator" / case / f"{ts}.nii.gz"
    if not f.exists():
        return np.zeros(ref_shape, bool)
    m = load_volume(f).data > 0
    if model == "medformer" and structure == "pancreas":  # evaluated as pancreas ∪ lesion on both sides
        fl = f.with_name(f"{LES}.nii.gz")
        if fl.exists():
            m |= load_volume(fl).data > 0
    return m


def _ref(case, structure):
    seg = PANTS / "LabelTe" / case / "segmentations"
    v = load_volume(seg / f"{structure}.nii.gz")
    m = v.data > 0
    if structure == "pancreas":  # evaluated as pancreas ∪ lesion (evaluate_models.UNION)
        m |= load_volume(seg / f"{LES}.nii.gz").data > 0
    return m, v


def _image(case):
    return load_volume(PANTS / "ImageTe" / case / "ct.nii.gz", kind="image")


# --------------------------------------------------------------------------------------------- organs
def organ_figures(res):
    common = {n: r.filter(labels=[o for o in ORGANS if o in r.labels]) for n, r in res.items()}
    save(P.metric_distribution(common, "dice", labels=ORGANS, figsize=(7.0, 2.9)), "dist_dice_organs")
    save(P.metric_distribution(common, "nsd", labels=ORGANS, figsize=(7.0, 2.9)), "dist_nsd_organs")
    save(P.metric_distribution(common, "hd95", labels=ORGANS, kind="box", figsize=(7.0, 2.9)), "dist_hd95_organs")
    names = list(res)
    if len(names) >= 2:
        a, b = names[0], names[1]
        c = compare(res[a], res[b], metrics=["dice", "nsd", "hd95"],
                    labels=["liver", "pancreas", "spleen", "kidney_left", "stomach", "aorta"])
        c.to_csv(EVAL / "report" / f"forest_{a.split()[0]}_vs_{b.split()[0]}.csv", index=False)
        save(P.comparison_forest(c, name_a=a, name_b=b), "forest_compare")
        stab = ranking_stability(res, "dice", label="pancreas", n_boot=500)
        save(P.ranking_stability_plot(stab), "ranking_stability_pancreas")
    first = res.get("MedFormer", next(iter(res.values())))
    save(P.metric_correlation(first.filter(labels=["pancreas"]),
                              ["dice", "iou", "nsd", "hd95", "assd", "masd", "precision", "recall",
                               "relative_volume_difference"]), "metric_correlation_pancreas")
    save(P.failure_quadrants(first, "pancreas"), "failure_quadrants_pancreas")
    save(P.failure_quadrants(first, LES) if LES in first.labels else plt.figure(), "failure_quadrants_lesion")


# --------------------------------------------------------------------------------------------- lesion
def lesion_numbers(r):
    w = r.wide()
    w = w[w.label == LES]
    pos, neg = w[w.ref_empty == 0], w[w.ref_empty == 1]
    les = r.lesions
    les = les[(les.label == LES)]
    return {"n_patients": int(len(w)), "n_tumour_patients": int(len(pos)), "n_tumour_free": int(len(neg)),
            "n_ref_lesions": int((les.kind == "ref").sum()),
            # false tumours = predicted components counted by the detection matching (min. size applied)
            "tumour_free_with_false_tumour": int(les[les.kind.str.startswith("pred")
                                                     & les.case_id.isin(neg.case_id)].case_id.nunique()),
            "false_tumours_in_tumour_free": int((les.kind.str.startswith("pred") & les.case_id.isin(neg.case_id)).sum()),
            "tumour_patient_dice_mean": float(pos.dice.mean()), "tumour_patient_dice_median": float(pos.dice.median()),
            "tumour_patient_nsd_mean": float(pos.nsd.mean()),
            "tumour_patients_missed_entirely": int((pos.tp_ref_lesions == 0).sum())}


def _frac(v):
    return v["tumour_free_with_false_tumour"] / v["n_tumour_free"]


def lesion_figures(res, scored):
    lres = {n: r for n, r in res.items() if LES in r.labels}
    if not lres:
        return {}
    nums = {n: lesion_numbers(r) for n, r in lres.items()}
    colors = color_for(list(res))
    # (a) per-tumour-patient Dice / NSD, (b) tumour-free patients with a false tumour
    tum = {n: r.filter(labels=[LES], cases=list(r.wide().query("label == @LES and ref_empty == 0").case_id))
           for n, r in lres.items()}
    with theme():
        fig = plt.figure(figsize=(7.0, 2.7), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
        P.metric_distribution(tum, "dice", labels=[LES], ax=fig.add_subplot(gs[0]))
        P.metric_distribution(tum, "nsd", labels=[LES], ax=fig.add_subplot(gs[1]))
        n_pos = next(iter(nums.values()))["n_tumour_patients"]
        for ax, t in zip(fig.axes[:2], ("DSC", "NSD at 2 mm")):
            ax.set_title(f"{t}, tumour patients (n = {n_pos})", fontsize=9.5)
            ax.set_xticklabels([])
        ax = fig.add_subplot(gs[2])
        for i, (n, v) in enumerate(nums.items()):
            ax.barh(i, _frac(v), color=colors[n], height=0.55)
            ax.text(_frac(v) + 0.02, i,
                    f"{v['tumour_free_with_false_tumour']} / {v['n_tumour_free']}", va="center", fontsize=8,
                    color=INK["primary"])
        ax.set_yticks(range(len(nums)))
        ax.set_yticklabels(list(nums))
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of patients ↓")
        ax.set_title("Tumour-free: false tumour", fontsize=9.5)
        ax.grid(True, axis="x")
    save(fig, "dist_dice_lesion")
    # Dice vs size over every reference tumour (lesion table: one row per reference tumour)
    with theme():
        fig, ax = plt.subplots(figsize=(3.6, 3.2), layout="constrained")
        for n, r in lres.items():
            le = r.lesions
            le = le[(le.label == LES) & (le.kind == "ref")]
            ax.scatter(le.volume_ml, le.dice, s=13, alpha=0.55, color=colors[n], edgecolors="none",
                       label=f"{n} ({len(le)} tumours)")
            x = np.log10(le.volume_ml)
            edges = np.quantile(x, np.linspace(0, 1, 8))
            idx = np.clip(np.digitize(x, edges[1:-1]), 0, 6)
            cx = [np.median(le.volume_ml.to_numpy()[idx == b]) for b in range(7) if np.any(idx == b)]
            cy = [np.median(le.dice.to_numpy()[idx == b]) for b in range(7) if np.any(idx == b)]
            ax.plot(cx, cy, color=colors[n], lw=2, marker="o", ms=4, mec=INK["surface"], label="binned median")
        ax.set_xscale("log")
        ax.set_xlabel("Reference tumour volume [mL]")
        ax.set_ylabel("Tumour DSC ↑")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True)
        ax.set_title("Tumour DSC vs size", fontsize=10)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    save(fig, "size_lesion_dice")
    bins = (0, 0.1, 0.5, 1, 5, 20, np.inf)
    fig = P.detection_by_size(lres, label=LES, bins_ml=bins, figsize=(3.6, 3.0))
    with theme():  # counts above the bars (the library writes them inside, unreadable on short bars)
        ax = fig.axes[0]
        for t in list(ax.texts):
            t.remove()
        le = next(iter(lres.values())).lesions
        le = le[(le.label == LES) & (le.kind == "ref")]
        for i, n in enumerate(le.groupby(pd.cut(le.volume_ml, bins=list(bins), include_lowest=True),
                                         observed=False).size()):
            ax.text(i, 1.02, f"n={n}", ha="center", va="bottom", fontsize=7.5, color=INK["secondary"])
        ax.set_ylim(0, 1.13)
        ax.set_title("Tumours found, by size", fontsize=10)
        ax.set_xlabel("Reference tumour volume [mL]")
    save(fig, "detection_by_size_lesion")
    if scored:
        one = next(iter(scored.values())) if len(scored) == 1 else scored
        with theme():
            fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
            P.froc_plot(one, LES, n_boot=500, ax=axs[0])
            P.roc_plot(one, LES, reference_point=RSUPER, ax=axs[1])
            P.pr_plot(one, LES, ax=axs[2])
            axs[0].set_title("FROC: lesion level")
            axs[1].set_title("ROC: patient level")
            axs[2].set_title("PR: lesion level")
        save(fig, "detection_curves_lesion")
    return nums


def select_cases(r):
    """Rule-based example cases from the full test set (one model's lesion table)."""
    le = r.lesions
    le = le[le.label == LES]
    ref = le[le.kind == "ref"]
    found = ref[ref.detected]
    med = found.dice.median()
    f = found.iloc[(found.dice - med).abs().argsort().iloc[0]]
    miss = ref[~ref.detected].sort_values("volume_ml", ascending=False).iloc[0]
    w = r.wide()
    w = w[w.label == LES]
    free = set(w[(w.ref_empty == 1) & (w.pred_empty == 0)].case_id)
    fp = le[le.kind.str.startswith("pred") & le.case_id.isin(free)].groupby("case_id").volume_ml.sum()
    fpc = fp.index[(fp - fp.median()).abs().argsort().iloc[0]]
    return {
        "found": {"case": f.case_id, "rule": "detected tumour whose DSC is closest to the median DSC of all "
                  "detected tumours", "dice": float(f.dice), "volume_ml": float(f.volume_ml), "median_dice": float(med)},
        "missed": {"case": miss.case_id, "rule": "largest reference tumour that no predicted component touches",
                   "volume_ml": float(miss.volume_ml)},
        "false_positive": {"case": fpc, "rule": "tumour-free patient whose total false-tumour volume is the median "
                           "over tumour-free patients with a false tumour", "volume_ml": float(fp[fpc]),
                           "median_ml": float(fp.median())},
    }


def qualitative(res, model="medformer"):
    name = NAMES[model]
    if name not in res or LES not in res[name].labels:
        return {}
    sel = select_cases(res[name])
    # found / missed / false-positive panel
    with theme():
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), constrained_layout=True)
        titles = {"found": "Found: {case}\nDSC {dice:.2f}, {volume_ml:.1f} mL",
                  "missed": "Missed: {case}\n{volume_ml:.1f} mL tumour",
                  "false_positive": "False tumour: {case}\ntumour-free, {volume_ml:.2f} mL predicted"}
        for ax, (k, s) in zip(axs, sel.items()):
            rm, rv = _ref(s["case"], LES)
            pm = _pred(model, s["case"], LES, rm.shape)
            viz.error_overlay(_image(s["case"]).data, pm, rm, affine=rv.affine, window="pancreas", ax=ax,
                              title=titles[k].format(**{**s, "case": s["case"].replace("PanTS_0000", "")}))
        viz.error_legend(fig, window="pancreas", note="cases chosen by rule from the 901 test CTs")
    save(fig, "lesion_cases")
    c = sel["found"]["case"]
    img = _image(c)
    rm, rv = _ref(c, LES)
    pm = _pred(model, c, LES, rm.shape)
    save(viz.triplanar(img.data, pm, rm, affine=rv.affine, window="pancreas", figsize=(7.0, 2.9),
                       title=f"{name} · pancreatic tumour · {c} (DSC {sel['found']['dice']:.2f})"), "triplanar_lesion")
    save(viz.surface_distance_map(pm, rm, img.spacing, clip_mm=5.0, figsize=(5.2, 2.7),
                                  title=f"{name} · tumour surface distance · {c}"), "surface_lesion")
    if OUT["png"]:
        viz.surface_distance_map(pm, rm, img.spacing, clip_mm=5.0, backend="plotly",
                                 html_path=str(OUT["png"] / "surface_lesion.html"),
                                 title=f"{name}: tumour surface distance ({c})")
    # pancreas views (secondary) on the same case
    pr, prv = _ref(c, "pancreas")
    pp = _pred(model, c, "pancreas", pr.shape)
    save(viz.triplanar(img.data, pp, pr, affine=prv.affine, window="pancreas", figsize=(4.4, 2.4),
                       title=f"{name} · pancreas"), "triplanar_pancreas")
    save(viz.surface_distance_map(pp, pr, img.spacing, clip_mm=10.0, figsize=(4.4, 2.5),
                                  title="Pancreas surface distance"), "surface_pancreas")
    return sel


def _calib_case(cdir):
    from segevalkit.metrics.calibration import _roi

    f = cdir / "predictions_raw" / f"{LES}.nii.gz"
    g = PANTS / "LabelTe" / cdir.name / "segmentations" / f"{LES}.nii.gz"
    if not f.exists() or not g.exists():
        return None
    p, gv = load_volume(f, kind="prob"), load_volume(g)
    if p.shape != gv.shape:
        return None
    pr = p.data.astype(np.float32)
    if pr.max() > 1.0:
        pr = pr / 255.0 if pr.max() <= 255 else pr / pr.max()
    y = gv.data > 0
    if not y.any() and not (pr >= 0.5).any():
        return None  # no band: neither a tumour nor a predicted tumour
    pp, yy = _roi(sek.PairContext(pr >= 0.5, y, gv.spacing, pr), "band", 10.0)
    return pp.astype(np.float32), yy.astype(bool)


def calibration(workers: int):
    """Reliability of MedFormer lesion probabilities, pooled over every test case with a band.

    The pooled voxels are cached (outputs/eval/pants/report/calibration_voxels.npz) so the
    figure can be re-rendered without reading the 901 probability volumes again.
    """
    from segevalkit.metrics.calibration import reliability_curve

    cache = EVAL / "report" / "calibration_voxels.npz"
    if cache.exists():
        z = np.load(cache)
        p, y, n = z["p"], z["y"], int(z["n"])
    else:
        rel = PRED / "medformer_pants/abdomenatlas/pants_pancreas_release"
        cdirs = sorted(d for d in rel.iterdir() if d.is_dir())
        ps, ys, n = [], [], 0
        with ProcessPoolExecutor(workers) as ex:
            for out in ex.map(_calib_case, cdirs, chunksize=4):
                if out is not None:
                    ps.append(out[0])
                    ys.append(out[1])
                    n += 1
        if not ps:
            return {}
        p, y = np.concatenate(ps), np.concatenate(ys)
        np.savez_compressed(cache, p=p, y=y, n=n)
    fig = P.reliability_diagram(p, y, figsize=(3.4, 3.2))
    ax = fig.axes[0]
    c = reliability_curve(p, y, 15)
    ok = c["count"] > 0
    w = c["count"] / c["count"].sum()
    ece = float(np.sum(w[ok] * np.abs(c["accuracy"][ok] - c["confidence"][ok])))
    gap = (c["confidence"] - c["accuracy"])[ok]
    for loc in ("left", "center", "right"):
        ax.set_title("", loc=loc)
    ax.set_title(f"MedFormer tumour probabilities\n{n} CTs, ECE {ece:.3f}", loc="left", fontsize=9)
    save(fig, "reliability_medformer_lesion")
    return {"n_cases": n, "n_voxels": int(p.size), "pooled_ece": ece, "mean_p": float(p.mean()),
            "frac_positive": float(y.mean()), "max_gap": float(gap.max()),
            "frac_bins_overconfident": float((gap > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default="docs/assets/figures/pants")
    ap.add_argument("--pdf", default="paper/figures")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-qualitative", action="store_true")
    ap.add_argument("--skip-calibration", action="store_true")
    a = ap.parse_args()
    OUT["png"], OUT["pdf"] = Path(a.png), Path(a.pdf)
    for p in OUT.values():
        p.mkdir(parents=True, exist_ok=True)
    res, scored = load_all(), load_scored()
    info = {"models": list(res), "scored_models": list(scored)}
    organ_figures(res)
    info["lesion"] = lesion_figures(res, scored)
    if not a.skip_qualitative:
        info["cases"] = qualitative(res)
    if not a.skip_calibration:
        info["calibration"] = calibration(a.workers)
    fj = EVAL / "report" / "figures.json"
    prev = json.loads(fj.read_text()) if fj.exists() else {}
    fj.write_text(json.dumps({**prev, **info}, indent=2, default=str))  # skipped parts keep their last values
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
