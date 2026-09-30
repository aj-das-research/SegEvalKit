"""Render every documentation / paper figure from the PanTS benchmark results.

Inputs are the results folders written by ``evaluate_models.py`` and the raw
PanTS images / labels / predictions; nothing is typed in by hand.

    python benchmarks/pants/make_figures.py --out docs/assets/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import segevalkit as sek  # noqa: E402
from segevalkit import plotting as P  # noqa: E402
from segevalkit import viz  # noqa: E402
from segevalkit.io import load_volume  # noqa: E402
from segevalkit.stats import compare, presence_detection, ranking_stability  # noqa: E402

STORE = Path("/l/users/abhijit.das/SegEvalKit")
EVAL = STORE / "outputs/eval/pants"
PANTS = STORE / "data/raw/PanTS"
PRED = STORE / "outputs/predictions"
NAMES = {"nnunet": "nnU-Net ResEnc-M", "medformer": "MedFormer", "totalseg": "TotalSegmentator"}


def save(fig, out: Path, name: str, dpi: int = 170):
    fig.savefig(out / f"{name}.png", dpi=dpi)
    fig.savefig(out / f"{name}.pdf")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print("wrote", name)


def load_all():
    res = {}
    for k, n in NAMES.items():
        if (EVAL / k / "per_case.csv").exists():
            r = sek.load_results(EVAL / k)
            r.meta["name"] = n
            res[n] = r
    return res


def _pred(model, case, structure, ref_shape):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_models import STRUCTURES

    _, ids, ts = STRUCTURES[structure]
    if model == "nnunet":
        v = load_volume(PRED / "nnunet_pants_regions" / f"{case}.nii.gz")
        return np.isin(v.data, ids)
    if model == "medformer":
        f = PRED / "medformer_pants" / case / "predictions" / f"{structure}.nii.gz"
    else:
        f = PRED / "totalsegmentator" / case / f"{ts}.nii.gz"
    if not f.exists():
        return np.zeros(ref_shape, bool)
    return load_volume(f).data > 0


def quantitative(res, out: Path):
    organs = ["liver", "spleen", "kidney_left", "pancreas", "stomach", "gall_bladder", "aorta", "postcava"]
    common = {n: r.filter(labels=organs) for n, r in res.items()}
    save(P.metric_distribution(common, "dice", labels=organs), out, "dist_dice_organs")
    save(P.metric_distribution(common, "nsd", labels=organs), out, "dist_nsd_organs")
    save(P.metric_distribution(common, "hd95", labels=organs, kind="box"), out, "dist_hd95_organs")
    save(P.ecdf(res, "dice", label="pancreas"), out, "ecdf_pancreas_dice")
    first = next(iter(res.values()))
    save(P.metric_heatmap(first.filter(labels=organs), "dice", max_cases=40), out, "heatmap_dice")
    save(P.metric_correlation(first.filter(labels=["pancreas"]),
                              ["dice", "iou", "nsd", "hd95", "assd", "masd", "precision", "recall",
                               "relative_volume_difference"]), out, "metric_correlation_pancreas")
    save(P.volume_agreement(first, "liver"), out, "volume_liver")
    save(P.bland_altman_plot(first, "pancreas"), out, "bland_altman_pancreas")
    save(P.metric_profile(res, ["dice", "nsd", "hd95", "assd"], label="pancreas"), out, "profile_pancreas")
    save(P.failure_quadrants(first, "pancreas"), out, "failure_quadrants_pancreas")
    lesion = {n: r for n, r in res.items() if "pancreatic_lesion" in r.labels}
    if lesion:
        save(P.metric_vs_size(lesion, "dice", label="pancreatic_lesion"), out, "size_lesion_dice")
        save(P.detection_by_size(lesion, label="pancreatic_lesion"), out, "detection_by_size_lesion")
    tub = {n: r for n, r in res.items() if "aorta" in r.labels}
    save(P.metric_distribution(tub, "cldice", labels=[x for x in ("aorta", "postcava", "veins",
                                                                  "superior_mesenteric_artery")
                                                      if x in first.labels]), out, "dist_cldice_vessels")
    names = list(res)
    if len(names) >= 2:
        a, b = names[0], names[1]
        c = compare(res[a], res[b], metrics=["dice", "nsd", "hd95"],
                    labels=["liver", "pancreas", "spleen", "kidney_left", "stomach", "aorta"])
        c.to_csv(out / "compare_table.csv", index=False)
        save(P.comparison_forest(c, name_a=a, name_b=b), out, "forest_compare")
        stab = ranking_stability(res, "dice", label="pancreas", n_boot=500)
        save(P.ranking_stability_plot(stab), out, "ranking_stability_pancreas")


def qualitative(res, out: Path, model: str = "nnunet"):
    name = NAMES[model]
    if name not in res:
        return
    r = res[name]
    # A representative case: median pancreas Dice among cases with a lesion.
    w = r.wide()
    les = w[(w.label == "pancreatic_lesion") & (w.ref_volume_ml > 2)].dropna(subset=["dice"])
    if not len(les):
        return
    case = les.iloc[(les["dice"] - les["dice"].median()).abs().argsort().iloc[0]]["case_id"]
    img = load_volume(PANTS / "ImageTe" / case / "ct.nii.gz", kind="image")
    ref_l = load_volume(PANTS / "LabelTe" / case / "segmentations" / "pancreatic_lesion.nii.gz")
    ref_p = load_volume(PANTS / "LabelTe" / case / "segmentations" / "pancreas.nii.gz").data > 0
    pl = _pred(model, case, "pancreatic_lesion", ref_l.shape)
    pp = _pred(model, case, "pancreas", ref_l.shape)
    rl = ref_l.data > 0
    aff = ref_l.affine
    d = float(w[(w.case_id == case) & (w.label == "pancreatic_lesion")]["dice"].iloc[0])
    save(viz.error_overlay(img.data, pl, rl, affine=aff, window="pancreas",
                           title=f"{case} · lesion · DSC {d:.2f}"), out, "overlay_lesion")
    save(viz.error_overlay(img.data, pp, ref_p, affine=aff, window="pancreas", style="contour",
                           title=f"{case} · pancreas contours"), out, "contour_pancreas")
    save(viz.triplanar(img.data, pp, ref_p, affine=aff, window="pancreas",
                       title=f"{name} · pancreas · {case}"), out, "triplanar_pancreas")
    save(viz.slice_montage(img.data, pl, rl, affine=aff, window="pancreas", n=8,
                           title=f"{name} · pancreatic lesion · {case}"), out, "montage_lesion")
    save(viz.error_projection(pp, ref_p, affine=aff, title=f"{name} · pancreas errors in 3D"), out,
         "projection_pancreas")
    save(viz.surface_distance_map(pp, ref_p, img.spacing, title=f"{name} · pancreas surface distance"), out,
         "surface_pancreas")
    viz.surface_distance_map(pp, ref_p, img.spacing, backend="plotly", html_path=str(out / "surface_pancreas.html"),
                             title=f"{name}: pancreas surface distance ({case})")
    # Worst pancreas cases gallery
    items = []
    for _, row in r.worst_cases("dice", "pancreas", k=4).iterrows():
        c = row["case_id"]
        rv = load_volume(PANTS / "LabelTe" / c / "segmentations" / "pancreas.nii.gz")
        im = load_volume(PANTS / "ImageTe" / c / "ct.nii.gz", kind="image")
        items.append({"image": im.data, "pred": _pred(model, c, "pancreas", rv.shape), "ref": rv.data > 0,
                      "affine": rv.affine, "title": f"{c} · DSC {row['dice']:.2f}"})
    save(viz.case_gallery(items, window="pancreas", title=f"{name}: four worst pancreas cases"), out,
         "gallery_worst_pancreas")
    (out / "qualitative_case.json").write_text(json.dumps({"case": case, "model": model}))


def calibration(out: Path):
    """Reliability diagram of MedFormer lesion probabilities pooled over cases (band ROI)."""
    rel = STORE / "outputs/predictions/medformer_pants"
    ps, ys = [], []
    for cdir in sorted(rel.glob("*"))[:120]:
        f = cdir / "predictions_raw" / "pancreatic_lesion.nii.gz"
        if not f.exists():
            continue
        p = load_volume(f, kind="prob")
        g = load_volume(PANTS / "LabelTe" / cdir.name / "segmentations" / "pancreatic_lesion.nii.gz")
        if p.shape != g.shape:
            continue
        pr = p.data.astype(np.float32)
        if pr.max() > 1.0:
            pr = pr / 255.0 if pr.max() <= 255 else pr / pr.max()
        y = g.data > 0
        band = sek.PairContext(pr >= 0.5, y, g.spacing, pr)
        from segevalkit.metrics.calibration import _roi

        pp, yy = _roi(band, "band", 10.0)
        ps.append(pp)
        ys.append(yy)
    if ps:
        save(P.reliability_diagram(np.concatenate(ps), np.concatenate(ys)), out, "reliability_medformer_lesion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/figures")
    ap.add_argument("--skip-qualitative", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = load_all()
    quantitative(res, out)
    rows = []
    for n, r in res.items():
        if "pancreatic_lesion" in r.labels:
            pdct = presence_detection(r, "pancreatic_lesion")
            rows.append({"model": n, **{k: pdct[k] for k in ("auc", "sensitivity", "specificity", "threshold",
                                                            "n_pos", "n_neg")}})
    pd.DataFrame(rows).to_csv(out / "presence_detection.csv", index=False)
    if not a.skip_qualitative:
        qualitative(res, out)
        calibration(out)


if __name__ == "__main__":
    main()
