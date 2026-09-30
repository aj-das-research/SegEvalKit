"""Build the documentation showcase from real model outputs on 8 PanTS test cases.

The cases (benchmarks/showcase/showcase_cases.csv) are stratified by lesion size:
two large, two medium, two small, two tumour-free. Predictions come from the
official nnU-Net, MedFormer and TotalSegmentator checkpoints. Everything this
script writes is an *illustration of the library on real data*, not a benchmark
result (eight cases are far too few for that; see the PanTS benchmark page).

    python benchmarks/showcase/make_showcase.py --out docs/assets/showcase

Outputs: evaluation folders (outputs/eval/showcase/<model>), every plot and
qualitative view as PNG, an interactive surface map, a sample HTML report, and a
JSON log of real printed outputs used by the README / quickstart walkthrough.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pants"))
import segevalkit as sek  # noqa: E402
from evaluate_models import PARAMS, SOURCES, labels_for, ORGAN  # noqa: E402
from segevalkit import plotting as P  # noqa: E402
from segevalkit import viz  # noqa: E402
from segevalkit.io import Source, load_volume  # noqa: E402
from segevalkit.plotting.theme import INK, theme  # noqa: E402
from segevalkit.stats import compare, rank_methods, ranking_stability  # noqa: E402

STORE = Path("/l/users/abhijit.das/SegEvalKit")
PANTS = STORE / "data/raw/PanTS"
EVAL = STORE / "outputs/eval/showcase"
HERE = Path(__file__).resolve().parent
NNU = {"pancreas": [17, 18, 19, 20, 21, 28], "pancreatic_lesion": [28], "liver": [14]}


def save(fig, out, name, dpi=160):
    fig.savefig(out / f"{name}.png", dpi=dpi)
    plt.close(fig)
    print("  figure", name)


def pred_mask(model, case, structure, shape):
    if model == "nnunet":
        v = load_volume(STORE / "outputs/predictions/nnunet_pants_regions" / f"{case}.nii.gz")
        return np.isin(v.data, NNU.get(structure, []))
    if model == "medformer":
        d = SOURCES["medformer"]["root"] / case / "predictions"
        if structure == "pancreas":  # pancreas ∪ lesion, as in evaluate_models.UNION
            return (load_volume(d / "pancreas.nii.gz").data > 0) | (load_volume(d / "pancreatic_lesion.nii.gz").data > 0)
        f = d / f"{structure}.nii.gz"
    else:
        ts = {"pancreas": "pancreas", "liver": "liver"}.get(structure)
        if ts is None:
            return np.zeros(shape, bool)
        f = SOURCES["totalseg"]["root"] / case / f"{ts}.nii.gz"
    return load_volume(f).data > 0 if f.exists() else np.zeros(shape, bool)


def ref_pancreas(case):
    """Reference pancreas ∪ lesion (PanTS annotates the lesion inconsistently w.r.t. the pancreas mask)."""
    d = PANTS / "LabelTe" / case / "segmentations"
    return (load_volume(d / "pancreas.nii.gz").data > 0) | (load_volume(d / "pancreatic_lesion.nii.gz").data > 0)


def evaluate(cases):
    ref = Source(PANTS / "LabelTe", layout="per_structure", subdir="segmentations")
    res = {}
    for m in ("nnunet", "medformer", "totalseg"):
        s = SOURCES[m]
        pred = Source(s["root"], layout=s["layout"], subdir=s.get("subdir"))
        prob = Source(s["root"], layout="per_structure", subdir=s["prob_subdir"], kind="prob") if s.get("prob_subdir") else None
        ev = sek.Evaluator(labels=labels_for(m), metrics=ORGAN, params=PARAMS, min_lesion_voxels=10,
                           alignment="ignore")  # PanTS label headers are unreliable; see evaluate_models.py
        r = ev.evaluate(pred, ref, prob=prob, cases=cases, n_workers=8, out_dir=EVAL / m, name=s["name"],
                        progress=False)
        res[s["name"]] = r
        print(" ", r)
    return res


def quantitative(res, out):
    organs = ["liver", "pancreas", "spleen", "kidney_left", "stomach", "aorta"]
    save(P.metric_distribution(res, "dice", labels=organs), out, "dist_dice")
    save(P.metric_distribution(res, "hd95", labels=organs, kind="box"), out, "dist_hd95_box")
    save(P.ecdf(res, "nsd", label="pancreas"), out, "ecdf_nsd_pancreas")
    first = res["nnU-Net ResEnc-M"]
    save(P.metric_heatmap(first, "dice"), out, "heatmap_dice")
    save(P.metric_correlation(first, ["dice", "iou", "nsd", "hd95", "assd", "masd", "precision", "recall",
                                      "relative_volume_difference"]), out, "metric_correlation")
    save(P.metric_vs_size({k: v for k, v in res.items() if "pancreatic_lesion" in v.labels}, "dice",
                          label="pancreatic_lesion"), out, "size_lesion_dice")
    save(P.volume_agreement(first, "liver"), out, "volume_agreement_liver")
    save(P.bland_altman_plot(first, "pancreas"), out, "bland_altman_pancreas")
    save(P.metric_profile(res, ["dice", "nsd", "hd95", "assd"], label="pancreas"), out, "metric_profile_pancreas")
    save(P.failure_quadrants(first, "pancreas", x_thr=0.8), out, "failure_quadrants_pancreas")
    lesion = {k: v for k, v in res.items() if "pancreatic_lesion" in v.labels}
    save(P.detection_by_size(lesion, label="pancreatic_lesion", bins_ml=(0, 1, 10, np.inf)), out, "detection_by_size")
    a, b = "nnU-Net ResEnc-M", "MedFormer"
    c = compare(res[a], res[b], metrics=["dice", "nsd", "hd95"], labels=["liver", "pancreas", "spleen", "stomach"])
    save(P.comparison_forest(c, name_a=a, name_b=b), out, "comparison_forest")
    stab = ranking_stability(res, "dice", label="pancreas", n_boot=300)
    save(P.ranking_stability_plot(stab), out, "ranking_stability")
    return c


def calibration(cases, out):
    ps, ys = [], []
    for cid in cases:
        f = SOURCES["medformer"]["root"] / cid / "predictions_raw" / "pancreatic_lesion.nii.gz"
        if not f.exists():
            continue
        p = load_volume(f, kind="prob").data
        g = load_volume(PANTS / "LabelTe" / cid / "segmentations" / "pancreatic_lesion.nii.gz")
        ctx = sek.PairContext(p >= 0.5, g.data > 0, g.spacing, p)
        from segevalkit.metrics.calibration import _roi

        pp, yy = _roi(ctx, "band", 10.0)
        ps.append(pp)
        ys.append(yy)
    if ps:
        save(P.reliability_diagram(np.concatenate(ps), np.concatenate(ys)), out, "reliability_medformer_lesion")


def qualitative(res, cases_df, out):
    w = res["nnU-Net ResEnc-M"].wide()
    large = cases_df[cases_df.stratum.str.startswith("large")].case_id.iloc[0]
    small = cases_df[cases_df.stratum.str.startswith("small")].case_id.iloc[0]
    for tag, cid in (("large", large), ("small", small)):
        img = load_volume(PANTS / "ImageTe" / cid / "ct.nii.gz", kind="image")
        rl = load_volume(PANTS / "LabelTe" / cid / "segmentations" / "pancreatic_lesion.nii.gz")
        rp = ref_pancreas(cid)
        pl = pred_mask("nnunet", cid, "pancreatic_lesion", rl.shape)
        pp = pred_mask("nnunet", cid, "pancreas", rl.shape)
        d = w[(w.case_id == cid) & (w.label == "pancreatic_lesion")]["dice"].iloc[0]
        save(viz.error_overlay(img.data, pl, rl.data > 0, affine=rl.affine, window="pancreas",
                               title=f"{cid} · lesion ({tag}) · DSC {d:.2f}"), out, f"overlay_lesion_{tag}")
        if tag == "large":
            save(viz.error_overlay(img.data, pp, rp, affine=rl.affine, window="pancreas", style="contour",
                                   title=f"{cid} · pancreas contours"), out, "contour_pancreas")
            save(viz.triplanar(img.data, pp, rp, affine=rl.affine, window="pancreas",
                               title=f"nnU-Net · pancreas · {cid}"), out, "triplanar_pancreas")
            save(viz.slice_montage(img.data, pl, rl.data > 0, affine=rl.affine, window="pancreas", n=8,
                                   title=f"nnU-Net · pancreatic lesion · {cid}"), out, "montage_lesion")
            save(viz.error_projection(pp, rp, affine=rl.affine, title=f"nnU-Net · pancreas errors in 3D · {cid}"),
                 out, "projection_pancreas")
            save(viz.surface_distance_map(pp, rp, rl.spacing, title=f"nnU-Net · pancreas surface distance · {cid}"),
                 out, "surface_pancreas")
            viz.surface_distance_map(pp, rp, rl.spacing, backend="plotly", html_path=str(out / "surface_pancreas.html"),
                                     title=f"nnU-Net: pancreas surface distance ({cid})")
            model_comparison(img, rl, rp, cid, out)
    # Worst pancreas cases (nnU-Net)
    items = []
    for _, row in res["nnU-Net ResEnc-M"].worst_cases("dice", "pancreas", k=4).iterrows():
        cid = row["case_id"]
        rv = load_volume(PANTS / "LabelTe" / cid / "segmentations" / "pancreas.nii.gz")
        im = load_volume(PANTS / "ImageTe" / cid / "ct.nii.gz", kind="image")
        items.append({"image": im.data, "pred": pred_mask("nnunet", cid, "pancreas", rv.shape), "ref": ref_pancreas(cid),
                      "affine": rv.affine, "title": f"{cid} · DSC {row['dice']:.2f}"})
    save(viz.case_gallery(items, window="pancreas", title="nnU-Net: the four lowest pancreas Dice in the showcase"),
         out, "gallery_worst_pancreas")


def model_comparison(img, rl, rp, cid, out):
    """The same slice, three models: pancreas (top) and lesion (bottom)."""
    models = [("nnunet", "nnU-Net"), ("medformer", "MedFormer"), ("totalseg", "TotalSegmentator")]
    with theme():
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), layout="constrained")
        for row, (struct, ref) in enumerate((("pancreas", rp), ("pancreatic_lesion", rl.data > 0))):
            for col, (m, name) in enumerate(models):
                pr = pred_mask(m, cid, struct, ref.shape)
                idx = viz.pick_slice(viz.to_canonical(ref, rl.affine)[0], viz.to_canonical(ref, rl.affine)[0],
                                     "axial", "ref")
                viz.error_overlay(img.data, pr, ref, affine=rl.affine, window="pancreas", index=idx, ax=axes[row][col],
                                  legend=False,
                                  title=f"{name} · {'pancreas ∪ lesion' if struct == 'pancreas' else 'lesion'}"
                                        + (" (no lesion class)" if m == "totalseg" and struct != "pancreas" else ""))
        from matplotlib.patches import Patch

        from segevalkit.plotting.theme import ERROR_COLORS

        fig.legend(handles=[Patch(color=ERROR_COLORS["tp"], label="True positive"),
                            Patch(color=ERROR_COLORS["fn"], label="Missed"),
                            Patch(color=ERROR_COLORS["fp"], label="Added")],
                   loc="outside lower center", ncol=3, frameon=False)
        fig.suptitle(f"Three official models, one slice ({cid})", x=0.01, ha="left", fontsize=12,
                     fontweight="bold", color=INK["primary"])
    save(fig, out, "model_comparison")


def walkthrough_log(res, cases, out):
    """Record real printed outputs for the README / quickstart walkthrough."""
    log = []

    def step(title, code, fn):
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn()
        log.append({"title": title, "code": code, "output": buf.getvalue().rstrip()})

    r = res["nnU-Net ResEnc-M"]
    pd.set_option("display.width", 110)
    step("summary", 'res.summary().query("metric in [\'dice\', \'nsd\', \'hd95\']")',
         lambda: print(r.summary().query("metric in ['dice', 'nsd', 'hd95'] and label in ['liver', 'pancreas', 'pancreatic_lesion']")
                       [["label", "metric", "n", "mean", "median", "ci_low", "ci_high"]].round(3).to_string(index=False)))
    step("wide", 'res.wide()[["case_id", "label", "dice", "nsd", "hd95", "ref_volume_ml"]].head(6)',
         lambda: print(r.wide().query("label == 'pancreas'")[["case_id", "label", "dice", "nsd", "hd95", "ref_volume_ml"]]
                       .round(3).to_string(index=False)))
    step("worst", 'res.worst_cases("dice", "pancreas", k=3)',
         lambda: print(r.worst_cases("dice", "pancreas", k=3).round(3).to_string(index=False)))
    step("lesions", 'res.lesions.query("kind == \'ref\'")[["case_id", "volume_ml", "detected", "dice"]]',
         lambda: print(r.lesions.query("kind == 'ref'")[["case_id", "volume_ml", "detected", "dice"]].round(3)
                       .to_string(index=False)))
    step("compare", 'compare(res_nnunet, res_medformer, metrics=["dice", "nsd", "hd95"], labels=["pancreas"])',
         lambda: print(compare(res["nnU-Net ResEnc-M"], res["MedFormer"], metrics=["dice", "nsd", "hd95"],
                               labels=["pancreas"])[["label", "metric", "n", "mean_a", "mean_b", "mean_diff",
                                                    "frac_a_better", "p_adjusted"]].round(3).to_string(index=False)))
    step("rank", 'rank_methods(results, "dice", label="pancreas")',
         lambda: print(rank_methods(res, "dice", label="pancreas").round(3).to_string(index=False)))
    (out / "walkthrough.json").write_text(json.dumps(log, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/showcase")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases_df = pd.read_csv(HERE / "showcase_cases.csv")
    cases = list(cases_df.case_id)
    print("evaluating 3 models on", len(cases), "cases")
    res = evaluate(cases)
    print("quantitative figures")
    quantitative(res, out)
    calibration(cases, out)
    print("qualitative figures")
    qualitative(res, cases_df, out)
    walkthrough_log(res, cases, out)
    from segevalkit.report import build_report

    build_report(res["nnU-Net ResEnc-M"], out / "report_nnunet.html", image_source=str(PANTS / "ImageTe"),
                 gallery_k=2, window="pancreas", title="Showcase report: nnU-Net on 8 PanTS test cases")
    print("done")


if __name__ == "__main__":
    main()
