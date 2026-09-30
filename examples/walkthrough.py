"""A guided walkthrough on real data: every step's output is recorded for the README.

Runs SegEvalKit on the PanTS test cases for which fresh model predictions exist
(official nnU-Net PanTS-regions checkpoint), prints each step's real output, and
writes the figures used in the README to docs/assets/walkthrough/.

    python examples/walkthrough.py --n 40 --out docs/assets/walkthrough
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

STORE = Path("/l/users/abhijit.das/SegEvalKit")
REF = STORE / "data/raw/PanTS/LabelTe"
IMG = STORE / "data/raw/PanTS/ImageTe"
PRED = STORE / "outputs/predictions/nnunet_pants_regions"


def step(title, code, fn, log):
    buf = io.StringIO()
    with redirect_stdout(buf):
        value = fn()
    out = buf.getvalue().rstrip()
    log.append({"title": title, "code": code, "output": out})
    print(f"\n### {title}\n{code}\n---\n{out}")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="docs/assets/walkthrough")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 12)
    log = []

    import segevalkit as sek
    from segevalkit import plotting as P
    from segevalkit import viz
    from segevalkit.io import Source, load_volume

    cases = sorted(p.name[:-7] for p in PRED.glob("*.nii.gz"))[: a.n]
    (out / "cases.txt").write_text("\n".join(cases))

    ev = step("Create an evaluator", '''ev = sek.Evaluator(
    labels={
        "pancreas":          {"pred": [17, 18, 19, 20, 21, 28], "ref_file": "pancreas.nii.gz"},
        "pancreatic_lesion": {"pred": 28, "ref_file": "pancreatic_lesion.nii.gz",
                              "metrics": ["default", "detection"]},
        "liver":             {"pred": 14, "ref_file": "liver.nii.gz"},
    },
    metrics="default", params={"nsd": {"tolerance_mm": 2.0}}, min_lesion_voxels=10,
)
print(ev.config.metrics)''', lambda: _mk(sek), log)

    res = step("Evaluate a folder of predictions", '''res = ev.evaluate("nnunet_predictions/", "PanTS/LabelTe/", cases=cases, n_workers=8, out_dir="eval/")
print(res)''', lambda: _eval(ev, cases, out), log)

    step("Summary with 95 % bootstrap CIs", '''res.summary().query("metric in ['dice', 'nsd', 'hd95']")[["label", "metric", "n", "mean", "median", "ci_low", "ci_high"]]''',
         lambda: print(res.summary().query("metric in ['dice', 'nsd', 'hd95']")[
             ["label", "metric", "n", "mean", "median", "ci_low", "ci_high"]].round(3).to_string(index=False)), log)

    step("Per-case table", '''res.wide()[["case_id", "label", "dice", "nsd", "hd95", "ref_volume_ml"]].head(6)''',
         lambda: print(res.wide()[["case_id", "label", "dice", "nsd", "hd95", "ref_volume_ml"]].head(6)
                       .round(3).to_string(index=False)), log)

    step("Worst cases", '''res.worst_cases("dice", "pancreas", k=3)''',
         lambda: print(res.worst_cases("dice", "pancreas", k=3).round(3).to_string(index=False)), log)

    step("Lesion-level table", '''res.lesions.query("kind == 'ref'")[["case_id", "volume_ml", "detected", "dice"]].head(5)''',
         lambda: print(res.lesions.query("kind == 'ref'")[["case_id", "volume_ml", "detected", "dice"]].head(5)
                       .round(3).to_string(index=False)), log)

    def plots():
        P.metric_distribution(res, "dice", labels=["liver", "pancreas", "pancreatic_lesion"]).savefig(out / "dice.png")
        P.metric_vs_size(res, "dice", label="pancreas").savefig(out / "size.png")
        P.failure_quadrants(res, "pancreas").savefig(out / "failures.png")
        print("wrote dice.png, size.png, failures.png")

    step("Plots", '''P.metric_distribution(res, "dice", labels=["liver", "pancreas", "pancreatic_lesion"]).savefig("dice.png")
P.metric_vs_size(res, "dice", label="pancreas").savefig("size.png")
P.failure_quadrants(res, "pancreas").savefig("failures.png")''', plots, log)

    def overlay():
        cid = res.worst_cases("dice", "pancreas", k=1)["case_id"].iloc[0]
        ref = load_volume(REF / cid / "segmentations" / "pancreas.nii.gz")
        img = load_volume(IMG / cid / "ct.nii.gz", kind="image")
        pred = load_volume(PRED / f"{cid}.nii.gz")
        import numpy as np

        viz.triplanar(img.data, np.isin(pred.data, [17, 18, 19, 20, 21, 28]), ref.data > 0, affine=ref.affine,
                      window="pancreas", title=f"{cid} · pancreas").savefig(out / "triplanar.png")
        print(f"wrote triplanar.png for {cid}")

    step("Look at the worst case", '''cid = res.worst_cases("dice", "pancreas", k=1)["case_id"][0]
viz.triplanar(ct.data, pred.data >= 17, ref.data > 0, affine=ref.affine, window="pancreas").savefig("triplanar.png")''',
         overlay, log)

    def report():
        from segevalkit.report import build_report

        p = build_report(res, out / "report.html", image_source=str(IMG), gallery_k=2)
        print(f"wrote {p.name} ({p.stat().st_size / 1e6:.1f} MB, self-contained)")

    step("HTML report", '''sek.report.build_report(res, "eval/report.html", image_source="PanTS/ImageTe/")''', report, log)
    (out / "walkthrough.json").write_text(json.dumps(log, indent=2))


def _mk(sek):
    ev = sek.Evaluator(
        labels={
            "pancreas": {"pred": [17, 18, 19, 20, 21, 28], "ref_file": "pancreas.nii.gz"},
            "pancreatic_lesion": {"pred": 28, "ref_file": "pancreatic_lesion.nii.gz",
                                  "metrics": ["default", "detection"]},
            "liver": {"pred": 14, "ref_file": "liver.nii.gz"},
        },
        metrics="default", params={"nsd": {"tolerance_mm": 2.0}}, min_lesion_voxels=10,
    )
    print(ev.config.metrics)
    return ev


def _eval(ev, cases, out):
    from segevalkit.io import Source

    res = ev.evaluate(Source(PRED, layout="flat"), Source(REF, layout="per_structure", subdir="segmentations"),
                      cases=cases, n_workers=8, out_dir=out / "eval", progress=False, name="nnU-Net ResEnc-M")
    print(res)
    return res


if __name__ == "__main__":
    main()
