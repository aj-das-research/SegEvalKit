"""Open one PanTS test case with all three official models in the interactive viewer.

Uses the same label mappings as the benchmark (benchmarks/pants/evaluate_models.py) and the
showcase evaluation folders for the stored per-case metrics.

    python benchmarks/showcase/view_pants.py --case PanTS_00009152            # serve on :8765
    python benchmarks/showcase/view_pants.py --case PanTS_00009152 --export docs/assets/showcase/viewer_9152.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pants"))
from evaluate_models import SOURCES, STORE, labels_for  # noqa: E402

from segevalkit.app import ViewerSession, export_html, serve  # noqa: E402

PANTS = STORE / "data/raw/PanTS"
KEYS = ["nnunet", "medformer", "totalseg"]


def session(case: str, models=KEYS, crop=None, max_dim=None) -> ViewerSession:
    ref_labels = {n: {"ref_file": s["ref_file"]} for n, s in labels_for("nnunet").items()}
    preds, pred_labels, results = {}, {}, {}
    for m in models:
        src = SOURCES[m]
        name = src["name"]
        root = src["root"]
        preds[name] = root / f"{case}.nii.gz" if src["layout"] == "flat" else root / case / (src.get("subdir") or "")
        pred_labels[name] = labels_for(m)
        ev = STORE / "outputs/eval/showcase" / m
        if (ev / "per_case_wide.csv").exists():
            results[name] = ev
    return ViewerSession(PANTS / "ImageTe" / case / "ct.nii.gz", ref=PANTS / "LabelTe" / case / "segmentations",
                         preds=preds, labels=ref_labels, pred_labels=pred_labels, results=results, case=case,
                         crop_margin_mm=crop, max_dim=max_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="PanTS_00009152")
    ap.add_argument("--export")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-dim", type=int, default=None)
    ap.add_argument("--crop", type=float, default=None)
    a = ap.parse_args()
    if a.export:
        s = session(a.case, crop=a.crop if a.crop is not None else 20.0, max_dim=a.max_dim or 224)
        print(s)
        out = export_html(s, a.export, focus=["pancreas"])
        print(out, f"{out.stat().st_size / 1e6:.1f} MB")
        return
    s = session(a.case, crop=a.crop, max_dim=a.max_dim)
    print(s)
    serve(s, "0.0.0.0", a.port, open_browser=False)


if __name__ == "__main__":
    main()
