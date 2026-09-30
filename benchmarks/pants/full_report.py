"""Self-contained HTML report of one model on the full PanTS test set (segevalkit.report.build_report).

    python benchmarks/pants/full_report.py --model medformer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import segevalkit as sek
from segevalkit.report import build_report

STORE = Path("/l/users/abhijit.das/SegEvalKit")
NAMES = {"nnunet": "nnU-Net ResEnc-M", "medformer": "MedFormer", "totalseg": "TotalSegmentator"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medformer")
    ap.add_argument("--out", default=str(STORE / "outputs/reports"))
    a = ap.parse_args()
    r = sek.load_results(STORE / "outputs/eval/pants" / a.model)
    r.meta["name"] = NAMES[a.model]
    out = Path(a.out) / f"pants_{a.model}_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = ["pancreatic_lesion", "pancreas", "liver", "spleen", "kidney_left", "kidney_right", "stomach",
              "gall_bladder", "duodenum", "aorta", "postcava"]
    r = r.filter(labels=[x for x in labels if x in r.labels])
    build_report(r, out, image_source=str(STORE / "data/raw/PanTS/ImageTe"), gallery_k=3, window="pancreas",
                 max_labels=len(r.labels),
                 title=f"{NAMES[a.model]} on the PanTS test set ({len(r.cases)} CTs)")
    print(out, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
