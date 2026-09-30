"""Re-evaluate the pancreatic lesion with per-lesion confidence scores (FROC / CPM, PR curves).

The organ evaluation (evaluate_models.py) is left untouched; this writes a
separate results folder per model with only the lesion, scored with the
detection metrics including lesion-wise HD95 / NSD and lesion AP. MedFormer's
lesion probabilities (``predictions_raw/``) give each predicted lesion its
maximum probability as the score; models without probabilities fall back to
the lesion volume.

    python benchmarks/pants/lesion_scores.py --models medformer --workers 30
"""

from __future__ import annotations

import argparse
import json

import segevalkit as sek
from segevalkit.io import Source

from evaluate_models import LESION, OUT, PARAMS, REF, SOURCES, labels_for

EXTRA = ["lesionwise_hd95", "lesionwise_nsd", "lesion_ap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["medformer"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    ref = Source(REF, layout="per_structure", subdir="segmentations")
    for m in a.models:
        s = SOURCES[m]
        pred = Source(s["root"], layout=s["layout"], subdir=s.get("subdir"))
        cases = [c for c in ref.case_ids if c in set(pred.case_ids)]
        lab = {"pancreatic_lesion": {**labels_for(m)["pancreatic_lesion"], "metrics": LESION + EXTRA}}
        ev = sek.Evaluator(labels=lab, metrics=LESION + EXTRA, params=PARAMS, min_lesion_voxels=10,
                           alignment="ignore", lesion_score="max")
        prob = Source(s["root"], layout="per_structure", subdir=s["prob_subdir"], kind="prob") \
            if s.get("prob_subdir") else None
        res = ev.evaluate(pred, ref, prob=prob, cases=cases, n_workers=a.workers,
                          out_dir=OUT / f"{m}_lesion_scores", name=s["name"])
        print(res, json.dumps({k: res.meta[k] for k in ("seconds",)}))


if __name__ == "__main__":
    main()
