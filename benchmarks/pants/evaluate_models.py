"""Evaluate the official baseline models on the PanTS test set with SegEvalKit.

Every model writes a different output convention; the label specs below map
each one onto the PanTS reference structures
(``LabelTe/<case>/segmentations/<structure>.nii.gz``):

* **nnU-Net PanTS-regions**: one multi-label map per case (``<case>.nii.gz``)
  with region-based ids from its ``dataset.json``; the pancreas region is
  the union of ids 17-21 and 28 (lesion).
* **MedFormer (PanTS)**: per-structure files ``<case>/predictions/<name>.nii.gz``
  with PanTS names, plus lesion probabilities in ``predictions_raw/``.
* **TotalSegmentator**: per-structure files ``<case>/<name>.nii.gz`` with its
  own names (``gallbladder``, ``inferior_vena_cava``, ``urinary_bladder``) and
  no lesion class.

Structures get metrics that fit them (organs: overlap + boundary; tubular:
+ clDice / Betti; lesion: + detection), following the SegEvalKit guide.

Usage:
    python benchmarks/pants/evaluate_models.py --models nnunet medformer totalseg --workers 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import segevalkit as sek
from segevalkit.io import Source

STORE = Path("/l/users/abhijit.das/SegEvalKit")
REF = STORE / "data/raw/PanTS/LabelTe"
PRED = STORE / "outputs/predictions"
OUT = STORE / "outputs/eval/pants"

ORGAN = ["dice", "iou", "nsd", "hd95", "assd", "masd", "precision", "recall", "relative_volume_difference",
         "absolute_volume_difference"]
TUBULAR = ORGAN + ["cldice", "betti0_error", "betti1_error"]
LESION = ["dice", "nsd", "hd95", "assd", "precision", "recall", "relative_volume_difference", "ece", "brier",
          "auprc", "soft_dice",
          "absolute_volume_difference", "lesion_recall", "lesion_precision", "lesion_f1", "lesionwise_dice",
          "panoptic_quality", "false_positive_lesions", "false_negative_lesions", "lesion_count_difference"]

# PanTS structure -> (kind, nnU-Net ids, TotalSegmentator file stem or None)
STRUCTURES = {
    "pancreas": ("organ", [17, 18, 19, 20, 21, 28], "pancreas"),
    "pancreatic_lesion": ("lesion", [28], None),
    "pancreatic_duct": ("tubular", [21], None),
    "liver": ("organ", [14], "liver"),
    "spleen": ("organ", [24], "spleen"),
    "kidney_left": ("organ", [12], "kidney_left"),
    "kidney_right": ("organ", [13], "kidney_right"),
    "stomach": ("organ", [25], "stomach"),
    "gall_bladder": ("organ", [11], "gallbladder"),
    "duodenum": ("organ", [8], "duodenum"),
    "colon": ("organ", [6], "colon"),
    "adrenal_gland_left": ("organ", [1], "adrenal_gland_left"),
    "adrenal_gland_right": ("organ", [2], "adrenal_gland_right"),
    "aorta": ("tubular", [3], "aorta"),
    "postcava": ("tubular", [22], "inferior_vena_cava"),
    "superior_mesenteric_artery": ("tubular", [26], None),
    "veins": ("tubular", [27], None),
    "common_bile_duct": ("tubular", [7], None),
}
METRICS = {"organ": ORGAN, "tubular": TUBULAR, "lesion": LESION}
PARAMS = {"nsd": {"tolerance_mm": 2.0}, "ece": {"roi": "band"}, "brier": {"roi": "band"}, "auprc": {"roi": "band"}}


# PanTS annotates the lesion inconsistently relative to the pancreas mask: in most cases the lesion lies
# inside pancreas.nii.gz, in others almost entirely outside it. The pancreas is therefore evaluated as
# pancreas ∪ lesion on both sides, which is also how the nnU-Net checkpoint defines its pancreas region.
UNION = {"pancreas": "pancreas.nii.gz+pancreatic_lesion.nii.gz"}


def labels_for(model: str):
    out = {}
    for name, (kind, ids, ts) in STRUCTURES.items():
        spec = {"ref_file": UNION.get(name, f"{name}.nii.gz"), "metrics": METRICS[kind]}
        if model == "nnunet":
            spec["pred"] = ids
        elif model == "medformer":
            if name == "pancreatic_duct":  # not predicted by this checkpoint
                continue
            spec["pred_file"] = UNION.get(name, f"{name}.nii.gz")
        elif model == "totalseg":
            if ts is None:
                continue
            spec["pred_file"] = f"{ts}.nii.gz"
        out[name] = spec
    return out


SOURCES = {
    "nnunet": dict(root=PRED / "nnunet_pants_regions", layout="flat", name="nnU-Net ResEnc-M"),
    # The official R-Super script nests outputs under <save_path>/<dataset>/<experiment>/.
    "medformer": dict(root=PRED / "medformer_pants" / "abdomenatlas" / "pants_pancreas_release",
                      layout="per_structure", subdir="predictions", prob_subdir="predictions_raw", name="MedFormer"),
    "totalseg": dict(root=PRED / "totalsegmentator", layout="per_structure", subdir="", name="TotalSegmentator"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SOURCES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cases", help="text file of case ids (default: all reference cases with a prediction)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ref = Source(REF, layout="per_structure", subdir="segmentations")
    for m in a.models:
        s = SOURCES[m]
        pred = Source(s["root"], layout=s["layout"], subdir=s.get("subdir"))
        cases = [c for c in ref.case_ids if c in set(pred.case_ids)]
        if a.cases:
            keep = set(Path(a.cases).read_text().split())
            cases = [c for c in cases if c in keep]
        if a.limit:
            cases = cases[: a.limit]
        # PanTS label headers disagree with their CT in 227 of 901 test cases (the aorta in 218), while the
        # voxel data is aligned with the CT (see `segevalkit audit`). All predictions are written on the CT
        # grid, so voxel correspondence is the correct pairing; "resample" would trust the broken headers.
        ev = sek.Evaluator(labels=labels_for(m), metrics=ORGAN, params=PARAMS, device=a.device,
                           min_lesion_voxels=10, alignment="ignore")
        prob = None
        if s.get("prob_subdir"):  # lesion probabilities -> calibration metrics
            prob = Source(s["root"], layout="per_structure", subdir=s["prob_subdir"], kind="prob")
        res = ev.evaluate(pred, ref, prob=prob, cases=cases, n_workers=a.workers, out_dir=OUT / m, name=s["name"])
        print(res, json.dumps({k: res.meta[k] for k in ("seconds",)}))


if __name__ == "__main__":
    main()
