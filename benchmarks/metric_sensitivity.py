"""Metric sensitivity study on real anatomy.

For real reference masks of structures with very different size and shape
(liver, kidney, pancreas, gallbladder, aorta, veins, pancreatic lesion) we
apply each controlled perturbation of :mod:`segevalkit.synthetic` at
increasing magnitude and record every metric. The resulting curves show, with
physical units on the x-axis, what each metric does and does not see: e.g.
that Dice barely moves for a 2 mm boundary error on the liver but collapses on
a lesion, that HD reacts to a single distant island while Dice ignores it, or
that only topology metrics notice a cut vessel.

Usage (CPU node):
    python benchmarks/metric_sensitivity.py --ref-root data/raw/PanTS/LabelTe \
        --out outputs/sensitivity --n-cases 12 --workers 12
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

STRUCTURES = {
    "liver": "large organ", "kidney_left": "compact organ", "pancreas": "elongated organ",
    "gall_bladder": "small organ", "aorta": "tubular", "veins": "tubular (branching)",
    "pancreatic_lesion": "lesion",
}
METRICS = ["dice", "iou", "nsd", "boundary_iou", "hd", "hd95", "assd", "masd", "relative_volume_difference",
           "lesion_f1", "centroid_distance"]
# Topology metrics (skeletons, Betti numbers) are the expensive family; they are
# computed where the literature says they matter plus one compact and one large
# organ as controls (Betti numbers are cheap, skeletons are not).
TOPOLOGY = {"aorta", "veins", "pancreatic_lesion", "kidney_left", "gall_bladder", "pancreas"}
PERTURBATIONS = {
    "erode": [0, 1, 2, 3, 5], "dilate": [0, 1, 2, 3, 5], "shift": [0, 1, 2, 4, 8],
    "boundary_noise": [0, 1, 2, 3, 5], "islands": [0, 1, 2, 4, 8], "holes": [0, 1, 2, 4, 8],
    "cut": [0, 2, 4, 8], "remove_slab": [0, 0.05, 0.1, 0.2, 0.4],
}


ONLY: list = []  # restrict to these metrics (supplementary runs); set from --only


def _one(args):
    case_id, structure, ref_root, out, only = args
    global ONLY
    ONLY = only
    part = Path(out) / ("parts" if not only else "parts_" + "_".join(only)) / f"{structure}__{case_id}.csv"
    if part.exists():
        return pd.read_csv(part)
    from segevalkit.io import load_volume
    from segevalkit.metrics.context import bbox
    from segevalkit.synthetic import sensitivity_study

    f = Path(ref_root) / case_id / "segmentations" / f"{structure}.nii.gz"
    if not f.exists():
        return None
    v = load_volume(f)
    m = v.data > 0
    if m.sum() < 50:
        return None
    # Work in a crop with an 80 mm margin: perturbations stay inside, EDTs stay small.
    margin = int(np.ceil(80 / min(v.spacing)))
    m = m[bbox(m, margin)]
    metrics = METRICS + (["cldice", "betti0_error", "betti1_error", "betti2_error"] if structure in TOPOLOGY else [])
    if ONLY:
        metrics = [m for m in metrics if m in ONLY]
    df = sensitivity_study([(case_id, m, v.spacing)], metrics, perturbations=PERTURBATIONS,
                           params={"nsd": {"tolerance_mm": 2.0}, "boundary_iou": {"width_mm": 2.0}}, seed=7)
    df["structure"] = structure
    df["ref_volume_ml"] = float(m.sum() * np.prod(v.spacing) / 1000)
    part.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(part, index=False)
    print(f"done {structure} {case_id}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=[], help="compute only these metrics (supplementary run)")
    ap.add_argument("--structures", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted(os.listdir(a.ref_root))
    rng = np.random.default_rng(a.seed)
    lesion_cases = [c for c in cases if (Path(a.ref_root) / c / "segmentations" / "pancreatic_lesion.nii.gz").exists()]
    pick = list(rng.choice(cases, size=min(a.n_cases, len(cases)), replace=False))
    jobs = []
    for s in (a.structures or STRUCTURES):
        pool_cases = pick
        if s == "pancreatic_lesion":
            # Lesion masks are empty in tumour-free patients; sample among cases that have one.
            from segevalkit.io import load_volume

            nonempty = []
            for c in rng.permutation(lesion_cases):
                if load_volume(Path(a.ref_root) / c / "segmentations" / f"{s}.nii.gz").data.any():
                    nonempty.append(c)
                if len(nonempty) >= a.n_cases:
                    break
            pool_cases = nonempty
        jobs += [(c, s, a.ref_root, str(out), a.only) for c in pool_cases]
    with ProcessPoolExecutor(a.workers) as ex:
        frames = [d for d in ex.map(_one, jobs) if d is not None]
    df = pd.concat(frames, ignore_index=True)
    dest = out / ("sensitivity.csv" if not a.only else f"sensitivity_{'_'.join(a.only)}.csv")
    df.to_csv(dest, index=False)
    print(df.groupby(["structure"])["case_id"].nunique())
    print(f"wrote {dest} ({len(df)} rows)")


if __name__ == "__main__":
    main()
