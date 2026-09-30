"""Patient-level tumour detection of MedFormer with the official R-Super protocol.

R-Super reports sensitivity / specificity / F1 on the PanTS test set from
``test_with_reports.py`` (predicted tumour volume in voxels after 1 mm
resampling and an erode-1 / dilate-2 opening) and
``calculate_sensitivity_specificity.py`` (a patient is positive when that volume
is >= a threshold T; ground truth = "number of pancreatic lesion instances" >= 1
in ``metadata_pants.csv``). This script runs their ``detection`` function,
imported unchanged from the R-Super checkout, on our MedFormer predictions and
writes the volume per case, so the report can apply their threshold grid.

    python benchmarks/pants/rsuper_detection.py --workers 16
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import pandas as pd

STORE = Path("/l/users/abhijit.das/SegEvalKit")
RS = STORE / "third_party/R-Super/rsuper_train"
PRED = STORE / "outputs/predictions/medformer_pants/abdomenatlas/pants_pancreas_release"
CT = STORE / "data/raw/PanTS/ImageTe"
OUT = STORE / "outputs/eval/pants/medformer/rsuper_detection.csv"

_spec = importlib.util.spec_from_file_location("rs_test_with_reports", RS / "test_with_reports.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def one(case: str) -> dict:
    spacing = nib.load(str(CT / case / "ct.nii.gz")).header.get_zooms()
    vol = _mod.detection(str(PRED / case / "predictions" / "pancreatic_lesion.nii.gz"), None, spacing, 0.5)
    return {"BDMAP_ID": case, "pancreatic tumor volume predicted": float(vol)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cases = sorted(p.name for p in PRED.iterdir() if p.is_dir())
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, cases, chunksize=4))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(len(rows), "cases ->", OUT)


if __name__ == "__main__":
    main()
