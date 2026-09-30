"""Pick a small, varied set of PanTS test cases for documentation examples.

Stratified by reference pancreatic-lesion volume (large / medium / small / none),
seeded, so the showcase set is reproducible. Writes showcase_cases.csv.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REF = Path("/l/users/abhijit.das/SegEvalKit/data/raw/PanTS/LabelTe")


def info(case):
    im = nib.load(REF / case / "segmentations" / "pancreatic_lesion.nii.gz")
    vox = float(np.prod(im.header.get_zooms()[:3]))
    les = np.asanyarray(im.dataobj) > 0.5
    return {"case_id": case, "lesion_ml": les.sum() * vox / 1000, "orientation": "".join(nib.aff2axcodes(im.affine)),
            "slice_mm": float(max(im.header.get_zooms()[:3]))}


if __name__ == "__main__":
    cases = sorted(p.name for p in REF.iterdir() if p.is_dir())
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        df = pd.DataFrame(list(ex.map(info, cases, chunksize=8)))
    df.to_csv(Path(__file__).with_name("pants_test_lesion_volumes.csv"), index=False)
    rng = np.random.default_rng(2026)
    strata = {"large (>10 mL)": df.lesion_ml > 10, "medium (1-10 mL)": (df.lesion_ml > 1) & (df.lesion_ml <= 10),
              "small (<1 mL)": (df.lesion_ml > 0) & (df.lesion_ml <= 1), "tumour-free": df.lesion_ml == 0}
    pick = []
    for name, m in strata.items():
        sub = df[m]
        for i in rng.choice(len(sub), size=min(2, len(sub)), replace=False):
            pick.append({**sub.iloc[i].to_dict(), "stratum": name})
    out = pd.DataFrame(pick)
    out.to_csv(Path(__file__).with_name("showcase_cases.csv"), index=False)
    print(df.lesion_ml.describe(), "\n", (df.lesion_ml > 0).sum(), "cases with a lesion")
    print(out.to_string(index=False))
