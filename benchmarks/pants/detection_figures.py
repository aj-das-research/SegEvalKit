"""FROC / ROC / PR figures and numbers for the PanTS lesion evaluation with scores.

Reads outputs/eval/pants/<model>_lesion_scores (written by lesion_scores.py) and
writes docs/assets/figures/{froc,roc,pr}_pants_lesion.png plus a JSON of the
numbers quoted in the docs.

    python benchmarks/pants/detection_figures.py --models medformer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import segevalkit as sek  # noqa: E402
from segevalkit import plotting as P  # noqa: E402
from segevalkit.stats import froc, lesion_pr, localized_presence, patient_pr  # noqa: E402

EVAL = Path("/l/users/abhijit.das/SegEvalKit/outputs/eval/pants")
NAMES = {"nnunet": "nnU-Net ResEnc-M", "medformer": "MedFormer"}
LABEL = "pancreatic_lesion"
# R-Super GitHub, rsuper_train/Merlin_demo.md / documents/demo_results.png, "PanTS test", standard segmentation
# baseline (the MedFormerPanTS checkpoint): sensitivity 76 %, specificity 91 %.
RSUPER = {"sensitivity": 0.76, "specificity": 0.91, "label": "R-Super GitHub (baseline)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["medformer"])
    ap.add_argument("--out", default="docs/assets/figures")
    a = ap.parse_args()
    out = Path(a.out)
    res = {}
    for m in a.models:
        r = sek.load_results(EVAL / f"{m}_lesion_scores")
        r.meta["name"] = NAMES[m]
        res[NAMES[m]] = r
    numbers = {}
    for name, r in res.items():
        f = froc(r, LABEL, n_boot=1000)
        lp = localized_presence(r, LABEL, score="pred_volume_ml", target_specificity=0.9)
        ll = localized_presence(r, LABEL, score="lesion", target_specificity=0.9)
        numbers[name] = {
            "score_type": f["score_type"], "n_ref": f["n_ref"], "n_cases": f["n_cases"],
            "cpm": f["cpm"], "cpm_ci": f["cpm_ci"], "sensitivity_at": f["sensitivity_at"],
            "max_sensitivity": f["max_sensitivity"], "lesion_ap": lesion_pr(r, LABEL)["ap"],
            "patient_ap": patient_pr(r, LABEL)["ap"], "patient_ap_localized": patient_pr(r, LABEL, localized=True)["ap"],
            "presence_volume": {k: lp[k] for k in ("auc", "auc_localized", "threshold", "specificity", "sensitivity",
                                                   "sensitivity_localized", "lesion_sensitivity", "n_pos", "n_neg")},
            "presence_lesion_score": {k: ll[k] for k in ("auc", "auc_localized", "threshold", "specificity",
                                                         "sensitivity", "sensitivity_localized", "lesion_sensitivity")},
        }
    one = next(iter(res.values())) if len(res) == 1 else res
    P.froc_plot(one, LABEL, n_boot=500, figsize=(4.4, 3.2)).savefig(out / "froc_pants_lesion.png", dpi=200)
    P.roc_plot(one, LABEL, reference_point=RSUPER, figsize=(4.0, 3.6)).savefig(out / "roc_pants_lesion.png", dpi=200)
    P.pr_plot(one, LABEL, figsize=(4.0, 3.4)).savefig(out / "pr_pants_lesion.png", dpi=200)
    (EVAL / "report" / "detection_curves.json").write_text(json.dumps(numbers, indent=2, default=float))
    print(json.dumps(numbers, indent=2, default=float))


if __name__ == "__main__":
    main()
