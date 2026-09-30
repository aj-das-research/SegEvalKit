"""Turn the PanTS evaluations into every benchmark artefact (tables, cohorts, docs page inputs, paper tables).

Reads outputs/eval/pants/<model>/ (written by evaluate_models.py) and the official
PanTS metadata, and writes outputs/eval/pants/report/:

    macro.csv          per-case mean / median / CI per model, structure, metric
    pooled.csv         pooled (micro) voxel and lesion metrics per model
    presence.csv       patient-level tumour detection (AUC, sens/spec at spec >= 0.9)
    compare_*.csv      paired comparisons between models (Holm-corrected)
    ranking.csv        aggregate-then-rank and rank-then-aggregate + bootstrap stability
    cohorts_*.csv      subgroup summaries and tests (phase, sex, scanner, country, slice, age, tumour status)
    numbers.json       headline numbers quoted by the docs page and the paper

    python benchmarks/pants/benchmark_report.py [--models nnunet medformer totalseg]
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import segevalkit as sek
from segevalkit.cohort import cohort_summary, cohort_tests, pooled_metrics
from segevalkit.stats import compare, presence_detection, rank_methods, ranking_stability

STORE = Path("/l/users/abhijit.das/SegEvalKit")
EVAL = STORE / "outputs/eval/pants"
META = STORE / "data/pants_meta/metadata.xlsx"
NAMES = {"nnunet": "nnU-Net ResEnc-M", "medformer": "MedFormer", "totalseg": "TotalSegmentator"}
ORGANS = ["pancreas", "liver", "spleen", "kidney_left", "kidney_right", "stomach", "gall_bladder", "duodenum",
          "aorta", "postcava"]
MAIN = ["dice", "nsd", "hd95"]


def metadata() -> pd.DataFrame:
    """Official PanTS metadata with harmonised cohort factors (mapping documented here)."""
    m = pd.read_excel(META)
    m = m[m["PanTS ID"] >= "PanTS_00009001"].copy()
    s = lambda c: m[c].astype(str).str.strip().replace({"nan": np.nan})  # noqa: E731
    out = pd.DataFrame({"case_id": m["PanTS ID"].astype(str).str.strip()})
    out["CT phase"] = s("ct phase")
    out["sex"] = s("sex").str.upper()
    man = s("manufacturer").str.upper()
    out["scanner"] = man.replace({"GE MEDICAL SYSTEMS": "GE", "PHILIPS MEDICAL SYSTEMS": "PHILIPS"}).str.title() \
        .replace({"Ge": "GE"})
    nat = s("site nationality")
    out["country"] = np.where(nat.str.contains(";", na=False), "multinational", nat)
    sp = m["spacing"].astype(str).map(lambda x: max(float(v) for v in re.findall(r"[\d.]+", x)) if re.findall(r"[\d.]+", x) else np.nan)
    out["slice thickness"] = pd.cut(sp.values, [0, 1.25, 3.0, 100], labels=["≤ 1.25 mm", "1.25–3 mm", "> 3 mm"]).astype(str)
    out["age"] = pd.cut(pd.to_numeric(m["age"], errors="coerce").values, [0, 50, 65, 120],
                        labels=["< 50", "50–65", "> 65"]).astype(str)
    out["tumour"] = np.where(m["tumor?"].astype(str).str.strip() == "1", "tumour", "no tumour")
    return out.replace({"nan": np.nan})


FACTORS = ["CT phase", "sex", "scanner", "country", "slice thickness", "age", "tumour"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(NAMES))
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    out = EVAL / "report"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for m in a.models:
        if (EVAL / m / "per_case.csv").exists():
            r = sek.load_results(EVAL / m)
            r.meta["name"] = NAMES[m]
            res[NAMES[m]] = r
    numbers = {"n_cases": {k: len(v.cases) for k, v in res.items()}}
    # macro
    macro = pd.concat([r.summary(n_boot=a.n_boot).assign(model=k) for k, r in res.items()])
    macro.to_csv(out / "macro.csv", index=False)
    # pooled
    pooled = pd.concat([pooled_metrics(r, n_boot=a.n_boot).assign(model=k) for k, r in res.items()])
    pooled.to_csv(out / "pooled.csv", index=False)
    # presence detection
    pres = []
    for k, r in res.items():
        if "pancreatic_lesion" in r.labels:
            p = presence_detection(r, "pancreatic_lesion")
            pres.append({"model": k, **{x: p[x] for x in ("auc", "sensitivity", "specificity", "threshold", "n_pos", "n_neg")}})
    pd.DataFrame(pres).to_csv(out / "presence.csv", index=False)
    # paired comparisons and ranking
    for (ka, ra), (kb, rb) in combinations(res.items(), 2):
        c = compare(ra, rb, metrics=MAIN, labels=[x for x in ORGANS if x in ra.labels and x in rb.labels])
        c.to_csv(out / f"compare_{ka.split()[0]}_vs_{kb.split()[0]}.csv", index=False)
    rank_rows = []
    for struct in ["pancreas", "liver", "spleen", "kidney_left", "aorta"]:
        rs = {k: v for k, v in res.items() if struct in v.labels}
        if len(rs) < 2:
            continue
        for scheme in ("aggregate-then-rank", "rank-then-aggregate"):
            rk = rank_methods(rs, "dice", label=struct, scheme=scheme)
            rank_rows += [{"structure": struct, "scheme": scheme, **row} for row in rk.to_dict("records")]
        stab = ranking_stability(rs, "dice", label=struct, n_boot=min(a.n_boot, 500))
        rank_rows.append({"structure": struct, "scheme": "bootstrap", "kendall_tau_median": float(np.nanmedian(stab["kendall_tau"]))})
    pd.DataFrame(rank_rows).to_csv(out / "ranking.csv", index=False)
    # cohorts
    md = metadata()
    md.to_csv(out / "cohort_factors.csv", index=False)
    for k, r in res.items():
        labels = [x for x in ("pancreas", "pancreatic_lesion", "liver") if x in r.labels]
        cohort_summary(r, md, FACTORS, metrics=MAIN, labels=labels, n_boot=a.n_boot).to_csv(
            out / f"cohorts_{k.split()[0]}.csv", index=False)
        cohort_tests(r, md, FACTORS, metrics=MAIN, labels=labels).to_csv(out / f"cohort_tests_{k.split()[0]}.csv", index=False)
    (out / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
