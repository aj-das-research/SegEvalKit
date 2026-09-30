"""Which metrics should I report? A problem-fingerprint recommender.

Following the Metrics Reloaded framework (Maier-Hein et al. 2024, *Nat
Methods* 21:195), the choice of metrics is derived from properties of the
problem, not from habit. Describe the problem with a `Fingerprint` and
`recommend` returns a metric set, each with the reason it was chosen and
the pitfall it guards against.

Examples:
    >>> from segevalkit.guide import Fingerprint, recommend
    >>> rec = recommend(Fingerprint(structure="small_lesion", multi_instance=True))
    >>> rec.metrics[:4]
    ['dice', 'nsd', 'masd', 'lesion_f1']

The combined rule of thumb from the literature: **one overlap metric + one
boundary metric**, plus detection metrics when objects are instances,
topology metrics when shape connectivity matters, volume agreement when
volume is the endpoint, and calibration metrics when probabilities are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..metrics import get_metric

__all__ = ["Fingerprint", "Recommendation", "recommend", "STRUCTURES", "AVOID"]

#: Supported structure types.
STRUCTURES = {
    "large_organ": "Large, compact organ (liver, kidney, spleen, lung, brain)",
    "small_structure": "Small compact structure (adrenal gland, lymph node, gallbladder)",
    "small_lesion": "Small lesion(s) (metastases, nodules, small tumours)",
    "large_lesion": "Large lesion (primary tumour, glioma, large mass)",
    "tubular": "Tubular / curvilinear (vessels, airways, ducts, nerves)",
    "hollow": "Hollow or thin-walled organ (colon, bladder wall, myocardium)",
}

#: Metrics that should not be primary endpoints for 3D segmentation, with reasons.
AVOID: Dict[str, str] = {
    "accuracy": "Dominated by true-negative background voxels; depends on the field of view.",
    "specificity": "≈1 for almost any 3D prediction: background dominates.",
    "auroc": "Voxel-level AUROC is dominated by easy background; prefer AUPRC or ECE in an ROI.",
    "volumetric_similarity": "Blind to position: a perfectly wrong-place mask can score 1.",
    "hd": "Set by a single outlier voxel; report HD95 alongside or instead.",
    "global_consistency_error": "Weak interpretation for binary tasks.",
    "adjusted_rand_index": "Weak interpretation for binary tasks; tracks Dice.",
}


@dataclass
class Fingerprint:
    """Properties of a segmentation problem that drive metric choice.

    Attributes:
        structure: One of `STRUCTURES`.
        multi_instance: Several separate objects per image whose *detection*
            matters (lesions, nodules, metastases).
        boundary_critical: Boundary position has clinical consequences
            (radiotherapy contouring, surgical planning).
        volumetry: Volume itself is the endpoint (tumour burden, atrophy).
        empty_references: Some cases legitimately contain no structure.
        probabilistic: The model outputs probabilities that will be used.
        fp_fn_asymmetric: Missing tissue is worse than adding it (or vice versa).
        noisy_reference: Reference annotations are imprecise (inter-rater
            disagreement comparable to the structure size).
        tolerance_mm: Acceptable boundary deviation (inter-rater variability);
            used as the NSD / boundary IoU tolerance.
        ranking: Results will be used to rank several methods.
    """

    structure: str = "large_organ"
    multi_instance: bool = False
    boundary_critical: bool = False
    volumetry: bool = False
    empty_references: bool = False
    probabilistic: bool = False
    fp_fn_asymmetric: bool = False
    noisy_reference: bool = False
    tolerance_mm: Optional[float] = None
    ranking: bool = False

    def __post_init__(self):
        if self.structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {sorted(STRUCTURES)}")


@dataclass
class Recommendation:
    """The recommended metrics with justification.

    Attributes:
        items: ``(metric, role, reason)`` triples in priority order; role is
            ``"primary"``, ``"secondary"`` or ``"diagnostic"``.
        params: Suggested metric parameters (e.g. NSD tolerance).
        empty_policy: Suggested empty-mask policy preset.
        statistics: Suggested statistical analysis.
        avoid: Metrics to avoid as primary endpoints, with reasons.
        notes: Extra caveats.
    """

    items: List[Tuple[str, str, str]] = field(default_factory=list)
    params: Dict[str, Dict] = field(default_factory=dict)
    empty_policy: str = "segevalkit"
    statistics: List[str] = field(default_factory=list)
    avoid: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def metrics(self) -> List[str]:
        return [m for m, _, _ in self.items]

    def add(self, metric: str, role: str, reason: str) -> None:
        """Add a metric; a later, stronger role (primary > secondary > diagnostic) upgrades an earlier one."""
        rank = {"primary": 0, "secondary": 1, "diagnostic": 2}
        for i, (m, r, _) in enumerate(self.items):
            if m == metric:
                if rank[role] < rank[r]:
                    self.items[i] = (metric, role, reason)
                return
        self.items.append((metric, role, reason))

    def table(self) -> pd.DataFrame:
        rows = []
        for m, role, reason in self.items:
            info = get_metric(m)
            rows.append({"metric": m, "name": info.display, "family": info.family, "role": role,
                         "why": reason, "reference": info.reference})
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        lines = ["| Metric | Role | Why |", "|---|---|---|"]
        for m, role, reason in self.items:
            lines.append(f"| {get_metric(m).display} (`{m}`) | {role} | {reason} |")
        if self.params:
            lines += ["", "**Parameters:** " + ", ".join(f"`{k}`: {v}" for k, v in self.params.items())]
        if self.statistics:
            lines += ["", "**Statistics:** " + "; ".join(self.statistics)]
        if self.notes:
            lines += [""] + [f"> {n}" for n in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:
        width = max(len(m) for m in self.metrics) if self.items else 10
        out = [f"{m:<{width}}  {role:<10} {reason}" for m, role, reason in self.items]
        return "\n".join(out)


def recommend(fp: Optional[Fingerprint] = None, **kwargs) -> Recommendation:
    """Recommend metrics for a problem described by a `Fingerprint` (or keyword arguments)."""
    fp = fp or Fingerprint(**kwargs)
    r = Recommendation()
    tol = fp.tolerance_mm
    s = fp.structure

    # 1. Overlap: always one, the choice depends on size and cost asymmetry.
    if s in ("large_organ", "large_lesion", "hollow"):
        r.add("dice", "primary", "Overlap is informative for large objects and is the field's common currency.")
    elif s == "tubular":
        r.add("cldice", "primary", "Tubular fingerprint: rewards connected, complete centrelines (Shit et al. 2021).")
        r.add("dice", "secondary", "Overlap for comparability; blind to breaks in thin structures on its own.")
    else:
        r.add("dice", "primary", "Overlap for comparability; unstable for small objects (one voxel moves it a lot), "
                                 "so never report it alone here.")
    if fp.fp_fn_asymmetric:
        r.add("fbeta", "secondary", "Unequal costs of missing vs adding tissue: F-beta with beta > 1 weights recall.")
        r.add("recall", "secondary", "Makes the under-segmentation component explicit.")
        r.add("precision", "secondary", "Makes the over-segmentation component explicit.")

    # 2. Boundary: NSD with a task tolerance + a robust distance.
    if tol is None:
        tol = {"large_organ": 2.0, "large_lesion": 2.0, "hollow": 1.0, "tubular": 1.0,
               "small_structure": 1.0, "small_lesion": 1.0}[s]
        r.notes.append(f"NSD tolerance defaulted to {tol} mm; set it from inter-rater variability for your task.")
    r.params["nsd"] = {"tolerance_mm": tol}
    r.add("nsd", "primary", f"Boundary agreement within a clinically acceptable {tol} mm; tolerates annotation "
                            "imprecision and approximates correction effort (Nikolov et al. 2021).")
    if s in ("small_lesion", "small_structure"):
        r.add("masd", "secondary", "Average boundary error; HD-type metrics are dominated by single voxels on small objects.")
    else:
        r.add("hd95", "secondary", "Near-worst-case boundary error: catches leakage and distant false positives.")
    if fp.boundary_critical:
        r.add("hd95", "primary", "Worst-case geometric error matters for dose / margins.")
        r.add("hd", "diagnostic", "Maximum error, for safety review (not as a ranking metric).")
        r.add("assd", "secondary", "Typical boundary error in mm, complementary to HD95.")
        r.params["boundary_iou"] = {"width_mm": tol}
        r.add("boundary_iou", "secondary", "Boundary-band IoU stays sensitive to boundary quality on large organs.")

    # 3. Instances.
    if fp.multi_instance or s == "small_lesion":
        r.add("lesion_f1", "primary", "Unit of analysis is the lesion: were they found, and only they?")
        r.add("lesion_recall", "secondary", "Detection sensitivity, independent of lesion size.")
        r.add("lesion_precision", "secondary", "False discoveries at lesion level.")
        r.add("lesionwise_dice", "secondary", "BraTS 2023-style: every lesion weighs equally, misses score 0.")
        r.add("lesionwise_hd95", "secondary", "BraTS 2023-style boundary error per lesion; misses and false lesions score the penalty.")
        r.add("panoptic_quality", "secondary", "Detection x segmentation quality in one number (Kirillov et al. 2019).")
        r.add("lesion_count_difference", "diagnostic", "Over-/under-counting (ISLES'22).")
        r.add("split_count", "diagnostic", "Fragmentation of single lesions.")
        r.add("merge_count", "diagnostic", "Neighbouring lesions merged into one.")
        r.statistics.append("Stratify detection by lesion size (plotting.detection_by_size)")

    # 4. Topology.
    if s in ("tubular", "hollow"):
        r.add("betti0_error", "secondary", "Counts breaks and spurious islands (connectivity).")
        r.add("betti1_error", "secondary", "Loops / tunnels opened or falsely closed.")
        if s == "hollow":
            r.add("betti2_error", "secondary", "Cavities: a filled-in lumen or spurious void.")

    # 5. Volume.
    if fp.volumetry:
        r.add("absolute_volume_difference", "primary", "Volume is the endpoint: report the error in mL.")
        r.add("relative_volume_difference", "primary", "Signed error reveals systematic over-/under-segmentation.")
        r.statistics += ["Bland-Altman bias and limits of agreement (stats.bland_altman)",
                         "ICC(2,1) for absolute agreement (stats.icc)"]
    else:
        r.add("relative_volume_difference", "diagnostic", "Signed volume bias, cheap to add, pairs with overlap.")

    # 6. Empty references.
    if fp.empty_references:
        r.empty_policy = "metrics_reloaded"
        r.notes.append("Empty references: report presence detection separately (cases with ref_empty) "
                       "and state the empty policy; Dice/HD are undefined when both masks are empty.")

    # 7. Probabilities.
    if fp.probabilistic:
        r.add("ece", "primary", "Are the probabilities trustworthy? Use roi='band' so background does not dominate.")
        r.add("brier", "secondary", "Proper scoring rule combining calibration and sharpness.")
        r.add("auprc", "secondary", "Discrimination that stays informative under class imbalance.")
        r.add("nll", "diagnostic", "Penalises confident errors strongly.")
        r.params.update({"ece": {"roi": "band"}, "brier": {"roi": "band"}})

    if fp.noisy_reference:
        r.notes.append("Noisy reference: favour tolerance-based metrics (NSD) and report inter-rater agreement "
                       "as the ceiling for any score.")

    r.statistics.insert(0, "Per-case values with mean, median, IQR and 95% bootstrap CI (stats.summarize)")
    if fp.ranking:
        r.statistics += ["Paired Wilcoxon signed-rank with Holm correction (stats.compare)",
                         "Ranking robustness under bootstrap with Kendall tau (stats.ranking_stability)",
                         "Report both aggregate-then-rank and rank-then-aggregate"]
    r.avoid = dict(AVOID)
    return r
