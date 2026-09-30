"""Presets for public volumetric segmentation benchmarks.

A preset records what you need to evaluate on a benchmark *the way the
benchmark does*: its label ids and region definitions, the on-disk layout of
its reference labels, its official metrics and their parameters (e.g. NSD
tolerance), and its empty-mask convention.

Examples:
    >>> from segevalkit.datasets import get_dataset
    >>> kits = get_dataset("kits23")
    >>> kits.labels["masses"]
    [2, 3]

From the command line:

```console
segevalkit datasets            # list
segevalkit datasets kits23     # details
segevalkit evaluate --dataset kits23 --pred preds/ --ref kits23/labels --out eval/
```

Official protocols evolve; each preset cites its source and the notes list
known deviations. Where a challenge uses a convention SegEvalKit implements
differently (e.g. surfel-weighted surface Dice), the preset says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

__all__ = ["DatasetPreset", "DATASETS", "get_dataset", "register_dataset"]


@dataclass
class DatasetPreset:
    """Evaluation preset for a dataset / benchmark.

    Attributes:
        key: Registry key.
        title: Full name.
        modality: ``"CT"``, ``"MR"``, ``"PET/CT"``, ``"CT+MR"``...
        anatomy: Region / targets.
        labels: Label spec (see `segevalkit.io.parse_labels`); for
            per-structure datasets a list of structure names (``None`` = all
            files found).
        metrics: Official (or de-facto standard) metrics.
        params: Metric parameters, e.g. NSD tolerance.
        empty_policy: `EmptyPolicy` preset name.
        ref_layout: Layout of the reference labels (``"flat"``, ``"folder"``,
            ``"per_structure"`` or ``"auto"``; ``None`` = auto).
        ref_file: Reference file name inside each case folder (``"folder"``
            layout).
        ref_subdir: Sub-folder holding one file per structure
            (``"per_structure"`` layout).
        cases: Approximate number of labelled cases.
        url: Download / project page.
        license: Data licence.
        citation: Primary citation.
        notes: Caveats and protocol details.
    """

    key: str
    title: str
    modality: str
    anatomy: str
    labels: Union[Dict[str, Any], List[str], None]
    metrics: List[str]
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    empty_policy: str = "segevalkit"
    ref_layout: Optional[str] = None
    ref_file: Optional[str] = None
    ref_subdir: Optional[str] = None
    cases: str = ""
    url: str = ""
    license: str = ""
    citation: str = ""
    notes: str = ""

    def label_summary(self) -> str:
        """Readable label description, e.g. ``liver=1, kidney=2+3 (NSD τ 1.1 mm)``."""
        if self.labels is None:
            return "all structure files found per case"
        if isinstance(self.labels, list):
            return ", ".join(self.labels)
        parts = []
        for k, v in self.labels.items():
            if isinstance(v, dict):
                ids = v.get("values", v.get("ref"))
                tau = v.get("params", {}).get("nsd", {}).get("tolerance_mm")
                txt = f"{k}={'+'.join(map(str, ids if isinstance(ids, list) else [ids]))}"
                parts.append(txt + (f" (NSD τ {tau:g} mm)" if tau is not None else ""))
            else:
                parts.append(f"{k}={'+'.join(map(str, v if isinstance(v, list) else [v]))}")
        return ", ".join(parts)

    def describe(self) -> str:
        lab = self.label_summary()
        lines = [
            f"{self.title}  [{self.key}]",
            f"  modality   {self.modality}",
            f"  anatomy    {self.anatomy}",
            f"  cases      {self.cases}",
            f"  labels     {lab}",
            f"  layout     {self.ref_layout or 'auto'}" + (f" ({self.ref_subdir or self.ref_file})" if (self.ref_subdir or self.ref_file) else ""),
            f"  metrics    {', '.join(self.metrics)}",
            f"  params     {self.params or '-'}",
            f"  empty      {self.empty_policy}",
            f"  url        {self.url}",
            f"  license    {self.license}",
            f"  cite       {self.citation}",
        ]
        if self.notes:
            lines.append(f"  notes      {self.notes}")
        return "\n".join(lines)


DATASETS: Dict[str, DatasetPreset] = {}


def register_dataset(preset: DatasetPreset) -> DatasetPreset:
    DATASETS[preset.key] = preset
    return preset


def get_dataset(key: str) -> DatasetPreset:
    k = key.lower().replace("-", "_")
    if k not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; available: {', '.join(sorted(DATASETS))}")
    return DATASETS[k]


_ORGAN = ["dice", "nsd", "hd95", "assd", "precision", "recall", "relative_volume_difference"]
_LESION = ["dice", "nsd", "hd95", "lesion_f1", "lesion_recall", "lesion_precision", "lesionwise_dice",
           "panoptic_quality", "absolute_volume_difference", "relative_volume_difference"]

# --------------------------------------------------------------- abdomen CT
register_dataset(DatasetPreset(
    key="pants", title="PanTS: Pancreatic Tumor Segmentation", modality="CT",
    anatomy="Pancreas (head / body / tail), pancreatic lesion and duct, 20+ surrounding structures",
    labels=None, metrics=_ORGAN + ["lesion_f1", "lesion_recall", "lesionwise_dice"],
    params={"nsd": {"tolerance_mm": 2.0}},
    ref_layout="per_structure", ref_subdir="segmentations",
    cases="36,390 CT volumes (9,000 train / 901 public test; per-voxel annotations)",
    url="https://github.com/MrGiovanni/PanTS",
    license="CC BY-NC-ND 4.0 (GitHub LICENSE; the HF card says CC BY-NC-SA 4.0): non-commercial",
    citation="Li et al. 2025, PanTS: The Pancreatic Tumor Segmentation Dataset, NeurIPS Datasets & Benchmarks",
    notes="Label headers are unreliable: in 227 of 901 test cases at least one label file (the aorta in 218) has "
          "an affine that disagrees with its CT although the voxel data is aligned with it; evaluate with "
          "alignment='ignore' and check with `segevalkit audit`. "
          "Evaluate from LabelTe/<case>/segmentations/<name>.nii.gz, not combined_labels.nii.gz: the "
          "per-structure masks overlap (pancreas contains head/body/tail and duct; veins overlap "
          "organs), so the combined map is lossy. The lesion is annotated inconsistently relative to the "
          "pancreas mask (sometimes inside it, sometimes outside), so evaluate the pancreas as pancreas "
          "\u222a lesion (ref_file 'pancreas.nii.gz+pancreatic_lesion.nii.gz'). Orientation varies per case. "
          "Most test cases are "
          "tumour-free: report patient-level presence detection (stats.presence_detection) next to lesion "
          "Dice. The official leaderboard reports patient-wise sensitivity/specificity/AUC, tumour-wise "
          "sensitivity and lesion DSC; the NSD tolerance and the detection rule are not published.",
))

register_dataset(DatasetPreset(
    key="totalsegmentator", title="TotalSegmentator CT (v2)", modality="CT",
    anatomy="117 anatomical structures (organs, bones, muscles, vessels)",
    labels=None, metrics=["dice", "nsd", "hd95", "assd"],
    params={"nsd": {"tolerance_mm": 3.0}},
    ref_layout="per_structure", ref_subdir="segmentations",
    cases="1,228 CT (v2); v3 (2026-09): 1,939 CT + 1,296 MR", url="https://github.com/wasserth/TotalSegmentator",
    license="CC BY 4.0 (dataset); model weights: see repository",
    citation="Wasserthal et al. 2023, Radiology: AI 5(5):e230024",
    notes="<case>/ct.nii.gz and <case>/segmentations/<structure>.nii.gz. The paper reports Dice and NSD.",
))

register_dataset(DatasetPreset(
    key="btcv", title="BTCV Multi-Atlas Labeling Beyond the Cranial Vault (abdomen)", modality="CT",
    anatomy="13 abdominal organs",
    labels={"spleen": 1, "right_kidney": 2, "left_kidney": 3, "gallbladder": 4, "esophagus": 5, "liver": 6,
            "stomach": 7, "aorta": 8, "inferior_vena_cava": 9, "portal_splenic_vein": 10, "pancreas": 11,
            "right_adrenal": 12, "left_adrenal": 13},
    metrics=["dice", "hd95", "nsd", "assd"], ref_layout="flat",
    cases="30 train / 20 test", url="https://www.synapse.org/#!Synapse:syn3193805",
    license="Synapse data use terms", citation="Landman et al. 2015, MICCAI Multi-Atlas Labeling Challenge",
))

register_dataset(DatasetPreset(
    key="amos", title="AMOS 2022: Abdominal Multi-Organ Segmentation", modality="CT+MR",
    anatomy="15 abdominal organs",
    labels={"spleen": 1, "right_kidney": 2, "left_kidney": 3, "gallbladder": 4, "esophagus": 5, "liver": 6,
            "stomach": 7, "aorta": 8, "postcava": 9, "pancreas": 10, "right_adrenal": 11, "left_adrenal": 12,
            "duodenum": 13, "bladder": 14, "prostate_uterus": 15},
    metrics=["dice", "nsd", "hd95"], params={"nsd": {"tolerance_mm": 1.0}}, ref_layout="flat",
    cases="Tr 200 CT + 40 MR, Va 100 CT + 20 MR, Ts 240", url="https://amos22.grand-challenge.org/",
    license="CC BY 4.0 (Zenodo; dataset.json says CC BY-SA 4.0)", citation="Ji et al. 2022, NeurIPS Datasets & Benchmarks",
    notes="Case id < 500 is CT, >= 500 is MR. dataset.json spells aorta 'arota'. Rank-then-aggregate ranking.",
))

register_dataset(DatasetPreset(
    key="flare22", title="FLARE 2022: Fast and Low-resource Abdominal oRgan sEgmentation", modality="CT",
    anatomy="13 abdominal organs",
    labels={name: {"values": [i], "params": {"nsd": {"tolerance_mm": tau}}} for i, (name, tau) in enumerate([
        ("liver", 5), ("right_kidney", 3), ("spleen", 3), ("pancreas", 5), ("aorta", 2), ("inferior_vena_cava", 2),
        ("right_adrenal", 2), ("left_adrenal", 2), ("gallbladder", 2), ("esophagus", 3), ("stomach", 5),
        ("duodenum", 7), ("left_kidney", 3)], start=1)},
    metrics=["dice", "nsd"], ref_layout="flat",
    cases="50 labelled + 2,000 unlabelled train; 200 validation; 800 test",
    url="https://flare22.grand-challenge.org/", license="CC BY-NC-SA 4.0",
    citation="Ma et al. 2024, Lancet Digital Health 6(11):e815",
    notes="Per-organ NSD tolerances from the official code. The official code also forces NSD to 0 when "
          "DSC < 0.2 and evaluates aorta / IVC / esophagus only on reference-labelled slices; SegEvalKit "
          "does not replicate these two rules.",
))

# Medical Segmentation Decathlon: labels from each task's dataset.json, NSD
# tolerances from Antonelli et al. 2022 (Nat Commun 13:4128).
_MSD = [
    ("msd_brain", "Task01 BrainTumour", "MR", "Glioma (MSD numbering: 1 edema, 2 non-enhancing, 3 enhancing; not the BraTS ids)",
     {"edema": 1, "non_enhancing_tumor": 2, "enhancing_tumor": 3, "tumor_core": [2, 3], "whole_tumor": [1, 2, 3]},
     5.0, "484 train / 266 test"),
    ("msd_heart", "Task02 Heart", "MR", "Left atrium", {"left_atrium": 1}, 4.0, "20 train / 10 test"),
    ("msd_liver", "Task03 Liver", "CT", "Liver and liver tumours",
     {"liver": 1, "cancer": 2, "liver_with_tumour": [1, 2]}, 7.0, "131 train / 70 test"),
    ("msd_hippocampus", "Task04 Hippocampus", "MR", "Anterior and posterior hippocampus",
     {"anterior": 1, "posterior": 2}, 1.0, "260 train / 130 test"),
    ("msd_prostate", "Task05 Prostate", "MR", "Peripheral and transition zones",
     {"peripheral_zone": 1, "transition_zone": 2}, 4.0, "32 train / 16 test"),
    ("msd_lung", "Task06 Lung", "CT", "Lung cancer", {"cancer": 1}, 2.0, "63 train / 32 test"),
    ("msd_pancreas", "Task07 Pancreas", "CT", "Pancreas and pancreatic tumour",
     {"pancreas": 1, "tumour": 2, "pancreas_with_tumour": [1, 2]}, 5.0, "281 train / 139 test"),
    ("msd_hepaticvessel", "Task08 HepaticVessel", "CT", "Hepatic vessels and tumours",
     {"vessel": 1, "tumour": 2}, 3.0, "303 train / 140 test"),
    ("msd_spleen", "Task09 Spleen", "CT", "Spleen", {"spleen": 1}, 3.0, "41 train / 20 test"),
    ("msd_colon", "Task10 Colon", "CT", "Primary colon cancer", {"colon_cancer": 1}, 4.0, "126 train / 64 test"),
]
for _key, _task, _mod, _anat, _labels, _tau, _cases in _MSD:
    _lesion = any(w in _anat.lower() for w in ("tumour", "cancer", "glioma"))
    register_dataset(DatasetPreset(
        key=_key, title=f"Medical Segmentation Decathlon {_task}", modality=_mod, anatomy=_anat, labels=_labels,
        metrics=["dice", "nsd", "hd95", "relative_volume_difference"] + (["lesion_f1", "lesion_recall"] if _lesion else []),
        params={"nsd": {"tolerance_mm": _tau}}, ref_layout="flat", cases=_cases + " (test labels withheld)",
        url="http://medicaldecathlon.com/", license="CC BY-SA 4.0",
        citation="Antonelli et al. 2022, Nat Commun 13:4128",
        notes=f"Official MSD ranking used DSC and NSD with a task-specific tolerance ({_tau:g} mm), with "
              "Wilcoxon-based significance ranking per region. Labels in labelsTr/<prefix>_<id>.nii.gz.",
    ))

register_dataset(DatasetPreset(
    key="lits", title="LiTS 2017: Liver Tumor Segmentation", modality="CT", anatomy="Liver and liver tumours",
    labels={"liver": 1, "tumour": 2, "liver_with_tumour": [1, 2]},
    metrics=["dice", "voe", "relative_volume_difference", "assd", "hd", "lesion_f1", "lesion_recall"],
    ref_layout="flat", cases="131 train / 70 test", url="https://competitions.codalab.org/competitions/17094",
    license="see challenge terms", citation="Bilic et al. 2023, Medical Image Analysis 84:102680",
    notes="The official code scores ASSD / MSD as 0 when either mask is empty (not penalised); SegEvalKit's "
          "default policy penalises it. Files: volume-N.nii / segmentation-N.nii.",
))

register_dataset(DatasetPreset(
    key="acdc", title="ACDC: Automated Cardiac Diagnosis Challenge", modality="MR",
    anatomy="Right ventricle, myocardium, left ventricle (ED and ES frames)",
    labels={"right_ventricle": 1, "myocardium": 2, "left_ventricle": 3},
    metrics=["dice", "hd", "hd95", "assd", "absolute_volume_difference"], ref_layout="flat",
    cases="100 train / 50 test", url="https://www.creatis.insa-lyon.fr/Challenge/acdc/",
    license="CC BY-NC-SA 4.0", citation="Bernard et al. 2018, IEEE TMI 37(11):2514",
    notes="Label order 1 RV, 2 MYO, 3 LV: the REVERSE of M&Ms (use the mnms preset there).",
))

register_dataset(DatasetPreset(
    key="mnms", title="M&Ms: Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Segmentation", modality="MR",
    anatomy="Left ventricle, myocardium, right ventricle",
    labels={"left_ventricle": 1, "myocardium": 2, "right_ventricle": 3},
    metrics=["dice", "hd", "hd95", "assd"], ref_layout="flat", cases="375 subjects",
    url="https://www.ub.edu/mnms/", license="see challenge terms",
    citation="Campello et al. 2021, IEEE TMI 40(12):3543",
    notes="Label order 1 LV, 2 MYO, 3 RV: the REVERSE of ACDC.",
))

register_dataset(DatasetPreset(
    key="segthor", title="SegTHOR: Segmentation of THoracic Organs at Risk", modality="CT",
    anatomy="Thoracic organs at risk", labels={"esophagus": 1, "heart": 2, "trachea": 3, "aorta": 4},
    metrics=["dice", "hd95", "assd", "masd", "cldice"], ref_layout="folder", ref_file="GT.nii.gz",
    cases="40 train / 20 test", url="https://competitions.codalab.org/competitions/21145",
    license="see challenge terms", citation="Lambert et al. 2020, IPTA",
    notes="The official distance is the 'average Hausdorff' (max of the directed mean distances).",
))

register_dataset(DatasetPreset(
    key="kits19", title="KiTS19: Kidney Tumor Segmentation 2019", modality="CT", anatomy="Kidneys and renal tumours",
    labels={"kidney_and_tumour": [1, 2], "tumour": [2]}, metrics=["dice", "nsd", "hd95"], ref_layout="folder",
    ref_file="segmentation.nii.gz", cases="210 train / 90 test", url="https://kits19.grand-challenge.org/",
    license="CC BY-NC-SA 4.0", citation="Heller et al. 2021, Medical Image Analysis 67:101821",
    notes="The official ranking uses the mean of the composite (kidney+tumour) and tumour Dice.",
))

register_dataset(DatasetPreset(
    key="kits23", title="KiTS23: Kidney and Kidney Tumor Segmentation", modality="CT",
    anatomy="Kidneys, renal tumours and cysts (hierarchical evaluation classes)",
    labels={"kidney_and_masses": {"values": [1, 2, 3], "params": {"nsd": {"tolerance_mm": 1.0331}}},
            "masses": {"values": [2, 3], "params": {"nsd": {"tolerance_mm": 1.1329}}},
            "tumor": {"values": [2], "params": {"nsd": {"tolerance_mm": 1.1498}}}},
    metrics=["dice", "nsd", "hd95"], ref_layout="folder",
    ref_file="segmentation.nii.gz", cases="489 train / 110 test", url="https://kits-challenge.org/kits23/",
    license="CC BY-NC-SA 4.0", citation="Heller et al. 2023, arXiv:2307.01984",
    notes="Hierarchical Evaluation Classes (HECs) with surface-Dice tolerances derived from inter-observer "
          "variability (1.0331 / 1.1329 / 1.1498 mm). The official surface Dice uses DeepMind-style surfel "
          "weighting; SegEvalKit's voxel NSD agrees closely but not exactly.",
))

# --------------------------------------------------------------- brain MR
register_dataset(DatasetPreset(
    key="brats2023", title="BraTS 2023 Adult Glioma", modality="MR",
    anatomy="Glioma sub-regions (evaluated as ET / TC / WT regions)",
    labels={"enhancing_tumor": [3], "tumor_core": [1, 3], "whole_tumor": [1, 2, 3]},
    metrics=["lesionwise_dice", "lesionwise_hd95", "dice", "hd95", "lesion_f1", "lesion_recall", "lesion_precision"],
    empty_policy="brats2023", ref_layout="auto",
    cases="1,251 train / 219 validation", url="https://www.synapse.org/brats2023",
    license="Synapse data use terms (CC BY-NC for most subsets)",
    citation="Kazerooni et al. 2023, arXiv:2305.17033; Baid et al. 2021, arXiv:2107.02314",
    notes="Label ids 1 = NCR, 2 = ED, 3 = ET. Official lesion-wise metrics dilate the reference before "
          "matching and drop lesions <= 50 mm^3 (set min_lesion_voxels accordingly); empty-case HD95 = 374 mm.",
))

register_dataset(DatasetPreset(
    key="isles22", title="ISLES 2022: Ischemic Stroke Lesion Segmentation", modality="MR",
    anatomy="Acute/sub-acute ischemic stroke lesions (DWI/ADC/FLAIR)", labels={"lesion": 1},
    metrics=["dice", "absolute_volume_difference", "lesion_f1", "lesion_count_difference"],
    ref_layout="auto", cases="250 train / 150 test", url="https://isles22.grand-challenge.org/",
    license="CC BY 4.0", citation="Hernandez Petzsche et al. 2022, Sci Data 9:762",
))

# --------------------------------------------------------------- PET/CT
register_dataset(DatasetPreset(
    key="autopet", title="autoPET: whole-body FDG-PET/CT lesion segmentation", modality="PET/CT",
    anatomy="Tumour lesions (melanoma, lymphoma, lung cancer)", labels={"lesion": 1},
    metrics=["dice", "false_positive_lesions", "false_negative_lesions", "lesion_f1",
             "absolute_volume_difference"],
    ref_layout="auto", cases="1,014 studies (autoPET I/II)", url="https://autopet.grand-challenge.org/",
    license="TCIA restricted / CC BY 4.0 (see challenge)", citation="Gatidis et al. 2022, Sci Data 9:601",
    notes="Official autoPET uses 18-connected components (evaluate with connectivity=18) and reports "
          "false-positive and false-negative *volumes* (mL); SegEvalKit's lesion table provides the "
          "per-component volumes to reproduce them. Tumour-free studies are scored by FP volume only.",
))

# --------------------------------------------------------------- vessels
register_dataset(DatasetPreset(
    key="topcow", title="TopCoW: Topology-Aware Anatomical Segmentation of the Circle of Willis",
    modality="CT+MR", anatomy="Circle of Willis arteries (13 vessel classes)", labels=None,
    metrics=["dice", "cldice", "betti0_error", "betti1_error", "hd95"], ref_layout="auto",
    cases="125 CTA + 125 MRA (2024)", url="https://topcow24.grand-challenge.org/",
    license="CC BY-NC 4.0", citation="Yang et al. 2023, arXiv:2312.17670",
    empty_policy="topcow",
    notes="Official HD95 uses 90 mm for missing classes (the `topcow` empty policy). "
          "clDice is computed on the merged binary vessel mask; Betti-0 error per class.",
))
