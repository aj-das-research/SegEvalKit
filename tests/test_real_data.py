"""Checks on real public datasets (skipped when data/raw is absent).

They exercise the layouts, geometry handling and presets on actual files:
PanTS and TotalSegmentator (per-structure, varying orientation, float-stored
masks) and MSD (flat multi-label). References are scored against perturbed
copies of themselves, so the expected behaviour is known.
"""

from pathlib import Path

import numpy as np
import pytest

import segevalkit as sek
from segevalkit import synthetic
from segevalkit.datasets import get_dataset
from segevalkit.io import Source, load_volume, parse_labels

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
pytestmark = pytest.mark.data


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"{p} not available")
    return p


def test_pants_per_structure_layout_and_float_masks():
    root = _need(RAW / "PanTS" / "LabelTe")
    src = Source(root, layout="per_structure", subdir="segmentations")
    assert len(src.case_ids) == 901
    cid = src.case_ids[0]
    assert {"pancreas", "pancreatic_lesion", "liver", "veins"} <= set(src.structures(cid))
    # Sub-part masks are stored so they load as 1.0000000591: they must become clean integer labels.
    v = load_volume(src.path(cid) / "pancreas_head.nii.gz")
    assert np.issubdtype(v.data.dtype, np.integer) and set(np.unique(v.data)) <= {0, 1}


def test_totalsegmentator_self_consistency():
    root = _need(RAW / "TotalSegmentator")
    src = Source(root, layout="per_structure", subdir="segmentations")
    cid = src.case_ids[0]
    lab = parse_labels(["liver"])[0]
    ref, vol = src.load_mask(cid, lab, "ref")
    if ref is None or not ref.any():
        pytest.skip("no liver in first case")
    pred = synthetic.dilate(ref, 2.0, vol.spacing)
    r = sek.compute_metrics(pred, ref, ["dice", "nsd", "hd"], spacing=vol.spacing,
                            params={"nsd": {"tolerance_mm": 2.0}})
    assert r["nsd"] == pytest.approx(1.0)
    assert 1.5 <= r["hd"] <= 2.0 + max(vol.spacing)
    assert 0.5 < r["dice"] < 1.0


def test_msd_liver_flat_layout_with_preset(tmp_path):
    root = _need(RAW / "MSD_Task03_Liver" / "labelsTr")
    ref = Source(root)
    assert ref.layout == "flat" and len(ref.case_ids) == 131
    cid = ref.case_ids[0]
    v = load_volume(root / f"{cid}.nii.gz")
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    sek.io.save_volume(v, pred_dir / f"{cid}.nii.gz")
    preset = get_dataset("msd_liver")
    ev = sek.Evaluator(labels=preset.labels, metrics=["dice", "nsd"], params=preset.params)
    res = ev.evaluate(pred_dir, root, cases=[cid], progress=False)
    w = res.wide()
    assert (w["dice"] == 1.0).all() and (w["nsd"] == 1.0).all()
    assert set(w["label"]) == {"liver", "cancer", "liver_with_tumour"}
