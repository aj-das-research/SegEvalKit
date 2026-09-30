import json

import numpy as np
import pytest

import segevalkit as sek
from segevalkit.io import Source, load_volume, parse_labels, save_volume
from conftest import cube

AFF = np.diag([0.8, 0.8, 2.0, 1.0])


def _write_flat(tmp_path, n=4, missing=()):
    gt, pr = tmp_path / "gt", tmp_path / "pr"
    gt.mkdir()
    pr.mkdir()
    for i in range(n):
        g = np.zeros((32, 32, 20), np.uint8)
        g[cube((32, 32, 20), (6, 6, 4), (12, 12, 8))] = 1
        g[cube((32, 32, 20), (22, 22, 12), (4, 4, 4))] = 2
        save_volume(g, gt / f"case{i}.nii.gz", AFF)
        if i not in missing:
            save_volume(np.roll(g, i, 0), pr / f"case{i}.nii.gz", AFF)
    return pr, gt


def test_volume_roundtrip(tmp_path):
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    save_volume(a, tmp_path / "x.nii.gz", AFF)
    v = load_volume(tmp_path / "x.nii.gz", kind="image")
    assert v.spacing == pytest.approx((0.8, 0.8, 2.0))
    np.testing.assert_allclose(v.data, a)


def test_parse_labels_forms():
    assert [lab.name for lab in parse_labels({"0": "background", "1": "liver", "2": "tumour"})] == ["liver", "tumour"]
    labs = parse_labels({"kidney": [2, 3], "p": {"ref": 1, "pred": 7}})
    assert labs[0].ref_values == (2, 3) and labs[1].pred_values == (7,)
    assert parse_labels(["liver"])[0].file("ref") == "liver.nii.gz"


def test_layout_detection(tmp_path):
    pr, gt = _write_flat(tmp_path, n=2)
    assert Source(gt).layout == "flat"
    ps = tmp_path / "ps" / "c0" / "segmentations"
    ps.mkdir(parents=True)
    save_volume(np.zeros((4, 4, 4), np.uint8), ps / "liver.nii.gz")
    s = Source(tmp_path / "ps")
    assert s.layout == "per_structure" and s.subdir == "segmentations" and s.structures() == ["liver"]


def test_evaluate_flat_end_to_end(tmp_path):
    pr, gt = _write_flat(tmp_path, n=4, missing=(3,))
    ev = sek.Evaluator(labels={"organ": 1, "lesion": 2}, metrics=["default", "detection"])
    with pytest.warns(RuntimeWarning, match="no prediction"):
        res = ev.evaluate(pr, gt, progress=False, out_dir=tmp_path / "out")
    w = res.wide()
    assert set(w["label"]) == {"organ", "lesion"}
    assert w.loc[(w.case_id == "case0") & (w.label == "organ"), "dice"].item() == 1.0
    assert w.loc[w.case_id == "case3", "dice"].eq(0).all()  # missing -> empty prediction
    for f in ("per_case.csv", "per_case_wide.csv", "summary.csv", "meta.json", "lesions.csv"):
        assert (tmp_path / "out" / f).exists()
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert meta["missing_pred"] == ["case3"]
    back = sek.load_results(tmp_path / "out")
    assert len(back.per_case()) == len(res.per_case())


def test_evaluate_parallel_matches_serial(tmp_path):
    pr, gt = _write_flat(tmp_path, n=4)
    ev = sek.Evaluator(labels={"organ": 1}, metrics=["dice", "hd95"])
    a = ev.evaluate(pr, gt, progress=False).wide().sort_values("case_id").reset_index(drop=True)
    b = ev.evaluate(pr, gt, progress=False, n_workers=2).wide().sort_values("case_id").reset_index(drop=True)
    np.testing.assert_allclose(a["dice"], b["dice"])
    np.testing.assert_allclose(a["hd95"], b["hd95"])


def test_per_structure_vs_multilabel(tmp_path):
    """A per-structure reference can be scored against a multi-label prediction."""
    pr, gt = _write_flat(tmp_path, n=2)
    ps = tmp_path / "ref_ps"
    for i in range(2):
        d = ps / f"case{i}" / "segmentations"
        d.mkdir(parents=True)
        g = load_volume(gt / f"case{i}.nii.gz").data
        save_volume((g == 1).astype(np.uint8), d / "organ.nii.gz", AFF)
    ev = sek.Evaluator(labels={"organ": {"pred": 1}}, metrics=["dice"])
    res = ev.evaluate(pr, ps, progress=False)
    assert res.wide().loc[lambda d: d.case_id == "case0", "dice"].item() == 1.0


def test_alignment_strict_and_resample(tmp_path):
    pr, gt = _write_flat(tmp_path, n=1)
    g = load_volume(gt / "case0.nii.gz").data
    shifted = AFF.copy()
    shifted[2, 3] = 2.0  # origin moved by one slice
    save_volume(g, pr / "case0.nii.gz", shifted)
    ev = sek.Evaluator(labels={"organ": 1}, metrics=["dice"])
    with pytest.warns(RuntimeWarning, match="failed"):
        res = ev.evaluate(pr, gt, progress=False)
    assert "case0" in res.meta["errors"]
    ev2 = sek.Evaluator(labels={"organ": 1}, metrics=["dice"], alignment="resample")
    d = ev2.evaluate(pr, gt, progress=False).wide()["dice"].item()
    assert 0.5 < d < 1.0


def test_evaluate_arrays():
    g = np.zeros((20, 20, 20), np.uint8)
    g[5:15, 5:15, 5:15] = 1
    res = sek.Evaluator(metrics=["dice"]).evaluate_arrays(g, g, spacing=(1, 1, 1))
    assert res.wide()["dice"].item() == 1.0


def test_per_label_metrics(tmp_path):
    pr, gt = _write_flat(tmp_path, n=2)
    ev = sek.Evaluator(labels={"organ": {"values": 1, "metrics": ["dice"]},
                               "lesion": {"values": 2, "metrics": "dice,lesion_f1"}}, metrics="default")
    res = ev.evaluate(pr, gt, progress=False)
    got = res.per_case().groupby("label")["metric"].unique().to_dict()
    assert list(got["organ"]) == ["dice"] and set(got["lesion"]) == {"dice", "lesion_f1"}
    assert len(res.lesions) and set(res.lesions["label"]) == {"lesion"}
