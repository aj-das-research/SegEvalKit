"""Interactive viewer (segevalkit.app): data model, HTTP endpoints, static export and ``view --demo``.

No browser is needed: the tests check what the page would receive.
"""

from __future__ import annotations

import gzip
import json
import threading
import urllib.error
import urllib.request

import nibabel as nib
import numpy as np
import pytest

from segevalkit.app import (
    ERROR_CODES,
    ViewerSession,
    demo_session,
    export_html,
    make_phantom,
    make_server,
)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("phantom")


@pytest.fixture(scope="module")
def demo(cache):
    return demo_session(cache)


def _nifti(b: bytes):
    return nib.Nifti1Image.from_bytes(gzip.decompress(b))


def _mz3_counts(b: bytes):
    raw = gzip.decompress(b)
    magic, attr = np.frombuffer(raw[:4], "<u2")
    nf, nv, _ = np.frombuffer(raw[4:16], "<u4")
    return int(magic), int(attr), int(nf), int(nv), len(raw)


def test_phantom_is_cached(cache):
    p1 = make_phantom(cache)
    t = p1["image"].stat().st_mtime_ns
    p2 = make_phantom(cache)
    assert p1 == p2 and p2["image"].stat().st_mtime_ns == t
    assert set(p1) == {"image", "ref", "Model A", "Model B", "Model C"}


def test_demo_manifest(demo):
    m = demo.manifest()
    names = [s["name"] for s in m["structures"]]
    assert names[0] == "pancreatic_lesion"  # lesions are listed first
    assert set(names) == {"liver", "spleen", "kidney_left", "kidney_right", "pancreas", "aorta", "stomach",
                          "pancreatic_lesion"}
    assert m["models"] == ["Model A", "Model B", "Model C"]
    les = m["lesions"]["pancreatic_lesion"]
    assert len(les["ref"]) == 1
    found = les["ref"][0]["found"]
    assert found == {"Model A": True, "Model B": True, "Model C": False}
    assert len(les["fp"]["Model B"]) == 1 and not les["fp"]["Model A"] and not les["fp"]["Model C"]
    liver = next(s for s in m["structures"] if s["name"] == "liver")
    assert all(0.9 < liver["models"][k]["dice"] <= 1 for k in m["models"])
    json.dumps(m)  # serialisable


def test_pancreas_region_includes_lesion(demo):
    # PHANTOM_LABELS define pancreas = ids {5, 8}: the reference region contains the lesion
    assert (demo.ref_masks["pancreatic_lesion"] & ~demo.ref_masks["pancreas"]).sum() == 0


def test_error_map_codes_match_counts(demo):
    s = next(x for x in demo.manifest()["structures"] if x["name"] == "pancreas")
    e = _nifti(demo.error_nifti("Model B", "pancreas")).get_fdata().astype(int)
    mb = s["models"]["Model B"]
    assert (e == ERROR_CODES["tp"]).sum() == mb["tp"]
    assert (e == ERROR_CODES["fn"]).sum() == mb["fn"]
    assert (e == ERROR_CODES["fp"]).sum() == mb["fp"]


def test_volumes_share_the_image_grid(demo):
    img = _nifti(demo.image_nifti())
    for path in ("vol/labels/ref.nii.gz", "vol/labels/Model A.nii.gz", "vol/error/Model C/liver.nii.gz"):
        v = _nifti(demo.resource(path)[0])
        assert v.shape == img.shape
        np.testing.assert_allclose(v.affine, img.affine)
    lab = np.asarray(_nifti(demo.labels_nifti("ref")).dataobj)
    assert set(np.unique(lab)) <= {0, *demo.index.values()}


def test_meshes(demo):
    magic, attr, nf, nv, n = _mz3_counts(demo.resource("mesh/ref/liver.mz3")[0])
    assert magic == 23117 and attr == 3 and nf > 100 and nv > 50 and n == 16 + 12 * nf + 12 * nv
    # missed lesion: the FN surface exists, the TP surface is empty
    assert _mz3_counts(demo.mesh("Model C", "pancreatic_lesion", "fn"))[2] > 0
    assert _mz3_counts(demo.mesh("Model C", "pancreatic_lesion", "tp"))[2] == 0


def test_resource_rejects_unknown(demo):
    for bad in ("vol/labels/nobody.nii.gz", "vol/error/Model A/spleenx.nii.gz", "mesh/ref/x.mz3",
                "mesh/Model A/zz/liver.mz3", "etc/passwd"):
        with pytest.raises(KeyError):
            demo.resource(bad)


def test_live_metrics(demo):
    r = demo.live_metrics("Model C", "pancreatic_lesion")
    assert r["dice"] == 0.0
    r = demo.live_metrics("Model A", "liver")
    assert 0.9 < r["dice"] <= 1 and r["hd95"] >= 0


def test_server_endpoints(demo):
    srv = make_server(demo, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(p):
        with urllib.request.urlopen(base + p, timeout=60) as r:
            return r.status, r.headers.get("Content-Type"), r.read()

    try:
        st, ct, body = get("/")
        assert st == 200 and "html" in ct and b"SEK_MODE" in body and b"static/niivue.umd.js" in body
        assert get("/static/app.js")[0] == 200
        assert len(get("/static/niivue.umd.js")[2]) > 1_000_000
        man = json.loads(get("/api/manifest")[2])
        assert man["case"] == "phantom" and len(man["structures"]) == 8
        met = json.loads(get("/api/metrics?model=Model%20A&structure=liver")[2])
        assert 0.9 < met["dice"] <= 1
        assert _nifti(get("/vol/error/Model%20B/pancreas.nii.gz")[2]).shape == tuple(demo.shape)
        assert _mz3_counts(get("/mesh/Model%20A/tp/pancreatic_lesion.mz3")[2])[2] > 0
        for bad in ("/api/metrics?model=X&structure=liver", "/vol/nope.nii.gz", "/static/../session.py",
                    "/static/%2e%2e/session.py"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(bad)
            assert e.value.code == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_export_is_self_contained(demo, tmp_path):
    out = export_html(demo, tmp_path / "demo.html")
    html = out.read_text()
    assert out.stat().st_size < 12e6
    assert "window.SEK_FILES" in html and "window.SEK_MANIFEST" in html and 'SEK_MODE = "static"' in html
    assert 'src="static/' not in html and 'href="static/' not in html  # assets are inlined
    files = json.loads(html[html.index("window.SEK_FILES = ") + 19: html.index(";</script>", html.index("window.SEK_FILES"))])
    assert "vol/image.nii.gz" in files and "vol/error/Model B/pancreatic_lesion.nii.gz" in files
    assert "mesh/Model B/fp/pancreatic_lesion.mz3" in files  # error surfaces of the lesion structure


def test_per_structure_folder_union_and_results(tmp_path):
    sh, aff = (20, 20, 10), np.diag([2.0, 2.0, 3.0, 1.0])
    ct = np.zeros(sh, np.int16)
    a = np.zeros(sh, np.uint8)
    a[3:8, 3:8, 2:6] = 1
    b = np.zeros(sh, np.uint8)
    b[8:12, 3:8, 2:6] = 1
    ref = tmp_path / "case1" / "segmentations"
    ref.mkdir(parents=True)
    nib.save(nib.Nifti1Image(ct, aff), str(tmp_path / "ct.nii.gz"))
    nib.save(nib.Nifti1Image(a, aff), str(ref / "a.nii.gz"))
    nib.save(nib.Nifti1Image(b, aff), str(ref / "b.nii.gz"))
    pred = a * 1 + b * 2
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), aff), str(tmp_path / "pred.nii.gz"))
    res = tmp_path / "eval"
    res.mkdir()
    (res / "per_case_wide.csv").write_text("case_id,label,dice,hd95,tp\ncase1,ab,0.5,3.0,10\nother,ab,0.1,9,1\n")
    s = ViewerSession(tmp_path / "ct.nii.gz", ref=ref, preds={"m": tmp_path / "pred.nii.gz"},
                      labels={"ab": {"ref_file": "a.nii.gz+b.nii.gz", "pred": [1, 2]}},
                      results={"m": res}, case="case1")
    assert s.structures == ["ab"]
    assert s.ref_masks["ab"].sum() == a.sum() + b.sum()
    assert s.manifest()["structures"][0]["models"]["m"]["dice"] == 1.0
    assert s.results == {"m": {"ab": {"dice": 0.5, "hd95": 3.0}}}  # raw counts are not shown
    # prediction on another grid is resampled onto the image grid
    small = nib.Nifti1Image(pred[::2, ::2].astype(np.uint8), np.diag([4.0, 4.0, 3.0, 1.0]))
    nib.save(small, str(tmp_path / "pred_small.nii.gz"))
    s2 = ViewerSession(tmp_path / "ct.nii.gz", ref=ref, preds={"m": tmp_path / "pred_small.nii.gz"},
                       labels={"ab": {"ref_file": "a.nii.gz+b.nii.gz", "pred": [1, 2]}}, case="case1")
    assert s2.pred_masks["m"]["ab"].shape == sh


def test_crop_and_stride(cache):
    s = demo_session(cache, crop_margin_mm=5.0, max_dim=64)
    assert max(s.shape) <= 64 and s.step >= 2
    img = _nifti(s.image_nifti())
    assert img.shape == s.shape
    # the region affine maps region voxel 0 to the crop origin
    lo = np.array([r.start for r in s.region])
    np.testing.assert_allclose(img.affine[:3, 3], (s._img.affine @ np.r_[lo, 1])[:3])


def test_cli_view_demo_export(tmp_path, monkeypatch):
    from segevalkit.cli import main

    monkeypatch.setenv("SEGEVALKIT_CACHE", str(tmp_path / "cache"))
    out = tmp_path / "v.html"
    assert main(["view", "--demo", "--export", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 2_000_000


def test_cli_view_needs_an_image():
    from segevalkit.cli import main

    assert main(["view", "--no-browser"]) == 2
