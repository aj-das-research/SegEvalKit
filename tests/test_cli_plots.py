import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from segevalkit.cli import main  # noqa: E402
from segevalkit.io import save_volume  # noqa: E402
from conftest import cube  # noqa: E402

AFF = np.diag([1.0, 1.0, 2.0, 1.0])


@pytest.fixture
def folders(tmp_path):
    gt, pr, im = tmp_path / "gt", tmp_path / "pr", tmp_path / "im"
    for d in (gt, pr, im):
        d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(6):
        g = np.zeros((32, 32, 16), np.uint8)
        g[cube((32, 32, 16), (6, 6, 3), (12 + i, 12, 8))] = 1
        g[cube((32, 32, 16), (24, 24, 10), (3, 3, 3))] = 2
        p = np.roll(g, i % 3, 0)
        save_volume(g, gt / f"c{i}.nii.gz", AFF)
        save_volume(p, pr / f"c{i}.nii.gz", AFF)
        img = (rng.normal(40, 30, g.shape) + 100 * (g > 0)).astype(np.float32)
        save_volume(img, im / f"c{i}_0000.nii.gz", AFF)
    return tmp_path


def test_cli_evaluate_report_compare(folders, capsys):
    out = folders / "out"
    assert main(["evaluate", "--pred", str(folders / "pr"), "--ref", str(folders / "gt"),
                 "--labels", "organ=1,lesion=2", "--metrics", "default,detection,topology",
                 "--out", str(out), "--report", "--images", str(folders / "im")]) == 0
    assert (out / "report.html").stat().st_size > 10_000
    assert main(["compare", str(out), str(out), "--out", str(folders / "cmp")]) == 0
    assert main(["metrics", "-v"]) == 0
    assert main(["recommend", "--structure", "tubular", "--markdown"]) == 0
    assert "Centreline Dice" in capsys.readouterr().out


def test_cli_visualize(folders):
    for kind in ("slice", "triplanar", "montage", "projection", "surface"):
        out = folders / f"{kind}.png"
        assert main(["visualize", "--pred", str(folders / "pr/c1.nii.gz"), "--ref", str(folders / "gt/c1.nii.gz"),
                     "--image", str(folders / "im/c1_0000.nii.gz"), "--label", "1", "--kind", kind,
                     "--out", str(out)]) == 0
        assert out.stat().st_size > 5000


def test_all_plots_render(folders):
    import segevalkit as sek
    from segevalkit import plotting as P
    from segevalkit.stats import compare, ranking_stability

    ev = sek.Evaluator(labels={"organ": 1, "lesion": 2}, metrics=["default", "detection"])
    a = ev.evaluate(folders / "pr", folders / "gt", progress=False, name="A")
    b = ev.evaluate(folders / "gt", folders / "gt", progress=False, name="B")
    figs = [
        P.metric_distribution({"A": a, "B": b}, "dice"),
        P.metric_distribution(a, "hd95", kind="box"),
        P.metric_heatmap(a, "dice"),
        P.metric_vs_size(a, "dice", label="organ"),
        P.metric_correlation(a),
        P.volume_agreement(a, "organ"),
        P.bland_altman_plot(a, "organ"),
        P.ecdf({"A": a, "B": b}, "dice", label="organ"),
        P.metric_profile({"A": a, "B": b}, ["dice", "nsd", "hd95"], label="organ"),
        P.comparison_forest(compare(a, b)),
        P.ranking_stability_plot(ranking_stability({"A": a, "B": b}, "dice", "organ", n_boot=20)),
        P.detection_by_size({"A": a, "B": b}),
        P.failure_quadrants(a, "organ"),
        P.reliability_diagram(np.random.rand(1000), np.random.rand(1000) > 0.5),
    ]
    assert all(f is not None for f in figs)
