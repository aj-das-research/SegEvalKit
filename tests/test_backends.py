"""The PyTorch backend must compute exactly what the NumPy/SciPy backend computes."""

import pytest

from segevalkit.metrics import compute_metrics
from segevalkit.metrics._backend import cuda_available, torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")

METRICS = ["dice", "iou", "hd", "hd95", "assd", "masd", "nsd", "precision", "recall"]


@pytest.mark.parametrize("spacing", [(1, 1, 1), (0.78, 0.78, 2.5)])
def test_torch_cpu_matches_numpy(pair, spacing):
    p, g = pair
    a = compute_metrics(p, g, METRICS, spacing=spacing, device="cpu")
    b = compute_metrics(p, g, METRICS, spacing=spacing, device="torch")
    for k in METRICS:
        assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-5), k


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
def test_cuda_matches_numpy(pair):
    p, g = pair
    a = compute_metrics(p, g, METRICS, spacing=(0.8, 0.8, 2.0), device="cpu")
    b = compute_metrics(p, g, METRICS, spacing=(0.8, 0.8, 2.0), device="cuda")
    for k in METRICS:
        assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-5), k
