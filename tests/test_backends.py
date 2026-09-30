"""The PyTorch backend must compute exactly what the NumPy/SciPy backend computes."""

import pytest

from segevalkit.metrics import compute_metrics
import numpy as np

from segevalkit.metrics._backend import _nn_dist, cuda_available, mps_available, torch_available

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


@pytest.mark.gpu
@pytest.mark.skipif(not mps_available(), reason="no Apple MPS device")
def test_mps_matches_numpy(pair):
    p, g = pair
    a = compute_metrics(p, g, METRICS, spacing=(0.8, 0.8, 2.0), device="cpu")
    b = compute_metrics(p, g, METRICS, spacing=(0.8, 0.8, 2.0), device="mps")
    for k in METRICS:
        assert b[k] == pytest.approx(a[k], rel=1e-5, abs=1e-4), k


def test_nn_dist_returns_cpu_float64():
    import torch

    a = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 1.0]])
    d = _nn_dist(a, b)
    assert d.dtype == torch.float64 and d.device.type == "cpu"
    assert np.allclose(d.numpy(), [1.0, np.sqrt(26.0)])


def test_unavailable_mps_raises():
    if mps_available():
        pytest.skip("MPS is available here")
    from segevalkit.metrics._backend import confusion

    with pytest.raises(RuntimeError, match="MPS"):
        confusion(np.ones((4, 4, 4), bool), np.ones((4, 4, 4), bool), device="mps")
