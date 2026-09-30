"""Compute backends: NumPy/SciPy on the CPU, PyTorch on the GPU.

The two backends compute *the same quantities*: the GPU path is an
acceleration, never a different definition. ``tests/test_backends.py`` checks
that they agree to floating-point precision.

* Surface distances on the CPU use an exact Euclidean distance transform
  (:func:`scipy.ndimage.distance_transform_edt`) of the other mask's surface,
  sampled at this mask's surface voxels.
* On the GPU the surface voxels become point clouds in millimetres and the
  nearest-neighbour distance is taken with a chunked :func:`torch.cdist`,
  which is exact and scales to organ-sized surfaces (10^5 points) in well
  under a second on an A100.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage

__all__ = [
    "confusion",
    "directed_surface_distances",
    "label_components",
    "skeleton",
    "torch_available",
    "cuda_available",
]


@lru_cache(maxsize=1)
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def cuda_available() -> bool:
    if not torch_available():
        return False
    import torch

    return torch.cuda.is_available()


def _is_gpu(device: str) -> bool:
    return device is not None and str(device) != "cpu"


def _torch_device(device: str):
    if not torch_available():
        raise RuntimeError(
            f"device={device!r} requested but PyTorch is not installed; "
            "install with `pip install segevalkit[gpu]` or use device='cpu'"
        )
    import torch

    # "torch" runs the PyTorch code path on the CPU (used by the test-suite to
    # check backend agreement on machines without a GPU).
    dev = torch.device("cpu" if device == "torch" else device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requested but CUDA is not available")
    return dev


# ---------------------------------------------------------------- confusion
def confusion(pred: np.ndarray, ref: np.ndarray, device: str = "cpu") -> Tuple[int, int, int, int]:
    """Return ``(TP, FP, FN, TN)`` for two boolean arrays."""
    if _is_gpu(device):
        import torch

        dev = _torch_device(device)
        p = torch.from_numpy(np.ascontiguousarray(pred)).to(dev, non_blocking=True)
        r = torch.from_numpy(np.ascontiguousarray(ref)).to(dev, non_blocking=True)
        tp = int(torch.count_nonzero(p & r))
        n_pred = int(torch.count_nonzero(p))
        n_ref = int(torch.count_nonzero(r))
    else:
        tp = int(np.count_nonzero(pred & ref))
        n_pred = int(np.count_nonzero(pred))
        n_ref = int(np.count_nonzero(ref))
    fp = n_pred - tp
    fn = n_ref - tp
    tn = int(pred.size) - tp - fp - fn
    return tp, fp, fn, tn


# ---------------------------------------------------------- surface distance
def _edt_to(surface_mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Distance (mm) from every voxel to the nearest voxel of ``surface_mask``."""
    return ndimage.distance_transform_edt(~surface_mask, sampling=spacing)


def directed_surface_distances(
    pred_surface: np.ndarray,
    ref_surface: np.ndarray,
    spacing: Sequence[float],
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Directed distances ``(pred→ref, ref→pred)`` between two surface masks, in mm."""
    if _is_gpu(device):
        return _surface_distances_torch(pred_surface, ref_surface, spacing, device)
    d_to_ref = _edt_to(ref_surface, spacing)
    d_to_pred = _edt_to(pred_surface, spacing)
    return d_to_ref[pred_surface].astype(np.float64), d_to_pred[ref_surface].astype(np.float64)


def _surface_distances_torch(pred_surface, ref_surface, spacing, device):
    import torch

    dev = _torch_device(device)
    sp = torch.tensor(spacing, dtype=torch.float32, device=dev)
    a = torch.from_numpy(np.argwhere(pred_surface)).to(dev).float() * sp
    b = torch.from_numpy(np.argwhere(ref_surface)).to(dev).float() * sp
    return _nn_dist(a, b).cpu().numpy(), _nn_dist(b, a).cpu().numpy()


def _nn_dist(a, b, budget: int = 1 << 27):
    """Distance from every row of ``a`` to its nearest row of ``b`` (chunked cdist)."""
    import torch

    chunk = max(1, budget // max(1, b.shape[0]))
    out = torch.empty(a.shape[0], dtype=torch.float64, device=a.device)
    for i in range(0, a.shape[0], chunk):
        # float32 cdist then a float64 sqrt of the squared minimum keeps the
        # result within ~1e-6 mm of the float64 EDT.
        d2 = torch.cdist(a[i:i + chunk], b, compute_mode="donot_use_mm_for_euclid_dist").pow_(2)
        out[i:i + chunk] = d2.min(dim=1).values.double().sqrt()
    return out


# --------------------------------------------------------------- components
def label_components(mask: np.ndarray, structure: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected-component labelling; uses ``cc3d`` when installed (5-10x faster)."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0
    try:
        import cc3d

        conn = {7: 6, 19: 18, 27: 26}[int(structure.sum())]
        lab, n = cc3d.connected_components(mask, connectivity=conn, return_N=True)
        return lab.astype(np.int32, copy=False), int(n)
    except ImportError:
        lab, n = ndimage.label(mask, structure=structure)
        return lab.astype(np.int32, copy=False), int(n)


# ----------------------------------------------------------------- skeleton
def skeleton(mask: np.ndarray) -> np.ndarray:
    """Topology-preserving 3D skeleton (Lee et al. 1994 thinning, scikit-image)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    from skimage.morphology import skeletonize

    skel = skeletonize(mask).astype(bool)
    # Lee thinning can delete *every* voxel of a component whose cross-section
    # is perfectly symmetric with even width (e.g. a 4x4 bar). A non-empty
    # object must keep a non-empty skeleton, so any component that vanished
    # keeps its most interior voxel (the maximum of its distance transform).
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
    kept = np.unique(lab[skel])
    missing = np.setdiff1d(np.arange(1, n + 1), kept)
    if missing.size:
        edt = ndimage.distance_transform_edt(mask)
        for comp in missing:
            flat = np.flatnonzero(lab == comp)
            skel.flat[flat[np.argmax(edt.flat[flat])]] = True
    return skel
