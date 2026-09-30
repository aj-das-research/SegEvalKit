"""The per-pair computation context.

Most segmentation metrics share expensive intermediates: the confusion counts,
the two boundary surfaces, the two sets of directed surface distances, the
connected components, the skeletons. :class:`PairContext` computes each of
these lazily, once, and caches it, so asking for twenty metrics costs little
more than asking for the most expensive one.

It also fixes the conventions that differ silently between existing tools:

* **Surface voxels** are foreground voxels with at least one 6-connected
  background neighbour (the image border counts as background).
* **Directed surface distances** are exact Euclidean distances, in millimetres,
  from every surface voxel of one mask to the nearest surface voxel of the
  other, computed with the physical voxel spacing.
* **Empty masks** follow an explicit, configurable :class:`EmptyPolicy`
  instead of whatever falls out of a division by zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage

from . import _backend

__all__ = ["EmptyPolicy", "PairContext"]

Number = Union[int, float]


@dataclass(frozen=True)
class EmptyPolicy:
    """How metrics behave when the prediction and/or reference is empty.

    Attributes:
        both_empty: ``"best"`` returns each metric's ideal value when *both*
            masks are empty (the structure is correctly absent, e.g. Dice = 1,
            HD = 0), which is the BraTS / KiTS convention. ``"nan"`` returns NaN
            so the case is excluded from aggregates (the nnU-Net convention).
        one_empty_distance: Value of distance metrics when exactly one mask is
            empty: ``"worst"`` uses the diagonal of the image in millimetres
            (a finite, size-aware penalty in the spirit of BraTS' 373.13 mm),
            ``"nan"`` excludes the case, and a number is used verbatim.
    """

    both_empty: str = "best"
    one_empty_distance: Union[str, float] = "worst"

    @classmethod
    def preset(cls, name: str) -> "EmptyPolicy":
        """Named conventions used by well-known evaluation code.

        ``"segevalkit"`` (default): best / image-diagonal penalty.
        ``"brats2023"``: best / 374 mm (BraTS 2023 lesion-wise code).
        ``"metrics_reloaded"``: best / worst (MetricsReloaded aggregation advice).
        ``"nan"`` (nnU-Net / MONAI ``ignore_empty``): exclude undefined cases.
        ``"topcow"``: best / 90 mm (TopCoW 2024 HD95 cap).
        """
        presets = {
            "segevalkit": cls("best", "worst"),
            "brats2023": cls("best", 374.0),
            "metrics_reloaded": cls("best", "worst"),
            "nan": cls("nan", "nan"),
            "nnunet": cls("nan", "nan"),
            "topcow": cls("best", 90.0),
        }
        try:
            return presets[name.lower()]
        except KeyError:
            raise ValueError(f"unknown empty policy preset {name!r}; choose from {sorted(presets)}") from None

    def __post_init__(self):
        if self.both_empty not in ("best", "nan"):
            raise ValueError("both_empty must be 'best' or 'nan'")
        if isinstance(self.one_empty_distance, str) and self.one_empty_distance not in ("worst", "nan"):
            raise ValueError("one_empty_distance must be 'worst', 'nan' or a number")


# 6-connected (face) and 26-connected (full) 3x3x3 structuring elements.
FACE = ndimage.generate_binary_structure(3, 1)
FULL = ndimage.generate_binary_structure(3, 3)


def _as_bool(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != bool:
        a = a != 0
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D, got shape {a.shape}")
    return a


def bbox(mask: np.ndarray, margin: int = 0) -> Tuple[slice, ...]:
    """Bounding box of ``mask`` as a tuple of slices, grown by ``margin`` voxels."""
    idx = []
    for ax in range(mask.ndim):
        any_ax = np.any(mask, axis=tuple(i for i in range(mask.ndim) if i != ax))
        nz = np.flatnonzero(any_ax)
        if nz.size == 0:
            return tuple(slice(0, 0) for _ in range(mask.ndim))
        lo = max(int(nz[0]) - margin, 0)
        hi = min(int(nz[-1]) + 1 + margin, mask.shape[ax])
        idx.append(slice(lo, hi))
    return tuple(idx)


def surface(mask: np.ndarray) -> np.ndarray:
    """Boolean surface of a binary mask (foreground voxels with a background face-neighbour)."""
    if not mask.any():
        return np.zeros_like(mask)
    eroded = ndimage.binary_erosion(mask, structure=FACE, border_value=0)
    return mask & ~eroded


class PairContext:
    """Lazily computed, cached intermediates for one (prediction, reference) pair.

    Args:
        pred: Predicted binary mask (any non-zero value is foreground).
        ref: Reference ("ground truth") binary mask of the same shape.
        spacing: Voxel spacing in millimetres, one value per axis. ``None``
            means isotropic 1 mm; distance metrics are then in voxel units.
        prob: Optional foreground probability map in ``[0, 1]`` (same shape),
            required by calibration metrics.
        device: ``"cpu"`` or a torch device string such as ``"cuda"`` /
            ``"cuda:1"``. On a GPU device, confusion counts and surface
            distances run in PyTorch; everything else stays on the CPU.
        empty: The :class:`EmptyPolicy`.
        connectivity: Connectivity used to define connected components
            (lesions / instances): 6, 18 or 26.
        min_component_voxels: Components smaller than this are ignored for
            detection metrics (noise suppression; 0 disables).
    """

    def __init__(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        spacing: Optional[Sequence[Number]] = None,
        prob: Optional[np.ndarray] = None,
        *,
        device: str = "cpu",
        empty: EmptyPolicy = EmptyPolicy(),
        connectivity: int = 26,
        min_component_voxels: int = 0,
    ) -> None:
        self.pred = _as_bool(pred, "pred")
        self.ref = _as_bool(ref, "ref")
        if self.pred.shape != self.ref.shape:
            raise ValueError(f"shape mismatch: pred {self.pred.shape} vs ref {self.ref.shape}")
        if spacing is None:
            spacing = (1.0,) * 3
        spacing = tuple(float(s) for s in spacing)
        if len(spacing) == 2:
            spacing = spacing + (1.0,)
        if len(spacing) != 3 or min(spacing) <= 0:
            raise ValueError(f"invalid spacing {spacing}")
        self.spacing: Tuple[float, float, float] = spacing
        if prob is not None:
            prob = np.asarray(prob, dtype=np.float32)
            if prob.ndim == 2:
                prob = prob[..., None]
            if prob.shape != self.ref.shape:
                raise ValueError(f"shape mismatch: prob {prob.shape} vs ref {self.ref.shape}")
        self.prob = prob
        self.device = device
        self.empty = empty
        if connectivity not in (6, 18, 26):
            raise ValueError("connectivity must be 6, 18 or 26")
        self.connectivity = connectivity
        self.min_component_voxels = int(min_component_voxels)
        self._memo: dict = {}

    def memo(self, key, compute):
        """Cache an arbitrary intermediate under ``key`` (for metric plug-ins)."""
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    # ------------------------------------------------------------------ basics
    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing
        return sx * sy * sz

    @property
    def n_voxels(self) -> int:
        return int(self.ref.size)

    @cached_property
    def counts(self) -> Tuple[int, int, int, int]:
        """``(TP, FP, FN, TN)`` voxel counts."""
        return _backend.confusion(self.pred, self.ref, self.device)

    @property
    def tp(self) -> int:
        return self.counts[0]

    @property
    def fp(self) -> int:
        return self.counts[1]

    @property
    def fn(self) -> int:
        return self.counts[2]

    @property
    def tn(self) -> int:
        return self.counts[3]

    @property
    def pred_empty(self) -> bool:
        return self.tp + self.fp == 0

    @property
    def ref_empty(self) -> bool:
        return self.tp + self.fn == 0

    @property
    def both_empty(self) -> bool:
        return self.pred_empty and self.ref_empty

    @property
    def one_empty(self) -> bool:
        return self.pred_empty != self.ref_empty

    def best_or_nan(self, best: float) -> float:
        """Value to return when both masks are empty, per the :class:`EmptyPolicy`."""
        return float(best) if self.empty.both_empty == "best" else float("nan")

    @cached_property
    def diagonal_mm(self) -> float:
        """Length of the image diagonal in mm (upper bound on any distance)."""
        return float(np.sqrt(sum((n * s) ** 2 for n, s in zip(self.ref.shape, self.spacing))))

    def distance_penalty(self) -> float:
        """Distance value for the one-mask-empty case, per the :class:`EmptyPolicy`."""
        p = self.empty.one_empty_distance
        if p == "worst":
            return self.diagonal_mm
        if p == "nan":
            return float("nan")
        return float(p)

    # --------------------------------------------------------------- surfaces
    @cached_property
    def _crop(self) -> Tuple[slice, ...]:
        """Union bounding box grown by one voxel: all surface computations live here."""
        return bbox(self.pred | self.ref, margin=1)

    def _cropped(self, a: np.ndarray) -> np.ndarray:
        c = a[self._crop]
        # Pad by one voxel of background so voxels on the true image border are
        # still classified as surface after cropping.
        return np.pad(c, 1, mode="constant", constant_values=False)

    @cached_property
    def pred_surface(self) -> np.ndarray:
        """Surface of the prediction, in the padded crop frame."""
        return surface(self._cropped(self.pred))

    @cached_property
    def ref_surface(self) -> np.ndarray:
        """Surface of the reference, in the padded crop frame."""
        return surface(self._cropped(self.ref))

    @cached_property
    def surface_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        """Directed surface distances ``(pred→ref, ref→pred)`` in mm.

        Each entry is a 1D array with one distance per surface voxel. Both are
        empty arrays if either mask is empty.
        """
        if self.pred_empty or self.ref_empty:
            return np.empty(0), np.empty(0)
        return _backend.directed_surface_distances(
            self.pred_surface, self.ref_surface, self.spacing, self.device
        )

    # ------------------------------------------------------------- components
    @cached_property
    def _structure(self) -> np.ndarray:
        return ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[self.connectivity])

    def _label(self, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        lab, n = _backend.label_components(mask, self._structure)
        if self.min_component_voxels > 0 and n > 0:
            sizes = np.bincount(lab.ravel())
            keep = sizes >= self.min_component_voxels
            keep[0] = False
            remap = np.zeros(n + 1, dtype=lab.dtype)
            remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=lab.dtype)
            lab, n = remap[lab], int(keep.sum())
        return lab, n

    @cached_property
    def pred_components(self) -> Tuple[np.ndarray, int]:
        """Connected components of the prediction: ``(label image, count)``."""
        return self._label(self.pred)

    @cached_property
    def ref_components(self) -> Tuple[np.ndarray, int]:
        """Connected components of the reference: ``(label image, count)``."""
        return self._label(self.ref)

    # -------------------------------------------------------------- skeletons
    @cached_property
    def pred_skeleton(self) -> np.ndarray:
        """Topological skeleton of the prediction, in the padded crop frame."""
        return _backend.skeleton(self._cropped(self.pred))

    @cached_property
    def ref_skeleton(self) -> np.ndarray:
        """Topological skeleton of the reference, in the padded crop frame."""
        return _backend.skeleton(self._cropped(self.ref))

    @cached_property
    def pred_cropped(self) -> np.ndarray:
        return self._cropped(self.pred)

    @cached_property
    def ref_cropped(self) -> np.ndarray:
        return self._cropped(self.ref)

    def __repr__(self) -> str:
        return (f"PairContext(shape={self.ref.shape}, spacing={self.spacing}, "
                f"pred_voxels={self.tp + self.fp}, ref_voxels={self.tp + self.fn}, "
                f"device={self.device!r})")
