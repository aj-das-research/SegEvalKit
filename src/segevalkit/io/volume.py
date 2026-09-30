"""Reading and writing volumes with their physical geometry.

A segmentation metric is only meaningful on the voxel grid it was computed on,
so SegEvalKit never separates an array from its geometry: every loader returns
a `Volume` (array + spacing + affine), and `check_alignment`
refuses to compare two volumes that do not live on the same grid unless you
explicitly ask for resampling.

Supported formats: NIfTI (``.nii``, ``.nii.gz``) via nibabel; anything
SimpleITK reads (``.mha``, ``.mhd``, ``.nrrd``, DICOM series...) when it is
installed; and NumPy ``.npy`` / ``.npz`` (spacing from an ``spacing`` key).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["Volume", "load_volume", "save_volume", "check_alignment", "resample_to", "NIFTI_SUFFIXES"]

PathLike = Union[str, os.PathLike]
NIFTI_SUFFIXES = (".nii.gz", ".nii")
_ITK_SUFFIXES = (".mha", ".mhd", ".nrrd", ".nhdr")


@dataclass
class Volume:
    """A 3D array with its voxel geometry.

    Attributes:
        data: Array indexed ``[i, j, k]`` in the file's voxel order.
        spacing: Physical voxel size along ``i, j, k`` in millimetres.
        affine: 4x4 voxel-to-world (RAS+, mm) matrix.
        path: Source file, if any.
    """

    data: np.ndarray
    spacing: Tuple[float, float, float]
    affine: np.ndarray = field(default_factory=lambda: np.eye(4))
    path: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    def __repr__(self) -> str:
        sp = ", ".join(f"{s:.3g}" for s in self.spacing)
        return f"Volume(shape={self.shape}, spacing=({sp}) mm, dtype={self.data.dtype}, path={self.path!r})"


def _strip(path: str) -> str:
    low = path.lower()
    for suf in NIFTI_SUFFIXES + _ITK_SUFFIXES + (".npz", ".npy"):
        if low.endswith(suf):
            return suf
    return Path(path).suffix.lower()


def _as_label_dtype(a: np.ndarray) -> np.ndarray:
    """Cast float-stored label maps (common in NIfTI) to the smallest exact integer type."""
    if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
        return a
    r = np.rint(a)
    if not np.allclose(r, a, atol=1e-3):
        return a.astype(np.float32, copy=False)  # genuinely continuous: a probability map
    lo, hi = float(r.min(initial=0)), float(r.max(initial=0))
    for t in (np.uint8, np.int16, np.int32):
        info = np.iinfo(t)
        if info.min <= lo and hi <= info.max:
            return r.astype(t)
    return r.astype(np.int64)


def load_volume(path: PathLike, *, kind: str = "label", spacing: Optional[Sequence[float]] = None) -> Volume:
    """Load a volume from disk.

    Args:
        path: File path.
        kind: ``"label"`` casts integer-valued float data to an integer type;
            ``"image"`` / ``"prob"`` keep ``float32``.
        spacing: Override spacing (needed for ``.npy``).
    """
    path = str(path)
    suf = _strip(path)
    if suf in NIFTI_SUFFIXES:
        import nibabel as nib

        img = nib.load(path)
        data = np.asanyarray(img.dataobj)
        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]
        sp = tuple(float(z) for z in img.header.get_zooms()[:3])
        affine = np.asarray(img.affine, dtype=np.float64)
    elif suf in _ITK_SUFFIXES:
        try:
            import SimpleITK as sitk
        except ImportError as exc:  # pragma: no cover
            raise ImportError(f"reading {suf} files needs SimpleITK: pip install segevalkit[sitk]") from exc
        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img).transpose(2, 1, 0)  # (z,y,x) -> (x,y,z)
        sp = tuple(float(s) for s in img.GetSpacing())
        d = np.asarray(img.GetDirection()).reshape(3, 3)
        affine = np.eye(4)
        affine[:3, :3] = d * np.asarray(sp)
        affine[:3, 3] = img.GetOrigin()
        affine = np.diag([-1, -1, 1, 1]) @ affine  # LPS -> RAS
    elif suf in (".npy", ".npz"):
        if suf == ".npy":
            data = np.load(path)
            sp = tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0)
        else:
            with np.load(path) as z:
                key = "data" if "data" in z else ("probabilities" if "probabilities" in z else z.files[0])
                data = z[key]
                sp = tuple(z["spacing"].tolist()) if "spacing" in z else tuple(spacing or (1.0, 1.0, 1.0))
        affine = np.diag(list(sp) + [1.0])
    else:
        raise ValueError(f"unsupported file type: {path}")
    if spacing is not None:
        sp = tuple(float(s) for s in spacing)
    data = _as_label_dtype(data) if kind == "label" else np.asarray(data, dtype=np.float32)
    return Volume(data=data, spacing=tuple(float(s) for s in sp), affine=affine, path=path)


def save_volume(data: Union[np.ndarray, Volume], path: PathLike, affine: Optional[np.ndarray] = None) -> None:
    """Write an array (or `Volume`) to NIfTI."""
    import nibabel as nib

    if isinstance(data, Volume):
        affine = data.affine if affine is None else affine
        data = data.data
    arr = np.asarray(data)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    nib.save(nib.Nifti1Image(arr, np.eye(4) if affine is None else affine), str(path))


def check_alignment(a: Volume, b: Volume, *, atol_mm: float = 1e-2) -> List[str]:
    """Return human-readable problems if ``a`` and ``b`` are not on the same voxel grid."""
    issues = []
    if a.shape != b.shape:
        issues.append(f"shape {a.shape} != {b.shape}")
    if not np.allclose(a.spacing, b.spacing, atol=1e-4, rtol=1e-3):
        issues.append(f"spacing {a.spacing} != {b.spacing}")
    if a.affine is not None and b.affine is not None and not np.allclose(a.affine, b.affine, atol=atol_mm):
        issues.append("affine differs (orientation or origin)")
    return issues


def resample_to(moving: Volume, target: Volume, *, order: int = 0) -> Volume:
    """Resample ``moving`` onto ``target``'s grid (nearest-neighbour for labels)."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    src = nib.Nifti1Image(np.asarray(moving.data), moving.affine)
    out = resample_from_to(src, (target.shape, target.affine), order=order)
    data = np.asanyarray(out.dataobj)
    if order == 0:
        data = data.astype(moving.data.dtype, copy=False)
    return Volume(data=data, spacing=target.spacing, affine=target.affine, path=moving.path)
