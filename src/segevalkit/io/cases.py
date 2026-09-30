"""Discovering and pairing cases on disk.

SegEvalKit understands the three layouts that cover virtually every public
volumetric dataset and every common model output:

``flat``
    One multi-label file per case directly in the root:
    ``root/<case>.nii.gz`` (nnU-Net, MSD, AMOS, BTCV, KiTS, FLARE outputs).
``folder``
    One multi-label file inside a folder per case:
    ``root/<case>/<file>`` (e.g. ``combined_labels.nii.gz``, ``label.nii.gz``).
``per_structure``
    One binary file per structure inside a folder per case, optionally in a
    sub-folder: ``root/<case>/[segmentations/]<structure>.nii.gz``
    (TotalSegmentator, AbdomenAtlas, PanTS).

``layout="auto"`` inspects the root and picks one; pass it explicitly when a
dataset is ambiguous. Case ids are the file stem (``flat``) or the folder name.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .labels import LabelSpec, extract_mask
from .volume import NIFTI_SUFFIXES, Volume, load_volume

__all__ = ["Source", "Case", "discover_cases"]

_SUFFIXES = NIFTI_SUFFIXES + (".mha", ".nrrd", ".npz", ".npy")
_KNOWN_FILES = ("combined_labels.nii.gz", "label.nii.gz", "labels.nii.gz", "seg.nii.gz",
                "segmentation.nii.gz", "mask.nii.gz", "prediction.nii.gz", "pred.nii.gz")
_PER_STRUCTURE_SUBDIRS = ("segmentations", "predictions", "masks", "labels")


def _stem(name: str) -> Optional[str]:
    low = name.lower()
    for suf in _SUFFIXES:
        if low.endswith(suf):
            return name[: -len(suf)]
    return None


def _is_vol(name: str) -> bool:
    return _stem(name) is not None


@dataclass
class Source:
    """A set of volumes on disk in one of the supported layouts.

    Args:
        root: Directory.
        layout: ``"flat"``, ``"folder"``, ``"per_structure"`` or ``"auto"``.
        file: For ``folder``: file name inside each case folder.
        subdir: For ``per_structure``: sub-folder holding the structure files
            (``""`` for none).
        strip: Regex removed from case ids (default: nnU-Net's ``_0000``
            channel suffix), so images and labels pair up.
        kind: ``"label"``, ``"image"`` or ``"prob"``.
    """

    root: Union[str, Path]
    layout: str = "auto"
    file: Optional[str] = None
    subdir: Optional[str] = None
    strip: str = r"_0000$"
    kind: str = "label"

    def __post_init__(self):
        self.root = Path(self.root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"not a directory: {self.root}")
        if self.layout == "auto":
            self.layout, self.file, self.subdir = self._detect()
        if self.layout not in ("flat", "folder", "per_structure"):
            raise ValueError(f"unknown layout {self.layout!r}")
        self._index: Dict[str, Path] = self._build_index()
        self._cache_key: Optional[str] = None
        self._cache: Optional[Volume] = None

    # ------------------------------------------------------------ discovery
    def _detect(self) -> Tuple[str, Optional[str], Optional[str]]:
        with os.scandir(self.root) as it:
            entries = sorted(it, key=lambda e: e.name)
        if any(e.is_file() and _is_vol(e.name) for e in entries):
            return "flat", None, None
        dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
        if not dirs:
            raise FileNotFoundError(f"no volumes or case folders found in {self.root}")
        probe = Path(dirs[0].path)
        for sub in _PER_STRUCTURE_SUBDIRS:
            if (probe / sub).is_dir():
                return "per_structure", self.file, sub
        vols = sorted(n for n in os.listdir(probe) if _is_vol(n))
        if self.file and self.file in vols:
            return "folder", self.file, None
        for known in _KNOWN_FILES:
            if known in vols:
                return "folder", known, None
        if len(vols) == 1:
            return "folder", vols[0], None
        return "per_structure", None, ""

    def _build_index(self) -> Dict[str, Path]:
        rx = re.compile(self.strip) if self.strip else None
        index: Dict[str, Path] = {}
        with os.scandir(self.root) as it:
            for e in it:
                if self.layout == "flat":
                    stem = _stem(e.name) if e.is_file() else None
                    if stem is None:
                        continue
                    cid = rx.sub("", stem) if rx else stem
                    index[cid] = Path(e.path)
                elif e.is_dir() and not e.name.startswith("."):
                    index[e.name] = Path(e.path)
        return dict(sorted(index.items()))

    @property
    def case_ids(self) -> List[str]:
        return list(self._index)

    def path(self, case_id: str) -> Path:
        base = self._index[case_id]
        if self.layout == "folder":
            return base / (self.file or "")
        if self.layout == "per_structure" and self.subdir:
            return base / self.subdir
        return base

    def structures(self, case_id: Optional[str] = None) -> List[str]:
        """Structure names available as per-structure files (for ``per_structure`` sources)."""
        if self.layout != "per_structure":
            return []
        case_id = case_id or self.case_ids[0]
        return sorted(s for s in (_stem(n) for n in os.listdir(self.path(case_id))) if s)

    # ---------------------------------------------------------------- loading
    def load(self, case_id: str) -> Volume:
        """Load the (multi-label) volume of a ``flat``/``folder`` source, cached per case."""
        if self.layout == "per_structure":
            raise ValueError("per_structure sources are loaded one structure at a time")
        if self._cache_key != case_id:
            self._cache = load_volume(self.path(case_id), kind=self.kind)
            self._cache_key = case_id
        return self._cache

    def load_mask(self, case_id: str, label: LabelSpec, side: str) -> Tuple[Optional[np.ndarray], Volume]:
        """Binary mask for ``label``; ``(None, geometry)`` if a per-structure file is missing."""
        if self.layout == "per_structure":
            f = self.path(case_id) / label.file(side)
            if not f.exists():
                alt = [p for p in (self.path(case_id) / (label.file(side).split(".")[0] + s) for s in _SUFFIXES) if p.exists()]
                if not alt:
                    return None, None
                f = alt[0]
            vol = load_volume(f, kind=self.kind)
            data = vol.data
            if self.kind == "prob":
                return data.astype(np.float32, copy=False), vol
            values = label.values(side)
            mask = extract_mask(data, values) if values else data != 0
            return mask, vol
        vol = self.load(case_id)
        values = label.values(side)
        if not values:
            raise ValueError(f"label {label.name!r} has no {side} ids but {self.root} is a multi-label layout")
        return extract_mask(vol.data, values), vol

    def __repr__(self) -> str:
        extra = f", file={self.file!r}" if self.file else ""
        extra += f", subdir={self.subdir!r}" if self.subdir else ""
        return f"Source({str(self.root)!r}, layout={self.layout!r}{extra}, n_cases={len(self._index)})"


@dataclass
class Case:
    """One evaluation unit: a case id and where its inputs live."""

    case_id: str
    has_pred: bool = True
    image: Optional[Path] = None


def discover_cases(
    pred: Source,
    ref: Source,
    image: Optional[Source] = None,
    *,
    cases: Optional[Sequence[str]] = None,
) -> Tuple[List[Case], Dict[str, List[str]]]:
    """Pair reference and prediction cases.

    The reference defines the case list (a model must not choose its own test
    set). Returns the cases and a report ``{"missing_pred": [...],
    "extra_pred": [...]}``; cases with a missing prediction are kept with
    ``has_pred=False`` so the evaluator can score them as empty predictions.
    """
    ref_ids = list(cases) if cases is not None else ref.case_ids
    unknown = [c for c in ref_ids if c not in ref._index]
    if unknown:
        raise KeyError(f"{len(unknown)} requested cases not in reference: {unknown[:5]}")
    pred_ids = set(pred.case_ids)
    out = []
    for cid in ref_ids:
        img = None
        if image is not None and cid in image._index:
            img = image.path(cid)
        out.append(Case(cid, has_pred=cid in pred_ids, image=img))
    report = {
        "missing_pred": [c for c in ref_ids if c not in pred_ids],
        "extra_pred": sorted(pred_ids - set(ref_ids)) if cases is None else [],
    }
    return out, report
