"""Data model of the interactive viewer: one case, its reference and any number of model predictions.

A `ViewerSession` loads the CT, the reference structures and every model's
prediction once, puts them on the image grid, and serves what the browser
needs: NIfTI volumes (image, label maps, per-structure error maps), surface
meshes for the 3D view, per-structure statistics and the lesion list.

Structures are resolved exactly as in the `~segevalkit.Evaluator`: a side is
either a multi-label map (structures are id sets) or a folder of
per-structure files (``"a.nii.gz+b.nii.gz"`` is a union), described by
`~segevalkit.io.LabelSpec`. Masks whose array shape equals the image are
compared in voxel correspondence (label headers are not always trustworthy,
see the pitfall "Unreliable label headers"); masks on another grid are
resampled onto the image grid with nearest-neighbour interpolation.
"""

from __future__ import annotations

import gzip
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..io.cases import Source, _is_vol, _stem
from ..io.labels import LabelSpec, extract_mask, parse_labels
from ..io.volume import Volume, load_volume, resample_to

__all__ = ["ViewerSession", "ERROR_CODES", "LESION_PATTERN"]

PathLike = Union[str, os.PathLike]

#: Values of the error maps served to the viewer.
ERROR_CODES = {"tp": 1, "fn": 2, "fp": 3}
#: Structure names treated as lesions (instance list, lesion-first display).
LESION_PATTERN = re.compile(r"lesion|tumou?r|metasta|nodule|cancer|cyst", re.I)

# Organ palette (distinct, colour-vision-aware hues; lesions use the first slot).
_PALETTE = ["#ff2d6f", "#f2a93b", "#b5653a", "#8e63d6", "#3f9fd6", "#44b37a", "#d6c13f", "#d65f9e",
            "#5ec4c4", "#a4b83f", "#c7853f", "#6f7fe0", "#e0766f", "#58a36b", "#b98ad6", "#8f9b5c",
            "#d19a6a", "#5f8fb0", "#c05f5f", "#7fb0d6", "#b0a05f", "#9f6fa0", "#6fb09f", "#d0b0e0"]
_WINDOWS = {"abdomen": (-160, 240), "liver": (-15, 185), "bone": (-400, 1400), "lung": (-1350, 150),
            "brain": (0, 80)}


@dataclass
class _Input:
    """Where one side's masks come from: a multi-label map file or a folder of per-structure files."""

    kind: str  # "map" | "files"
    path: Path


def _case_stem(name: str) -> str:
    return re.sub(r"_0000$", "", _stem(name) or name)


def resolve_input(path: PathLike, case: Optional[str] = None, kind: str = "label") -> _Input:
    """Resolve a file, a per-structure folder, or a dataset folder plus case id."""
    p = Path(path)
    if p.is_file():
        return _Input("map", p)
    if not p.is_dir():
        raise FileNotFoundError(f"no such file or folder: {p}")
    vols = sorted(n for n in os.listdir(p) if _is_vol(n) and not n.startswith("."))
    if vols:
        if case is not None:
            hit = [n for n in vols if _case_stem(n) == case]
            if hit:
                return _Input("map", p / hit[0])
        if kind == "image":
            if len(vols) == 1:
                return _Input("map", p / vols[0])
            raise ValueError(f"{p} holds {len(vols)} images; pass --case")
        return _Input("files", p)
    src = Source(p, kind=kind)
    ids = src.case_ids
    if case is None:
        if len(ids) != 1:
            raise ValueError(f"{p} holds {len(ids)} cases; pass --case")
        case = ids[0]
    if case not in ids:
        raise KeyError(f"case {case!r} not found in {p}")
    q = src.path(case)
    if kind == "image" and q.is_dir():
        vols = sorted(n for n in os.listdir(q) if _is_vol(n))
        pref = [n for n in vols if _stem(n) in ("ct", "image", "img", "t1", "t2", "flair")]
        q = q / (pref or vols)[0]
    return _Input("map" if q.is_file() else "files", q)


def _find_structure_file(folder: Path, name: str) -> Optional[Path]:
    f = folder / name
    if f.exists():
        return f
    base = name.split(".")[0]
    for suf in (".nii.gz", ".nii", ".mha", ".nrrd"):
        if (folder / (base + suf)).exists():
            return folder / (base + suf)
    return None


def _nifti_bytes(data: np.ndarray, affine: np.ndarray, level: int = 3) -> bytes:
    import nibabel as nib

    img = nib.Nifti1Image(np.ascontiguousarray(data), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    return gzip.compress(img.to_bytes(), compresslevel=level)


def _mz3_bytes(verts: np.ndarray, faces: np.ndarray) -> bytes:
    """Gzip-compressed MZ3 mesh (Surf Ice / NiiVue): header, int32 faces, float32 vertices."""
    head = np.array([23117, 3], "<u2").tobytes() + np.array([len(faces), len(verts), 0], "<u4").tobytes()
    raw = head + np.ascontiguousarray(faces, "<i4").tobytes() + np.ascontiguousarray(verts, "<f4").tobytes()
    return gzip.compress(raw, compresslevel=6)


class ViewerSession:
    """One case prepared for the viewer.

    Args:
        image: CT/MR file, or a dataset folder (with ``case``).
        ref: Reference: a multi-label map, a folder of per-structure masks,
            or a dataset folder (with ``case``). Optional: without it the
            viewer shows the predictions only.
        preds: ``{model name: path}``, each resolved like ``ref``.
        labels: Structures to show, in any form accepted by
            `~segevalkit.io.parse_labels` (ids, regions, per-structure files,
            ``"a.nii.gz+b.nii.gz"`` unions). Default: every id / file of the
            reference.
        pred_labels: Per-model label specs for models whose output convention
            differs (e.g. nnU-Net ids, TotalSegmentator file names). Structures
            a model's spec does not mention fall back to ``labels``.
        results: ``{model name: results folder}`` of SegEvalKit evaluations;
            the viewer shows this case's stored metrics.
        case: Case id when a dataset folder is given (also used to look up
            ``results``).
        lesions: Names of lesion structures (default: names matching
            `LESION_PATTERN`).
        crop_margin_mm: Crop the grid to all structures plus this margin
            (``None`` keeps the full field of view).
        max_dim: Downsample (voxel stride) so that no axis exceeds this size
            (``None`` keeps full resolution).
        min_lesion_voxels: Connected components smaller than this are ignored
            in the lesion list.
    """

    def __init__(self, image: PathLike, ref: Optional[PathLike] = None,
                 preds: Optional[Mapping[str, PathLike]] = None, labels: Any = None,
                 pred_labels: Optional[Mapping[str, Any]] = None,
                 results: Optional[Mapping[str, PathLike]] = None, case: Optional[str] = None,
                 lesions: Optional[Sequence[str]] = None, crop_margin_mm: Optional[float] = None,
                 max_dim: Optional[int] = None, min_lesion_voxels: int = 10):
        self._lock = threading.Lock()
        self._cache: Dict[Tuple, Any] = {}
        img_in = resolve_input(image, case, kind="image")
        self.image_path = img_in.path
        self.case = case or _case_stem(img_in.path.name)
        self._img = load_volume(img_in.path, kind="image")
        self.spacing = tuple(float(s) for s in self._img.spacing)
        self.full_shape = tuple(int(s) for s in self._img.shape)
        self.min_lesion_voxels = int(min_lesion_voxels)
        #: Marching-cubes step (voxels); 2 makes meshes about four times smaller.
        self.mesh_step = 1

        self._ref_in = resolve_input(ref, case) if ref is not None else None
        self._pred_in = {str(k): resolve_input(v, case) for k, v in (preds or {}).items()}
        self.models: List[str] = list(self._pred_in)
        self._map_cache: Dict[str, Volume] = {}

        specs = parse_labels(labels) if labels is not None else self._default_labels()
        self.labels: Dict[str, LabelSpec] = {s.name: s for s in specs}
        self._pred_specs: Dict[str, Dict[str, LabelSpec]] = {}
        for m in self.models:
            own = {s.name: s for s in parse_labels((pred_labels or {}).get(m))} if (pred_labels or {}).get(m) else {}
            self._pred_specs[m] = {n: own.get(n, s) for n, s in self.labels.items()}

        # masks on the full image grid
        self.ref_masks: Dict[str, np.ndarray] = {}
        if self._ref_in is not None:
            for n, s in self.labels.items():
                mk = self._mask(self._ref_in, s, "ref")
                if mk is not None:
                    self.ref_masks[n] = mk
        self.pred_masks: Dict[str, Dict[str, np.ndarray]] = {}
        for m, inp in self._pred_in.items():
            self.pred_masks[m] = {}
            for n, s in self._pred_specs[m].items():
                mk = self._mask(inp, s, "pred")
                if mk is not None:
                    self.pred_masks[m][n] = mk
        names = [n for n in self.labels if n in self.ref_masks or any(n in d for d in self.pred_masks.values())]
        lesion_set = set(lesions) if lesions is not None else {n for n in names if LESION_PATTERN.search(n)}
        # display order: lesions first, then by reference volume (largest first)
        vol = {n: int(self.ref_masks[n].sum()) if n in self.ref_masks else
               max(int(d[n].sum()) for d in self.pred_masks.values() if n in d) for n in names}
        self.structures: List[str] = sorted(names, key=lambda n: (n not in lesion_set, -vol[n]))
        self.lesions = [n for n in self.structures if n in lesion_set]
        self.index = {n: i + 1 for i, n in enumerate(self.structures)}
        self.colors = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(self.structures)}

        self._set_region(crop_margin_mm, max_dim)
        self.results = self._read_results(results or {})

    # ------------------------------------------------------------------ inputs
    def _default_labels(self) -> List[LabelSpec]:
        inp = self._ref_in or (next(iter(self._pred_in.values())) if self._pred_in else None)
        if inp is None:
            return []
        if inp.kind == "files":
            return [LabelSpec(s) for s in sorted(_stem(n) for n in os.listdir(inp.path) if _is_vol(n))]
        vals = [int(v) for v in np.unique(self._load_map(inp).data) if v != 0]
        return [LabelSpec(f"label_{v}", (v,), (v,)) for v in vals]

    def _fit(self, vol: Volume) -> np.ndarray:
        """Put a label volume on the image grid (voxel correspondence when shapes agree)."""
        if tuple(vol.shape) == self.full_shape:
            return vol.data
        return resample_to(vol, self._img, order=0).data

    def _load_map(self, inp: _Input) -> Volume:
        key = str(inp.path)
        if key not in self._map_cache:
            v = load_volume(inp.path)
            self._map_cache[key] = Volume(self._fit(v), self._img.spacing, self._img.affine, str(inp.path))
        return self._map_cache[key]

    def _mask(self, inp: _Input, spec: LabelSpec, side: str) -> Optional[np.ndarray]:
        values = spec.values(side)
        if inp.kind == "map":
            if not values:
                return None
            return extract_mask(self._load_map(inp).data, values)
        mask = None
        for name in spec.file(side).split("+"):
            f = _find_structure_file(inp.path, name.strip())
            if f is None:
                continue
            data = self._fit(load_volume(f))
            part = extract_mask(data, values) if values else data != 0
            mask = part if mask is None else (mask | part)
        return mask

    def _read_results(self, results: Mapping[str, PathLike]) -> Dict[str, Dict[str, Dict[str, float]]]:
        import pandas as pd

        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for m, d in results.items():
            f = Path(d) / "per_case_wide.csv"
            if not f.exists():
                continue
            w = pd.read_csv(f)
            w = w[w["case_id"].astype(str) == str(self.case)]
            skip = {"case_id", "label", "ref_empty", "pred_empty", "tp", "fp", "fn", "tn", "n_ref_lesions",
                    "n_pred_lesions", "tp_ref_lesions", "tp_pred_lesions"}
            out[m] = {str(r["label"]): {k: float(v) for k, v in r.items()
                                        if k not in skip and isinstance(v, (int, float, np.floating)) and np.isfinite(v)}
                      for _, r in w.iterrows()}
        return out

    # ------------------------------------------------------------------ region
    def _set_region(self, margin_mm: Optional[float], max_dim: Optional[int]) -> None:
        lo = np.zeros(3, int)
        hi = np.asarray(self.full_shape, int)
        if margin_mm is not None:
            masks = list(self.ref_masks.values()) + [m for d in self.pred_masks.values() for m in d.values()]
            any_ = np.zeros(self.full_shape, bool)
            for mk in masks:
                any_ |= mk
            if any_.any():
                idx = np.nonzero(any_)
                pad = np.ceil(float(margin_mm) / np.asarray(self.spacing)).astype(int)
                lo = np.maximum(np.array([i.min() for i in idx]) - pad, 0)
                hi = np.minimum(np.array([i.max() for i in idx]) + pad + 1, self.full_shape)
        ext = hi - lo
        # stride per axis, so an axis that is already small (e.g. thick slices) keeps full resolution
        steps = np.ones(3, int)
        if max_dim is not None:
            steps = np.maximum(1, np.ceil(ext / float(max_dim))).astype(int)
        self.region = tuple(slice(int(a), int(b), int(k)) for a, b, k in zip(lo, hi, steps))
        self.steps = tuple(int(k) for k in steps)
        self.step = int(steps.max())
        t = np.eye(4)
        t[:3, :3] = np.diag(steps.astype(float))
        t[:3, 3] = lo
        self.affine = self._img.affine @ t
        self.shape = tuple(len(range(*s.indices(n))) for s, n in zip(self.region, self.full_shape))

    def _r(self, a: np.ndarray) -> np.ndarray:
        return a[self.region]

    def vox2mm(self, vox: Sequence[float]) -> List[float]:
        """Region voxel coordinates to world millimetres (RAS)."""
        return [float(x) for x in (self.affine @ np.r_[np.asarray(vox, float), 1.0])[:3]]

    # ------------------------------------------------------------------ products
    def _cached(self, key: Tuple, fn):
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        val = fn()
        with self._lock:
            self._cache[key] = val
        return val

    def _paint(self, masks: Mapping[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(self.shape, np.uint16)
        # large structures first, lesions last, so small structures stay visible
        for n in sorted(masks, key=lambda n: (n in self.lesions, -int(masks[n].sum()))):
            out[self._r(masks[n])] = self.index[n]
        return out

    def image_nifti(self) -> bytes:
        def build():
            a = np.clip(np.rint(self._r(self._img.data)), -32768, 32767).astype(np.int16)
            return _nifti_bytes(a, self.affine)
        return self._cached(("image",), build)

    def labels_nifti(self, source: str = "ref") -> bytes:
        """Label map (values = structure index) of the reference or of one model."""
        masks = self.ref_masks if source == "ref" else self.pred_masks[source]
        return self._cached(("labels", source), lambda: _nifti_bytes(self._paint(masks), self.affine))

    def error_array(self, model: str, structure: str) -> np.ndarray:
        g = self.ref_masks.get(structure, np.zeros(self.full_shape, bool))
        p = self.pred_masks[model].get(structure, np.zeros(self.full_shape, bool))
        g, p = self._r(g), self._r(p)
        e = np.zeros(self.shape, np.uint8)
        e[g & p] = ERROR_CODES["tp"]
        e[g & ~p] = ERROR_CODES["fn"]
        e[~g & p] = ERROR_CODES["fp"]
        return e

    def error_nifti(self, model: str, structure: str) -> bytes:
        return self._cached(("error", model, structure),
                            lambda: _nifti_bytes(self.error_array(model, structure), self.affine))

    def mesh(self, source: str, structure: str, part: Optional[str] = None) -> bytes:
        """Surface mesh (gzipped MZ3, world mm) of a reference / predicted structure or of its TP/FN/FP part."""
        def build():
            if part is None:
                src = self.ref_masks if source == "ref" else self.pred_masks[source]
                m = self._r(src[structure]) if structure in src else np.zeros(self.shape, bool)
            else:
                m = self.error_array(source, structure) == ERROR_CODES[part]
            return self._mesh_from_mask(m)
        return self._cached(("mesh", source, structure, part, self.mesh_step), build)

    def _mesh_from_mask(self, m: np.ndarray) -> bytes:
        from scipy import ndimage
        from skimage.measure import marching_cubes

        if m.sum() < 2:
            return _mz3_bytes(np.zeros((0, 3)), np.zeros((0, 3), int))
        idx = np.nonzero(m)
        lo = np.maximum(np.array([i.min() for i in idx]) - 2, 0)
        hi = np.minimum(np.array([i.max() for i in idx]) + 3, m.shape)
        sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        sub = np.pad(sub, 1)
        big = sub.size > 4e6
        sm = ndimage.gaussian_filter(sub, 0.8)
        top = float(sm.max())
        if top <= 1e-6:
            return _mz3_bytes(np.zeros((0, 3)), np.zeros((0, 3), int))
        # tiny components fall below 0.5 after smoothing: contour them at half their peak instead
        step = max(self.mesh_step, 2 if big else 1) if top > 0.6 else 1
        verts, faces, _, _ = marching_cubes(sm, level=min(0.5, 0.5 * top), step_size=step)
        verts = verts - 1 + lo
        world = (self.affine @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
        return _mz3_bytes(world, faces.astype(np.int32))

    # ------------------------------------------------------------------ statistics
    def _extent(self, m: np.ndarray) -> Optional[Dict[str, Any]]:
        mr = self._r(m)
        if not mr.any():
            return None
        idx = np.nonzero(mr)
        c = [float(i.mean()) for i in idx]
        lo = [int(i.min()) for i in idx]
        hi = [int(i.max()) for i in idx]
        corners = [self.vox2mm([a, b, cc]) for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for cc in (lo[2], hi[2])]
        corners = np.asarray(corners)
        return {"centroid_vox": c, "centroid_mm": self.vox2mm(c),
                "min_mm": corners.min(0).tolist(), "max_mm": corners.max(0).tolist()}

    def live_metrics(self, model: str, structure: str) -> Dict[str, float]:
        """Dice, NSD (2 mm), HD95, ASSD and RVD for one structure, computed on the full grid."""
        def build():
            from ..metrics import compute_metrics

            g = self.ref_masks.get(structure, np.zeros(self.full_shape, bool))
            p = self.pred_masks.get(model, {}).get(structure, np.zeros(self.full_shape, bool))
            r = compute_metrics(p, g, ["dice", "nsd", "hd95", "assd", "relative_volume_difference"],
                                spacing=self.spacing)
            return {k: float(v) for k, v in r.items()}
        return self._cached(("live", model, structure), build)

    def lesion_table(self) -> Dict[str, Any]:
        """Reference lesions (found / missed per model) and each model's false-positive components."""
        def build():
            from scipy import ndimage

            st = np.ones((3, 3, 3), bool)
            vox_ml = float(np.prod(self.spacing)) / 1000.0
            out: Dict[str, Any] = {}
            for n in self.lesions:
                g = self.ref_masks.get(n, np.zeros(self.full_shape, bool))
                gl, ng = ndimage.label(g, st)
                refs = []
                for k in range(1, ng + 1):
                    comp = gl == k
                    if comp.sum() < self.min_lesion_voxels:
                        continue
                    row = {"id": k, "ml": float(comp.sum() * vox_ml), "found": {}, "dice": {}}
                    ext = self._extent(comp)
                    row.update(ext or {})
                    for m in self.models:
                        if n not in self.pred_masks[m]:  # the model has no such class
                            row["found"][m], row["dice"][m] = None, None
                            continue
                        p = self.pred_masks[m][n]
                        inter = int((comp & p).sum())
                        row["found"][m] = inter > 0
                        pl, _ = ndimage.label(p, st)
                        ids = np.unique(pl[comp & p])
                        pm = np.isin(pl, ids[ids > 0])
                        row["dice"][m] = float(2 * (comp & pm).sum() / (comp.sum() + pm.sum())) if inter else 0.0
                    refs.append(row)
                fps = {}
                for m in self.models:
                    if n not in self.pred_masks[m]:
                        continue
                    p = self.pred_masks[m][n]
                    pl, npr = ndimage.label(p, st)
                    rows = []
                    for k in range(1, npr + 1):
                        comp = pl == k
                        if comp.sum() < self.min_lesion_voxels or (comp & g).any():
                            continue
                        rows.append({"ml": float(comp.sum() * vox_ml), **(self._extent(comp) or {})})
                    fps[m] = rows
                out[n] = {"ref": refs, "fp": fps}
            return out
        return self._cached(("lesions",), build)

    def manifest(self) -> Dict[str, Any]:
        """Everything the page needs besides the volumes and meshes."""
        vox_ml = float(np.prod(self.spacing)) / 1000.0
        structs = []
        for n in self.structures:
            g = self.ref_masks.get(n)
            row: Dict[str, Any] = {"name": n, "index": self.index[n], "color": self.colors[n],
                                   "lesion": n in self.lesions, "in_ref": g is not None,
                                   "ref_ml": float(g.sum() * vox_ml) if g is not None else None, "models": {}}
            ext = self._extent(g) if g is not None and g.any() else None
            for m in self.models:
                p = self.pred_masks[m].get(n)
                if p is None:
                    row["models"][m] = None
                    continue
                gg = g if g is not None else np.zeros(self.full_shape, bool)
                tp = int((gg & p).sum())
                fp = int(p.sum()) - tp
                fn = int(gg.sum()) - tp
                den = 2 * tp + fp + fn
                row["models"][m] = {"pred_ml": float(p.sum() * vox_ml), "tp": tp, "fp": fp, "fn": fn,
                                    "dice": float(2 * tp / den) if den else 1.0}
                if ext is None and p.any():
                    ext = self._extent(p)
            row["extent"] = ext
            structs.append(row)
        from .. import __version__
        from ..plotting.theme import ERROR_COLORS

        return {
            "version": __version__, "case": self.case, "image": self.image_path.name,
            "shape": list(self.shape), "spacing": list(self.spacing), "step": self.step, "steps": list(self.steps),
            "has_ref": self._ref_in is not None, "models": self.models, "structures": structs,
            "lesions": self.lesion_table() if self.lesions else {},
            "results": self.results, "windows": _WINDOWS,
            "error_colors": {k: ERROR_COLORS[k] for k in ("tp", "fn", "fp")},
        }

    # ------------------------------------------------------------------ routing
    def resource(self, path: str) -> Tuple[bytes, str]:
        """Bytes and content type of a data path used by the page (shared by server and export)."""
        parts = [p for p in path.strip("/").split("/") if p]
        if parts[:1] == ["vol"]:
            if parts[1:] == ["image.nii.gz"]:
                return self.image_nifti(), "application/gzip"
            if len(parts) == 3 and parts[1] == "labels":
                src = parts[2].removesuffix(".nii.gz")
                if src != "ref" and src not in self.models:
                    raise KeyError(path)
                return self.labels_nifti(src), "application/gzip"
            if len(parts) == 4 and parts[1] == "error":
                m, s = parts[2], parts[3].removesuffix(".nii.gz")
                if m not in self.models or s not in self.index:
                    raise KeyError(path)
                return self.error_nifti(m, s), "application/gzip"
        if parts[:1] == ["mesh"] and len(parts) in (3, 4):
            src, s = parts[1], parts[-1].removesuffix(".mz3")
            part = parts[2] if len(parts) == 4 else None
            if (src != "ref" and src not in self.models) or s not in self.index or (part and part not in ERROR_CODES):
                raise KeyError(path)
            return self.mesh(src, s, part), "application/octet-stream"
        raise KeyError(path)

    def __repr__(self) -> str:
        return (f"ViewerSession(case={self.case!r}, structures={len(self.structures)}, models={self.models}, "
                f"grid={self.shape}, steps={self.steps})")
