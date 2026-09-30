"""Label specifications: which structures to evaluate and how to find them.

A structure can be stored two ways, and prediction and reference need not
agree:

* as one or more integer values in a **multi-label map** (nnU-Net, MSD, AMOS,
  BTCV...). Several values form a *region*: BraTS' whole tumour is
  ``{1, 2, 3}``, "kidney" can be ``{left, right}``;
* as a **binary file per structure** (TotalSegmentator, AbdomenAtlas, PanTS:
  ``<case>/segmentations/<name>.nii.gz``).

:class:`LabelSpec` records both, separately for the prediction and the
reference, so a model that writes ``pancreas = 7`` in one file can be scored
against a dataset that ships ``segmentations/pancreas.nii.gz``.

Accepted forms for :func:`parse_labels` (all equivalent in a YAML config)::

    {"liver": 1, "tumour": 2}                 # same id in pred and ref
    {"kidney": [2, 3]}                         # region = union of ids
    {"pancreas": {"ref": 1, "pred": 7}}        # different ids per side
    {"pancreas": {"ref_file": "pancreas.nii.gz", "pred": 7}}
    ["liver", "pancreas"]                      # per-structure files by name
    {"0": "background", "1": "liver"}          # MSD dataset.json style
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

__all__ = ["LabelSpec", "parse_labels", "extract_mask"]


@dataclass(frozen=True)
class LabelSpec:
    """How to obtain one structure's binary mask from prediction and reference.

    Attributes:
        name: Structure name used in all outputs.
        ref_values / pred_values: Integer ids forming the structure in a
            multi-label map (``()`` when the structure comes from its own file).
        ref_file / pred_file: File name of the per-structure binary mask inside
            a case folder (defaults to ``<name>.nii.gz``).
        params: Per-structure metric parameter overrides, e.g.
            ``{"nsd": {"tolerance_mm": 1.0}}`` (tolerances should be
            structure-specific; Metrics Reloaded).
        metrics: Per-structure metric list (names / sets); empty means the
            evaluator's global list. Lets one run score clDice only on
            vessels and lesion-wise metrics only on tumours.
    """

    name: str
    ref_values: Tuple[int, ...] = ()
    pred_values: Tuple[int, ...] = ()
    ref_file: Optional[str] = None
    pred_file: Optional[str] = None
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metrics: Tuple[str, ...] = ()

    def values(self, side: str) -> Tuple[int, ...]:
        return self.ref_values if side == "ref" else self.pred_values

    def file(self, side: str) -> str:
        f = self.ref_file if side == "ref" else self.pred_file
        return f or f"{self.name}.nii.gz"


def _ids(v: Union[int, Iterable[int], None]) -> Tuple[int, ...]:
    if v is None:
        return ()
    if isinstance(v, (int, np.integer)):
        return (int(v),)
    if isinstance(v, str):
        return tuple(int(x) for x in v.replace("+", ",").split(",") if x.strip())
    return tuple(int(x) for x in v)


def parse_labels(spec: Union[Mapping, Iterable[str], None]) -> List[LabelSpec]:
    """Normalise any supported label description into a list of :class:`LabelSpec`."""
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    if isinstance(spec, Mapping):
        items = list(spec.items())
        # MSD / nnU-Net-v1 style {"1": "liver"}: invert.
        if items and all(isinstance(k, str) and k.lstrip("-").isdigit() and isinstance(v, str) for k, v in items):
            items = [(v, int(k)) for k, v in items]
        out = []
        for name, v in items:
            name = str(name)
            if name.lower() in ("background", "bg") and _ids(v if not isinstance(v, Mapping) else v.get("ref")) == (0,):
                continue
            if isinstance(v, Mapping):
                ref = _ids(v.get("ref", v.get("values")))
                pred = _ids(v.get("pred", v.get("values", v.get("ref"))))
                mets = v.get("metrics", ())
                mets = tuple(mets.split(",")) if isinstance(mets, str) else tuple(mets)
                out.append(LabelSpec(name, ref, pred, v.get("ref_file", v.get("file")),
                                     v.get("pred_file", v.get("file")), dict(v.get("params", {})), mets))
            else:
                ids = _ids(v)
                out.append(LabelSpec(name, ids, ids))
        return out
    return [LabelSpec(str(n)) for n in spec]


def extract_mask(label_map: np.ndarray, values: Tuple[int, ...]) -> np.ndarray:
    """Binary mask of all voxels whose value is in ``values``."""
    if len(values) == 1:
        return label_map == values[0]
    return np.isin(label_map, values)


def labels_from_map(label_map: np.ndarray, names: Optional[Dict[int, str]] = None) -> List[LabelSpec]:
    """One :class:`LabelSpec` per non-zero value present in ``label_map``."""
    vals = [int(v) for v in np.unique(label_map) if v != 0]
    names = names or {}
    return [LabelSpec(names.get(v, f"label_{v}"), (v,), (v,)) for v in vals]
