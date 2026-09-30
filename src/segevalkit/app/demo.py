"""A synthetic abdominal phantom for trying the viewer without any data (``segevalkit view --demo``).

Nothing is downloaded and nothing large ships with the package: the phantom is
generated on the fly (about a second) into a cache folder. It has a body
outline, spine, liver, spleen, both kidneys, stomach, aorta, a pancreas with a
hypodense lesion, and three synthetic "models" whose errors are made with
`segevalkit.synthetic`:

* **Model A**: small boundary noise everywhere; finds the lesion.
* **Model B**: over-segments (organs dilated), shifts the pancreas, and adds a
  false-positive "lesion" in the liver.
* **Model C**: under-segments the kidneys, misses part of the pancreas and
  misses the lesion.

The phantom is an illustration of the viewer, not an anatomical model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .. import synthetic as syn

__all__ = ["make_phantom", "demo_session", "PHANTOM_LABELS"]

#: Label ids of the phantom label maps (reference and models).
PHANTOM_IDS = {"liver": 1, "spleen": 2, "kidney_right": 3, "kidney_left": 4, "pancreas": 5, "aorta": 6,
               "stomach": 7, "pancreatic_lesion": 8}
#: Structures shown by the demo; the pancreas region includes its lesion (ids 5 and 8).
PHANTOM_LABELS = {**{n: v for n, v in PHANTOM_IDS.items() if n != "pancreas"}, "pancreas": [5, 8]}
_VERSION = "1"


def _cache_dir() -> Path:
    base = os.environ.get("SEGEVALKIT_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "segevalkit")
    return Path(base) / f"demo_phantom_v{_VERSION}"


def _ellipsoid(grid, c, r):
    x, y, z = grid
    return ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2 <= 1.0


def make_phantom(folder: Optional[Union[str, Path]] = None, seed: int = 0, overwrite: bool = False) -> Dict[str, Path]:
    """Write the phantom CT, reference and three model label maps as NIfTI; return their paths.

    Coordinates are millimetres in RAS (x: patient left, y: anterior, z: superior);
    the grid is 144 x 112 x 56 voxels of 1.5 x 1.5 x 3 mm.
    """
    import nibabel as nib

    folder = Path(folder) if folder is not None else _cache_dir()
    names = {"image": "ct.nii.gz", "ref": "reference.nii.gz", "Model A": "model_a.nii.gz",
             "Model B": "model_b.nii.gz", "Model C": "model_c.nii.gz"}
    paths = {k: folder / v for k, v in names.items()}
    if not overwrite and all(p.exists() for p in paths.values()):
        return paths
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape, sp = (144, 112, 56), (1.5, 1.5, 3.0)
    ax = [(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, sp)]
    grid = np.meshgrid(*ax, indexing="ij")
    x, y, z = grid

    body = (x / 100.0) ** 2 + (y / 72.0) ** 2 <= 1.0
    spine = ((x / 13.0) ** 2 + ((y + 42) / 14.0) ** 2 <= 1.0)
    ref = np.zeros(shape, np.uint8)
    organs = {
        "liver": _ellipsoid(grid, (-45, 8, 22), (48, 52, 42)) & ~_ellipsoid(grid, (-5, 30, 50), (30, 30, 30)),
        "spleen": _ellipsoid(grid, (62, -18, 30), (18, 26, 30)),
        "kidney_right": _ellipsoid(grid, (-40, -32, -10), (14, 18, 30)),
        "kidney_left": _ellipsoid(grid, (42, -30, -6), (14, 18, 30)),
        "stomach": _ellipsoid(grid, (40, 28, 30), (26, 20, 26)),
        "aorta": ((x - 12) / 8.0) ** 2 + ((y + 20) / 8.0) ** 2 <= 1.0,
    }
    # pancreas: a curved capsule from the duodenal loop (patient right) to the splenic hilum (left)
    t = np.linspace(0, 1, 40)
    cx, cy, cz = -30 + 85 * t, 5 - 22 * t ** 2, -2 + 16 * t
    rad = 11 - 4 * t
    panc = np.zeros(shape, bool)
    for a, b, c, r in zip(cx, cy, cz, rad):
        panc |= (x - a) ** 2 + (y - b) ** 2 + ((z - c) * 1.4) ** 2 <= r ** 2
    lesion = _ellipsoid(grid, (-12, 3, 2), (9, 8, 9)) & panc
    organs["pancreas"] = panc & ~lesion
    organs["pancreatic_lesion"] = lesion
    for n in ("liver", "spleen", "kidney_right", "kidney_left", "stomach", "pancreas", "aorta", "pancreatic_lesion"):
        m = organs[n] & body
        organs[n] = m
        ref[m] = PHANTOM_IDS[n]

    hu = np.full(shape, -1000.0)
    hu[body] = -90.0  # fat
    muscle = body & ~((x / 92.0) ** 2 + (y / 64.0) ** 2 <= 1.0)
    hu[muscle] = 45.0
    hu[(ref > 0)] = 45.0
    for n, v in (("liver", 62), ("spleen", 52), ("kidney_right", 160), ("kidney_left", 160), ("stomach", 25),
                 ("pancreas", 45), ("aorta", 210), ("pancreatic_lesion", 12)):
        hu[organs[n]] = v
    hu[spine] = 650.0
    hu[spine & (((x / 8.0) ** 2 + ((y + 42) / 9.0) ** 2) <= 1.0)] = 280.0
    from scipy import ndimage

    hu = ndimage.gaussian_filter(hu, (0.8, 0.8, 0.5)) + rng.normal(0, 12, shape)
    ct = np.clip(np.rint(hu), -1024, 3071).astype(np.int16)

    def model(fn):
        out = np.zeros(shape, np.uint8)
        for n in ("liver", "spleen", "kidney_right", "kidney_left", "stomach", "pancreas", "aorta", "pancreatic_lesion"):
            m = fn(n, organs[n])
            if m is not None:
                out[m & body] = PHANTOM_IDS[n]
        return out

    def model_a(n, m):
        return syn.boundary_noise(m, 1.5, sp, rng=rng) if m.sum() > 50 else m

    def model_b(n, m):
        if n == "pancreatic_lesion":  # found, slightly too large, plus a false-positive "lesion" in the liver
            return syn.dilate(m, 1.5, sp) | _ellipsoid(grid, (-60, 10, 25), (6, 6, 7))
        if n == "pancreas":
            return syn.shift(m, 4.0, sp, rng=rng, axis=0)
        return syn.dilate(m, 2.0, sp)

    def model_c(n, m):
        if n == "pancreatic_lesion":
            return None  # missed
        if n == "pancreas":
            return syn.remove_slab(m | organs["pancreatic_lesion"], 0.25, sp, rng=rng, axis=0)
        if n.startswith("kidney"):
            return syn.erode(m, 2.5, sp)
        return syn.boundary_noise(m, 2.0, sp, rng=rng)

    aff = np.diag([sp[0], sp[1], sp[2], 1.0])
    aff[:3, 3] = [a[0] for a in ax]
    for key, arr in (("image", ct), ("ref", ref), ("Model A", model(model_a)), ("Model B", model(model_b)),
                     ("Model C", model(model_c))):
        img = nib.Nifti1Image(arr, aff)
        img.set_qform(aff, code=1)
        img.set_sform(aff, code=1)
        nib.save(img, str(paths[key]))
    (folder / "README.txt").write_text(
        "Synthetic abdominal phantom generated by segevalkit.app.demo.make_phantom (not anatomy, not patient data).\n")
    return paths


def demo_session(folder: Optional[Union[str, Path]] = None, **kwargs):
    """A `ViewerSession` on the phantom with its three synthetic models."""
    from .session import ViewerSession

    p = make_phantom(folder)
    preds = {k: v for k, v in p.items() if k.startswith("Model")}
    return ViewerSession(p["image"], ref=p["ref"], preds=preds, labels=PHANTOM_LABELS, case="phantom",
                         min_lesion_voxels=5, **kwargs)
