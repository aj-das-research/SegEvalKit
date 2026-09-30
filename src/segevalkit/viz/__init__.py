"""Qualitative visualisation of segmentation errors.

Numbers say *how much* is wrong; pictures say *what* is wrong. All views use
the same error encoding (see :data:`segevalkit.plotting.theme.ERROR_COLORS`):

* **violet**: true positive (agreement),
* **orange**: false negative (reference tissue the model missed),
* **teal**: false positive (tissue the model added),

and show slices in radiological convention (patient right on screen left)
after reorienting to RAS with the NIfTI affine, with the correct physical
aspect ratio from the voxel spacing.
"""

from .overlay import (
    WINDOWS,
    apply_window,
    case_gallery,
    error_overlay,
    error_projection,
    pick_slice,
    slice_montage,
    to_canonical,
    triplanar,
)
from .surface import surface_distance_map

__all__ = [
    "WINDOWS",
    "apply_window",
    "case_gallery",
    "error_overlay",
    "error_projection",
    "pick_slice",
    "slice_montage",
    "surface_distance_map",
    "to_canonical",
    "triplanar",
]
