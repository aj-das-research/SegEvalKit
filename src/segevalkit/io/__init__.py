"""Input/output: volumes with geometry, label specifications and case discovery."""

from .cases import Case, Source, audit_geometry, discover_cases
from .labels import LabelSpec, extract_mask, labels_from_map, parse_labels
from .volume import Volume, check_alignment, load_volume, resample_to, save_volume

__all__ = [
    "Case",
    "audit_geometry",
    "Source",
    "discover_cases",
    "LabelSpec",
    "extract_mask",
    "labels_from_map",
    "parse_labels",
    "Volume",
    "check_alignment",
    "load_volume",
    "resample_to",
    "save_volume",
]
