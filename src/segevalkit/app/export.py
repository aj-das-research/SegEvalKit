"""Export a `ViewerSession` as one self-contained HTML file (no Python, no network needed to open it).

The page embeds NiiVue, the viewer script, the manifest, and every volume and
mesh the viewer can request, base64-encoded. Volumes are cropped around the
structures and downsampled (``max_dim``) to keep the file small enough to
e-mail; the 3D view shows every structure, and the error surfaces
(TP / FN / FP) of each model for the lesion structures and for ``focus``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable, Optional, Union

from .server import _sanitize, page_html
from .session import ERROR_CODES, ViewerSession

__all__ = ["export_html"]


def export_html(session: ViewerSession, out: Union[str, Path], focus: Optional[Iterable[str]] = None,
                meshes: bool = True, error_meshes: str = "lesions", mesh_step: int = 2) -> Path:
    """Write the viewer and all its data into one HTML file.

    Args:
        session: A session, ideally built with ``crop_margin_mm`` and
            ``max_dim`` (e.g. 20 mm and 224) so the file stays small.
        out: Output ``.html`` path.
        focus: Structures whose TP/FN/FP surfaces are embedded in addition
            to the lesions.
        meshes: Embed surface meshes for the 3D view.
        error_meshes: ``"lesions"`` (lesions + ``focus``), ``"all"`` or ``"none"``.
        mesh_step: Marching-cubes step for the embedded meshes (2 keeps the file small).

    Returns:
        The written path.
    """
    out = Path(out)
    session.mesh_step = int(mesh_step)
    files = {"vol/image.nii.gz": session.image_nifti()}
    if session.ref_masks:
        files["vol/labels/ref.nii.gz"] = session.labels_nifti("ref")
    for m in session.models:
        files[f"vol/labels/{m}.nii.gz"] = session.labels_nifti(m)
        for s in session.structures:
            files[f"vol/error/{m}/{s}.nii.gz"] = session.error_nifti(m, s)
    if meshes:
        sel = set(session.lesions) | set(focus or ())
        if error_meshes == "all":
            sel = set(session.structures)
        elif error_meshes == "none":
            sel = set()
        for s in session.structures:
            if s in session.ref_masks:
                files[f"mesh/ref/{s}.mz3"] = session.mesh("ref", s)
            for m in session.models:
                if s in session.pred_masks[m]:
                    files[f"mesh/{m}/{s}.mz3"] = session.mesh(m, s)
                if s in sel:
                    for part in ERROR_CODES:
                        files[f"mesh/{m}/{part}/{s}.mz3"] = session.mesh(m, s, part)
    manifest = session.manifest()
    manifest["live"] = {m: {s: session.live_metrics(m, s) for s in session.structures
                            if s in session.pred_masks[m] or s in session.ref_masks} for m in session.models}
    manifest["embedded_meshes"] = sorted(k for k in files if k.startswith("mesh/"))
    blob = {k: base64.b64encode(v).decode("ascii") for k, v in files.items()}
    embed = ("<script>window.SEK_MANIFEST = " + json.dumps(_sanitize(manifest)) + ";\n"
             "window.SEK_FILES = " + json.dumps(blob) + ";</script>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html(mode="static", embed=embed, inline_assets=True), encoding="utf-8")
    return out
