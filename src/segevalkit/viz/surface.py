"""3D surface-distance maps: where on the organ is the boundary wrong, and in which direction?"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from ..metrics.context import bbox
from ..plotting.theme import INK, diverging_cmap, theme

__all__ = ["surface_distance_map", "signed_distance_at_surface"]


def _signed_distance(ref: np.ndarray, spacing) -> np.ndarray:
    """Signed distance to the reference boundary in mm: > 0 outside (over-segmentation), < 0 inside."""
    outside = ndimage.distance_transform_edt(~ref, sampling=spacing)
    inside = ndimage.distance_transform_edt(ref, sampling=spacing)
    return outside - inside


def signed_distance_at_surface(pred: np.ndarray, ref: np.ndarray, spacing: Sequence[float] = (1, 1, 1),
                               step: int = 1):
    """Mesh the prediction and sample the signed distance to the reference at every vertex.

    Returns ``(verts_mm, faces, dist_mm)``. Positive distances are where the
    prediction bulges outside the reference; negative where it falls short.
    """
    from skimage.measure import marching_cubes

    pred = np.asarray(pred, bool)
    ref = np.asarray(ref, bool)
    crop = bbox(pred | ref, margin=3)
    p = np.pad(pred[crop], 1)
    r = np.pad(ref[crop], 1)
    sdt = _signed_distance(r, spacing)
    verts, faces, _, _ = marching_cubes(p.astype(np.float32), level=0.5, step_size=step)
    dist = ndimage.map_coordinates(sdt, verts.T, order=1, mode="nearest")
    offset = np.array([s.start for s in crop]) - 1
    verts_mm = (verts + offset) * np.asarray(spacing)
    return verts_mm, faces, dist


def surface_distance_map(pred: np.ndarray, ref: np.ndarray, spacing: Sequence[float] = (1, 1, 1), *,
                         clip_mm: Optional[float] = None, backend: str = "matplotlib", step: int = 1,
                         title: Optional[str] = None, views: Sequence[tuple] = ((20, -60), (20, 120)),
                         html_path: Optional[str] = None, figsize=(9.0, 4.2)):
    """Render the predicted surface coloured by signed distance to the reference.

    Args:
        clip_mm: Colour range ±clip (default: 95th percentile of |distance|, at least 2 mm).
        backend: ``"matplotlib"`` (static, two viewpoints) or ``"plotly"``
            (interactive; saved to ``html_path`` when given).
        step: Marching-cubes step size; >1 for faster previews of large organs.

    Colour: purple = prediction outside the reference (over-segmentation),
    orange = inside (under-segmentation), grey = on the boundary.
    """
    verts, faces, dist = signed_distance_at_surface(pred, ref, spacing, step)
    if clip_mm is None:
        clip_mm = max(2.0, float(np.percentile(np.abs(dist), 95)))
    cmap = diverging_cmap()
    if backend == "plotly":
        import plotly.graph_objects as go

        stops = [[i / 6, c] for i, c in enumerate(["#b4541c", "#e2712f", "#f3c3a3", INK["neutral"],
                                                    "#d6c7fb", "#6d3fd6", "#2e1766"])]
        fig = go.Figure(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=np.clip(dist, -clip_mm, clip_mm), cmin=-clip_mm, cmax=clip_mm, colorscale=stops,
            colorbar=dict(title="Signed distance [mm]"), flatshading=False,
            lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15),
            hovertemplate="%{intensity:.2f} mm<extra></extra>"))
        fig.update_layout(title=title or "Surface distance to reference", scene=dict(aspectmode="data"),
                          paper_bgcolor="white", font=dict(color=INK["primary"]))
        if html_path:
            fig.write_html(html_path, include_plotlyjs="cdn")
        return fig
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    face_d = dist[faces].mean(axis=1)
    norm = plt.Normalize(-clip_mm, clip_mm)
    colors = cmap(norm(face_d))
    with theme():
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        for k, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
            mesh = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none", linewidth=0)
            ax.add_collection3d(mesh)
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi - lo)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=fig.axes, fraction=0.03, pad=0.02, shrink=0.8)
        cb.set_label("Signed distance to reference [mm]\n← under-segmented · over-segmented →")
        cb.outline.set_visible(False)
        fig.suptitle(title or "Surface distance to reference", x=0.01, ha="left", fontsize=11,
                     fontweight="semibold", color=INK["primary"])
    return fig
