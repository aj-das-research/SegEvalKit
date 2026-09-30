"""Slice-based error overlays."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..plotting.theme import ERROR_COLORS, INK, theme

__all__ = ["WINDOWS", "apply_window", "to_canonical", "pick_slice", "error_overlay", "triplanar",
           "slice_montage", "error_projection", "case_gallery", "error_legend"]

#: CT display windows as (level, width) in HU.
WINDOWS: Dict[str, Tuple[float, float]] = {
    "abdomen": (40, 400), "soft_tissue": (50, 350), "liver": (60, 160), "pancreas": (40, 300),
    "lung": (-600, 1500), "bone": (400, 1800), "brain": (40, 80), "mediastinum": (50, 350),
    "angio": (300, 600),
}

# Axis index of each view in RAS space (sagittal = x, coronal = y, axial = z).
_VIEW_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def apply_window(image: np.ndarray, window: Union[str, Tuple[float, float], None] = "abdomen") -> np.ndarray:
    """Map intensities to [0, 1]. ``window``: preset name, ``(level, width)``, or ``None`` / ``"auto"``
    for a 0.5-99.5 percentile stretch (MR)."""
    img = np.asarray(image, dtype=np.float32)
    if window is None or window == "auto":
        lo, hi = np.percentile(img, [0.5, 99.5])
    else:
        level, width = WINDOWS[window] if isinstance(window, str) else window
        lo, hi = level - width / 2, level + width / 2
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def to_canonical(data: np.ndarray, affine: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient an array to RAS+ using its affine; returns ``(data, spacing)``."""
    if affine is None:
        return data, np.ones(3)
    import nibabel as nib

    ornt = nib.orientations.io_orientation(affine)
    out = nib.orientations.apply_orientation(data, ornt)
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    spacing = zooms[ornt[:, 0].astype(int).argsort()]
    return out, spacing


def _slice(vol: np.ndarray, view: str, index: int) -> np.ndarray:
    """2D slice in radiological display orientation (rows top→bottom, cols screen left→right)."""
    ax = _VIEW_AXIS[view]
    s = np.take(vol, index, axis=ax)
    # Remaining axes: sagittal (y, z), coronal (x, z), axial (x, y).
    # Transpose so the second remaining axis (A for axial, S otherwise) is
    # vertical and flip it so it increases upwards; then flip columns: for
    # axial/coronal this puts the patient's right (+x in RAS) on the screen's
    # left (radiological convention), for sagittal it puts anterior on the left.
    return s.T[::-1, ::-1]


def _aspect(spacing, view: str) -> float:
    ax = _VIEW_AXIS[view]
    rest = [s for i, s in enumerate(spacing) if i != ax]
    return rest[1] / rest[0]


def pick_slice(pred: Optional[np.ndarray], ref: np.ndarray, view: str = "axial", mode: str = "error") -> int:
    """Choose a slice index: ``"error"`` (most FP+FN voxels), ``"ref"`` (largest
    reference area) or ``"center"`` (reference centroid)."""
    ax = _VIEW_AXIS[view]
    other = tuple(i for i in range(3) if i != ax)
    if mode == "error" and pred is not None:
        err = (pred != ref).sum(axis=other)
        if err.any():
            return int(np.argmax(err))
        mode = "ref"
    if mode == "center" and ref.any():
        return int(round(np.argwhere(ref)[:, ax].mean()))
    area = ref.sum(axis=other)
    if area.any():
        return int(np.argmax(area))
    return ref.shape[ax] // 2


def _prep(image, pred, ref, affine):
    arrays = [a for a in (image, pred, ref) if a is not None]
    shape = arrays[0].shape
    for a in arrays:
        if a.shape != shape:
            raise ValueError("image, pred and ref must share a shape")
    out = []
    spacing = np.ones(3)
    for a in (image, pred, ref):
        if a is None:
            out.append(None)
        else:
            c, spacing = to_canonical(np.asarray(a), affine)
            out.append(c)
    return out[0], out[1], out[2], spacing


def _rgba(mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    from matplotlib.colors import to_rgba

    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[mask] = to_rgba(color, alpha)
    return rgba


def _draw(ax, img2d, p2d, r2d, aspect, window, style="fill", alpha=0.45, zoom=None, title=None):
    if img2d is not None:
        ax.imshow(apply_window(img2d, window), cmap="gray", aspect=aspect, interpolation="bilinear")
    else:
        ax.imshow(np.zeros(r2d.shape), cmap="gray", aspect=aspect, vmin=0, vmax=1)
    if style in ("fill", "both") and r2d is not None:
        if p2d is None:
            ax.imshow(_rgba(r2d, ERROR_COLORS["ref"], alpha), aspect=aspect, interpolation="nearest")
        else:
            tp, fn, fp = p2d & r2d, r2d & ~p2d, p2d & ~r2d
            for m, key in ((tp, "tp"), (fn, "fn"), (fp, "fp")):
                if m.any():
                    ax.imshow(_rgba(m, ERROR_COLORS[key], alpha if key == "tp" else min(1.0, alpha + 0.3)),
                              aspect=aspect, interpolation="nearest")
    if style in ("contour", "both"):
        if r2d is not None and r2d.any():
            ax.contour(r2d.astype(float), levels=[0.5], colors=[ERROR_COLORS["ref"]], linewidths=1.4)
        if p2d is not None and p2d.any():
            ax.contour(p2d.astype(float), levels=[0.5], colors=[ERROR_COLORS["fp"]], linewidths=1.2,
                       linestyles="--")
    if zoom is not None:
        (r0, r1), (c0, c1) = zoom
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, loc="left")


def _zoom_box(masks, margin: int = 12):
    m = np.zeros_like(masks[0], dtype=bool)
    for x in masks:
        if x is not None:
            m |= x
    if not m.any():
        return None
    rr, cc = np.nonzero(m)
    return ((max(rr.min() - margin, 0), min(rr.max() + margin, m.shape[0] - 1)),
            (max(cc.min() - margin, 0), min(cc.max() + margin, m.shape[1] - 1)))


def _window_note(window, has_image: bool) -> str:
    if not has_image:
        return "Background: no image given"
    if window is None:
        return "Grey: image intensity (full range)"
    if isinstance(window, str):
        lvl, wid = WINDOWS[window]
        return f"Grey: CT, {window.replace('_', ' ')} window (L {lvl:g} / W {wid:g} HU)"
    lo, hi = window
    return f"Grey: image, window {lo:g} to {hi:g}"


def _legend(fig, show_pred: bool, style: str = "fill", note: Optional[str] = None):
    """Figure legend explaining every colour and line, placed below the axes.

    ``note`` is a first line of notation (what the grey background shows).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    if not show_pred:
        handles.append(Patch(color=ERROR_COLORS["ref"], label="Reference"))
    elif style in ("fill", "both"):
        handles += [Patch(color=ERROR_COLORS["tp"], label="Agreement (TP)"),
                    Patch(color=ERROR_COLORS["fn"], label="Missed: under-segmented (FN)"),
                    Patch(color=ERROR_COLORS["fp"], label="Added: over-segmented (FP)")]
    if show_pred and style in ("contour", "both"):
        handles += [Line2D([], [], color=ERROR_COLORS["ref"], lw=1.6, label="Reference outline"),
                    Line2D([], [], color=ERROR_COLORS["fp"], lw=1.4, ls="--", label="Prediction outline")]
    # "outside" reserves space below the axes in the constrained layout, so the legend
    # never overlaps an image.
    ncol = 1 if fig.get_figwidth() < 6 else min(len(handles), 3)
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False,
               fontsize=8, title=note, title_fontsize=7.5)


def error_legend(fig, *, style: str = "fill", window="abdomen", has_image: bool = True, show_pred: bool = True,
                 note: Optional[str] = None):
    """Add the standard error-colour key below a figure you composed yourself
    (e.g. a grid of `error_overlay` panels drawn with ``legend=False``)."""
    extra = f" · {note}" if note else ""
    _legend(fig, show_pred, style, _window_note(window, has_image) + extra)
    return fig


def error_overlay(image: Optional[np.ndarray], pred: Optional[np.ndarray], ref: np.ndarray, *,
                  affine: Optional[np.ndarray] = None, view: str = "axial", index: Optional[int] = None,
                  window="abdomen", style: str = "fill", zoom: bool = True, ax=None, title: Optional[str] = None,
                  legend: bool = True, figsize=(4.2, 5.2)):
    """One slice with TP / FN / FP colour-coded (``style="fill"``), or as reference
    (lavender) and prediction (teal, dashed) contours (``style="contour"``).

    Colours: violet = agreement (TP), orange = missed reference voxels
    (under-segmentation, FN), teal = added voxels (over-segmentation, FP); the
    grey background is the image in the given display window. ``legend=True``
    (default, own figure only) writes this below the image."""
    import matplotlib.pyplot as plt

    img, p, r, sp = _prep(image, None if pred is None else pred.astype(bool), ref.astype(bool), affine)
    index = pick_slice(p, r, view) if index is None else index
    with theme():
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize, constrained_layout=True)
        s = lambda a: None if a is None else _slice(a, view, index)  # noqa: E731
        box = _zoom_box([s(r), s(p)]) if zoom else None
        _draw(ax, s(img), s(p), s(r), _aspect(sp, view), window, style=style, zoom=box,
              title=title or f"{view.capitalize()} slice {index}")
        if legend and len(fig.axes) == 1:
            _legend(fig, pred is not None, style, _window_note(window, image is not None))
    return fig


def triplanar(image, pred, ref, *, affine=None, window="abdomen", mode: str = "error", style: str = "fill",
              zoom: bool = True, title: Optional[str] = None, legend: bool = True, figsize=(10.5, 4.1)):
    """Axial, coronal and sagittal slices through the region of largest error.

    Same colours as `error_overlay`; ``legend=False`` drops the colour key."""
    import matplotlib.pyplot as plt

    img, p, r, sp = _prep(image, None if pred is None else pred.astype(bool), ref.astype(bool), affine)
    with theme():
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        for ax, view in zip(axes, ("axial", "coronal", "sagittal")):
            idx = pick_slice(p, r, view, mode)
            s = lambda a: None if a is None else _slice(a, view, idx)  # noqa: E731
            box = _zoom_box([s(r), s(p)]) if zoom else None
            _draw(ax, s(img), s(p), s(r), _aspect(sp, view), window, style=style, zoom=box,
                  title=f"{view.capitalize()} · {idx}")
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK["primary"])
        if legend:
            _legend(fig, pred is not None, style, _window_note(window, image is not None))
    return fig


def slice_montage(image, pred, ref, *, affine=None, view: str = "axial", n: int = 8, window="abdomen",
                  style: str = "fill", zoom: bool = True, title: Optional[str] = None, ncols: int = 4,
                  legend: bool = True):
    """``n`` evenly spaced slices through the extent of the reference ∪ prediction.

    Panel titles are slice indices; colours as in `error_overlay`."""
    import matplotlib.pyplot as plt

    img, p, r, sp = _prep(image, None if pred is None else pred.astype(bool), ref.astype(bool), affine)
    ax_i = _VIEW_AXIS[view]
    other = tuple(i for i in range(3) if i != ax_i)
    u = r | p if p is not None else r
    present = np.nonzero(u.any(axis=other))[0]
    if present.size == 0:
        present = np.arange(r.shape[ax_i])
    idxs = np.unique(np.linspace(present.min(), present.max(), n).round().astype(int))
    nrows = int(np.ceil(len(idxs) / ncols))
    with theme():
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.7 * nrows + 0.4), constrained_layout=True,
                                 squeeze=False)
        # One zoom box for the whole montage so slices are comparable.
        proj = u.any(axis=ax_i)
        box = None
        if zoom:
            box = _zoom_box([_slice(np.expand_dims(proj, ax_i), view, 0)])
        for k, ax in enumerate(axes.ravel()):
            if k >= len(idxs):
                ax.axis("off")
                continue
            s = lambda a: None if a is None else _slice(a, view, idxs[k])  # noqa: E731
            _draw(ax, s(img), s(p), s(r), _aspect(sp, view), window, style=style, zoom=box, title=f"{idxs[k]}")
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK["primary"])
        if legend:
            note = _window_note(window, image is not None) + f" · panel titles: {view} slice index"
            _legend(fig, pred is not None, style, note)
    return fig


def error_projection(pred, ref, *, affine=None, title: Optional[str] = None, legend: bool = True,
                     figsize=(10.5, 3.9)):
    """Maximum-intensity projections of FN and FP voxels along the three axes.

    A one-glance summary of *where in 3D* a case fails, useful for spotting
    distant false-positive islands that a single slice would miss. A pixel is
    coloured when any voxel along the line of sight has that label: light grey
    is the reference's silhouette, violet agreement, and orange (missed) and
    teal (added) are drawn on top, so an error anywhere along the ray shows.
    """
    import matplotlib.pyplot as plt

    _, p, r, sp = _prep(None, pred.astype(bool), ref.astype(bool), affine)
    with theme():
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        for ax, view in zip(axes, ("axial", "coronal", "sagittal")):
            a = _VIEW_AXIS[view]
            proj = lambda m: _slice(np.expand_dims(m.any(axis=a), a), view, 0)  # noqa: E731
            ref2, tp2, fn2, fp2 = proj(r), proj(p & r), proj(r & ~p), proj(p & ~r)
            asp = _aspect(sp, view)
            ax.imshow(np.full(ref2.shape, 1.0), cmap="gray", vmin=0, vmax=1, aspect=asp)
            ax.imshow(_rgba(ref2, INK["neutral"], 1.0), aspect=asp, interpolation="nearest")
            ax.imshow(_rgba(tp2, ERROR_COLORS["tp"], 0.25), aspect=asp, interpolation="nearest")
            ax.imshow(_rgba(fn2, ERROR_COLORS["fn"], 0.9), aspect=asp, interpolation="nearest")
            ax.imshow(_rgba(fp2, ERROR_COLORS["fp"], 0.9), aspect=asp, interpolation="nearest")
            box = _zoom_box([ref2, fn2 | fp2 | tp2], margin=6)
            if box:
                ax.set_xlim(box[1][0], box[1][1])
                ax.set_ylim(box[0][1], box[0][0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{view.capitalize()} projection", fontsize=9, loc="left")
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK["primary"])
        if legend:
            from matplotlib.patches import Patch

            handles = [Patch(facecolor=INK["neutral"], edgecolor=INK["muted"], lw=0.6, label="Reference silhouette"),
                       Patch(color=ERROR_COLORS["tp"], alpha=0.35, label="Agreement (TP)"),
                       Patch(color=ERROR_COLORS["fn"], label="Missed: under-segmented (FN)"),
                       Patch(color=ERROR_COLORS["fp"], label="Added: over-segmented (FP)")]
            fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8,
                       title="Projection through the whole volume: a pixel shows a colour if any voxel along "
                             "the line of sight has it (errors drawn on top)", title_fontsize=7.5)
    return fig


def case_gallery(items: Sequence[Dict], *, view: str = "axial", window="abdomen", style: str = "fill",
                 ncols: int = 4, title: Optional[str] = None, legend: bool = True):
    """Grid of error overlays for several cases.

    ``items``: dicts with keys ``image`` (optional), ``pred``, ``ref``,
    ``affine`` (optional) and ``title`` (e.g. ``"case_012 · DSC 0.41"``).
    Typical use is the *k* worst cases from
    `worst_cases`. Each panel shows the slice with the most error; colours as
    in `error_overlay`.
    """
    import matplotlib.pyplot as plt

    n = len(items)
    nrows = int(np.ceil(n / ncols))
    with theme():
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 3.0 * nrows + 0.4), constrained_layout=True,
                                 squeeze=False)
        for ax, it in zip(axes.ravel(), items):
            img, p, r, sp = _prep(it.get("image"), it["pred"].astype(bool), it["ref"].astype(bool), it.get("affine"))
            idx = pick_slice(p, r, view)
            s = lambda a: None if a is None else _slice(a, view, idx)  # noqa: E731
            _draw(ax, s(img), s(p), s(r), _aspect(sp, view), window, style=style,
                  zoom=_zoom_box([s(r), s(p)]), title=it.get("title"))
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK["primary"])
        if legend:
            has_img = any(it.get("image") is not None for it in items)
            _legend(fig, True, style, _window_note(window, has_img) + f" · each panel: {view} slice with most error")
    return fig
