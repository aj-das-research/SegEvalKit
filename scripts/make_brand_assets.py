"""Render the README banner (docs/assets/banner.png) in the SegEvalKit brand: STIX serif, purple gradient,
the overlap mark. Run: python scripts/make_brand_assets.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
W, H = 12.0, 2.6


def lens(cx1, cx2, cy, r):
    d = (cx2 - cx1) / 2
    y = np.sqrt(r * r - d * d)
    t1 = np.linspace(np.arcsin(y / r), -np.arcsin(y / r), 60)
    right = np.c_[cx1 + r * np.cos(t1), cy + r * np.sin(t1)]
    t2 = np.linspace(np.pi + np.arcsin(-y / r) * -1, np.pi - np.arcsin(y / r) * -1, 60)
    left = np.c_[cx2 + r * np.cos(np.linspace(np.pi + np.arcsin(y / r), np.pi - np.arcsin(y / r), 60)),
                 cy + r * np.sin(np.linspace(np.pi + np.arcsin(y / r), np.pi - np.arcsin(y / r), 60))]
    verts = np.r_[right, left, right[:1]]
    return MPath(verts, [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 2) + [MPath.CLOSEPOLY])


def banner():
    plt.rcParams.update({"font.family": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(W, H), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    # gradient background inside a rounded rectangle
    g = np.linspace(0, 1, 512)[None, :]
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("b", ["#1c0f40", "#2e1766", "#472594", "#6d3fd6"])
    im = ax.imshow(np.vstack([g] * 8), extent=[0, W, 0, H], cmap=cmap, aspect="auto", zorder=0)
    clip = FancyBboxPatch((0.02, 0.02), W - 0.04, H - 0.04, boxstyle="round,pad=0,rounding_size=0.28",
                          transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    # the mark: two contours and their filled overlap (lens under the strokes)
    cy, r = H / 2 + 0.02, 0.5
    c1, c2 = 1.05, 1.53
    ax.add_patch(PathPatch(lens(c1, c2, cy, r), fc="#ffffff", ec="none", zorder=3, alpha=0.95))
    ax.add_patch(Circle((c2, cy), r, fill=False, lw=4.2, ec="#b89cf5", zorder=4))
    ax.add_patch(Circle((c1, cy), r, fill=False, lw=4.2, ec="#ffffff", zorder=5))
    # wordmark and tagline
    ax.text(2.45, cy + 0.2, "SegEvalKit", fontsize=50, color="white", fontweight="bold", va="center", zorder=5)
    ax.text(2.48, cy - 0.5, "Explicit, tested evaluation of volumetric medical image segmentation",
            fontsize=17, color="#e6dcff", style="italic", va="center", zorder=5)
    chips = ["56 metrics", "CT · MR · NIfTI", "GPU distances", "statistics & ranking", "plots · reports"]
    x = 2.5
    for c in chips:
        t = ax.text(x + 0.12, 0.42, c, fontsize=11.5, color="white", va="center", zorder=6)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        ax.add_patch(FancyBboxPatch((x, 0.27), bb.width + 0.24, 0.3, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc="#ffffff22", ec="#ffffff55", lw=0.8, zorder=5))
        x += bb.width + 0.42
    fig.savefig(OUT / "banner.png", dpi=220, transparent=True)
    print("wrote", OUT / "banner.png")


if __name__ == "__main__":
    banner()
