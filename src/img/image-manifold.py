"""A manifold of images: 32x32 images of a "7" varying only by their rotation and their size (latent variables chapter).

Each image is a point in the 1024-dimensional pixel space. As only two factors vary, the images lie on a 2D surface.
Left: projection of the images onto the first three principal components of the pixel space (PCA), with a few images.
Right: the same images arranged by their two factors (the latent space).

Run from anywhere: `python src/img/image-manifold.py` writes `image-manifold.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageDraw, ImageFilter
from sklearn.decomposition import PCA

SIZE = 32  # images are SIZE x SIZE pixels
UPSAMPLING = 4  # drawn at a higher resolution, then downsampled (anti-aliasing)
ANGLES = np.radians(np.linspace(-90, 90, 41))  # first factor: rotation
SCALES = np.linspace(0.5, 1.0, 21)  # second factor: size


def seven(angle, scale):
    """Grayscale image (values in [0, 1]) of a "7" rotated by `angle` and scaled by `scale`."""
    big = SIZE * UPSAMPLING
    image = Image.new("L", (big, big), 0)
    corners = np.array([[-0.45, -0.55], [0.45, -0.55], [-0.1, 0.6]]) * scale * big * 0.75  # top bar, then diagonal
    c, s = np.cos(angle), np.sin(angle)
    corners = corners @ np.array([[c, -s], [s, c]]).T + big / 2
    ImageDraw.Draw(image).line([tuple(p) for p in corners], fill=255, width=int(big * 0.09 * scale), joint="curve")
    image = image.filter(ImageFilter.GaussianBlur(big * 0.05))
    return np.asarray(image.resize((SIZE, SIZE), Image.LANCZOS), dtype=float) / 255


images = np.array([[seven(a, s) for a in ANGLES] for s in SCALES])  # (scales, angles, SIZE, SIZE)
pca = PCA(3).fit(images.reshape(-1, SIZE * SIZE))
points = pca.transform(images.reshape(-1, SIZE * SIZE)).reshape(len(SCALES), len(ANGLES), 3)

# Palette: categorical slot 1 of the dataviz reference palette, plus text/grid tokens
SURFACE = "#2a78d6"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig = plt.figure(figsize=(15, 6.5))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax_grid = fig.add_subplot(1, 2, 2)

# Left: the surface in the space of the first three principal components
ax3d.plot_surface(points[..., 0], points[..., 1], points[..., 2], color=SURFACE, alpha=0.75, shade=True,
                  rstride=1, cstride=1, lw=0.2, edgecolor="white")
ax3d.view_init(elev=40, azim=120)
ax3d.set_xlabel("PC 1")
ax3d.set_ylabel("PC 2")
ax3d.set_zlabel("PC 3")
ax3d.set_xticklabels([])
ax3d.set_yticklabels([])
ax3d.set_zticklabels([])
ax3d.set_title("Images in the pixel space (projected to 3D)", color=TEXT, fontsize=11, pad=30)

# A few images attached to their point on the surface
fig.canvas.draw()
MID = len(ANGLES) // 2  # index of the unrotated images
# (size index, rotation index): offset of the thumbnail in points, chosen by hand to avoid overlaps
shown = {(-1, 0): (-55, 20), (-1, -1): (50, 40), (-1, MID): (-60, -45), (0, MID): (-5, -65), (0, -1): (60, -45)}
for (i, j), offset in shown.items():
    x2, y2, _ = proj3d.proj_transform(*points[i, j], ax3d.get_proj())
    box = AnnotationBbox(OffsetImage(images[i, j], cmap="gray_r", zoom=1.4), (x2, y2), xybox=offset,
                         xycoords="data", boxcoords="offset points", pad=0.2,
                         bboxprops={"edgecolor": MUTED, "lw": 0.8},
                         arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax3d.add_artist(box)
    ax3d.plot(*points[i, j], "o", color=TEXT, ms=4, zorder=10)

# Right: the same images arranged by their two factors
rows, cols = SCALES[::-5], ANGLES[::5]  # largest size on top
mosaic = np.vstack([np.hstack([np.pad(seven(a, s), 1, constant_values=0) for a in cols]) for s in rows])
ax_grid.imshow(mosaic, cmap="gray_r", extent=(-0.5, len(cols) - 0.5, -0.5, len(rows) - 0.5))
ax_grid.set_xticks(range(len(cols)), [f"{np.degrees(a):.0f}°" for a in cols])
ax_grid.set_yticks(range(len(rows)), [f"{s:.2f}" for s in rows[::-1]])
ax_grid.set_xlabel("latent variable $z_1$: rotation")
ax_grid.set_ylabel("latent variable $z_2$: size")
ax_grid.set_title("The same images in the 2D latent space", color=TEXT, fontsize=11)
for side in ["top", "right", "left", "bottom"]:
    ax_grid.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
