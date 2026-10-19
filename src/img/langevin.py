"""Langevin dynamics on the 2D mixture of two Gaussians of `score.py` (HMC chapter).

400 independent chains start on a regular grid; snapshots of their positions at a few steps.
Inspired by the animation of Yang Song's blog post <https://yang-song.net/blog/2021/score/>.

Run from anywhere: `python src/img/langevin.py` writes `langevin.png` next to this file.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import COMPONENTS_2D, p_2d, score_2d  # noqa: E402

EPSILON = 0.05  # step size
SNAPSHOTS = [0, 10, 100, 1000]

# Initial states on a regular 20 x 20 grid
g = np.linspace(-4.5, 4.5, 20)
x = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)

rng = np.random.default_rng(0)
positions = {0: x.copy()}
for t in range(1, SNAPSHOTS[-1] + 1):
    x = x + EPSILON * score_2d(x) + np.sqrt(2 * EPSILON) * rng.normal(size=x.shape)
    if t in SNAPSHOTS:
        positions[t] = x.copy()

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
TARGET, SAMPLE = "#2a78d6", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(1, len(SNAPSHOTS), figsize=(16, 4.4), sharey=True)
c = np.linspace(-5, 5, 300)
C1, C2 = np.meshgrid(c, c)
log_p = np.log(p_2d(np.stack([C1, C2], axis=-1)))

for ax, t in zip(axes, SNAPSHOTS):
    ax.contour(C1, C2, log_p, levels=12, colors=TARGET, linewidths=1, linestyles="solid", alpha=0.4)
    ax.scatter(positions[t][:, 0], positions[t][:, 1], color=SAMPLE, s=8, alpha=0.8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_title(f"$t = {t}$", color=TEXT, fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("$x_2$")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
