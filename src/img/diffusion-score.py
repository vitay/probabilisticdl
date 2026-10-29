"""Score of the noisy distributions q(x_t) of the forward process on the 2D mixture of `score.py` (diffusion chapter).

At high noise levels, q(x_t) is close to a standard normal distribution and its score is a simple field pointing to the
origin, defined everywhere; as the noise decreases, the score progressively separates the two modes.
Uses the exact functions of `diffusion-forward.py`.

Run from anywhere: `python src/img/diffusion-score.py` writes `diffusion-score.png` next to this file.
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
forward = importlib.import_module("diffusion-forward")

STEPS = [1000, 400, 200, 0]

# Palette: categorical slots 1-2 of the dataviz reference palette, plus text/grid tokens
DENSITY, SCORE = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(1, len(STEPS), figsize=(15, 4.1), sharey=True)
g = np.linspace(-5, 5, 300)
G1, G2 = np.meshgrid(g, g)
grid = np.stack([G1, G2], axis=-1)
q = np.linspace(-4.5, 4.5, 14)
Q1, Q2 = np.meshgrid(q, q)
points = np.stack([Q1, Q2], axis=-1)

for ax, t in zip(axes, STEPS):
    ax.contour(G1, G2, np.log(forward.density(grid, t)), levels=10, colors=DENSITY, linewidths=1,
               linestyles="solid", alpha=0.5)
    s = forward.score(points, t)
    ax.quiver(Q1, Q2, s[..., 0], s[..., 1], color=SCORE, angles="xy", width=0.005)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_title(f"Score of $q(x_t)$, $t = {t}$", color=TEXT, fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("$x_2$")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
