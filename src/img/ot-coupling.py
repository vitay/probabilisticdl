"""Independent vs. optimal coupling between samples of two distributions (optimal transport chapter).

N samples of a standard normal distribution (source) and N samples of the 2D mixture of `score.py` (target) are
paired either at random (independent coupling) or by the optimal assignment minimizing the sum of squared distances.

Run from anywhere: `python src/img/ot-coupling.py` writes `ot-coupling.png` next to this file.
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
forward = importlib.import_module("diffusion-forward")

N = 40

rng = np.random.default_rng(0)
source = rng.normal(size=(N, 2))
target = forward.sample_data(N, rng)

random_pairs = rng.permutation(N)
cost_matrix = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=-1)
_, optimal_pairs = linear_sum_assignment(cost_matrix)
couplings = [("Independent coupling (random pairs)", random_pairs),
             ("Optimal coupling (optimal assignment)", optimal_pairs)]

# Palette: categorical slots 1-2 of the dataviz reference palette, plus text/grid tokens
SOURCE, TARGET = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
for ax, (title, pairs) in zip(axes, couplings):
    cost = np.mean(np.sum((source - target[pairs]) ** 2, axis=1))
    print(f"{title}: average squared distance {cost:.2f}")
    for i in range(N):
        ax.plot([source[i, 0], target[pairs[i], 0]], [source[i, 1], target[pairs[i], 1]], color=MUTED, lw=0.8,
                alpha=0.7)
    ax.scatter(source[:, 0], source[:, 1], color=SOURCE, s=25, zorder=3, label="samples of $p$ (normal)")
    ax.scatter(target[:, 0], target[:, 1], color=TARGET, s=25, zorder=3, label="samples of $q$ (mixture)")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_title(f"{title}\naverage squared distance: {cost:.1f}", color=TEXT, fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("$x_2$")
axes[0].legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
