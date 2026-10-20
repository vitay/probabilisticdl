"""Cross-entropy between two distributions (information theory chapter).

Samples of p are drawn on the x-axis; their likelihood q(x_i) under the other distribution is shown as vertical segments.
The cross-entropy H(p, q) is the average of -log q(x_i): it is high when the samples of p are unlikely under q.

Run from anywhere: `python src/img/cross-entropy.py` writes `cross-entropy.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MU_P, SIGMA_P = 0.0, 1.0
MU_Q, SIGMA_Q = 2.5, 1.2
N = 12


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


samples = np.random.default_rng(1).normal(MU_P, SIGMA_P, N)

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
P, Q, LIK = "#2a78d6", "#1baf7a", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.linspace(-4, 7, 1000)
ax.plot(x, gaussian(x, MU_P, SIGMA_P), color=P, lw=2, label="$p(x)$")
ax.plot(x, gaussian(x, MU_Q, SIGMA_Q), color=Q, lw=2, label="$q(x)$")
ax.vlines(samples, 0, gaussian(samples, MU_Q, SIGMA_Q), color=LIK, lw=2,
          label="likelihood $q(x_i)$ of the samples of $p$")
ax.scatter(samples, np.zeros(N), color=P, s=30, zorder=3, clip_on=False, label="samples $x_i \\sim p(x)$")
ax.set_xlim(-4, 7)
ax.set_ylim(0, 0.45)
ax.set_xlabel("$x$")
ax.set_yticks([])
ax.legend(frameon=False, loc="upper right")
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
