"""Mixture vs. displacement interpolation between two normal distributions (optimal transport chapter).

p = N(-2, 0.5) and q = N(2, 1). The mixture interpolation (1 - t) p + t q fades one distribution out and the other in;
the displacement interpolation moves each sample along a straight line, x_t = (1 - t) x + t T(x), with the optimal map
T: between normal distributions, it is again normal, with mean (1 - t) mu_p + t mu_q and std (1 - t) s_p + t s_q.

Run from anywhere: `python src/img/ot-interpolation.py` writes `ot-interpolation.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MU_P, S_P = -2.0, 0.5
MU_Q, S_Q = 2.0, 1.0
TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


# Palette: categorical slots 1-2 of the dataviz reference palette, plus text/grid tokens
P, Q = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

x = np.linspace(-4.5, 6, 800)
interpolations = {
    "Mixture: $(1 - t) \\, p + t \\, q$":
        lambda t: (1 - t) * gaussian(x, MU_P, S_P) + t * gaussian(x, MU_Q, S_Q),
    "Displacement: $x_t = (1 - t) \\, x + t \\, T(x)$":
        lambda t: gaussian(x, (1 - t) * MU_P + t * MU_Q, (1 - t) * S_P + t * S_Q),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
offset = 0.9  # vertical offset between successive times
for ax, (title, density) in zip(axes, interpolations.items()):
    for k, t in enumerate(TIMES):
        color = (1 - t) * np.array(plt.matplotlib.colors.to_rgb(P)) + t * np.array(plt.matplotlib.colors.to_rgb(Q))
        base = (len(TIMES) - 1 - k) * offset  # t = 0 at the top
        ax.fill_between(x, base, base + density(t), color=color, alpha=0.35, lw=0)
        ax.plot(x, base + density(t), color=color, lw=1.8)
        ax.text(-4.4, base + 0.08, f"$t = {t:g}$", color=MUTED, fontsize=10)
    ax.set_xlim(-4.5, 6)
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_title(title, color=TEXT, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
