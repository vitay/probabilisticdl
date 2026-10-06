"""Inverse transform sampling of the exponential distribution (Monte Carlo chapter).

Run from anywhere: `python src/img/inverse-transform.py` writes `inverse-transform.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

lam = 1.0
N = 10000

rng = np.random.default_rng(42)
samples = -np.log(1 - rng.uniform(0, 1, N)) / lam

# Evenly spaced quantiles for the left panel: equal steps in u give unequal steps in x
u_grid = np.arange(0.05, 1.0, 0.1)
x_grid = -np.log(1 - u_grid) / lam

x = np.linspace(0, 5, 500)
pdf = lam * np.exp(-lam * x)
cdf = 1 - np.exp(-lam * x)

# Palette: categorical slots 1-2 of the dataviz reference palette, plus text/grid tokens
DIST, SAMPLE = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

# Left: from u on the y-axis to x = F^{-1}(u) on the x-axis
for u, xi in zip(u_grid, x_grid):
    ax1.plot([0, xi, xi], [u, u, 0], color=SAMPLE, lw=1, alpha=0.8)
ax1.plot(x, cdf, color=DIST, lw=2)
ax1.scatter(np.zeros_like(u_grid), u_grid, color=SAMPLE, s=25, zorder=3, clip_on=False)
ax1.scatter(x_grid, np.zeros_like(x_grid), color=SAMPLE, s=25, zorder=3, clip_on=False)
ax1.text(3.2, 0.88, "$F(x) = 1 - e^{-\\lambda x}$", color=DIST)
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 1.02)
ax1.set_xlabel("$x = F^{-1}(u)$")
ax1.set_ylabel("$u \\sim \\mathcal{U}(0, 1)$")
ax1.set_title("Evenly spaced $u$ give unevenly spaced $x$", color=TEXT)

# Right: histogram of the samples against the pdf
ax2.hist(samples, bins=np.linspace(0, 5, 41), density=True, color=SAMPLE, alpha=0.5, lw=0,
         label=f"histogram of {N} samples $x = F^{{-1}}(u)$")
ax2.plot(x, pdf, color=DIST, lw=2, label="$p(x) = \\lambda e^{-\\lambda x}$")
ax2.set_xlim(0, 5)
ax2.set_xlabel("$x$")
ax2.set_ylabel("density")
ax2.set_title(f"Exponential distribution, $\\lambda = {lam:g}$", color=TEXT)
ax2.legend(frameon=False, loc="upper right")

for ax in (ax1, ax2):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
