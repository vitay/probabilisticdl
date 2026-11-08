"""Central limit theorem: distribution of the average of N samples of an exponential distribution (Bayesian chapter).

The average of N i.i.d. samples of an exponential distribution with rate 1 (mean 1, std 1) follows exactly a gamma
distribution with shape N and rate N; it is compared to the normal approximation N(1, 1 / sqrt(N)) of the CLT.

Run from anywhere: `python src/img/central-limit-theorem.py` writes `central-limit-theorem.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, norm

Ns = [1, 2, 10, 50]

# Palette: categorical slots 1-2 of the dataviz reference palette, plus text/grid tokens
EXACT, NORMAL = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(1, len(Ns), figsize=(13, 3.4), sharex=True)
x = np.linspace(0.001, 3, 1000)

for ax, N in zip(axes, Ns):
    exact = gamma.pdf(x, a=N, scale=1 / N)
    ax.fill_between(x, exact, color=EXACT, alpha=0.15, lw=0)
    ax.plot(x, exact, color=EXACT, lw=2, label="distribution of the average $\\bar{x}$")
    ax.plot(x, norm.pdf(x, 1, 1 / np.sqrt(N)), color=NORMAL, lw=2, ls="--",
            label="normal approximation $\\mathcal{N}(\\mu, \\sigma / \\sqrt{N})$")
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (1, 3)))

    ax.set_title(f"$N = {N}$", color=TEXT)
    ax.set_xlabel("$\\bar{x}$")
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

axes[0].set_ylabel("density")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1))

fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
