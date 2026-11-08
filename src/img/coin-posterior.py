"""Posterior of the probability of heads of a coin after observing only heads (Bayesian chapter).

Prior Beta(5, 5) (coins are roughly fair); after k heads in N flips the posterior is Beta(5 + k, 5 + N - k).
The MLE after only heads is rho = 1 whatever N.

Run from anywhere: `python src/img/coin-posterior.py` writes `coin-posterior.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

a0, b0 = 5, 5
Ns = [3, 30]

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
PRIOR, POST3, POST30 = "#2a78d6", "#1baf7a", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, ax = plt.subplots(figsize=(7, 3.6))
rho = np.linspace(0, 1, 1000)

ax.plot(rho, beta.pdf(rho, a0, b0), color=PRIOR, lw=2, ls="--", label="prior $p(\\rho)$")
for N, color in zip(Ns, [POST3, POST30]):
    post = beta.pdf(rho, a0 + N, b0)
    ax.fill_between(rho, post, color=color, alpha=0.15, lw=0)
    ax.plot(rho, post, color=color, lw=2, label=f"posterior after {N} heads in {N} flips")
ax.axvline(1, color=MUTED, lw=1, ls=(0, (1, 3)))
ax.text(1.0, 1.01, "MLE $\\hat{\\rho} = 1$", transform=ax.transAxes, ha="center", va="bottom", color=MUTED,
        fontsize=10)

ax.set_xlabel("probability of heads $\\rho$")
ax.set_ylabel("density")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, None)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
