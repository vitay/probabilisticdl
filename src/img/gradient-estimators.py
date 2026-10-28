"""Distribution of single-sample gradient estimates (gradient estimators chapter).

Gradient of E_{z ~ N(mu, 1)}[f(z)] with respect to mu, for f(z) = (z - 2)^2 + 10 and mu = 0 (true gradient: 2 (mu - 2) = -4).
The constant offset of f does not change the gradient, but increases the variance of the plain score-function estimator.
- reparameterization: z = mu + xi, estimate f'(z) = 2 (z - 2);
- score function: estimate f(z) * d/dmu log N(z; mu, 1) = f(z) (z - mu);
- score function with a baseline b = E[f(z)] = (mu - 2)^2 + 11: estimate (f(z) - b) (z - mu).

Run from anywhere: `python src/img/gradient-estimators.py` writes `gradient-estimators.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MU = 0.0
OFFSET = 10.0  # constant added to f
N = 100_000  # number of single-sample estimates


def f(z):
    return (z - 2) ** 2 + OFFSET


true_gradient = 2 * (MU - 2)
baseline = (MU - 2) ** 2 + 1 + OFFSET

z = np.random.default_rng(0).normal(MU, 1.0, N)
estimators = [
    ("Reparameterization", 2 * (z - 2)),
    ("Score function", f(z) * (z - MU)),
    ("Score function with baseline", (f(z) - baseline) * (z - MU)),
]
for name, g in estimators:
    print(f"{name}: mean {g.mean():.2f}, std {g.std():.2f}")

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
bins = np.linspace(-50, 30, 161)
for ax, (name, g), color in zip(axes, estimators, COLORS):
    ax.hist(g, bins=bins, density=True, color=color, alpha=0.6)
    ax.axvline(true_gradient, color=TEXT, lw=1.5, ls="--", label=f"true gradient ({true_gradient:g})")
    ax.set_title(f"{name}\nstandard deviation: {g.std():.1f}", color=TEXT, fontsize=11)
    ax.set_xlabel("single-sample estimate of the gradient")
    ax.set_yticks([])
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
