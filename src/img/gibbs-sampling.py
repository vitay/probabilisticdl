"""Gibbs sampling on two bivariate normal distributions with unit variances (MCMC chapter).

Left: uncorrelated coordinates (rho = 0). Right: strongly correlated coordinates (rho = 0.95).
Both chains start at the same point and perform the same number of sweeps (one update of each coordinate per sweep).

Run from anywhere: `python src/img/gibbs-sampling.py` writes `gibbs-sampling.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_SWEEPS = 30
X0 = np.array([-3.0, -3.0])  # initial state
CORRELATIONS = [0.0, 0.95]


def gibbs(rho, seed):
    """Gibbs sampling of N(0, [[1, rho], [rho, 1]]). Returns all intermediate points (two per sweep)."""
    rng = np.random.default_rng(seed)
    s = np.sqrt(1 - rho**2)  # standard deviation of each conditional
    x1, x2 = X0
    path = [(x1, x2)]
    for _ in range(N_SWEEPS):
        x1 = rng.normal(rho * x2, s)  # x1 | x2
        path.append((x1, x2))
        x2 = rng.normal(rho * x1, s)  # x2 | x1
        path.append((x1, x2))
    return np.array(path)


# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
TARGET, START, SAMPLE = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
grid = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(grid, grid)

for ax, rho in zip(axes, CORRELATIONS):
    path = gibbs(rho, seed=0)

    # Target: contours of the Mahalanobis distance at 1, 2 and 3 standard deviations
    mahalanobis = (X1**2 - 2 * rho * X1 * X2 + X2**2) / (1 - rho**2)
    ax.contour(X1, X2, mahalanobis, levels=[1, 4, 9], colors=TARGET, linewidths=1.2, alpha=0.7)

    # Chain: axis-aligned moves, samples at the end of each sweep
    ax.plot(path[:, 0], path[:, 1], color=SAMPLE, lw=1.2, alpha=0.8)
    ax.scatter(path[2::2, 0], path[2::2, 1], color=SAMPLE, s=18, zorder=3, label="samples (end of each sweep)")
    ax.scatter(*X0, color=START, s=50, zorder=4, label="initial state")

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Correlation $\\rho = {rho:g}$: {N_SWEEPS} sweeps", color=TEXT, fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

axes[0].legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
