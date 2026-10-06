"""Convergence of the Monte Carlo estimate of pi (Monte Carlo chapter).

Run from anywhere: `python src/img/montecarlo-convergence.py` writes `montecarlo-convergence.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_max = 100_000
n_runs_shown = 5   # individual runs on the left
n_runs_error = 200  # runs averaged for the error on the right

rng = np.random.default_rng(42)

# Standard deviation of f(x) = 4 * indicator(||x|| <= 1) under the uniform distribution on the square
sigma_f = np.sqrt(np.pi * (4 - np.pi))

N = np.arange(1, N_max + 1)


def running_estimates(n_runs):
    """Running Monte Carlo estimates of pi, one row per run."""
    x = rng.uniform(-1, 1, (n_runs, N_max, 2))
    f = 4.0 * ((x**2).sum(axis=-1) <= 1)
    return np.cumsum(f, axis=1) / N


# Palette: categorical slot 1 of the dataviz reference palette, plus text/grid tokens
RUN = "#2a78d6"
TEXT, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

# Left: a few runs converging towards pi, inside the +/- 1 standard error band
estimates = running_estimates(n_runs_shown)
ax1.fill_between(N, np.pi - sigma_f / np.sqrt(N), np.pi + sigma_f / np.sqrt(N), color=BAND, lw=0,
                 label="$\\pi \\pm \\sigma_f / \\sqrt{N}$")
for run in estimates:
    ax1.plot(N, run, color=RUN, lw=1, alpha=0.8)
ax1.plot([], [], color=RUN, lw=1, label="independent runs of $\\hat{f}_N$")
ax1.axhline(np.pi, color=TEXT, lw=1, ls="--", label="$\\pi$")
ax1.set_xscale("log")
ax1.set_ylim(2.4, 3.9)
ax1.set_xlabel("number of samples $N$")
ax1.set_ylabel("estimate of $\\pi$")
ax1.legend(frameon=False, loc="upper right")

# Right: root mean square error over many runs vs. the theoretical sigma_f / sqrt(N)
errors = running_estimates(n_runs_error) - np.pi
rmse = np.sqrt((errors**2).mean(axis=0))
ax2.plot(N, sigma_f / np.sqrt(N), color=TEXT, lw=1, ls="--", label="$\\sigma_f / \\sqrt{N} \\approx 1.64 / \\sqrt{N}$")
ax2.plot(N, rmse, color=RUN, lw=2, label=f"measured error ({n_runs_error} runs)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("number of samples $N$")
ax2.set_ylabel("root mean square error")
ax2.legend(frameon=False, loc="upper right")

for ax in (ax1, ax2):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
