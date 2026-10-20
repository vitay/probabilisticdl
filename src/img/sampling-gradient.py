"""Why sampling is not differentiable (VAE chapter).

A sample z of N(mu, sigma) has a loss gradient dL/dz > 0: z should decrease.
When z is sampled directly, shifting mu gives a new, independent sample, which can land anywhere.

Run from anywhere: `python src/img/sampling-gradient.py` writes `sampling-gradient.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MU, SIGMA = 0.0, 1.0
SHIFT = -0.6  # change of mu
XI = 0.8  # the sample is z = MU + SIGMA * XI
N_NEW = 8  # possible new samples after the shift (left panel)


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
DIST, GRAD, SAMPLE = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, ax_sample = plt.subplots(figsize=(8, 4.8))
axes = [ax_sample]
x = np.linspace(-3.5, 3.5, 500)
z = MU + SIGMA * XI
peak = gaussian(0, 0, SIGMA)

for ax in axes:
    # Original distribution and the sample z, with the direction in which z should move
    ax.plot(x, gaussian(x, MU, SIGMA), color=DIST, lw=2, label="$\\mathcal{N}(\\mu, \\sigma)$")
    ax.plot(x, gaussian(x, MU + SHIFT, SIGMA), color=DIST, lw=1.5, ls="--",
            label="$\\mathcal{N}(\\mu + \\Delta\\mu, \\sigma)$")
    ax.axvline(MU, color=DIST, lw=1, ls=":", ymax=0.75)
    ax.axvline(MU + SHIFT, color=DIST, lw=1, ls=":", ymax=0.75)
    ax.annotate("", xy=(MU + SHIFT, 0.47), xytext=(MU, 0.47),
                arrowprops={"arrowstyle": "->", "color": DIST, "lw": 1.5})
    ax.text(MU + SHIFT / 2, 0.485, "$\\Delta\\mu$", color=DIST, ha="center")
    ax.scatter(z, 0, color=SAMPLE, s=60, zorder=4, clip_on=False)
    ax.text(z, -0.045, "$z$", color=SAMPLE, ha="center", va="top", fontsize=13)
    ax.annotate("", xy=(z - 0.8, 0.07), xytext=(z, 0.07),
                arrowprops={"arrowstyle": "->", "color": GRAD, "lw": 2})
    ax.text(z - 0.4, 0.085, "$\\dfrac{\\partial \\mathcal{L}}{\\partial z} > 0$: $z$ should decrease",
            color=GRAD, ha="center", va="bottom", fontsize=10)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.08, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("$z$")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


# Left: a new, independent sample after the shift can land anywhere
new_samples = np.random.default_rng(3).normal(MU + SHIFT, SIGMA, N_NEW)
ax_sample.scatter(new_samples, np.zeros(N_NEW), marker="|", color=MUTED, s=200, lw=1.5, zorder=3,
                  label="possible new samples $z'$")
ax_sample.text(MU + SHIFT, 0.2, "Will the new sample $z'$\nbe smaller than $z$?", color=TEXT, fontsize=11,
               ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
ax_sample.legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
