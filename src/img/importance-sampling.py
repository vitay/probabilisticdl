"""Importance sampling with a good proposal and two runs of a too narrow proposal (Monte Carlo chapter).

The seeds of the two bad runs are chosen to show the two typical failures: no sample in the second mode
(72% of runs with this proposal), or a few samples there with huge weights.

Same bimodal target as `rejection-sampling.py`.
Run from anywhere: `python src/img/importance-sampling.py` writes `importance-sampling.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N = 50


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def p(x):
    """Normalized target: a mixture of two Gaussians."""
    return 0.6 * gaussian(x, -1.5, 0.6) + 0.4 * gaussian(x, 1.5, 0.8)


# (title, mean and std of the proposal, seed)
runs = [
    ("Good proposal", 0.0, 2.0, 42),
    ("Bad proposal, run A (missed mode)", -1.5, 0.8, 5),
    ("Bad proposal, run B (degenerate weights)", -1.5, 0.8, 3),
]
mean_p = 0.6 * -1.5 + 0.4 * 1.5  # true E_p[x] = -0.3

x = np.linspace(-6, 6, 2000)

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
TARGET, PROPOSAL, SAMPLE = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

for col, (title, mu_q, sigma_q, seed) in enumerate(runs):
    ax_density, ax_weights = axes[0, col], axes[1, col]

    samples = np.random.default_rng(seed).normal(mu_q, sigma_q, N)
    w = p(samples) / gaussian(samples, mu_q, sigma_q)
    w_bar = w / w.sum()
    ess = w.sum() ** 2 / (w**2).sum()
    estimate = (w_bar * samples).sum()  # self-normalized estimate of E_p[x]

    # Top: target and proposal, samples as ticks on the x-axis
    ax_density.plot(x, p(x), color=TARGET, lw=2, label="target $p(x)$")
    ax_density.plot(x, gaussian(x, mu_q, sigma_q), color=PROPOSAL, lw=2, ls="--", label="proposal $q(x)$")
    ax_density.plot(samples, np.zeros_like(samples), "|", color=SAMPLE, ms=12, mew=1.5, label=f"{N} samples $x_i \\sim q(x)$")
    ax_density.set_ylim(-0.03, 0.75)
    ax_density.set_title(f"{title}, $q(x) = \\mathcal{{N}}({mu_q:g}, {sigma_q:g})$\n"
                         f"ESS = {ess:.1f} / {N},   estimate of $\\mathbb{{E}}_p[x]$ = {estimate:.2f} (true: {mean_p:.1f})",
                         color=TEXT, fontsize=11)

    # Bottom: normalized importance weights at the position of each sample
    ax_weights.vlines(samples, 0, w_bar, color=SAMPLE, lw=1.5)
    ax_weights.scatter(samples, w_bar, color=SAMPLE, s=12, zorder=3)
    ax_weights.axhline(1 / N, color=MUTED, lw=1, ls=":")
    ax_weights.text(-5.9, 1 / N, "  equal weights $1/N$", va="bottom", color=MUTED, fontsize=9)
    ax_weights.set_ylim(0, 1.15 * w_bar.max())
    ax_weights.set_xlim(-6, 6)
    ax_weights.set_xlabel("$x$")

axes[0, 0].legend(frameon=False, loc="upper right")
axes[0, 0].set_ylabel("density")
axes[1, 0].set_ylabel("normalized weight $\\bar{w}_i$")

for ax in axes.flat:
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
