"""Prior, likelihood and posterior of the mean of a normal distribution (MAP chapter).

Run from anywhere: `python src/img/map-posterior.py` writes `map-posterior.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Setting of the MAP section: data ~ N(mu_true, sigma) with sigma known, prior N(mu0, sigma0) on the mean
mu_true, sigma = 3.0, 2.0
mu0, sigma0 = 0.0, 1.0
Ns = [1, 5, 50]

rng = np.random.default_rng(42)
samples = rng.normal(mu_true, sigma, max(Ns))

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
PRIOR, LIKELIHOOD, POSTERIOR = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


x = np.linspace(-3, 6, 1000)

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(1, len(Ns), figsize=(12, 3.6), sharex=True, sharey=True)

for ax, N in zip(axes, Ns):
    xbar = samples[:N].mean()

    # Likelihood of mu, normalized: N(xbar, sigma / sqrt(N))
    s_lik = sigma / np.sqrt(N)

    # Posterior: N(mu_MAP, sigma_N)
    mu_map = (N * sigma0**2 * xbar + sigma**2 * mu0) / (N * sigma0**2 + sigma**2)
    s_post = 1.0 / np.sqrt(N / sigma**2 + 1 / sigma0**2)

    ax.plot(x, gaussian(x, mu0, sigma0), color=PRIOR, lw=2, ls="--", label="prior $p(\\mu)$")
    ax.plot(x, gaussian(x, xbar, s_lik), color=LIKELIHOOD, lw=2, ls=":", label="likelihood $p(X | \\mu)$ (normalized)")
    post = gaussian(x, mu_map, s_post)
    ax.fill_between(x, post, color=POSTERIOR, alpha=0.15, lw=0)
    ax.plot(x, post, color=POSTERIOR, lw=2, label="posterior $p(\\mu | X)$")

    # Estimates as thin vertical lines, true mean dotted
    for value, color in [(xbar, LIKELIHOOD), (mu_map, POSTERIOR)]:
        ax.axvline(value, color=color, lw=1, alpha=0.6)
    ax.axvline(mu_true, color=MUTED, lw=1, ls=(0, (1, 3)))

    ax.set_title(f"$N = {N}$", color=TEXT)
    ax.text(0.02, 0.97, f"$\\bar{{x}} = {xbar:.2f}$\n$\\mu_\\mathrm{{MAP}} = {mu_map:.2f}$",
            transform=ax.transAxes, va="top", ha="left", color=TEXT, fontsize=10)
    ax.set_xlabel("$\\mu$")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

axes[0].set_ylabel("density")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))

fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
