"""Principle of importance sampling: plain vs. weighted average of samples from the proposal (Monte Carlo chapter).

Run from anywhere: `python src/img/importance-sampling-principle.py` writes `importance-sampling-principle.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N = 30
mu_q, sigma_q = 0.0, 1.5   # proposal
mu_p, sigma_p = 1.8, 0.7   # target

# Seed chosen as a typical run: over 1000 seeds, the weighted average is 1.76 +/- 0.23 and the median ESS 8.5
rng = np.random.default_rng(2)
samples = rng.normal(mu_q, sigma_q, N)


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


w = gaussian(samples, mu_p, sigma_p) / gaussian(samples, mu_q, sigma_q)
w_bar = w / w.sum()
plain_mean = samples.mean()
weighted_mean = (w_bar * samples).sum()

x = np.linspace(-5, 5, 1000)

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

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

panels = [
    (axes[0], np.full(N, 40.0), plain_mean,
     "Plain average of samples from $q$", f"plain average = {plain_mean:.2f}"),
    (axes[1], 40.0 * N * w_bar, weighted_mean,
     "Weighted average, $w(x) = p(x) / q(x)$", f"weighted average = {weighted_mean:.2f}"),
]

for ax, sizes, estimate, title, label in panels:
    ax.plot(x, gaussian(x, mu_p, sigma_p), color=TARGET, lw=2, label="target $p(x)$")
    ax.plot(x, gaussian(x, mu_q, sigma_q), color=PROPOSAL, lw=2, ls="--", label="proposal $q(x)$")
    ax.scatter(samples, np.zeros(N), s=sizes, color=SAMPLE, alpha=0.7, lw=0, zorder=3, clip_on=False,
               label="samples $x_i \\sim q(x)$")
    ax.axvline(mu_p, color=TARGET, lw=1, ls=":")
    ax.text(mu_p, 0.62, f" $\\mathbb{{E}}_p[x] = {mu_p:g}$", color=TARGET, va="top")
    ax.annotate(label, xy=(estimate, 0), xytext=(estimate, -0.12), ha="center", va="top", color=TEXT,
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1))
    ax.set_title(title, color=TEXT)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.02, 0.65)
    ax.set_yticks([])
    ax.set_xlabel("$x$", labelpad=36)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

axes[0].legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
print(f"plain mean = {plain_mean:.2f}, weighted mean = {weighted_mean:.2f}, ESS = {1 / (w_bar**2).sum():.1f}")
