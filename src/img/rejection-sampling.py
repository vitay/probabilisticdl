"""Rejection sampling of an unnormalized bimodal distribution (Monte Carlo chapter).

Run from anywhere: `python src/img/rejection-sampling.py` writes `rejection-sampling.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_shown = 500       # proposals drawn on the left
N_hist = 50_000     # proposals used for the histogram on the right

rng = np.random.default_rng(42)


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def p_tilde(x):
    """Unnormalized target: a mixture of two Gaussians, scaled so that Z = 3."""
    return 3 * (0.6 * gaussian(x, -1.5, 0.6) + 0.4 * gaussian(x, 1.5, 0.8))


Z = 3.0
mu_q, sigma_q = 0.0, 2.0  # proposal q = N(0, 2)

x = np.linspace(-6, 6, 2000)
M = (p_tilde(x) / gaussian(x, mu_q, sigma_q)).max()


def rejection_sampling(n):
    """Returns the proposals, their heights u * M q(x) and the acceptance mask."""
    xs = rng.normal(mu_q, sigma_q, n)
    heights = rng.uniform(0, 1, n) * M * gaussian(xs, mu_q, sigma_q)
    return xs, heights, heights <= p_tilde(xs)


# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
TARGET, REJECTED, ACCEPTED = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

# Left: proposals under the envelope, accepted if below the target
xs, heights, accepted = rejection_sampling(N_shown)
ax1.scatter(xs[~accepted], heights[~accepted], s=10, color=REJECTED, lw=0, label="rejected")
ax1.scatter(xs[accepted], heights[accepted], s=10, color=ACCEPTED, lw=0, label="accepted")
ax1.plot(x, M * gaussian(x, mu_q, sigma_q), color=MUTED, lw=2, ls="--", label="envelope $M \\, q(x)$")
ax1.plot(x, p_tilde(x), color=TARGET, lw=2, label="target $\\tilde{p}(x)$")
ax1.set_xlim(-6, 6)
ax1.set_ylim(0, 2.2)
ax1.set_xlabel("$x$")
ax1.set_title(f"{N_shown} proposals: {accepted.sum()} accepted ({accepted.mean():.0%})", color=TEXT)
ax1.legend(frameon=False, loc="upper center", ncol=2)

# Right: histogram of the accepted samples against the normalized target
xs, _, accepted = rejection_sampling(N_hist)
ax2.hist(xs[accepted], bins=np.linspace(-6, 6, 61), density=True, color=ACCEPTED, alpha=0.5, lw=0,
         label=f"histogram of {accepted.sum()} accepted samples")
ax2.plot(x, p_tilde(x) / Z, color=TARGET, lw=2, label="normalized target $p(x) = \\tilde{p}(x) / Z$")
ax2.set_xlim(-6, 6)
ax2.set_ylim(0, 0.55)
ax2.set_xlabel("$x$")
ax2.set_ylabel("density")
ax2.set_title(f"Acceptance rate: {accepted.mean():.1%} (theory $Z / M$ = {Z / M:.1%})", color=TEXT)
ax2.legend(frameon=False, loc="upper center", ncol=1)

for ax in (ax1, ax2):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
print(f"M = {M:.3f}, Z / M = {Z / M:.3f}")
