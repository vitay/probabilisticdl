"""Fitting a normal distribution to a bimodal target with the forward and the reverse KL (KL chapter).

Forward KL(p || q): the optimum matches the mean and variance of p (moment matching).
Reverse KL(q || p): minimized numerically over (mu, log sigma) from two initializations, one near each mode.
The KL divergences are computed by numerical integration on a grid.

Same weights and widths as the bimodal target of `rejection-sampling.py`, with the modes moved apart (at -2 and 2):
with the original modes (at -1.5 and 1.5), the reverse KL also prefers a single broad normal distribution.
Run from anywhere: `python src/img/forward-reverse-kl.py` writes `forward-reverse-kl.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

x = np.linspace(-8, 8, 4001)
dx = x[1] - x[0]
COMPONENTS = [(0.6, -2.0, 0.6), (0.4, 2.0, 0.8)]  # (weight, mean, std)


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


p = sum(w * gaussian(x, mu, s) for w, mu, s in COMPONENTS)


def kl(a, b):
    """KL(a || b) by numerical integration, for densities evaluated on the grid."""
    mask = a > 1e-300
    return np.sum(a[mask] * (np.log(a[mask]) - np.log(np.maximum(b[mask], 1e-300)))) * dx


# Forward KL: moment matching
mean = sum(w * mu for w, mu, _ in COMPONENTS)
var = sum(w * (s**2 + mu**2) for w, mu, s in COMPONENTS) - mean**2
q_forward = gaussian(x, mean, np.sqrt(var))

# Reverse KL: numerical minimization from two initializations
fits = []
for mu0 in [-2.0, 2.0]:
    res = minimize(lambda t: kl(gaussian(x, t[0], np.exp(t[1])), p), x0=[mu0, 0.0], method="Nelder-Mead")
    fits.append((res.x[0], np.exp(res.x[1])))
    print(f"reverse KL fit from {mu0}: mu = {res.x[0]:.2f}, sigma = {np.exp(res.x[1]):.2f}, KL(q||p) = {res.fun:.3f}")
print(f"forward KL fit: mu = {mean:.2f}, sigma = {np.sqrt(var):.2f}, KL(p||q) = {kl(p, q_forward):.3f}")

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
TARGET, FIT, FIT2 = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)

ax_f.fill_between(x, p, color=TARGET, alpha=0.15, lw=0)
ax_f.plot(x, p, color=TARGET, lw=2, label="target $p(x)$")
ax_f.plot(x, q_forward, color=FIT, lw=2,
          label=f"$q(x) = \\mathcal{{N}}({mean:.1f}, {np.sqrt(var):.1f})$, KL$(p || q)$ = {kl(p, q_forward):.2f}")
ax_f.set_title("Forward KL: $\\min_q \\, \\text{KL}(p || q)$ (mass-covering)", color=TEXT, fontsize=11)

ax_r.fill_between(x, p, color=TARGET, alpha=0.15, lw=0)
ax_r.plot(x, p, color=TARGET, lw=2, label="target $p(x)$")
for (mu, s), color in zip(fits, [FIT, FIT2]):
    q = gaussian(x, mu, s)
    ax_r.plot(x, q, color=color, lw=2, label=f"$q(x) = \\mathcal{{N}}({mu:.1f}, {s:.1f})$, KL$(q || p)$ = {kl(q, p):.2f}")
ax_r.set_title("Reverse KL: $\\min_q \\, \\text{KL}(q || p)$ (mode-seeking, two local optima)", color=TEXT, fontsize=11)

for ax in (ax_f, ax_r):
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("$x$")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
ax_f.set_ylabel("density")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
