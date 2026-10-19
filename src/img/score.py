"""Score of a 1D and a 2D mixture of two Gaussians (HMC chapter).

Left: the bimodal target of the Monte Carlo chapter (top) and its score d/dx log p(x) (bottom).
Right: a 2D mixture of two isotropic Gaussians, with the density as contours and the score as a vector field.
Inspired by the figure of Yang Song's blog post <https://yang-song.net/blog/2021/score/>.

Same 1D bimodal target as `rejection-sampling.py`; the 2D mixture is reused in `langevin.py`.
Run from anywhere: `python src/img/score.py` writes `score.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


# 1D target: (weight, mean, std) of each component
COMPONENTS_1D = [(0.6, -1.5, 0.6), (0.4, 1.5, 0.8)]
# 2D target: (weight, mean, std) of each isotropic component
COMPONENTS_2D = [(0.6, np.array([-2.0, -2.0]), 1.0), (0.4, np.array([2.0, 2.0]), 1.0)]


def p_1d(x):
    return sum(w * gaussian(x, mu, s) for w, mu, s in COMPONENTS_1D)


def score_1d(x):
    """d/dx log p(x) = sum_k r_k(x) * (-(x - mu_k) / s_k^2), with the responsibilities r_k(x)."""
    return sum(w * gaussian(x, mu, s) * (-(x - mu) / s**2) for w, mu, s in COMPONENTS_1D) / p_1d(x)


def p_2d(x):
    """Density of the 2D mixture, x of shape (..., 2)."""
    return sum(w * np.exp(-np.sum((x - mu) ** 2, axis=-1) / (2 * s**2)) / (2 * np.pi * s**2)
               for w, mu, s in COMPONENTS_2D)


def score_2d(x):
    weighted = [w * np.exp(-np.sum((x - mu) ** 2, axis=-1) / (2 * s**2)) / (2 * np.pi * s**2)
                for w, mu, s in COMPONENTS_2D]
    return sum(r[..., None] * (-(x - mu) / s**2) for r, (_, mu, s) in zip(weighted, COMPONENTS_2D)) \
        / sum(weighted)[..., None]


if __name__ == "__main__":
    # Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
    TARGET, SCORE, MODE = "#2a78d6", "#eb6834", "#1baf7a"
    TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

    plt.rcParams.update({
        "font.size": 11,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
    })

    fig = plt.figure(figsize=(14, 6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.2)
    left = outer[0].subgridspec(2, 1, hspace=0.1)
    ax_p = fig.add_subplot(left[0])
    ax_s = fig.add_subplot(left[1], sharex=ax_p)
    ax_2d = fig.add_subplot(outer[1])

    # Left: 1D density and score
    x = np.linspace(-5, 5, 1000)
    score = score_1d(x)
    ax_p.plot(x, p_1d(x), color=TARGET, lw=2)
    ax_p.set_ylabel("density $p(x)$")
    ax_p.set_title("Bimodal target and its score", color=TEXT, fontsize=11)
    ax_p.tick_params(labelbottom=False)
    ax_s.plot(x, score, color=SCORE, lw=2)
    ax_s.axhline(0, color=MUTED, lw=1)
    ax_s.set_ylim(-6, 6)
    ax_s.set_xlim(-5, 5)
    ax_s.set_xlabel("$x$")
    ax_s.set_ylabel("score $\\nabla_x \\log p(x)$")
    # Zeros of the score: the two modes and the bottom of the valley
    zeros = x[1:][np.sign(score[1:]) != np.sign(score[:-1])]
    for z in zeros:
        for ax in (ax_p, ax_s):
            ax.axvline(z, color=MUTED, lw=1, ls=":")
    ax_s.text(zeros[0], -5.5, " mode", color=MUTED, fontsize=9)
    ax_s.text(zeros[1], -5.5, " valley", color=MUTED, fontsize=9)
    ax_s.text(zeros[2], -5.5, " mode", color=MUTED, fontsize=9)

    # Right: 2D density as contours, score as a vector field
    g = np.linspace(-5, 5, 300)
    X1, X2 = np.meshgrid(g, g)
    grid = np.stack([X1, X2], axis=-1)
    ax_2d.contour(X1, X2, np.log(p_2d(grid)), levels=12, colors=TARGET, linewidths=1, linestyles="solid", alpha=0.6)
    q = np.linspace(-4.5, 4.5, 16)
    Q1, Q2 = np.meshgrid(q, q)
    arrows = score_2d(np.stack([Q1, Q2], axis=-1))
    ax_2d.quiver(Q1, Q2, arrows[..., 0], arrows[..., 1], color=SCORE, angles="xy", scale_units="xy", scale=8,
                 width=0.004)
    for _, mu, _ in COMPONENTS_2D:
        ax_2d.scatter(*mu, color=MODE, s=40, zorder=3)
    ax_2d.set_xlim(-5, 5)
    ax_2d.set_ylim(-5, 5)
    ax_2d.set_aspect("equal")
    ax_2d.set_xlabel("$x_1$")
    ax_2d.set_ylabel("$x_2$")
    ax_2d.set_title("Score of a 2D mixture of two Gaussians", color=TEXT, fontsize=11)

    for ax in (ax_p, ax_s, ax_2d):
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
