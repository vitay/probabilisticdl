"""Classifier-free guidance on a 2D mixture of two overlapping Gaussians (conditional diffusion chapter).

The data are two classes, two isotropic Gaussians at (-1, -1) and (1, 1) with weights 0.6 and 0.4, which overlap
(guidance has no visible effect on well-separated classes, as the classifier p(c | x) is then certain everywhere).
The condition c is the class (here the upper-right one). The conditional noisy distribution
q(x_t | c) is a single Gaussian, the unconditional one q(x_t) the mixture: both scores are exact.
Guided score: gamma * score(x_t | c) + (1 - gamma) * score(x_t), used in DDPM sampling
(gamma = 0: unconditional, gamma = 1: conditional, gamma > 1: guided).

Run from anywhere: `python src/img/classifier-free-guidance.py` writes `classifier-free-guidance.png` next to this file.
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
forward = importlib.import_module("diffusion-forward")

T, BETAS, ALPHAS, abar = forward.T, forward.BETAS, forward.ALPHAS, forward.abar
COMPONENTS = [(0.6, np.array([-1.0, -1.0]), 1.0), (0.4, np.array([1.0, 1.0]), 1.0)]  # (weight, mean, std) per class
CONDITION = 1  # index of the class used as condition (upper-right)
GAMMAS = [0.0, 1.0, 3.0]
N = 1000


def noisy(t):
    """(weight, mean, variance) of the classes of q(x_t)."""
    a = abar(t)
    return [(w, np.sqrt(a) * mu, a * s**2 + 1 - a) for w, mu, s in COMPONENTS]


def log_densities(x, t):
    return [np.log(w) - np.sum((x - m) ** 2, axis=-1) / (2 * v) - np.log(2 * np.pi * v) for w, m, v in noisy(t)]


def conditional_score(x, t):
    _, m, v = noisy(t)[CONDITION]
    return -(x - m) / v


def unconditional_score(x, t):
    logs = np.stack(log_densities(x, t), axis=-1)
    resp = np.exp(logs - logs.max(axis=-1, keepdims=True))
    resp /= resp.sum(axis=-1, keepdims=True)
    return sum(resp[..., k, None] * (-(x - m) / v) for k, (_, m, v) in enumerate(noisy(t)))


def sample(gamma, rng):
    x = rng.normal(size=(N, 2))
    for t in range(T, 0, -1):
        score = gamma * conditional_score(x, t) + (1 - gamma) * unconditional_score(x, t)
        eps = -np.sqrt(1 - abar(t)) * score
        mean = (x - BETAS[t - 1] / np.sqrt(1 - abar(t)) * eps) / np.sqrt(ALPHAS[t - 1])
        x = mean + (np.sqrt(BETAS[t - 1]) * rng.normal(size=x.shape) if t > 1 else 0.0)
    return x


# Palette: categorical slots 1-2 of the dataviz reference palette, plus text/grid tokens
DENSITY, SAMPLE = "#2a78d6", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

titles = {0.0: "unconditional", 1.0: "conditional", 3.0: "guided"}
fig, axes = plt.subplots(1, len(GAMMAS), figsize=(12, 4.2), sharey=True)
g = np.linspace(-5, 5, 300)
G1, G2 = np.meshgrid(g, g)
grid = np.stack([G1, G2], axis=-1)
for ax, gamma in zip(axes, GAMMAS):
    x = sample(gamma, np.random.default_rng(0))
    print(f"gamma = {gamma}: mean {x.mean(axis=0).round(2)}, std {x.std(axis=0).round(2)}")
    for _, m, v in noisy(0):  # one set of contours per class
        ax.contour(G1, G2, np.sum((grid - m) ** 2, axis=-1) / v, levels=[1, 4], colors=DENSITY, linewidths=1,
                   linestyles="solid", alpha=0.5)
    ax.scatter(x[:, 0], x[:, 1], color=SAMPLE, s=4, alpha=0.6)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_title(f"$\\gamma = {gamma:g}$ ({titles[gamma]})", color=TEXT, fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("$x_2$")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
