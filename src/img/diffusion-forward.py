"""Forward diffusion process on the 2D mixture of two Gaussians of `score.py` (diffusion chapter).

With a Gaussian mixture as data distribution, every noisy marginal q(x_t) is also a Gaussian mixture, with
means sqrt(abar_t) mu_k and variances abar_t s_k^2 + 1 - abar_t. Its density, score and the optimal noise predictor
are therefore known exactly; the functions below are reused by `diffusion-reverse.py`.
Figures: samples of q(x_t) at a few steps, with the contours of the exact density (`diffusion-forward.png`), and an
animation of the forward Markov chain simulated step by step (`diffusion-forward.gif`).

Run from anywhere: `python src/img/diffusion-forward.py` writes `diffusion-forward.png` and `diffusion-forward.gif`
next to this file.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).parent))
from score import COMPONENTS_2D  # noqa: E402  (weight, mean, std) of each isotropic component

T = 1000
BETAS = np.linspace(1e-4, 0.02, T)  # linear schedule of the DDPM paper, beta_1 ... beta_T
ALPHAS = 1 - BETAS
ALPHA_BARS = np.cumprod(ALPHAS)


def abar(t):
    """abar_t for t = 0 ... T (abar_0 = 1: no noise)."""
    return 1.0 if t == 0 else ALPHA_BARS[t - 1]


def noisy_components(t):
    """(weight, mean, variance) of the components of q(x_t)."""
    a = abar(t)
    return [(w, np.sqrt(a) * mu, a * s**2 + 1 - a) for w, mu, s in COMPONENTS_2D]


def density(x, t):
    """Density of q(x_t), x of shape (..., 2)."""
    return sum(w * np.exp(-np.sum((x - m) ** 2, axis=-1) / (2 * v)) / (2 * np.pi * v)
               for w, m, v in noisy_components(t))


def score(x, t):
    """Exact score of q(x_t): responsibility-weighted scores of the Gaussian components."""
    comps = noisy_components(t)
    dens = [w * np.exp(-np.sum((x - m) ** 2, axis=-1) / (2 * v)) / (2 * np.pi * v) for w, m, v in comps]
    total = sum(dens)
    return sum(d[..., None] * (-(x - m) / v) for d, (_, m, v) in zip(dens, comps)) / total[..., None]


def optimal_epsilon(x, t):
    """Noise prediction of a perfect network: epsilon* = - sqrt(1 - abar_t) * score."""
    return -np.sqrt(1 - abar(t)) * score(x, t)


def sample_data(n, rng):
    weights = np.array([w for w, _, _ in COMPONENTS_2D])
    k = rng.choice(len(COMPONENTS_2D), size=n, p=weights)
    means = np.array([mu for _, mu, _ in COMPONENTS_2D])[k]
    stds = np.array([s for _, _, s in COMPONENTS_2D])[k]
    return means + stds[:, None] * rng.normal(size=(n, 2))


if __name__ == "__main__":
    STEPS = [0, 100, 200, 400, 1000]
    N = 1000

    rng = np.random.default_rng(0)
    x0 = sample_data(N, rng)

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

    fig, axes = plt.subplots(1, len(STEPS), figsize=(17, 3.9), sharey=True)
    g = np.linspace(-5, 5, 300)
    G1, G2 = np.meshgrid(g, g)
    grid = np.stack([G1, G2], axis=-1)
    for ax, t in zip(axes, STEPS):
        eps = rng.normal(size=x0.shape)
        xt = np.sqrt(abar(t)) * x0 + np.sqrt(1 - abar(t)) * eps  # one-shot sampling of q(x_t | x_0)
        ax.contour(G1, G2, np.log(density(grid, t)), levels=10, colors=DENSITY, linewidths=1, linestyles="solid",
                   alpha=0.5)
        ax.scatter(xt[:, 0], xt[:, 1], color=SAMPLE, s=4, alpha=0.6)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_title(f"$t = {t}$, $\\bar{{\\alpha}}_t = {abar(t):.2f}$", color=TEXT, fontsize=11)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("$x_2$")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")

    # Animation: the forward Markov chain x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) eps, simulated step by step
    FRAME_EVERY = 20  # steps between two frames
    rng = np.random.default_rng(1)
    x = sample_data(N, rng)
    frames = [(0, x.copy())]
    for t in range(1, T + 1):
        x = np.sqrt(1 - BETAS[t - 1]) * x + np.sqrt(BETAS[t - 1]) * rng.normal(size=x.shape)
        if t % FRAME_EVERY == 0:
            frames.append((t, x.copy()))
    log_densities = {t: np.log(density(grid, t)) for t, _ in frames}

    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    def draw(k):
        t, xt = frames[k]
        ax.clear()
        ax.contour(G1, G2, log_densities[t], levels=10, colors=DENSITY, linewidths=1, linestyles="solid", alpha=0.5)
        ax.scatter(xt[:, 0], xt[:, 1], color=SAMPLE, s=4, alpha=0.6)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"Forward process: $t = {t}$, $\\bar{{\\alpha}}_t = {abar(t):.2f}$", color=TEXT, fontsize=11)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(left=0.15)
    hold = [len(frames) - 1] * 8  # pause on the last frame before looping
    animation = FuncAnimation(fig, draw, frames=list(range(len(frames))) + hold)
    animation.save(Path(__file__).with_suffix(".gif"), writer=PillowWriter(fps=8), dpi=90,
                   savefig_kwargs={"facecolor": "white"})

