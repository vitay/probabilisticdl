"""Reverse diffusion process (DDPM sampling) on the 2D mixture of `score.py` (diffusion chapter).

The noise predictor is the exact optimal one, epsilon*(x_t, t) = - sqrt(1 - abar_t) * score of q(x_t), computed in closed
form for the Gaussian mixture (functions of `diffusion-forward.py`): this is what a perfect network would learn.
Samples start from N(0, I) and follow the DDPM reverse step with sigma_t^2 = beta_t.
Figures: samples at a few steps with the contours of the exact q(x_t) (`diffusion-reverse.png`), and an animation of the
reverse process (`diffusion-reverse.gif`).

Run from anywhere: `python src/img/diffusion-reverse.py` writes `diffusion-reverse.png` and `diffusion-reverse.gif`
next to this file.
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).parent))
forward = importlib.import_module("diffusion-forward")
T, BETAS, ALPHAS, abar, density, optimal_epsilon = (forward.T, forward.BETAS, forward.ALPHAS, forward.abar,
                                                     forward.density, forward.optimal_epsilon)

N = 1000
STEPS = [1000, 600, 400, 200, 0]  # snapshots, in the order of generation
FRAME_EVERY = 20  # steps between two frames of the animation

# DDPM sampling: x_{t-1} = (x_t - beta_t / sqrt(1 - abar_t) * eps*(x_t, t)) / sqrt(alpha_t) + sigma_t * z
rng = np.random.default_rng(0)
x = rng.normal(size=(N, 2))
history = {T: x.copy()}
for t in range(T, 0, -1):
    eps = optimal_epsilon(x, t)
    mean = (x - BETAS[t - 1] / np.sqrt(1 - abar(t)) * eps) / np.sqrt(ALPHAS[t - 1])
    noise = rng.normal(size=x.shape) if t > 1 else 0.0  # no noise in the last step
    x = mean + np.sqrt(BETAS[t - 1]) * noise
    history[t - 1] = x.copy()
lower_left = (history[0].sum(axis=1) < 0).mean()
print(f"fraction of the final samples in the lower-left mode: {lower_left:.2f} (target: 0.60)")

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

g = np.linspace(-5, 5, 300)
G1, G2 = np.meshgrid(g, g)
grid = np.stack([G1, G2], axis=-1)


def draw(ax, t, title):
    ax.contour(G1, G2, np.log(density(grid, t)), levels=10, colors=DENSITY, linewidths=1, linestyles="solid",
               alpha=0.5)
    ax.scatter(history[t][:, 0], history[t][:, 1], color=SAMPLE, s=4, alpha=0.6)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_title(title, color=TEXT, fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


# Static figure
fig, axes = plt.subplots(1, len(STEPS), figsize=(17, 3.9), sharey=True)
for ax, t in zip(axes, STEPS):
    draw(ax, t, f"$t = {t}$")
axes[0].set_ylabel("$x_2$")
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")

# Animation
frames = list(range(T, -1, -FRAME_EVERY))
fig, ax = plt.subplots(figsize=(4.8, 4.8))
fig.subplots_adjust(left=0.15)


def update(k):
    ax.clear()
    draw(ax, frames[k], f"Reverse process: $t = {frames[k]}$")
    ax.set_ylabel("$x_2$")


hold = [len(frames) - 1] * 8  # pause on the last frame before looping
animation = FuncAnimation(fig, update, frames=list(range(len(frames))) + hold)
animation.save(Path(__file__).with_suffix(".gif"), writer=PillowWriter(fps=8), dpi=90,
               savefig_kwargs={"facecolor": "white"})
