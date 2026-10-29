"""Unconditional vs. conditional generation with a diffusion model (conditional diffusion chapter).

One-dimensional toy "image" distribution with two modes, standing for cat images (top) and dog images (bottom).
Deterministic reverse trajectories (DDIM with the exact scores) start from the same noise samples at t = T:
without condition, they split between the two modes; conditioned on "cat", all of them end in the cat mode.
The horizontal axis is the diffusion step, from the noise (left) to the data (right).
Icons: Twemoji (https://github.com/jdecked/twemoji), CC-BY 4.0.

Run from anywhere: `python src/img/conditional-generation.py` writes `conditional-generation.png` next to this file.
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

sys.path.insert(0, str(Path(__file__).parent))
forward = importlib.import_module("diffusion-forward")
abar, T = forward.abar, forward.T

COMPONENTS = [(0.5, 2.5, 0.5), (0.5, -2.5, 0.5)]  # (weight, mean, std) of the cat and dog modes
CAT = 0
N_TRAJECTORIES = 20
STEPS = np.linspace(T, 0, 201).astype(int)  # DDIM steps


def noisy(t):
    a = abar(t)
    return [(w, np.sqrt(a) * mu, a * s**2 + 1 - a) for w, mu, s in COMPONENTS]


def density(x, t, condition=None):
    comps = noisy(t) if condition is None else [(1.0, *noisy(t)[condition][1:])]
    return sum(w * np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v) for w, m, v in comps)


def score(x, t, condition=None):
    comps = noisy(t)
    if condition is not None:
        _, m, v = comps[condition]
        return -(x - m) / v
    logs = np.stack([np.log(w) - (x - m) ** 2 / (2 * v) - 0.5 * np.log(v) for w, m, v in comps], axis=-1)
    resp = np.exp(logs - logs.max(axis=-1, keepdims=True))
    resp /= resp.sum(axis=-1, keepdims=True)
    return sum(resp[..., k] * (-(x - m) / v) for k, (_, m, v) in enumerate(comps))


def ddim_trajectories(x, condition=None):
    path = [x.copy()]
    for t, t_next in zip(STEPS[:-1], STEPS[1:]):
        eps = -np.sqrt(1 - abar(t)) * score(x, t, condition)
        x0_hat = (x - np.sqrt(1 - abar(t)) * eps) / np.sqrt(abar(t))
        x = np.sqrt(abar(t_next)) * x0_hat + np.sqrt(1 - abar(t_next)) * eps
        path.append(x.copy())
    return np.array(path)  # (steps, trajectories)


start = np.sort(np.random.default_rng(0).normal(size=N_TRAJECTORIES))
paths = {None: ddim_trajectories(start), CAT: ddim_trajectories(start, CAT)}

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
DENSITY, CAT_COLOR, DOG_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

icons = {name: plt.imread(Path(__file__).parent / f"twemoji-{name}.png") for name in ["cat", "dog"]}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
y = np.linspace(-4.5, 4.5, 500)
time = T - STEPS  # horizontal axis: progress of the generation, from 0 (noise) to T (data)
width = 250  # horizontal extent of the density profiles drawn at both ends

for ax, (condition, path) in zip(axes, paths.items()):
    # Density profiles: noise on the left, (conditional) data distribution on the right
    noise = density(y, T)
    ax.fill_betweenx(y, -width * noise / noise.max(), 0, color=DENSITY, alpha=0.25, lw=0)
    data = density(y, 0, condition)
    ax.fill_betweenx(y, T, T + width * data / data.max(), color=DENSITY, alpha=0.25, lw=0)
    for k in range(N_TRAJECTORIES):
        color = CAT_COLOR if path[-1, k] > 0 else DOG_COLOR
        ax.plot(time, path[:, k], color=color, lw=1.2, alpha=0.8)
    ax.add_artist(AnnotationBbox(OffsetImage(icons["cat"], zoom=0.14), (T + width + 130, 2.5), frameon=False))
    ax.add_artist(AnnotationBbox(OffsetImage(icons["dog"], zoom=0.14), (T + width + 130, -2.5), frameon=False))
    ax.set_xlim(-width - 20, T + width + 220)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xticks([0, T], ["noise $x_T$", "sample $x_0$"])
    ax.set_yticks([])
    ax.set_xlabel("generation (reverse process)")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
axes[0].set_title("Unconditional generation: cat or dog?", color=TEXT, fontsize=11)
axes[1].set_title("Generation conditioned on \"cat\"", color=TEXT, fontsize=11)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
