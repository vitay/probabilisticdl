"""The common thread of the book: fitting a model p_theta to an unknown data distribution p (introduction).

A bimodal data distribution p(x) (dog images on one mode, cat images on the other), known only through samples
(the training set), and a parameterized model p_theta(x) to bring as close as possible to it, as measured by a
divergence. Redrawn after a figure of the author's introductory AI lecture.
Icons: Twemoji (https://github.com/jdecked/twemoji), CC-BY 4.0.

Run from anywhere: `python src/img/common-thread.py` writes `common-thread.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


DATA = [(0.6, -1.6, 0.8), (0.4, 1.8, 0.6)]  # (weight, mean, std): dogs, cats
MODEL = [(0.55, -1.3, 0.9), (0.45, 1.6, 0.75)]  # a slightly different mixture
N = 18


def mixture(x, components):
    return sum(w * gaussian(x, m, s) for w, m, s in components)


rng = np.random.default_rng(2)
k = rng.choice(2, size=N, p=[DATA[0][0], DATA[1][0]])
samples = np.sort(np.array([DATA[i][1] for i in k]) + np.array([DATA[i][2] for i in k]) * rng.normal(size=N))

# Palette: categorical slots 1-2 of the dataviz reference palette, plus text/grid tokens
DATA_COLOR, MODEL_COLOR = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 12,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.linspace(-4.5, 4.5, 800)
p, q = mixture(x, DATA), mixture(x, MODEL)
ax.plot(x, p, color=DATA_COLOR, lw=2.5)
ax.plot(x, q, color=MODEL_COLOR, lw=2.5)
ax.fill_between(x, p, color=DATA_COLOR, alpha=0.08, lw=0)

# Samples of the data distribution: crosses on the axis, dashed lines up to the density
ax.vlines(samples, 0, mixture(samples, DATA), color=MUTED, lw=0.8, ls=":")
ax.scatter(samples, np.zeros(N), marker="x", color=TEXT, s=40, lw=1.5, zorder=3, clip_on=False)

# Labels
ax.annotate("data distribution $p(x)$\n(unknown)", xy=(-2.3, mixture(-2.3, DATA)), xytext=(-4.4, 0.33),
            color=DATA_COLOR, arrowprops={"arrowstyle": "->", "color": DATA_COLOR})
ax.annotate("model $p_\\theta(x)$", xy=(2.45, mixture(2.45, MODEL)), xytext=(2.9, 0.27),
            color=MODEL_COLOR, arrowprops={"arrowstyle": "->", "color": MODEL_COLOR})
xv = 0.2  # the two curves differ most in the valley between the modes
ax.annotate("", xy=(xv, mixture(xv, DATA)), xytext=(xv, mixture(xv, MODEL)),
            arrowprops={"arrowstyle": "<->", "color": TEXT, "lw": 1.5})
ax.annotate("make them close:\nminimize a divergence,\ne.g. KL$(p \\, || \\, p_\\theta)$",
            xy=(xv, 0.5 * (mixture(xv, DATA) + mixture(xv, MODEL))), xytext=(xv, 0.27), ha="center", va="bottom",
            color=TEXT, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})

# A few samples illustrated by icons
icons = {name: plt.imread(Path(__file__).parent / f"twemoji-{name}.png") for name in ["dog", "cat"]}
for xi, name in [(samples[1], "dog"), (samples[6], "dog"), (samples[-7], "cat"), (samples[-1], "cat")]:
    ax.add_artist(AnnotationBbox(OffsetImage(icons[name], zoom=0.1), (xi, 0), xybox=(0, -45),
                                 xycoords="data", boxcoords="offset points", frameon=False,
                                 arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8}))

ax.set_xlim(-4.5, 4.5)
ax.set_ylim(0, 0.42)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("training set: samples $x_i \\sim p(x)$, e.g. images", labelpad=62, color=TEXT)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
