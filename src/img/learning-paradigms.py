"""The learning paradigms as probabilistic models (learning paradigms chapter).

Four panels, each input -> network f_theta -> output, where the output is a probability distribution or a sample:
(a) classification: an image -> p_theta(y | x) over classes;
(b) density estimation: an image -> its likelihood p_theta(x);
(c) generation: a noise sample z ~ N(0, I) -> a new sample x ~ p_theta(x);
(d) autoregressive modeling: the beginning of a sentence -> p_theta(next token | previous tokens).
Icons: Twemoji (https://github.com/jdecked/twemoji), CC-BY 4.0.

Run from anywhere: `python src/img/learning-paradigms.py` writes `learning-paradigms.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent

# Palette: categorical slots 1-2 of the dataviz reference palette, plus text/grid tokens
NETWORK, OUTPUT = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

icons = {name: plt.imread(HERE / f"twemoji-{name}.png") for name in ["cat", "dog"]}
fig = plt.figure(figsize=(13, 5.6))
canvas = fig.add_axes([0, 0, 1, 1])
canvas.set_xlim(0, 2)
canvas.set_ylim(0, 2)
canvas.axis("off")

# Panel origins (lower-left corner in canvas coordinates), each panel is 1 x 1
PANELS = {"a": (0, 1), "b": (1, 1), "c": (0, 0), "d": (1, 0)}
TITLES = {
    "a": "(a) Classification: $p_\\theta(y \\, | \\, x)$",
    "b": "(b) Density estimation: $p_\\theta(x)$",
    "c": "(c) Generation: $x \\sim p_\\theta(x)$",
    "d": "(d) Autoregressive modeling: $p_\\theta(x_{t+1} \\, | \\, x_1, \\ldots, x_t)$",
}


def inset(panel, x, y, w, h):
    """Axes at relative position (x, y, w, h) inside a panel (figure coordinates)."""
    px, py = PANELS[panel]
    return fig.add_axes([(px + x) / 2, (py + y) / 2, w / 2, h / 2])


def network(panel, label="$f_\\theta$"):
    px, py = PANELS[panel]
    box = FancyBboxPatch((px + 0.40, py + 0.33), 0.2, 0.26, boxstyle="round,pad=0.02", fc=NETWORK, ec="none",
                         alpha=0.85)
    canvas.add_patch(box)
    canvas.text(px + 0.5, py + 0.46, label, ha="center", va="center", color="white", fontsize=14)
    for x0, x1 in [(0.30, 0.39), (0.61, 0.68)]:
        canvas.add_patch(FancyArrowPatch((px + x0, py + 0.46), (px + x1, py + 0.46), arrowstyle="-|>",
                                         mutation_scale=15, color=MUTED, lw=1.5))
    canvas.text(px + 0.5, py + 0.8, TITLES[panel], ha="center", va="center", fontsize=12)


def image(panel, x, y, img, label):
    ax = inset(panel, x, y, 0.24, 0.3)
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    px, py = PANELS[panel]
    canvas.text(px + x + 0.12, py + y - 0.04, label, ha="center", va="top", color=MUTED, fontsize=10)


def bars(panel, labels, probs, title):
    ax = inset(panel, 0.79, 0.28, 0.2, 0.36)
    ax.barh(range(len(labels)), probs, color=OUTPUT, alpha=0.8)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title(title, fontsize=10, color=MUTED)
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)


# (a) Classification
network("a")
image("a", 0.04, 0.3, icons["cat"], "input $x$")
bars("a", ["cat", "dog", "car"], [0.85, 0.13, 0.02], "$p_\\theta(y \\, | \\, x)$")

# (b) Density estimation
network("b")
image("b", 0.04, 0.3, icons["dog"], "sample $x$")
ax = inset("b", 0.72, 0.3, 0.25, 0.32)
grid = np.linspace(-4, 4, 300)
density = 0.6 * np.exp(-(grid + 1.3) ** 2 / 0.8) + 0.4 * np.exp(-(grid - 1.6) ** 2 / 0.6)
ax.plot(grid, density, color=OUTPUT, lw=2)
ax.vlines(-0.6, 0, 0.6 * np.exp(-(0.7) ** 2 / 0.8) + 0.4 * np.exp(-(2.2) ** 2 / 0.6), color=MUTED, ls=":")
ax.scatter([-0.6], [0.6 * np.exp(-(0.7) ** 2 / 0.8) + 0.4 * np.exp(-(2.2) ** 2 / 0.6)], color=OUTPUT, s=30, zorder=3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(0, 0.7)
ax.set_title("likelihood $p_\\theta(x)$", fontsize=10, color=MUTED)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)

# (c) Generation
network("c")
noise = np.random.default_rng(0).normal(size=(24, 24))
image("c", 0.04, 0.3, noise, "noise $z \\sim \\mathcal{N}(0, I)$")
image("c", 0.73, 0.3, icons["cat"], "new sample $x$")

# (d) Autoregressive modeling
network("d")
px, py = PANELS["d"]
canvas.text(px + 0.16, py + 0.46, "\"The cat\nsat on\nthe ...\"", ha="center", va="center", fontsize=12,
            style="italic", color=TEXT)
canvas.text(px + 0.16, py + 0.24, "previous tokens", ha="center", va="top", color=MUTED, fontsize=10)
bars("d", ["mat", "sofa", "roof", "dog"], [0.55, 0.25, 0.15, 0.05], "next token")

fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
