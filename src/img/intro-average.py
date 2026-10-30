"""One answer or many: mean-square-error regression vs. a probabilistic model (introduction).

Inverse problem with two answers: y = +sqrt(x) or y = -sqrt(x) (with equal probability), plus noise.
Left: the mean-square-error regression predicts the conditional mean E[y | x] = 0, between the two branches,
where there is no data. Right: the conditional distribution p(y | x) (computed exactly) and samples of it,
which lie on both branches.

Run from anywhere: `python src/img/intro-average.py` writes `intro-average.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N = 300
NOISE = 0.15

rng = np.random.default_rng(0)
x = rng.uniform(0.05, 4, N)
y = rng.choice([-1, 1], size=N) * np.sqrt(x) + NOISE * rng.normal(size=N)

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
DATA, MEAN, MODEL = "#52514e", "#eb6834", "#2a78d6"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, (ax_m, ax_p) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
xs = np.linspace(0.05, 4, 300)

# Left: data and the mean-square-error prediction (the conditional mean, zero for this problem)
ax_m.scatter(x, y, color=DATA, s=8, alpha=0.5, label="training data")
ax_m.plot(xs, np.zeros_like(xs), color=MEAN, lw=2.5, label="prediction minimizing the MSE")
ax_m.annotate("no data here", xy=(3.0, 0.05), xytext=(2.3, 0.75), color=MEAN,
              arrowprops={"arrowstyle": "->", "color": MEAN})
ax_m.set_title("Deterministic model: a single answer (the average)", color=TEXT, fontsize=11)

# Right: the conditional distribution p(y | x) and samples of it
Y = np.linspace(-2.6, 2.6, 300)
XX, YY = np.meshgrid(xs, Y)
density = 0.5 * (np.exp(-(YY - np.sqrt(XX)) ** 2 / (2 * NOISE**2)) + np.exp(-(YY + np.sqrt(XX)) ** 2 / (2 * NOISE**2)))
ax_p.contourf(XX, YY, density, levels=12, cmap="Blues", alpha=0.6)
ax_p.scatter(x, y, color=DATA, s=8, alpha=0.3)
xq = np.linspace(0.3, 3.9, 25)
samples = rng.choice([-1, 1], size=xq.size) * np.sqrt(xq) + NOISE * rng.normal(size=xq.size)
ax_p.scatter(xq, samples, color=MODEL, s=30, zorder=3, label="samples of $p(y | x)$")
ax_p.set_title("Probabilistic model: a distribution of answers $p(y | x)$", color=TEXT, fontsize=11)

for ax in (ax_m, ax_p):
    ax.set_xlim(0, 4.1)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("input $x$")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
ax_m.set_ylabel("output $y$")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
