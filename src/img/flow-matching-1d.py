"""Marginal flow vs. optimal transport in one dimension (flow matching chapter).

Data distribution and exact marginal velocity of `flow.py` (straight conditional paths, independent coupling).
Top row: the conditional paths of a minibatch of noise and data samples, paired at random (left, they cross) or by
the optimal coupling, obtained in 1D by sorting both sets (right, they do not cross).
Bottom row: the resulting flows. Left: trajectories of the marginal flow with random pairs, which are curved.
Right: straight lines from each noise sample x_0 to its optimal transport destination T(x_0) = F_1^{-1}(Phi(x_0))
(quantile matching). In one dimension, a flow cannot make two particles cross, so the marginal flow ends exactly at
T(x_0): both reach the same end points, along different paths.
Figure: `flow-matching-1d.png`.

Run from anywhere: `python src/img/flow-matching-1d.py` writes `flow-matching-1d.png` next to this file.
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
flow = importlib.import_module("flow")

N = 16

x0 = np.sort(np.random.default_rng(0).normal(size=N))
times, flow_path = flow.integrate(x0, n_steps=1000, t_end=0.9999)


def data_cdf(y):
    return sum(w * norm.cdf(y, m, s) for w, m, s in flow.COMPONENTS)


ot_target = np.array([brentq(lambda y: data_cdf(y) - norm.cdf(a), -10, 10) for a in x0])
ot_path = (1 - times[:, None]) * x0[None, :] + times[:, None] * ot_target[None, :]
print(f"largest difference between the end points: {np.abs(flow_path[-1] - ot_target).max():.4f}")

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
DENSITY, FLOW, OT = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

y = np.linspace(-4, 4, 400)
width = 0.15
panels = [("Resulting flow (random pairs): curved trajectories", flow_path, FLOW),
          ("Resulting flow (optimal pairs): straight trajectories", ot_path, OT)]


def draw(axes, n):
    """Draw the first n time steps of the trajectories in both panels (all of them in the figure)."""
    for ax, (title, path, color) in zip(axes, panels):
        ax.clear()
        for t, side in [(0.0, -1), (0.9999, 1)]:
            d = flow.marginal_density(y, t)
            ax.fill_betweenx(y, t, t + side * width * d / d.max(), color=DENSITY, alpha=0.3, lw=0)
        for k in range(N):
            ax.plot(times[:n], path[:n, k], color=color, lw=1.4, alpha=0.9)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-4, 4)
        ax.set_xticks([0, 1], ["noise", "data"])
        ax.set_yticks([])
        ax.set_xlabel("time $t$")
        ax.set_title(title, color=TEXT, fontsize=11)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)


# Static figure: conditional paths of a minibatch (top), resulting flows (bottom)
rng = np.random.default_rng(1)
batch_x0 = rng.normal(size=N)
k = rng.choice(len(flow.COMPONENTS), size=N, p=[w for w, _, _ in flow.COMPONENTS])
batch_x1 = np.array([flow.COMPONENTS[i][1] for i in k]) + np.array([flow.COMPONENTS[i][2] for i in k]) * rng.normal(size=N)
pairings = [("Random pairs: conditional paths cross", batch_x0, batch_x1, FLOW),
            ("Optimal pairs (sorted): conditional paths do not cross", np.sort(batch_x0), np.sort(batch_x1), OT)]

fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharey=True)
for ax, (title, a, b, color) in zip(axes[0], pairings):
    for t, side in [(0.0, -1), (0.9999, 1)]:
        d = flow.marginal_density(y, t)
        ax.fill_betweenx(y, t, t + side * width * d / d.max(), color=DENSITY, alpha=0.3, lw=0)
    for i in range(N):
        ax.plot([0, 1], [a[i], b[i]], color=color, lw=1.2, alpha=0.9)
    ax.scatter(np.zeros(N), a, color=MUTED, s=12, zorder=3)
    ax.scatter(np.ones(N), b, color=color, s=12, zorder=3)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-4, 4)
    ax.set_xticks([0, 1], ["noise $x_0$", "data $x_1$"])
    ax.set_yticks([])
    ax.set_title(title, color=TEXT, fontsize=11)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
draw(axes[1], len(times))
axes[0, 0].set_ylabel("training pairs of a minibatch")
axes[1, 0].set_ylabel("resulting flow")
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
