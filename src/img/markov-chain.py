"""A 3-state Markov chain (weather) converging to its stationary distribution (MCMC chapter).

Run from anywhere: `python src/img/markov-chain.py` writes `markov-chain.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

states = ["sunny", "cloudy", "rainy"]
# T[i, j] = probability of going from state i to state j
T = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.4, 0.4],
])
stationary = np.array([6, 4, 3]) / 13

n_steps = 12
n_long = 10_000

rng = np.random.default_rng(42)

# Palette: categorical slots 1-3 of the dataviz reference palette (one per state), plus text/grid tokens
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, MUTED, GRID, NODE = "#0b0b0b", "#52514e", "#e4e3df", "#f4f3ef"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})

# Left: the chain as a graph
pos = np.array([[0.0, 1.0], [-0.9, -0.55], [0.9, -0.55]])
radius = 0.32
for i, (name, (x, y)) in enumerate(zip(states, pos)):
    ax0.add_patch(Circle((x, y), radius, facecolor=NODE, edgecolor=COLORS[i], lw=2.5, zorder=2))
    ax0.text(x, y + 0.06, name, ha="center", va="center", color=TEXT, zorder=3)
    ax0.text(x, y - 0.11, f"stay {T[i, i]:g}", ha="center", va="center", color=MUTED, fontsize=9, zorder=3)
for i in range(3):
    for j in range(3):
        if i == j:
            continue
        start, end = pos[i], pos[j]
        direction = (end - start) / np.linalg.norm(end - start)
        arrow = FancyArrowPatch(start + radius * direction, end - radius * direction,
                                connectionstyle="arc3,rad=0.25", arrowstyle="-|>", mutation_scale=14,
                                color=MUTED, lw=1.2, zorder=1)
        ax0.add_patch(arrow)
        # Label near the middle of the curved arrow, pushed to its side
        middle = (start + end) / 2
        normal = np.array([direction[1], -direction[0]])
        label_pos = middle + 0.27 * normal
        ax0.text(*label_pos, f"{T[i, j]:g}", ha="center", va="center", color=TEXT, fontsize=10,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1))
ax0.set_xlim(-1.5, 1.5)
ax0.set_ylim(-1.15, 1.5)
ax0.set_aspect("equal")
ax0.axis("off")
ax0.set_title("Transition probabilities", color=TEXT)

# Middle: distribution p_t of the states vs. t, from two initial states
for start, ls, marker in [(0, "-", "o"), (2, "--", "s")]:
    p = np.zeros(3)
    p[start] = 1.0
    history = [p]
    for _ in range(n_steps):
        p = p @ T
        history.append(p)
    history = np.array(history)
    for k in range(3):
        ax1.plot(range(n_steps + 1), history[:, k], color=COLORS[k], ls=ls, lw=2, marker=marker, ms=4)
for k in range(3):
    ax1.text(n_steps + 0.3, stationary[k], f"{states[k]}  {stationary[k]:.2f}", va="center", color=TEXT, fontsize=10)
ax1.plot([], [], color=MUTED, ls="-", marker="o", ms=4, label="starting sunny")
ax1.plot([], [], color=MUTED, ls="--", marker="s", ms=4, label="starting rainy")
ax1.set_xlim(0, n_steps + 3.2)
ax1.set_ylim(0, 1.02)
ax1.set_xticks(range(0, n_steps + 1, 2))
ax1.set_xlabel("step $t$")
ax1.set_ylabel("$p_t(x)$")
ax1.set_title("Distribution of the chain", color=TEXT)
ax1.legend(frameon=False, loc="upper right")

# Right: fraction of time spent in each state by a single long chain
x = 0
counts = np.zeros(3)
fractions = []
for t in range(n_long):
    counts[x] += 1
    fractions.append(counts / (t + 1))
    x = rng.choice(3, p=T[x])
fractions = np.array(fractions)
for k in range(3):
    ax2.plot(range(1, n_long + 1), fractions[:, k], color=COLORS[k], lw=2, label=states[k])
    ax2.axhline(stationary[k], color=COLORS[k], lw=1, ls=":")
ax2.set_xscale("log")
ax2.set_ylim(0, 1.02)
ax2.set_xlabel("number of steps")
ax2.set_ylabel("fraction of time in each state")
ax2.set_title("One long chain, starting sunny", color=TEXT)
ax2.legend(frameon=False, loc="upper right")

for ax in (ax1, ax2):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
