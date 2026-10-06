"""Monte Carlo estimate of pi with uniform samples in the square [-1, 1]^2 (Monte Carlo chapter).

Run from anywhere: `python src/img/montecarlo-pi.py` writes `montecarlo-pi.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

Ns = [100, 1000, 10000]
marker_sizes = [12, 4, 1]

rng = np.random.default_rng(42)
samples = rng.uniform(-1, 1, (max(Ns), 2))
inside = (samples**2).sum(axis=1) <= 1

# Palette: categorical slots 1-2 of the dataviz reference palette, plus text tokens
INSIDE, OUTSIDE = "#2a78d6", "#eb6834"
TEXT, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

fig, axes = plt.subplots(1, len(Ns), figsize=(12, 4.3))

for ax, N, size in zip(axes, Ns, marker_sizes):
    x, ins = samples[:N], inside[:N]
    pi_hat = 4 * ins.mean()

    ax.scatter(*x[ins].T, s=size, color=INSIDE, lw=0, label="inside the circle")
    ax.scatter(*x[~ins].T, s=size, color=OUTSIDE, lw=0, label="outside the circle")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=TEXT, lw=1))
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color=MUTED, lw=1))

    ax.set_title(f"$N = {N}$:   $\\hat{{\\pi}} = 4 \\times {ins.sum()} / {N} = {pi_hat:.3f}$", color=TEXT)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    for side in ax.spines.values():
        side.set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, markerscale=2, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))

fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
