"""KL, Jensen-Shannon and Wasserstein distances between two shifted uniform distributions (optimal transport chapter).

p is uniform on [0, 1], q uniform on [d, d + 1]. In closed form:
- KL(p || q) is infinite as soon as d > 0 (part of p lies outside the support of q);
- JS(p, q) = d log 2 for d < 1, and log 2 for d >= 1 (constant: no gradient once the supports are disjoint);
- W2(p, q) = d (the optimal transport is a translation).

Run from anywhere: `python src/img/ot-vs-kl.py` writes `ot-vs-kl.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SHIFTS = [0.5, 1.5, 3.0]

# Palette: categorical slots 1-4 of the dataviz reference palette, plus text/grid tokens
P, Q, JS, W = "#2a78d6", "#eb6834", "#1baf7a", "#9d5ce6"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, (ax_d, ax_c) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})

# Left: p and three shifted versions of q, all of density 1 (outlines, to see them overlap)
ax_d.fill_between([0, 1], 0, 1, color=P, alpha=0.35, lw=0, label="$p$")
for k, (d, style) in enumerate(zip(SHIFTS, ["-", "--", ":"])):
    ax_d.plot([d, d, d + 1, d + 1], [0, 1, 1, 0], color=Q, lw=2, ls=style,
              label=f"$q$, shift $d = {d:g}$")
ax_d.set_xlim(-0.5, 4.5)
ax_d.set_ylim(0, 1.5)
ax_d.set_xlabel("$x$")
ax_d.set_ylabel("density")
ax_d.set_title("Two uniform distributions shifted by $d$", color=TEXT, fontsize=11)
ax_d.legend(frameon=False, loc="upper right")

# Right: divergences as functions of the shift
d = np.linspace(0, 3, 301)
ax_c.plot(d, np.minimum(d, 1) * np.log(2), color=JS, lw=2, label="Jensen-Shannon")
ax_c.plot(d, d, color=W, lw=2, label="Wasserstein $W_2$")
ax_c.axhline(np.log(2), color=JS, lw=1, ls=":")
ax_c.text(3.0, np.log(2) + 0.05, "$\\log 2$", color=JS, ha="right")
ax_c.text(1.6, 2.6, "KL$(p || q) = \\infty$ for all $d > 0$", color=MUTED, ha="center")
ax_c.set_xlim(0, 3)
ax_c.set_ylim(0, 3.1)
ax_c.set_xlabel("shift $d$")
ax_c.set_title("Divergence between $p$ and $q$", color=TEXT, fontsize=11)
ax_c.legend(frameon=False, loc="center right")

for ax in (ax_d, ax_c):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
