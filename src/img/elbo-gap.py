"""Schematic of ELBO maximization (variational inference chapter).

The log-evidence log p(x) is split into the ELBO and the gap KL(q(z|x) || p(z|x)).
Updating the encoder (phi) leaves log p(x) unchanged and closes the gap; updating the decoder (theta) raises the ELBO,
and with it the evidence. The values are arbitrary: only their relations matter.

Run from anywhere: `python src/img/elbo-gap.py` writes `elbo-gap.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# (label, log-evidence, ELBO) at each step, arbitrary units
STEPS = [
    ("initial\nmodel", 5.0, 2.0),
    ("update $\\phi$\n(encoder)", 5.0, 4.4),
    ("update $\\theta$\n(decoder)", 7.0, 5.8),
    ("update $\\phi$\n(encoder)", 7.0, 6.7),
]

# Palette: categorical slots 1-2 of the dataviz reference palette, plus text/grid tokens
ELBO, GAP = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": TEXT,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, ax = plt.subplots(figsize=(10, 5))
width = 0.5
for k, (label, evidence, elbo) in enumerate(STEPS):
    ax.bar(k, elbo, width, color=ELBO, label="ELBO$(\\theta, \\phi)$" if k == 0 else None)
    ax.bar(k, evidence - elbo, width, bottom=elbo, color=GAP, alpha=0.35, edgecolor=GAP, hatch="//",
           label="gap: KL$(q_\\phi(z|x) \\,||\\, p_\\theta(z|x))$" if k == 0 else None)
    ax.plot([k - width / 2, k + width / 2], [evidence, evidence], color=TEXT, lw=2.5,
            label="evidence $\\log p_\\theta(x)$" if k == 0 else None)
    if k > 0:  # dotted line showing whether the evidence changed
        ax.plot([k - 1 + width / 2, k - width / 2], [STEPS[k - 1][1], evidence], color=MUTED, lw=1, ls=":")

ax.set_xticks(range(len(STEPS)), [s[0] for s in STEPS])
ax.set_yticks([])
ax.set_ylabel("log-likelihood (arbitrary units)")
ax.set_ylim(0, 8.5)
ax.legend(frameon=False, loc="upper left")
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
