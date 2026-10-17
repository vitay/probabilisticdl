"""Random-walk Metropolis on the bimodal target of the Monte Carlo chapter (MCMC chapter).

Left: the first steps of the chain (trace), with the rejected proposals. Right: histogram of a long chain against the target.

Same bimodal target as `rejection-sampling.py`, unnormalized with Z = 3.
Run from anywhere: `python src/img/metropolis-hastings.py` writes `metropolis-hastings.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N = 10000  # length of the chain
N_TRACE = 200  # steps shown in the trace
STEP = 1.0  # standard deviation s of the random-walk proposal
X0 = 0.0  # initial state


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def p(x):
    """Normalized target: a mixture of two Gaussians."""
    return 0.6 * gaussian(x, -1.5, 0.6) + 0.4 * gaussian(x, 1.5, 0.8)


def p_tilde(x):
    """Unnormalized target, Z = 3."""
    return 3 * p(x)


# Random-walk Metropolis
rng = np.random.default_rng(1)
chain = np.empty(N + 1)
proposals = np.empty(N)
accepted = np.empty(N, dtype=bool)
chain[0] = X0
for t in range(N):
    x_prop = rng.normal(chain[t], STEP)
    proposals[t] = x_prop
    accepted[t] = rng.uniform() < min(1, p_tilde(x_prop) / p_tilde(chain[t]))
    chain[t + 1] = x_prop if accepted[t] else chain[t]

print(f"acceptance rate: {accepted.mean():.2f}, mean of the chain: {chain.mean():.2f} (true: -0.3)")

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
TARGET, REJECTED, SAMPLE = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, (ax_trace, ax_hist) = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True,
                                        gridspec_kw={"width_ratios": [3, 1]})

# Left: trace of the first steps, rejected proposals as crosses
t = np.arange(N_TRACE + 1)
ax_trace.step(t, chain[: N_TRACE + 1], where="post", color=SAMPLE, lw=1.5, label="chain $x_t$")
rejected = ~accepted[:N_TRACE]
ax_trace.scatter(t[1:][rejected], proposals[:N_TRACE][rejected], marker="x", color=REJECTED, s=18, lw=1.2,
                 label="rejected proposals")
ax_trace.set_xlim(0, N_TRACE)
ax_trace.set_xlabel("step $t$")
ax_trace.set_ylabel("$x$")
ax_trace.set_title(f"First {N_TRACE} steps of the chain, proposal $q(x'|x) = \\mathcal{{N}}(x, {STEP:g})$",
                   color=TEXT, fontsize=11)
ax_trace.legend(frameon=False, loc="lower right", ncol=2)

# Right: histogram of the whole chain against the target, rotated to share the y-axis of the trace
x = np.linspace(-5, 5, 1000)
ax_hist.hist(chain, bins=60, range=(-5, 5), density=True, orientation="horizontal", color=SAMPLE, alpha=0.5,
             label=f"{N} steps")
ax_hist.plot(p(x), x, color=TARGET, lw=2, label="target $p(x)$")
ax_hist.set_ylim(-5, 5)
ax_hist.set_xlabel("density")
ax_hist.set_title(f"Acceptance rate: {accepted.mean():.0%}", color=TEXT, fontsize=11)
ax_hist.legend(frameon=False, loc="upper right")

for ax in (ax_trace, ax_hist):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
