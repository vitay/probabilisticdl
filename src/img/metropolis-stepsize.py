"""Effect of the step size of random-walk Metropolis on the bimodal target (MCMC chapter).

One row per step size (too small, good, too large), all chains starting in the tail at x0 = 8.
Left: trace of the first steps, with the burn-in period shaded. Middle: histogram of the chain after burn-in.
Right: autocorrelation of the chain after burn-in.

Same bimodal target as `rejection-sampling.py`, unnormalized with Z = 3.
Run from anywhere: `python src/img/metropolis-stepsize.py` writes `metropolis-stepsize.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N = 10000  # length of each chain
BURN_IN = 500  # discarded steps
N_TRACE = 1000  # steps shown in the trace
MAX_LAG = 200  # lags shown in the autocorrelation
X0 = 8.0  # initial state, far in the tail of the target
STEPS = [("too small", 0.1), ("good", 3.0), ("too large", 50.0)]


def gaussian(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def p(x):
    """Normalized target: a mixture of two Gaussians."""
    return 0.6 * gaussian(x, -1.5, 0.6) + 0.4 * gaussian(x, 1.5, 0.8)


def p_tilde(x):
    """Unnormalized target, Z = 3."""
    return 3 * p(x)


def metropolis(step, seed):
    """Random-walk Metropolis with the proposal N(x, step). Returns the chain and the acceptance rate."""
    rng = np.random.default_rng(seed)
    chain = np.empty(N + 1)
    chain[0] = X0
    n_accepted = 0
    for t in range(N):
        x_prop = rng.normal(chain[t], step)
        if rng.uniform() < min(1, p_tilde(x_prop) / p_tilde(chain[t])):
            chain[t + 1] = x_prop
            n_accepted += 1
        else:
            chain[t + 1] = chain[t]
    return chain, n_accepted / N


def autocorrelation(x, max_lag):
    x = x - x.mean()
    return np.array([np.mean(x[: len(x) - k] * x[k:]) for k in range(max_lag + 1)]) / x.var()


def ess(x):
    """Effective sample size N / (1 + 2 sum_k rho_k), summing the autocorrelations until the first negative one."""
    rho = autocorrelation(x, len(x) // 2)
    first_negative = np.argmax(rho[1:] < 0) + 1
    return len(x) / (1 + 2 * rho[1:first_negative].sum())


# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
TARGET, BURN, SAMPLE = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(3, 3, figsize=(15, 9), gridspec_kw={"width_ratios": [3, 1, 1.5]})
x = np.linspace(-5, 9, 1000)

for row, (name, step) in enumerate(STEPS):
    ax_trace, ax_hist, ax_acf = axes[row]
    chain, acceptance = metropolis(step, seed=0)
    samples = chain[BURN_IN + 1:]
    n_eff = ess(samples)
    print(f"s = {step:g}: acceptance {acceptance:.2f}, ESS {n_eff:.0f} / {len(samples)}, mean {samples.mean():.2f}")

    # Left: trace of the first steps, burn-in shaded
    ax_trace.axvspan(0, BURN_IN, color=BURN, alpha=0.1, lw=0, label="burn-in (discarded)")
    ax_trace.step(np.arange(N_TRACE + 1), chain[: N_TRACE + 1], where="post", color=SAMPLE, lw=1.2)
    ax_trace.set_xlim(0, N_TRACE)
    ax_trace.set_ylim(-5, 9)
    ax_trace.set_ylabel("$x$")
    ax_trace.set_title(f"Step size $s = {step:g}$ ({name}): acceptance rate {acceptance:.0%}, "
                       f"ESS = {n_eff:.0f} / {len(samples)}", color=TEXT, fontsize=11, loc="left")

    # Middle: histogram after burn-in against the target, rotated to share the y-axis of the trace
    ax_hist.hist(samples, bins=60, range=(-5, 9), density=True, orientation="horizontal", color=SAMPLE, alpha=0.5)
    ax_hist.plot(p(x), x, color=TARGET, lw=2, label="target $p(x)$")
    ax_hist.set_ylim(-5, 9)
    ax_hist.set_xlim(0, 0.75)
    ax_hist.set_yticklabels([])

    # Right: autocorrelation after burn-in
    rho = autocorrelation(samples, MAX_LAG)
    ax_acf.fill_between(np.arange(MAX_LAG + 1), rho, color=SAMPLE, alpha=0.5, lw=0)
    ax_acf.plot(np.arange(MAX_LAG + 1), rho, color=SAMPLE, lw=1.5)
    ax_acf.set_xlim(0, MAX_LAG)
    ax_acf.set_ylim(-0.2, 1.05)
    ax_acf.set_ylabel("autocorrelation $\\rho_k$")

axes[0, 0].legend(frameon=False, loc="upper right")
axes[0, 1].legend(frameon=False, loc="upper right")
axes[2, 0].set_xlabel("step $t$")
axes[2, 1].set_xlabel("density")
axes[2, 2].set_xlabel("lag $k$")

for ax in axes.flat:
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
