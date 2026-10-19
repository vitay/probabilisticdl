"""Random-walk Metropolis on the posterior of the mean and standard deviation of a normal distribution (MCMC chapter).

Data: N = 20 samples from N(3, 2), as in the MAP figure. Priors: mu ~ N(0, 5), sigma ~ half-normal of scale 5.
Left: posterior samples of (mu, sigma) with their marginal histograms.
Right: posterior predictive distribution (average of the normal densities of the samples) against the data.

Run from anywhere: `python src/img/mcmc-posterior.py` writes `mcmc-posterior.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_DATA = 20
N_STEPS = 20000
BURN_IN = 1000
STEP = 0.7  # standard deviation of the random-walk proposal, in both directions
THETA0 = np.array([0.0, 1.0])  # initial state (mu, sigma)

rng = np.random.default_rng(0)
data = rng.normal(3, 2, N_DATA)


def log_p_tilde(theta):
    """Unnormalized log-posterior: log-likelihood + log-prior (constants dropped)."""
    mu, sigma = theta
    if sigma <= 0:
        return -np.inf
    log_likelihood = -N_DATA * np.log(sigma) - np.sum((data - mu) ** 2) / (2 * sigma**2)
    log_prior = -mu**2 / (2 * 5**2) - sigma**2 / (2 * 5**2)
    return log_likelihood + log_prior


# Random-walk Metropolis, in the log domain
chain = np.empty((N_STEPS + 1, 2))
chain[0] = THETA0
log_p = log_p_tilde(THETA0)
n_accepted = 0
for t in range(N_STEPS):
    candidate = rng.normal(chain[t], STEP)
    log_p_candidate = log_p_tilde(candidate)
    if np.log(rng.uniform()) < log_p_candidate - log_p:
        chain[t + 1], log_p = candidate, log_p_candidate
        n_accepted += 1
    else:
        chain[t + 1] = chain[t]
samples = chain[BURN_IN + 1:]
mu, sigma = samples[:, 0], samples[:, 1]

print(f"data: mean {data.mean():.2f}, std {data.std():.2f}")
print(f"acceptance rate {n_accepted / N_STEPS:.2f}")
for name, s in [("mu", mu), ("sigma", sigma)]:
    low, high = np.percentile(s, [2.5, 97.5])
    print(f"{name}: posterior mean {s.mean():.2f}, 95% interval [{low:.2f}, {high:.2f}]")

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
TARGET, DATA, SAMPLE = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig = plt.figure(figsize=(14, 6))
outer = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.25)
inner = outer[0].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
ax_joint = fig.add_subplot(inner[1, 0])
ax_mu = fig.add_subplot(inner[0, 0], sharex=ax_joint)
ax_sigma = fig.add_subplot(inner[1, 1], sharey=ax_joint)
ax_pred = fig.add_subplot(outer[1])

# Left: posterior samples and marginals
thinned = samples[::10]  # only for readability of the scatter plot
ax_joint.scatter(thinned[:, 0], thinned[:, 1], color=SAMPLE, s=6, alpha=0.4)
ax_joint.set_xlabel("mean $\\mu$")
ax_joint.set_ylabel("standard deviation $\\sigma$")
ax_mu.hist(mu, bins=60, color=SAMPLE, alpha=0.6)
ax_sigma.hist(sigma, bins=60, color=SAMPLE, alpha=0.6, orientation="horizontal")
ax_mu.set_title("Posterior samples $(\\mu, \\sigma) \\sim p(\\mu, \\sigma | X)$", color=TEXT, fontsize=11)
for ax in (ax_mu, ax_sigma):
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_xticks([] if ax is ax_sigma else ax.get_xticks())
    ax.set_yticks([] if ax is ax_mu else ax.get_yticks())

# Right: posterior predictive, average of N(x; mu_t, sigma_t) over the chain, with a few individual densities
x = np.linspace(-6, 12, 500)
densities = np.exp(-((x[:, None] - mu[None, ::10]) ** 2) / (2 * sigma[None, ::10] ** 2)) \
    / np.sqrt(2 * np.pi * sigma[None, ::10] ** 2)
for k in np.random.default_rng(1).choice(densities.shape[1], 30, replace=False):
    ax_pred.plot(x, densities[:, k], color=SAMPLE, lw=0.8, alpha=0.4)
ax_pred.plot([], [], color=SAMPLE, lw=0.8, alpha=0.6, label="$\\mathcal{N}(\\mu_t, \\sigma_t)$ for some samples")
ax_pred.plot(x, densities.mean(axis=1), color=TARGET, lw=2.5, label="posterior predictive $p(x | X)$")
ax_pred.plot(data, np.zeros_like(data), "|", color=DATA, ms=14, mew=2, label=f"{N_DATA} data points")
ax_pred.set_xlim(-6, 12)
ax_pred.set_ylim(-0.01, None)
ax_pred.set_xlabel("$x$")
ax_pred.set_ylabel("density")
ax_pred.set_title("Posterior predictive distribution", color=TEXT, fontsize=11)
ax_pred.legend(frameon=False, loc="upper right")

for ax in (ax_joint, ax_mu, ax_sigma, ax_pred):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
