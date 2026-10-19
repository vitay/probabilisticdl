"""Random-walk Metropolis, Gibbs sampling and HMC on a correlated bivariate normal distribution (HMC chapter).

Same target and initial state as `gibbs-sampling.py` (unit variances, rho = 0.95, x0 = (-3, -3)), 30 iterations each.
For HMC, the first leapfrog trajectories are drawn.

Run from anywhere: `python src/img/hmc.py` writes `hmc.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RHO = 0.95
X0 = np.array([-3.0, -3.0])
N_ITER = 30
STEP_RW = 0.5  # standard deviation of the random-walk proposal
EPSILON, L = 0.15, 15  # leapfrog step size and number of steps
N_TRAJECTORIES = 3  # trajectories drawn

PRECISION = np.linalg.inv(np.array([[1, RHO], [RHO, 1]]))


def U(x):
    """Potential energy -log p(x), up to a constant."""
    return 0.5 * x @ PRECISION @ x


def grad_U(x):
    return PRECISION @ x


def random_walk_metropolis(rng):
    x, path, n_accepted = X0, [X0], 0
    for _ in range(N_ITER):
        candidate = rng.normal(x, STEP_RW)
        if np.log(rng.uniform()) < U(x) - U(candidate):
            x, n_accepted = candidate, n_accepted + 1
        path.append(x)
    return np.array(path), n_accepted / N_ITER


def gibbs(rng):
    s = np.sqrt(1 - RHO**2)
    x1, x2 = X0
    path = [(x1, x2)]
    for _ in range(N_ITER):
        x1 = rng.normal(RHO * x2, s)
        path.append((x1, x2))
        x2 = rng.normal(RHO * x1, s)
        path.append((x1, x2))
    return np.array(path)


def hmc(rng):
    """Returns the samples, the leapfrog trajectories and the acceptance rate."""
    x, samples, trajectories, n_accepted = X0, [X0], [], 0
    for _ in range(N_ITER):
        v = rng.normal(size=2)  # resample the momentum
        x_new, v_new = x.copy(), v.copy()
        trajectory = [x_new.copy()]
        v_new = v_new - EPSILON / 2 * grad_U(x_new)  # leapfrog: half step on v,
        for step in range(L):
            x_new = x_new + EPSILON * v_new  # full step on x,
            trajectory.append(x_new.copy())
            if step < L - 1:
                v_new = v_new - EPSILON * grad_U(x_new)  # full step on v,
        v_new = v_new - EPSILON / 2 * grad_U(x_new)  # final half step on v
        delta_H = (U(x_new) + v_new @ v_new / 2) - (U(x) + v @ v / 2)
        if np.log(rng.uniform()) < -delta_H:
            x, n_accepted = x_new, n_accepted + 1
        samples.append(x)
        trajectories.append(np.array(trajectory))
    return np.array(samples), trajectories, n_accepted / N_ITER


rw_path, rw_acceptance = random_walk_metropolis(np.random.default_rng(0))
gibbs_path = gibbs(np.random.default_rng(0))
hmc_samples, hmc_trajectories, hmc_acceptance = hmc(np.random.default_rng(0))
print(f"acceptance: random walk {rw_acceptance:.2f}, HMC {hmc_acceptance:.2f}")

# Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
TARGET, START, SAMPLE = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
grid = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(grid, grid)
mahalanobis = (X1**2 - 2 * RHO * X1 * X2 + X2**2) / (1 - RHO**2)

ax_rw, ax_gibbs, ax_hmc = axes
ax_rw.plot(rw_path[:, 0], rw_path[:, 1], color=SAMPLE, lw=1.2, alpha=0.8)
ax_rw.scatter(rw_path[1:, 0], rw_path[1:, 1], color=SAMPLE, s=18, zorder=3)
ax_rw.set_title(f"Random-walk Metropolis ($s = {STEP_RW:g}$)", color=TEXT, fontsize=11)

ax_gibbs.plot(gibbs_path[:, 0], gibbs_path[:, 1], color=SAMPLE, lw=1.2, alpha=0.8)
ax_gibbs.scatter(gibbs_path[2::2, 0], gibbs_path[2::2, 1], color=SAMPLE, s=18, zorder=3)
ax_gibbs.set_title("Gibbs sampling", color=TEXT, fontsize=11)

for trajectory in hmc_trajectories[:N_TRAJECTORIES]:  # only the first ones, for readability
    ax_hmc.plot(trajectory[:, 0], trajectory[:, 1], ".-", color=MUTED, lw=1, ms=3, alpha=0.8)
ax_hmc.plot([], [], ".-", color=MUTED, lw=1, ms=3, label=f"first {N_TRAJECTORIES} leapfrog trajectories")
ax_hmc.scatter(hmc_samples[1:, 0], hmc_samples[1:, 1], color=SAMPLE, s=18, zorder=3, label="samples")
ax_hmc.set_title(f"Hamiltonian Monte Carlo ($\\epsilon = {EPSILON:g}$, $L = {L}$)", color=TEXT, fontsize=11)
ax_hmc.legend(frameon=False, loc="upper left")

for ax in axes:
    ax.contour(X1, X2, mahalanobis, levels=[1, 4, 9], colors=TARGET, linewidths=1.2, alpha=0.7)
    ax.scatter(*X0, color=START, s=50, zorder=4)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
ax_rw.set_ylabel("$x_2$")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
