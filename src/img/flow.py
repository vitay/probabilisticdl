"""Schematic of a flow in one dimension (flow matching chapter).

A velocity field v_t(x) in the (t, x) plane transports the noise distribution p_0 = N(0, 1) (left) onto a bimodal
data distribution p_1 (right). The field is the exact marginal velocity of flow matching with straight paths
x_t = (1 - t) x_0 + t x_1 and independent coupling: v_t(x) = (E[x_1 | x_t = x] - x) / (1 - t), computed in closed form
for a Gaussian mixture. The functions are reused by `flow-matching-1d.py`.

Run from anywhere: `python src/img/flow.py` writes `flow.png` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COMPONENTS = [(0.5, -2.0, 0.4), (0.5, 2.0, 0.4)]  # (weight, mean, std) of the 1D data distribution


def marginal_density(x, t):
    """Density of x_t = (1 - t) x_0 + t x_1: mixture of N(t m_k, sqrt(t^2 s_k^2 + (1 - t)^2))."""
    return sum(w * np.exp(-(x - t * m) ** 2 / (2 * (t**2 * s**2 + (1 - t) ** 2)))
               / np.sqrt(2 * np.pi * (t**2 * s**2 + (1 - t) ** 2)) for w, m, s in COMPONENTS)


def velocity(x, t):
    """Exact marginal velocity v_t(x) = (E[x_1 | x_t = x] - x) / (1 - t), for t < 1."""
    logs, means = [], []
    for w, m, s in COMPONENTS:
        var = t**2 * s**2 + (1 - t) ** 2  # variance of x_t given the component
        logs.append(np.log(w) - (x - t * m) ** 2 / (2 * var) - 0.5 * np.log(var))
        means.append(m + t * s**2 / var * (x - t * m))  # E[x_1 | x_t, component]
    logs = np.stack(logs)
    resp = np.exp(logs - logs.max(axis=0))
    resp /= resp.sum(axis=0)
    expected_x1 = np.sum(resp * np.stack(means), axis=0)
    return (expected_x1 - x) / (1 - t)


def integrate(x0, n_steps=500, t_end=0.999):
    """Euler integration of dx/dt = v_t(x) from t = 0, returning the times and the trajectories."""
    times = np.linspace(0, t_end, n_steps + 1)
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for t, t_next in zip(times[:-1], times[1:]):
        x = x + (t_next - t) * velocity(x, t)
        path.append(x.copy())
    return times, np.array(path)


if __name__ == "__main__":
    N_TRAJECTORIES = 14

    # Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
    DENSITY, FIELD, TRAJ = "#2a78d6", "#52514e", "#eb6834"
    TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

    plt.rcParams.update({
        "font.size": 11,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.linspace(-4, 4, 400)
    width = 0.18  # horizontal extent of the density profiles

    # Velocity field: arrows in the (t, x) plane, direction (1, v_t(x)), normalized length
    T, X = np.meshgrid(np.linspace(0.04, 0.94, 16), np.linspace(-3.6, 3.6, 17))
    V = velocity(X, T)
    norm = np.sqrt(1 + V**2)
    ax.quiver(T, X, 1 / norm, V / norm, color=FIELD, alpha=0.35, angles="xy", scale_units="xy", scale=12,
              width=0.003)

    # Densities at t = 0, 0.5 and 1
    for t, alpha in [(0.0, 0.35), (0.5, 0.2), (0.999, 0.35)]:
        d = marginal_density(y, t)
        side = -1 if t == 0 else 1
        ax.fill_betweenx(y, t, t + side * width * d / d.max(), color=DENSITY, alpha=alpha, lw=0)
    ax.text(-0.1, 3.75, "noise $p_0$", ha="center", color=DENSITY)
    ax.text(0.5 + 0.09, 3.75, "$p_{0.5}$", ha="center", color=DENSITY)
    ax.text(1.09, 3.75, "data $p_1$", ha="center", color=DENSITY)

    # Trajectories
    x0 = np.sort(np.random.default_rng(0).normal(size=N_TRAJECTORIES))
    times, path = integrate(x0)
    for k in range(N_TRAJECTORIES):
        ax.plot(times, path[:, k], color=TRAJ, lw=1.5, alpha=0.9)
    ax.plot([], [], color=TRAJ, lw=1.5, label="trajectories $dx / dt = v_t(x)$")
    ax.plot([], [], color=FIELD, marker=r"$\rightarrow$", ls="none", ms=12, alpha=0.6, label="velocity field $v_t(x)$")

    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-4, 4.1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([])
    ax.set_xlabel("time $t$")
    ax.set_ylabel("$x$")
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.32))
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
