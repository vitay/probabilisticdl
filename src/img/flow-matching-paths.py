"""Conditional paths vs. marginal flow on the 2D mixture of `score.py` (flow matching chapter).

Straight conditional paths x_t = (1 - t) x_0 + t x_1 join random pairs of a noise sample x_0 ~ N(0, I) and a data
sample x_1 (independent coupling). The marginal velocity v_t(x) = (E[x_1 | x_t = x] - x) / (1 - t) is computed exactly
for the Gaussian mixture; its trajectories start from the same noise samples.
Figures: paths and trajectories (`flow-matching-paths.png`), and an animation showing that both families of particles
have the same distribution p_t at all times (`flow-matching-paths.gif`). The functions are reused by
`flow-matching-sampling.py`.

Run from anywhere: `python src/img/flow-matching-paths.py` writes `flow-matching-paths.png` and
`flow-matching-paths.gif` next to this file.
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).parent))
from score import COMPONENTS_2D  # noqa: E402  (weight, mean, std) of each isotropic component

forward = importlib.import_module("diffusion-forward")
sample_data = forward.sample_data


def component_stats(t):
    """(weight, mean, variance) of the components of p_t, the distribution of x_t = (1 - t) x_0 + t x_1."""
    return [(w, t * mu, t**2 * s**2 + (1 - t) ** 2) for w, mu, s in COMPONENTS_2D]


def density(x, t):
    return sum(w * np.exp(-np.sum((x - m) ** 2, axis=-1) / (2 * v)) / (2 * np.pi * v)
               for w, m, v in component_stats(t))


def velocity(x, t):
    """Exact marginal velocity (E[x_1 | x_t = x] - x) / (1 - t), x of shape (..., 2), t < 1."""
    logs, means = [], []
    for (w, mu, s), (_, m, v) in zip(COMPONENTS_2D, component_stats(t)):
        logs.append(np.log(w) - np.sum((x - m) ** 2, axis=-1) / (2 * v) - np.log(v))
        means.append(mu + t * s**2 / v * (x - m))  # E[x_1 | x_t, component]
    logs = np.stack(logs, axis=-1)
    resp = np.exp(logs - logs.max(axis=-1, keepdims=True))
    resp /= resp.sum(axis=-1, keepdims=True)
    expected_x1 = sum(resp[..., k, None] * means[k] for k in range(len(means)))
    return (expected_x1 - x) / (1 - t)


def integrate(x0, n_steps, t_end=1.0):
    """Euler integration of the marginal flow; the last step stops just before t = 1."""
    times = np.linspace(0, t_end, n_steps + 1)
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for t, t_next in zip(times[:-1], times[1:]):
        x = x + (t_next - t) * velocity(x, min(t, 0.999))
        path.append(x.copy())
    return times, np.array(path)


if __name__ == "__main__":
    N_LINES = 30  # paths drawn in the static figure
    N_PARTICLES = 500  # particles in the animation

    # Palette: categorical slots 1-3 of the dataviz reference palette, plus text/grid tokens
    DENSITY, PATH, NOISE = "#2a78d6", "#eb6834", "#52514e"
    TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

    plt.rcParams.update({
        "font.size": 11,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
    })

    g = np.linspace(-5, 5, 250)
    G1, G2 = np.meshgrid(g, g)
    grid = np.stack([G1, G2], axis=-1)

    def style(ax):
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    # Static figure
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(N_LINES, 2))
    x1 = sample_data(N_LINES, rng)
    _, path = integrate(x0, 200)
    fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax in (ax_c, ax_m):
        ax.contour(G1, G2, np.log(density(grid, 0.999)), levels=8, colors=DENSITY, linewidths=1,
                   linestyles="solid", alpha=0.4)
        ax.scatter(x0[:, 0], x0[:, 1], color=NOISE, s=15, zorder=3, label="noise samples $x_0$")
        style(ax)
    for i in range(N_LINES):
        ax_c.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]], color=PATH, lw=1, alpha=0.8)
        ax_m.plot(path[:, i, 0], path[:, i, 1], color=PATH, lw=1.2, alpha=0.9)
    ax_c.scatter(x1[:, 0], x1[:, 1], color=PATH, s=15, zorder=3, label="data samples $x_1$")
    ax_m.scatter(path[-1, :, 0], path[-1, :, 1], color=PATH, s=15, zorder=3, label="end points $x(1)$")
    ax_c.set_title("Conditional paths: straight lines between random pairs", color=TEXT, fontsize=11)
    ax_m.set_title("Marginal flow: trajectories of $dx/dt = v_t(x)$", color=TEXT, fontsize=11)
    ax_c.set_ylabel("$x_2$")
    ax_c.legend(frameon=False, loc="upper left")
    ax_m.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")

    # Animation: particles on the conditional paths (left) and on the marginal flow (right), same p_t
    rng = np.random.default_rng(1)
    p0 = rng.normal(size=(N_PARTICLES, 2))
    p1 = sample_data(N_PARTICLES, rng)
    times, flow_path = integrate(p0, 200)
    frame_steps = list(range(0, 201, 5))
    fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.9, wspace=0.08)

    def update(k):
        step = frame_steps[k]
        t = times[step]
        for ax, x, title in [(ax_c, (1 - t) * p0 + t * p1, "Conditional paths"),
                             (ax_m, flow_path[step], "Marginal flow")]:
            ax.clear()
            ax.contour(G1, G2, np.log(density(grid, min(t, 0.999))), levels=8, colors=DENSITY, linewidths=1,
                       linestyles="solid", alpha=0.4)
            ax.scatter(x[:, 0], x[:, 1], color=PATH, s=4, alpha=0.6)
            style(ax)
            ax.set_title(f"{title}, $t = {t:.2f}$", color=TEXT, fontsize=11)
        ax_c.set_ylabel("$x_2$")

    hold = [len(frame_steps) - 1] * 8
    animation = FuncAnimation(fig, update, frames=list(range(len(frame_steps))) + hold)
    animation.save(Path(__file__).with_suffix(".gif"), writer=PillowWriter(fps=8), dpi=80,
                   savefig_kwargs={"facecolor": "white"})
