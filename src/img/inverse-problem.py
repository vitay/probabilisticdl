"""An inverse problem on the manifold of `image-manifold.py` (latent variables chapter).

Only a horizontal strip of three rows through the middle of an image of a "7" is observed, with Gaussian noise.
With a uniform prior over the rotation and the size, the posterior p(z | y) is computed on a grid of the latent space.
Panels: the true image (unknown), the observation, the posterior, and a few posterior samples (plausible restorations).

Run from anywhere: `python src/img/inverse-problem.py` writes `inverse-problem.png` next to this file.
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
manifold = importlib.import_module("image-manifold")
seven, SIZE = manifold.seven, manifold.SIZE

TRUE_ANGLE, TRUE_SCALE = 0.0, 0.8  # latent variables of the true image
SIGMA = 0.1  # standard deviation of the observation noise
N_SAMPLES = 6

angles = np.linspace(-90, 90, 73)
scales = np.linspace(0.5, 1.0, 26)
decoded = np.array([[seven(np.radians(a), s) for a in angles] for s in scales])  # (scales, angles, SIZE, SIZE)

# Observation: a noisy horizontal strip through the middle of the image
rows = np.arange(SIZE)[:, None] * np.ones((1, SIZE))
mask = (rows >= 12) & (rows < 15)
rng = np.random.default_rng(0)
x_true = seven(np.radians(TRUE_ANGLE), TRUE_SCALE)
y = x_true + SIGMA * rng.normal(size=x_true.shape)

# Posterior on the grid: Gaussian likelihood of the observed pixels, uniform prior
log_likelihood = -np.sum(((decoded - y) ** 2) * mask, axis=(2, 3)) / (2 * SIGMA**2)
posterior = np.exp(log_likelihood - log_likelihood.max())
posterior /= posterior.sum()

# A few samples of the posterior
flat = rng.choice(posterior.size, size=N_SAMPLES, p=posterior.ravel())
samples = [np.unravel_index(k, posterior.shape) for k in flat]

# Palette: text/grid tokens of the dataviz reference palette, sequential blues for the posterior
TEXT, MUTED, GRID, TRUE = "#0b0b0b", "#52514e", "#e4e3df", "#eb6834"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig = plt.figure(figsize=(16, 4.6))
outer = fig.add_gridspec(1, 4, width_ratios=[1, 1, 2.2, 2.2], wspace=0.35)
ax_true = fig.add_subplot(outer[0])
ax_obs = fig.add_subplot(outer[1])
ax_post = fig.add_subplot(outer[2])
inner = outer[3].subgridspec(2, N_SAMPLES // 2, wspace=0.1, hspace=0.3)

ax_true.imshow(x_true, cmap="gray_r", vmin=0, vmax=1)
ax_true.set_title("True image $x$\n(unknown)", color=TEXT, fontsize=11)

observed = np.where(mask, y, np.nan)
cmap = plt.get_cmap("gray_r").copy()
cmap.set_bad("#c9c8c4")  # unobserved pixels
ax_obs.imshow(observed, cmap=cmap, vmin=0, vmax=1)
ax_obs.set_title("Observation $y$\n(noisy strip)", color=TEXT, fontsize=11)
for ax in (ax_true, ax_obs):
    ax.set_xticks([])
    ax.set_yticks([])

ax_post.imshow(posterior, origin="lower", aspect="auto", cmap="Blues",
               extent=(angles[0], angles[-1], scales[0], scales[-1]))
ax_post.scatter(TRUE_ANGLE, TRUE_SCALE, marker="x", color=TRUE, s=60, lw=2, label="true latent variables")
ax_post.set_xlim(-45, 45)  # the posterior is negligible outside
ax_post.set_xlabel("rotation $z_1$ (degrees)")
ax_post.set_ylabel("size $z_2$")
ax_post.set_title("Posterior $p(z | y)$", color=TEXT, fontsize=11)
ax_post.legend(frameon=False, loc="upper right")

for k, (i, j) in enumerate(samples):
    ax = fig.add_subplot(inner[k // (N_SAMPLES // 2), k % (N_SAMPLES // 2)])
    ax.imshow(decoded[i, j], cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{angles[j]:.0f}°, {scales[i]:.2f}", fontsize=9)
    if k == 1:
        ax.set_title("Samples of the posterior (plausible restorations)", color=TEXT, fontsize=11)

fig.savefig(Path(__file__).with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
