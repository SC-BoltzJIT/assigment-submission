"""Animation of SOR convergence with a sink object (Assignment 1.6.K).

Shows the concentration field evolving iteration-by-iteration until
the SOR solver converges to the steady state.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import scienceplots  # noqa: F401

styles = ["science"] if shutil.which("latex") else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.methods import METHODS
from scicomp3.objects.shapes import construct_rectangle
from scicomp3.objects.insulator import get_insulator_grid
from scicomp3.objects.sink import get_sink_grid


# Parameters
N = 50
omega = 1.9
tol = 1e-5
max_iter = 100_000
save_every = 5  # save a frame every N iterations

grid = Grid2D(N=N, L=1.0)

# Initial guess
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# Sink object (same as a1_6_sinks_sor.py)
sink_coords = construct_rectangle(21, 25, 24, 26)
is_insulator = get_insulator_grid(N, None)
is_sink = get_sink_grid(N, sink_coords)

y = c0.copy()
y[is_sink] = 0

# Build step function
make_step = METHODS["sor"]
step_func = make_step(is_insulator, is_sink, omega=omega)

# Run iteration loop, capturing snapshots
snapshots = [y.copy()]
iterations = [0]

for k in range(max_iter):
    y_old = y.copy()
    y = step_func(y, omega=omega)
    apply_diffusion_bc(y)

    delta = np.max(np.abs(y - y_old))

    if (k + 1) % save_every == 0 or delta < tol:
        snapshots.append(y.copy())
        iterations.append(k + 1)

    if delta < tol:
        print(f"Converged at iteration {k + 1}, delta={delta:.2e}")
        break

print(f"Captured {len(snapshots)} frames")

# Create animation
fig, ax = plt.subplots(figsize=(4, 4))

im = ax.pcolormesh(grid.X, grid.Y, snapshots[0], shading="auto",
                   cmap="gist_heat", vmin=0, vmax=1)
fig.colorbar(
    im, ax=ax,
    label=r"$c(x,y)$",
    location="top",
    orientation="horizontal",
    fraction=0.05,
    pad=0.06,
)
ax.tick_params(axis="both", which="minor", direction="out", length=1)
ax.tick_params(axis="both", which="major", direction="out", length=2.5)
ax.set_xlabel(r"$x$ [m]")
ax.set_ylabel(r"$y$ [m]")
title = ax.set_title(f"Iteration {iterations[0]}")
ax.set_aspect("equal")
fig.suptitle(f"SOR with sink object ($\\omega = {omega:.1f}$)")

plt.tight_layout()


def update(frame):
    im.set_array(snapshots[frame].ravel())
    title.set_text(f"Iteration {iterations[frame]}")
    return [im, title]


anim = FuncAnimation(fig, update, frames=len(snapshots), interval=50, blit=True)

# Save
out_dir = Path(__file__).parent.parent / "images" / "gifs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "a1_6_sinks_sor.gif"

print(f"Saving animation to {out_path}...")
anim.save(out_path, writer="pillow", fps=20)
print(f"Animation saved: {out_path}")

plt.show()
