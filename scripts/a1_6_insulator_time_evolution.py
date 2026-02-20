"""Time evolution of diffusion with insulating objects (Assignment 1.6.L).

Solves the time-dependent diffusion equation (Assignment 1.2) with three
domains side-by-side to show how an insulating object changes the transient
behaviour and final steady state:
  1. No object
  2. Sink object (c=0 inside)
  3. Insulating object (zero-flux Neumann BC at surface)

The diffusion PDE  dc/dt = D nabla^2 c  is integrated forward in time
using forward Euler (eq. 7 of the assignment).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc, diffusion_stable_dt
from scicomp3.ode.solver import solve_ivp
from scicomp3.objects.shapes import construct_rectangle
from scicomp3.objects.insulator import get_insulator_grid
from scicomp3.objects.sink import get_sink_grid


# --- Parameters ---
N = 50
D = 1.0
T_sim = 0.5

grid = Grid2D(N=N, L=1.0)
dt = diffusion_stable_dt(D, grid.dx)
dx = grid.dx

# Object: centred rectangle
obj_coords = construct_rectangle(20, 30, 20, 30)


# --- Build RHS and post_step for each case ---

def make_diffusion_rhs(is_insulator, is_sink):
    """Return a diffusion RHS function that respects insulator/sink objects.

    For insulator points: dcdt = 0 (concentration frozen).
    For free points next to an insulator: the insulator neighbour is excluded
    from the Laplacian stencil (Neumann zero-flux), with the neighbour count
    adjusted accordingly.
    For sink points: dcdt = 0 (concentration forced to 0 via post_step).
    """
    has_insulator = np.any(is_insulator)
    has_sink = np.any(is_sink)

    if has_insulator:
        # Precompute: for each free point, how many non-insulator neighbours?
        neighbour_count = (
            (~np.roll(is_insulator, -1, axis=0)).astype(float) +
            (~np.roll(is_insulator,  1, axis=0)).astype(float) +
            (~np.roll(is_insulator, -1, axis=1)).astype(float) +
            (~np.roll(is_insulator,  1, axis=1)).astype(float)
        )
        # Avoid division by zero at insulator points (value unused)
        neighbour_count[is_insulator] = 1.0

    def rhs(t, c, D, dx):
        c_ip = np.roll(c, -1, axis=0)
        c_im = np.roll(c,  1, axis=0)
        c_jp = np.roll(c, -1, axis=1)
        c_jm = np.roll(c,  1, axis=1)

        if has_insulator:
            # Zero out contributions from insulator neighbours
            ins_float = is_insulator.astype(float)
            insulator_contrib = (
                np.roll(c * ins_float, -1, axis=0) +
                np.roll(c * ins_float,  1, axis=0) +
                np.roll(c * ins_float, -1, axis=1) +
                np.roll(c * ins_float,  1, axis=1)
            )
            neighbour_sum = (c_ip + c_im + c_jp + c_jm) - insulator_contrib
            laplacian = (neighbour_sum - neighbour_count * c) / dx**2
            # Insulator points: no change
            laplacian[is_insulator] = 0.0
        else:
            laplacian = (c_ip + c_im + c_jp + c_jm - 4 * c) / dx**2

        if has_sink:
            laplacian[is_sink] = 0.0

        return D * laplacian

    return rhs


def make_post_step(is_insulator, is_sink, c_insulator_init):
    """Return a post_step callback that enforces BCs + object constraints."""
    has_insulator = np.any(is_insulator)
    has_sink = np.any(is_sink)

    def post_step(t, y):
        apply_diffusion_bc(y)
        if has_sink:
            y[is_sink] = 0.0
        if has_insulator:
            y[is_insulator] = c_insulator_init[is_insulator]
        return y

    return post_step


# --- Run three simulations ---
save_interval = 50

cases = {}

# Case 1: no object
print("Solving: no object...")
is_ins_none = get_insulator_grid(N, None)
is_snk_none = get_sink_grid(N, None)
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)
rhs_none = make_diffusion_rhs(is_ins_none, is_snk_none)
ps_none = make_post_step(is_ins_none, is_snk_none, c0)
cases["No object"] = solve_ivp(
    rhs_none, t_span=(0, T_sim), y0=c0,
    method="forward_euler", dt=dt, args=(D, dx),
    post_step=ps_none, save_interval=save_interval,
)

# Case 2: sink
print("Solving: sink object...")
is_ins_sink = get_insulator_grid(N, None)
is_snk_sink = get_sink_grid(N, obj_coords)
c0_sink = np.zeros(grid.shape)
apply_diffusion_bc(c0_sink)
c0_sink[is_snk_sink] = 0.0
rhs_sink = make_diffusion_rhs(is_ins_sink, is_snk_sink)
ps_sink = make_post_step(is_ins_sink, is_snk_sink, c0_sink)
cases["Sink (c=0)"] = solve_ivp(
    rhs_sink, t_span=(0, T_sim), y0=c0_sink,
    method="forward_euler", dt=dt, args=(D, dx),
    post_step=ps_sink, save_interval=save_interval,
)

# Case 3: insulator
print("Solving: insulator object...")
is_ins_ins = get_insulator_grid(N, obj_coords)
is_snk_ins = get_sink_grid(N, None)
c0_ins = np.zeros(grid.shape)
apply_diffusion_bc(c0_ins)
# Insulator points keep their initial value (0) throughout
rhs_ins = make_diffusion_rhs(is_ins_ins, is_snk_ins)
ps_ins = make_post_step(is_ins_ins, is_snk_ins, c0_ins)
cases["Insulator (dc/dn=0)"] = solve_ivp(
    rhs_ins, t_span=(0, T_sim), y0=c0_ins,
    method="forward_euler", dt=dt, args=(D, dx),
    post_step=ps_ins, save_interval=save_interval,
)

for name, res in cases.items():
    print(f"  {name}: {len(res.t)} frames, t_max={res.t[-1]:.4f}")

# --- Animation: 3 concentration fields side by side ---
labels = list(cases.keys())
results = list(cases.values())
n_frames = len(results[0].t)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ims = []
title_texts = []
for col, (ax, label) in enumerate(zip(axes, labels)):
    im = ax.pcolormesh(grid.X, grid.Y, results[col].y[0],
                       shading="auto", cmap="hot", vmin=0, vmax=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ttl = ax.set_title(f"{label}\nt = {results[col].t[0]:.4f}")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="Concentration c")
    ims.append(im)
    title_texts.append(ttl)

plt.tight_layout()


def update(frame):
    updated = []
    for col in range(3):
        ims[col].set_array(results[col].y[frame].ravel())
        title_texts[col].set_text(
            f"{labels[col]}\nt = {results[col].t[frame]:.4f}")
        updated.extend([ims[col], title_texts[col]])
    return updated


anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=True)

# Save
out_dir = Path(__file__).parent.parent / "images" / "gifs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "a1_6_insulator_time_evolution.gif"

print(f"Saving animation ({n_frames} frames) to {out_path}...")
anim.save(out_path, writer="pillow", fps=20)
print(f"Animation saved: {out_path}")

plt.show()
