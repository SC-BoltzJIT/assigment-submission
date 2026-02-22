"""
Assignment 1.2 - Time Dependent Diffusion Equation: time-evolution

Solves the 2D diffusion equation:
    ∂c/∂t = D∇²c

Boundary conditions:
    c(x, y=1, t) = 1  (top)
    c(x, y=0, t) = 0  (bottom)
    periodic in x direction

Initial condition:
    c(x, y, t=0) = 0 for 0 ≤ x ≤ 1, 0 ≤ y < 1

Tasks F: Plot at several times.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# import matplotlib as mpl
import scienceplots

# matplotlib.use("TkAgg")
plt.style.use("science")
# plt.rcParams.update({"font.size": 10})

from scicomp3.core.grid import Grid2D
from scicomp3.ode.solver import solve_ivp
from scicomp3.pde.diffusion import (
    diffusion2d_rhs,
    apply_diffusion_bc,
    diffusion_stable_dt,
    analytical_solution,
)

# Output directory
output_dir = Path(__file__).parent.parent / "images" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)


# Parameters
N = 100  # grid intervals
D = 1.0  # diffusion coefficient
T_sim = 1.0  # total simulation time

# Target times for plotting
target_times = [0, 0.001, 0.01, 0.1, 1.0]

# Create grid
grid = Grid2D(N=N, L=1.0)

# Setup diffusion IVP
dt = diffusion_stable_dt(D, grid.dx)
c0 = np.zeros((N + 1, N + 1))
apply_diffusion_bc(c0)


def enforce_bc(t, y):
    apply_diffusion_bc(y)
    return y


# Solve diffusion equation
print(f"Solving diffusion equation (N={N}, D={D}, T_sim={T_sim})...")
result = solve_ivp(
    diffusion2d_rhs,
    t_span=(0, T_sim),
    y0=c0,
    method="forward_euler",
    dt=dt,
    args=(D, grid.dx),
    post_step=enforce_bc,
    save_interval=10,
)
t, c_history = result.t, result.y
print(f"Completed: {len(t)} saved time points")

# --- Figure 1: 2D concentration at several times (Task F) ---
fig1, axes = plt.subplots(
    5, figsize=(3, 7), sharex=True, sharey=True, constrained_layout=True
)

for idx, t_target in enumerate(target_times):
    # Find closest time in saved data
    i_closest = np.argmin(np.abs(t - t_target))
    t_actual = t[i_closest]
    c = c_history[i_closest]

    ax = axes[idx]
    im = ax.pcolormesh(
        grid.X, grid.Y, c, shading="auto", cmap="gist_heat", vmin=0, vmax=1
    )

    ax.tick_params(axis="both", which="minor", direction="out", length=1)
    ax.tick_params(axis="both", which="major", direction="out", length=2.5)
    # give the bottom plot an x-label
    if idx == len(target_times) - 1:
        ax.set_xlabel(r"$x$ [m]")
        ax.set_xticks(
            [np.min(grid.x), (np.max(grid.x) - np.min(grid.x)) / 2, np.max(grid.x)]
        )
    ax.set_ylabel(r"$y$ [m]")
    ax.text(
        x=grid.x[-1],
        y=grid.y[0],
        s=rf"$t\approx$ {t_target:.3f} [s]",
        fontsize=8,
        color="white",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    ax.set_aspect("equal")

    if idx == 0:
        plt.colorbar(
            im,
            ax=ax,
            label=r"$c(x,y;t)$ [m$^{-2}$]",
            location="top",
            orientation="horizontal",
            fraction=0.05,
            pad=0.06,
        )
filename = (
    output_dir / f"a1_2_diffusion_2d_dt={np.round(dt, 6)}_Tsim={T_sim}_D={D}_N={N}.png"
)
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Saved: {filename}")

plt.show()

# Print summary
print("\nSummary:")
print(f"  Grid: {N}x{N} intervals ({N+1}x{N+1} points)")
print(f"  Diffusion coefficient D = {D}")
print(f"  Simulation time: {T_sim}")
print(f"  Time steps saved: {len(t)}")

# Final error at t=1
c_final_numerical = c_history[-1][N // 2, :]
c_final_analytical = analytical_solution(grid.y, t[-1], D=D)
final_error = np.max(np.abs(c_final_numerical - c_final_analytical))
print(f"  Final error at t={t[-1]:.2f}: {final_error:.2e}")
