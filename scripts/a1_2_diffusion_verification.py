"""
Assignment 1.2 - Time Dependent Diffusion Equation: comparison to analytical result

Solves the 2D diffusion equation:
    ∂c/∂t = D∇²c

Boundary conditions:
    c(x, y=1, t) = 1  (top)
    c(x, y=0, t) = 0  (bottom)
    periodic in x direction

Initial condition:
    c(x, y, t=0) = 0 for 0 ≤ x ≤ 1, 0 ≤ y < 1

Task E: Test correctness.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib
import scienceplots

matplotlib.use("TkAgg")
plt.style.use("science")
plt.rcParams.update({"font.size": 10})

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
N = 50  # grid intervals
D = 1.0  # diffusion coefficient
T_sim = 1.0  # total simulation time
n_terms = 100  # number of terms in the analytical solution sum

# Target times for plotting
target_times = [0.000, 0.001, 0.01, 0.1, 1.0]

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

# --- Figure 2: Compare with analytical solution (Task E) ---
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# Plot c(y) profiles at different times
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(target_times[0:])))

for idx, (t_target, color) in enumerate(zip(target_times[0:], colors)):
    i_closest = np.argmin(np.abs(t - t_target))
    t_actual = t[i_closest]
    c = c_history[i_closest]

    # Take profile at x = 0.5 (mid-grid)
    mid_i = N // 2
    c_numerical = c[mid_i, :]

    # Analytical solution
    c_analytical = analytical_solution(grid.y, t_actual, D=D, n_terms=n_terms)

    ax1.plot(
        grid.y,
        c_numerical,
        "o",
        color=color,
        markersize=2,
        label=rf"$t={t_actual:.3f}$ [s]",
    )
    ax1.plot(
        grid.y,
        c_analytical,
        "-",
        color=color,
        linewidth=1,
        # label=rf"$t={t_actual:.3f}$ (ana)",
    )
h, l = ax1.get_legend_handles_labels()
analytical_label = [
    Line2D(
        [0],
        [0],
        alpha=0.7,
        color="gray",
        lw=1,
    )
]
handles = h + analytical_label
labels = l + ["analytical sol." "\n" r"per $t$"]
ax1.set_ylabel(r"Concentration $c(x,y;t)$ [m$^{-2}$]")
ax1.set_xlabel(r"$y$ [m]")
ax1.set_title(r"Concentration profile $c(y)$ at $x=0.5$ [m]")
ax1.legend(
    handles=handles,
    labels=labels,
    loc="upper left",
    fontsize=8,
    ncol=1,
    alignment="right",
)
ax1.grid(True, alpha=0.3)

# Plot error vs analytical solution
ax2 = axes[1]
errors = []
times_for_error = []

for i, ti in enumerate(t):
    if ti > 0:  # skip t=0
        c_numerical = c_history[i][N // 2, :]
        c_analytical = analytical_solution(grid.y, ti, D=D, n_terms=n_terms)
        error = np.max(np.abs(c_numerical - c_analytical))
        errors.append(error)
        times_for_error.append(ti)

ax2.loglog(times_for_error, errors, "b-", linewidth=2)
ax2.set_xlabel(r"Time $t$ [s]")
ax2.set_ylabel(r"Max error $|c_{\text{num}} - c_{\text{ana}}|$ [m$^{-2}$]")
ax2.set_title("Numerical error over time")
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
filename = (
    output_dir
    / f"a1_2_diffusion_verification_dt={np.round(dt, 6)}_Tsim={T_sim}_D={D}_N={N}_nterms={n_terms}.png"
)
plt.savefig(filename, dpi=300, bbox_inches="tight")
print("Saved: ", filename)

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
