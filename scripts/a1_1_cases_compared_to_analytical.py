"""
Assignment 1.1 - Compute numerical deviations from analytical expectation

Runs the two initial conditions that have an analytical solution
and generates plots of the errors through time:
- Case i: sin(2πx)
- Case ii: sin(5πx)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import scienceplots

matplotlib.use("TkAgg")
plt.style.use("science")
plt.rcParams.update({"font.size": 16})

from scicomp3.core.grid import Grid1D
from scicomp3.ode.solver import solve_ivp
from scicomp3.pde.wave import (
    wave1d_rhs,
    initial_condition_case_i,
    initial_condition_case_ii,
    analytical_vibration_sol,
)


def fixed_ends(t, y):
    """Enforce fixed boundary conditions: psi=0 at both ends."""
    y[0, 0] = 0
    y[-1, 0] = 0
    return y


# Parameters
c = 1.0
L = 1.0
N = 90  # number grid points (N_intervals + 1)
dt = 1e-3
T_sims = [5e1, 2e1]  # suffificent simulation time to see error accumulation

# Setup grid
grid = Grid1D(N=N - 1, L=L)

# Define test cases
test_cases = [
    (r"Case i: sin($2\pi x$)", T_sims[0], initial_condition_case_i),
    (r"Case ii: sin($5\pi x$)", T_sims[1], initial_condition_case_ii),
]

# Output directory
output_dir = Path(__file__).parent.parent / "images" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Run each case
combined_results = []
modes = [2, 5]
for i, (name, T_sim, ic_func) in enumerate(test_cases):
    print(f"Running {name}...")

    psi0 = ic_func(grid.x)
    v0 = np.zeros(N)
    y0 = np.column_stack([psi0, v0])

    # compute numerical solution
    result = solve_ivp(
        wave1d_rhs,
        t_span=(0, T_sim),
        y0=y0,
        method="symplectic_euler",
        dt=dt,
        args=(c, L, N),
        post_step=fixed_ends,
    )

    # compute analytical solution
    analytical_amplitudes = []
    errors = []
    time = result.t
    amplitudes = result.y[:, :, 0]
    for j in range(len(time)):
        t = time[j]
        analytical_sol = analytical_vibration_sol(
            grid.x, t=t, ns=[modes[i]], cos_amps=[1], sin_amps=[0], c=c, L=L
        )
        analytical_amplitudes.append(analytical_sol)
        errors.append(np.sum(np.abs(amplitudes[j] - analytical_sol)))

    combined_results.append((name, result, analytical_amplitudes, errors))

# Plot the numerical solution of each case next to the analytical
for i, (name, result, analytical_amplitudes, errors) in enumerate(combined_results):
    amplitudes = result.y[:, :, 0]

    fig, ax = plt.subplots(figsize=(3, 5))
    final_idx_to_plot = int((T_sims[i] / dt) * 0.005)
    for j in np.linspace(0, final_idx_to_plot, 5):
        j = int(j)
        ax.plot(
            grid.x,
            amplitudes[j],
            color=plt.cm.cividis(j / final_idx_to_plot),
            linewidth=3,
            alpha=0.7,
            label="numerical" if j == 0 else "",
        )
        ax.plot(
            grid.x,
            analytical_amplitudes[j],
            color=plt.cm.cividis(j / final_idx_to_plot),
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="analytical" if j == 0 else "",
        )

    ax.set_xlabel(r"Position $x$ [m]")
    ax.set_ylabel(r"Amplitude $\Psi$ [m]")
    fig.suptitle(name, bbox=dict(facecolor="none", edgecolor="black", pad=3.0))

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="cividis"),
        ax=ax,
        label="Time [s]",
        ticks=np.linspace(0, 1, 3),
        location="top",
        orientation="horizontal",
    )
    cbar.ax.set_xticklabels(
        [f"{t:.2f}" for t in np.linspace(0, result.t[final_idx_to_plot], 3)]
    )
    if i == 0:
        ax.legend(fontsize=12, loc="upper right")

    filename = (
        output_dir
        / f"a1_vibrating_string_num_vs_anal_case={i+1}_dt={dt}_Tsim={T_sim}_c={c}_L={L}_N={N}.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")

plt.show()


# Plot the errors
for i, (name, result, analytical_amplitudes, errors) in enumerate(combined_results):
    amplitudes = result.y[:, :, 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.t, errors)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Sum of abs. errors [m]")
    fig.suptitle(name, bbox=dict(facecolor="none", edgecolor="black", pad=3.0))

    filename = (
        output_dir
        / f"a1_num_error_over_time_case={i+1}_dt={dt}_Tsim={T_sim}_c={c}_L={L}_N={N}.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")

plt.show()
