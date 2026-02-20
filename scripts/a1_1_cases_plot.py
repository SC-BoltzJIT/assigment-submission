"""
Assignment 1.1 - Plot All Test Cases

Runs all three initial conditions and generates plots:
- Case i: sin(2πx)
- Case ii: sin(5πx)
- Case iii: sin(5πx) localized to [1/5, 2/5]
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
    initial_condition_case_iii,
)


def fixed_ends(t, y):
    """Enforce fixed boundary conditions: psi=0 at both ends."""
    y[0, 0] = 0
    y[-1, 0] = 0
    return y


# Parameters
c = 1
L = 1
N = 90  # number grid points (N_intervals + 1)
dt = 1e-3
# T_sims = [2.25, 2.25, 1]  # sufficient simulation times to show the blow-up for the forward Euler
T_sims = [
    5e-1,
    2e-1,
    10e-1,
]  # suitable simulation times per case for plotting of symplectic Euler

# Setup grid
grid = Grid1D(N=N - 1, L=L)

# Define test cases
test_cases = [
    (r"Case i: sin($2\pi x$)", T_sims[0], initial_condition_case_i),
    (r"Case ii: sin($5\pi x$)", T_sims[1], initial_condition_case_ii),
    ("Case iii: Localized", T_sims[2], initial_condition_case_iii),
]

# Output directory
output_dir = Path(__file__).parent.parent / "images" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Run each case
results = []
for name, T_sim, ic_func in test_cases:
    print(f"Running {name}...")

    psi0 = ic_func(grid.x)
    v0 = np.zeros(N)
    y0 = np.column_stack([psi0, v0])

    result = solve_ivp(
        wave1d_rhs,
        t_span=(0, T_sim),
        y0=y0,
        method="symplectic_euler",
        dt=dt,
        args=(c, L, N),
        post_step=fixed_ends,
    )
    results.append((name, result))

# Plot each case
for i, (name, result) in enumerate(results):
    amplitudes = result.y[:, :, 0]

    fig, ax = plt.subplots(figsize=(4, 5))
    # times_to_plot = np.arange(len(result.t) - 200, len(result.t), 50)
    times_to_plot = range(0, len(result.t), len(result.t) // 20)
    # for j in range(0, len(result.t), len(result.t) // 20):
    for j in range(len(times_to_plot)):
        k = times_to_plot[j]
        ax.plot(
            grid.x, amplitudes[k], color=plt.cm.cividis(j / (len(times_to_plot) - 1))
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
        [
            f"{t:.2f}"
            for t in np.linspace(np.min(times_to_plot), np.max(times_to_plot), 3)
        ]
    )

    filename = (
        output_dir
        / f"vibrating_string_over_time_case={i+1}_dt={dt}_Tsim={np.round(T_sims[i], 2)}_c={c}_L={L}_N={N}.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")

plt.show()
