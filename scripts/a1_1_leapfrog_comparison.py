"""
Assignment 1.1 (Optional) - Leapfrog vs Symplectic Euler comparison

Compares the second-order Leapfrog (Velocity Verlet / Kick-Drift-Kick)
integrator against the first-order Symplectic Euler for the wave equation.

Produces three figures:
1. Error accumulation over time (cases i & ii)
2. Energy conservation over time (cases i & ii)
3. Case iii qualitative snapshot comparison
"""

import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("TkAgg")

if shutil.which("latex"):
    import scienceplots
    plt.style.use("science")

plt.rcParams.update({"font.size": 16})

from scicomp3.core.grid import Grid1D
from scicomp3.ode.solver import solve_ivp
from scicomp3.pde.wave import (
    wave1d_rhs,
    initial_condition_case_i,
    initial_condition_case_ii,
    initial_condition_case_iii,
    analytical_vibration_sol,
)


def fixed_ends(t, y):
    """Enforce fixed boundary conditions: psi=0 at both ends."""
    y[0, 0] = 0
    y[-1, 0] = 0
    return y


def compute_energy(y, dx, c):
    """Compute total energy (kinetic + potential) of the wave."""
    psi = y[:, 0]
    v = y[:, 1]
    KE = 0.5 * np.sum(v**2) * dx
    dpsi = np.gradient(psi, dx)
    PE = 0.5 * c**2 * np.sum(dpsi**2) * dx
    return KE + PE


# Parameters — use dt=5e-3 where temporal error differences are visible
c = 1.0
L = 1.0
N = 90
dt = 5e-3
T_sim = 20.0
dx = L / (N + 1)

# Setup grid
grid = Grid1D(N=N - 1, L=L)

# Define test cases with analytical solutions
test_cases = [
    ("Case i: $\\sin(2\\pi x)$", initial_condition_case_i, [2], [1], [0]),
    ("Case ii: $\\sin(5\\pi x)$", initial_condition_case_ii, [5], [1], [0]),
]

methods = ["symplectic_euler", "leapfrog"]
method_labels = {
    "symplectic_euler": "Symplectic Euler (1st order)",
    "leapfrog": "Leapfrog (2nd order)",
}

# Output directory
output_dir = Path(__file__).parent.parent / "images" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Save interval to keep memory manageable
save_interval = max(1, int(0.1 / dt))

# ============================================================
# Figure 1: Error accumulation over time
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

all_results = {}  # cache for energy plot

for case_idx, (name, ic_func, ns, cos_amps, sin_amps) in enumerate(test_cases):
    ax = axes[case_idx]

    for method in methods:
        print(f"Running {name} with {method}...")

        psi0 = ic_func(grid.x)
        v0 = np.zeros(N)
        y0 = np.column_stack([psi0, v0])

        result = solve_ivp(
            wave1d_rhs,
            t_span=(0, T_sim),
            y0=y0,
            method=method,
            dt=dt,
            args=(c, L, N),
            post_step=fixed_ends,
            save_interval=save_interval,
        )
        all_results[(case_idx, method)] = result

        # Compute error at each saved time step
        errors = []
        for j in range(len(result.t)):
            analytical_sol = analytical_vibration_sol(
                grid.x, t=result.t[j],
                ns=ns, cos_amps=cos_amps, sin_amps=sin_amps, c=c, L=L,
            )
            errors.append(np.sum(np.abs(result.y[j, :, 0] - analytical_sol)))

        ax.plot(result.t, errors, label=method_labels[method], linewidth=1.5)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Sum of abs. errors [m]")
    ax.set_title(name)
    ax.legend(fontsize=11)

fig.suptitle(
    f"Error accumulation: Leapfrog vs Symplectic Euler ($N={N}$, $\\Delta t={dt}$)",
    fontsize=14,
)
plt.tight_layout()

filename = output_dir / f"a1_leapfrog_vs_symplectic_error_dt={dt}_T={T_sim}_N={N}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Saved: {filename}")

plt.show()

# ============================================================
# Figure 2: Energy conservation over time
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for case_idx, (name, ic_func, ns, cos_amps, sin_amps) in enumerate(test_cases):
    ax = axes[case_idx]

    for method in methods:
        result = all_results[(case_idx, method)]
        energies = np.array([
            compute_energy(result.y[j], dx, c) for j in range(len(result.t))
        ])
        E0 = energies[0]
        relative_energy = (energies - E0) / E0

        ax.plot(result.t, relative_energy, label=method_labels[method], linewidth=1.5)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Relative energy error $(E - E_0) / E_0$")
    ax.set_title(name)
    ax.legend(fontsize=11)

fig.suptitle(
    f"Energy conservation: Leapfrog vs Symplectic Euler ($N={N}$, $\\Delta t={dt}$)",
    fontsize=14,
)
plt.tight_layout()

filename = output_dir / f"a1_leapfrog_vs_symplectic_energy_dt={dt}_T={T_sim}_N={N}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Saved: {filename}")

plt.show()

# ============================================================
# Figure 3: Case iii qualitative snapshot comparison
# ============================================================
print("Running Case iii with both methods...")

T_iii = 5.0
psi0_iii = initial_condition_case_iii(grid.x)
v0_iii = np.zeros(N)
y0_iii = np.column_stack([psi0_iii, v0_iii])

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
snapshot_times = [0.0, 0.5, 1.0, 2.0, 4.0]

for m_idx, method in enumerate(methods):
    ax = axes[m_idx]

    result = solve_ivp(
        wave1d_rhs,
        t_span=(0, T_iii),
        y0=y0_iii,
        method=method,
        dt=dt,
        args=(c, L, N),
        post_step=fixed_ends,
    )

    for snap_t in snapshot_times:
        idx = np.argmin(np.abs(result.t - snap_t))
        ax.plot(grid.x, result.y[idx, :, 0], label=f"$t={snap_t:.1f}$s", linewidth=1.5)

    ax.set_xlabel(r"Position $x$ [m]")
    ax.set_ylabel(r"Amplitude $\Psi$ [m]")
    ax.set_title(method_labels[method])
    ax.legend(fontsize=10)

fig.suptitle(
    f"Case iii: $\\sin(5\\pi x)$ pulse — snapshots ($N={N}$, $\\Delta t={dt}$)",
    fontsize=14,
)
plt.tight_layout()

filename = output_dir / f"a1_leapfrog_vs_symplectic_case_iii_dt={dt}_T={T_iii}_N={N}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Saved: {filename}")

plt.show()
