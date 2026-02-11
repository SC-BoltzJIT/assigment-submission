"""
Assignment 1.1 - Plot All Test Cases

Runs all three initial conditions and generates plots:
- Case i: sin(2πx)
- Case ii: sin(5πx)
- Case iii: sin(5πx) localized to [1/5, 2/5]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scicomp3 import Grid1D, solve_ivp, wave1d_rhs
from scicomp3.pde.wave import (
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
N = 90
dt = 1e-3
T_sim = 10

# Setup grid
grid = Grid1D(N=N, L=L)

# Define test cases
test_cases = [
    ("Case i: sin(2πx)", initial_condition_case_i),
    ("Case ii: sin(5πx)", initial_condition_case_ii),
    ("Case iii: Localized", initial_condition_case_iii),
]

# Output directory
output_dir = Path(__file__).parent.parent / "images" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Run each case
results = []
for name, ic_func in test_cases:
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

    fig, ax = plt.subplots(figsize=(6, 4))
    for j in range(0, len(result.t), len(result.t)//51):
        ax.plot(grid.x, amplitudes[j], color=plt.cm.cividis(j/len(result.t)))

    ax.set_xlabel('Position along string (x)')
    ax.set_ylabel('String amplitude (Ψ)')
    ax.set_title(f'Vibrating string over time - {name}')

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cividis'), ax=ax, label='Time', ticks=np.linspace(0, 1, 3))
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in np.linspace(0, T_sim, 3)])

    filename = output_dir / f"vibrating_string_over_time_case={i}_dt={dt}_Tsim={T_sim}_c={c}_L={L}_N={N}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")

plt.show()
