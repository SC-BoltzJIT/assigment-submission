"""
Comparison between various omega values to find the optimal omega value
that minimises the number of iterations needed

This file generates the data for a1_6_omega_for_various_N_plot.pys
"""

import numpy as np
from joblib import dump
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.omega import search_for_optimal_omega

def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y

# Parameters
N_values = np.arange(5, 100, 10)

grids = [Grid2D(N=N, L=1.0) for N in N_values]

omega_values = np.empty_like(N_values, dtype=float)
n_iterations = np.empty_like(N_values, dtype=int)

for i, (N, grid) in enumerate(zip(N_values, grids)):
    # Initial guess: zero everywhere, then apply BCs
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)

    # Solve
    print(f"Finding optimal omega for N={N} ...")
    omega, n_iter, _, iter_list = search_for_optimal_omega(c0, post_step=fixed_bc)
    print(f"Found omega after {len(iter_list)} tries, "\
          f"totalling {sum(iter_list[:-1])} iterations.")

    # Save
    omega_values[i] = omega
    n_iterations[i] = n_iter


# Save data to file
results = {"N_values": N_values,
           "omega_values": omega_values,
           "n_iterations": n_iterations}

out_dir = Path(__file__).parent.parent / "data"
out_dir.mkdir(parents=True, exist_ok=True)
filename = "n_vs_omega.pkl"
dump(results, out_dir / filename)
