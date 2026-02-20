"""
Comparison between various omega values to find the optimal omega value
that minimises the number of iterations needed
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.omega import search_for_optimal_omega

def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y

# Parameters
#N_values = np.logspace(1, 2.5, 4)
N_values = np.arange(5, 200, 15)

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
          f"totalling {sum(iter_list)} iterations.")

    # Save
    omega_values[i] = omega
    n_iterations[i] = n_iter


# Create the plot
COLOUR_OMEGA = "tab:green"
COLOUR_ITER  = "tab:blue"

fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

ax.plot(N_values, omega_values, marker="o", color=COLOUR_OMEGA)
ax.set_xlabel("$N$ (grid size)")
ax.set_ylabel("Optimal omega ($\\omega$)", color=COLOUR_OMEGA)
ax.tick_params(axis="y", colors=COLOUR_OMEGA)
ax.grid(True)

ax2 = plt.twinx(ax)
ax2.plot(N_values, n_iterations, marker="o", color=COLOUR_ITER)
ax2.set_ylabel("Iterations at optimal omega ($\\omega$)", color=COLOUR_ITER)
ax2.tick_params(axis="y", colors=COLOUR_ITER)

ax2.spines["left"].set_color(COLOUR_OMEGA)
ax2.spines["right"].set_color(COLOUR_ITER)
ax.set_title("Optimal omega ($\\omega$) and iterations needed for various $N$")

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
filename = "a1_6_N_vs_omega.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()
