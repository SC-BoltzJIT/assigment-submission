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
N = 50
grid = Grid2D(N=N, L=1.0)

# Initial guess: zero everywhere, then apply BCs
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)


omega_optimal, n_iter_opimal, omega_values, n_iterations = search_for_optimal_omega(c0, post_step=fixed_bc, tol=1e-5)



# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

ax.semilogy(omega_values, n_iterations, marker="o",
            label="Iterations needed")
ax.plot(omega_optimal, n_iter_opimal, marker="*",
        label=f"Optimal $\\omega$ = {omega_optimal:.3f}, "\
              f"Number of iterations = {n_iter_opimal}")
ax.set_xlabel("omega ($\\omega$)")
ax.set_ylabel("Number of iterations needed")
ax.set_title("Number of Iterations Needed for various values of omega ($\\omega$)")

ax.legend()
ax.grid(True)

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
filename = "a1_6_seeking_optimal_omega.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()
