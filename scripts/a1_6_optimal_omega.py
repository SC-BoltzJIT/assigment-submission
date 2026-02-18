"""
Comparison between various omega values to find the optimal omega value
that minimises the number of iterations needed
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.solver import solve_bvp

def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y

# Parameters
N = 50
grid = Grid2D(N=N, L=1.0)
n_omega_values = 41
omega_min = 0.0
omega_max = 2.0

omega_values = np.linspace(omega_min, omega_max, n_omega_values)
n_iterations = np.empty(n_omega_values)

# Initial guess: zero everywhere, then apply BCs
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

for i, omega in enumerate(omega_values):
    # Solve
    print(f"Solving problem {i}/{n_omega_values} ...")
    result = solve_bvp(c0, method="sor", post_step=fixed_bc, tol=1e-5, omega=omega)
    print(f"Finished computation in {result.n_iter} iterations.")

    # Save 
    n_iterations[i] = result.n_iter


# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

ax.semilogy(omega_values, n_iterations, marker="o",
            label="Iterations needed")
ax.set_xlabel("omega ($\\omega$)")
ax.set_ylabel("Number of iterations needed")
ax.set_title("Number of Iterations Needed for various values of omega ($\\omega$)")

ax.legend()
ax.grid(True)

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "a1_6_optimal_omega.png", dpi=150)
print(f"Saved to {out_dir / 'a1_6_optimal_omega.png'}")

plt.show()