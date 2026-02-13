"""Jacobi iteration for the 2D Laplace equation (Assignment 1.6.H).

Solves the steady-state diffusion equation with Jacobi iteration
on an N=50 grid and compares to the analytical solution c(y) = y.
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

# Initial guess: zero everywhere, then apply BCs
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# Solve
result = solve_bvp(c0, method="jacobi", post_step=fixed_bc, tol=1e-5)

print(f"Jacobi iteration: converged={result.converged}, "
      f"iterations={result.n_iter}, "
      f"final delta={result.delta_history[-1]:.2e}")

# Compare to analytical c(y) = y
c_analytical = grid.Y
error = np.max(np.abs(result.y - c_analytical))
print(f"Max error vs c(y)=y: {error:.2e}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. Concentration field
ax = axes[0]
im = ax.pcolormesh(grid.X, grid.Y, result.y, shading="auto", cmap="viridis")
fig.colorbar(im, ax=ax)
ax.set_title("Jacobi solution c(x, y)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")

# 2. Profile c(y) at x=0.5 vs analytical
ax = axes[1]
mid_i = N // 2
ax.plot(grid.y, result.y[mid_i, :], "o-", markersize=3, label="Jacobi")
ax.plot(grid.y, grid.y, "--", label="Analytical c=y")
ax.set_xlabel("y")
ax.set_ylabel("c")
ax.set_title("Profile c(y) at x = 0.5")
ax.legend()

# 3. Convergence history
ax = axes[2]
ax.semilogy(result.delta_history)
ax.set_xlabel("Iteration k")
ax.set_ylabel(r"$\delta$ (max-norm)")
ax.set_title(f"Convergence ({result.n_iter} iterations)")
ax.grid(True)

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "a1_6_jacobi.png", dpi=150)
print(f"Saved to {out_dir / 'a1_6_jacobi.png'}")

plt.show()
