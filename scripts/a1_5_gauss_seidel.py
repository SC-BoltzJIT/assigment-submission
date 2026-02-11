"""
Assignment 1.5 - Gauss-Seidel Iteration

Solves the 2D Laplace equation ∇²c = 0 using Gauss-Seidel iteration.

Boundary conditions:
    c(x, y=1) = 1  (top)
    c(x, y=0) = 0  (bottom)
    periodic in x direction

The analytical steady-state solution is c(y) = y (linear gradient).
"""

import numpy as np
import matplotlib.pyplot as plt

from scicomp3 import Grid2D, solve_laplace


# Parameters
N = 50  # grid intervals (N+1 points in each direction)
tol = 1e-5  # convergence tolerance

# Create grid
grid = Grid2D(N=N, L=1.0)

# Solve using Gauss-Seidel
print("Solving with Gauss-Seidel iteration...")
result = solve_laplace(grid, method="gauss_seidel", tol=tol)
print(result.message)

# Analytical solution: c(y) = y
c_analytical = grid.Y

# Compute error
error = np.max(np.abs(result.c - c_analytical))
print(f"Maximum error vs analytical solution: {error:.2e}")

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot 1: Numerical solution
ax1 = axes[0]
im1 = ax1.pcolormesh(grid.X, grid.Y, result.c, shading='auto', cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Numerical solution (Gauss-Seidel)')
ax1.set_aspect('equal')
fig.colorbar(im1, ax=ax1, label='c')

# Plot 2: Concentration profile c(y) at x=0.5
ax2 = axes[1]
mid_i = N // 2
ax2.plot(result.c[mid_i, :], grid.y, 'b-', label='Numerical', linewidth=2)
ax2.plot(grid.y, grid.y, 'r--', label='Analytical (c=y)', linewidth=2)
ax2.set_xlabel('Concentration c')
ax2.set_ylabel('y')
ax2.set_title('Concentration profile at x=0.5')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Convergence history
ax3 = axes[2]
ax3.semilogy(result.delta_history)
ax3.axhline(y=tol, color='r', linestyle='--', label=f'tolerance={tol}')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('δ (max change)')
ax3.set_title(f'Convergence ({result.iterations} iterations)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/a1_5_gauss_seidel.png', dpi=150)
plt.show()

print(f"\nSummary:")
print(f"  Grid: {N}x{N} intervals ({N+1}x{N+1} points)")
print(f"  Method: Gauss-Seidel")
print(f"  Iterations: {result.iterations}")
print(f"  Final δ: {result.delta_history[-1]:.2e}")
print(f"  Max error: {error:.2e}")
