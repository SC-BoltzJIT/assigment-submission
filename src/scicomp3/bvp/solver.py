"""Iterative solvers for the 2D Laplace equation.

Solves ∇²c = 0 with boundary conditions:
    c(x, y=1) = 1  (top)
    c(x, y=0) = 0  (bottom)
    periodic in x direction
"""

import numpy as np
from ..core.result import BVPResult
from ..core.grid import Grid2D
from .methods import METHODS


def apply_boundary_conditions(c):
    """Apply boundary conditions to the concentration field.

    Boundary conditions:
        c(x, y=0) = 0  (bottom, j=0)
        c(x, y=1) = 1  (top, j=N)
        periodic in x direction (handled in iteration)

    Args:
        c: Concentration field array, modified in-place
    """
    c[:, 0] = 0   # bottom boundary
    c[:, -1] = 1  # top boundary


def solve_laplace(grid, method="gauss_seidel", tol=1e-5, max_iter=100000,
                  omega=1.0, c0=None):
    """Solve the 2D Laplace equation using iterative methods.

    Solves ∇²c = 0 on a square domain with:
        c(x, y=1) = 1  (top)
        c(x, y=0) = 0  (bottom)
        periodic in x direction

    Args:
        grid: Grid2D object defining the spatial discretization
        method: Iteration method ("jacobi", "gauss_seidel", or "sor")
        tol: Convergence tolerance (stop when max change < tol)
        max_iter: Maximum number of iterations
        omega: Relaxation parameter for SOR (ignored for other methods)
        c0: Initial guess for concentration (default: zeros)

    Returns:
        BVPResult with solution and convergence info
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    N = grid.N

    # Initialize concentration field
    if c0 is not None:
        c = c0.copy()
    else:
        c = np.zeros((N + 1, N + 1))

    # Apply boundary conditions
    apply_boundary_conditions(c)

    # For Jacobi, we need a second array
    if method == "jacobi":
        c_new = np.zeros_like(c)
        apply_boundary_conditions(c_new)

    # Track convergence history
    delta_history = []

    # Iteration loop
    step_func = METHODS[method]
    converged = False

    for k in range(max_iter):
        if method == "jacobi":
            delta = step_func(c, c_new, grid.dx)
            # Swap arrays
            c, c_new = c_new, c
            apply_boundary_conditions(c_new)
        elif method == "sor":
            delta = step_func(c, grid.dx, omega)
        else:
            delta = step_func(c, grid.dx)

        delta_history.append(delta)

        if delta < tol:
            converged = True
            break

    iterations = k + 1
    if converged:
        message = f"Converged after {iterations} iterations (delta={delta:.2e} < tol={tol})"
    else:
        message = f"Did not converge after {max_iter} iterations (delta={delta:.2e})"

    return BVPResult(
        c=c,
        converged=converged,
        iterations=iterations,
        delta_history=np.array(delta_history),
        message=message,
    )
