"""Iterative methods for solving the Laplace equation.

Implements Jacobi, Gauss-Seidel, and SOR methods for solving:
    ∇²c = 0
with boundary conditions:
    c(x, y=1) = 1  (top)
    c(x, y=0) = 0  (bottom)
    periodic in x direction
"""

import numpy as np


def gauss_seidel_step(c, dx):
    """Perform one Gauss-Seidel iteration.

    The iteration proceeds along rows (incrementing i for fixed j):
        c^{k+1}_{i,j} = (1/4) * (c^k_{i+1,j} + c^{k+1}_{i-1,j} + c^k_{i,j+1} + c^{k+1}_{i,j-1})

    New values are used as soon as they are calculated (in-place update).

    Args:
        c: Current concentration field (N+1 x N+1 array), modified in-place
        dx: Grid spacing (not used in this method, kept for interface consistency)

    Returns:
        delta: Maximum change in any grid point (convergence measure)
    """
    N = c.shape[0] - 1  # number of intervals
    delta = 0.0

    # Interior points only (j=1 to N-1, i=0 to N with periodic BC in x)
    for j in range(1, N):  # y direction (skip boundaries at j=0 and j=N)
        for i in range(N + 1):  # x direction (all points, with periodic BC)
            # Periodic boundary conditions in x
            i_plus = (i + 1) % (N + 1)
            i_minus = (i - 1) % (N + 1)

            c_old = c[i, j]
            c_new = 0.25 * (c[i_plus, j] + c[i_minus, j] + c[i, j + 1] + c[i, j - 1])
            c[i, j] = c_new

            delta = max(delta, abs(c_new - c_old))

    return delta


def jacobi_step(c, c_new, dx):
    """Perform one Jacobi iteration.

    The Jacobi iteration uses only old values:
        c^{k+1}_{i,j} = (1/4) * (c^k_{i+1,j} + c^k_{i-1,j} + c^k_{i,j+1} + c^k_{i,j-1})

    Requires two separate arrays for old and new values.

    Args:
        c: Current concentration field (N+1 x N+1 array)
        c_new: Array to store new values (N+1 x N+1 array)
        dx: Grid spacing (not used in this method, kept for interface consistency)

    Returns:
        delta: Maximum change in any grid point (convergence measure)
    """
    N = c.shape[0] - 1
    delta = 0.0

    for j in range(1, N):
        for i in range(N + 1):
            i_plus = (i + 1) % (N + 1)
            i_minus = (i - 1) % (N + 1)

            c_new[i, j] = 0.25 * (c[i_plus, j] + c[i_minus, j] + c[i, j + 1] + c[i, j - 1])
            delta = max(delta, abs(c_new[i, j] - c[i, j]))

    return delta


def sor_step(c, dx, omega):
    """Perform one Successive Over Relaxation (SOR) iteration.

    SOR is obtained from Gauss-Seidel by over-correction:
        c^{k+1}_{i,j} = (ω/4) * (c^k_{i+1,j} + c^{k+1}_{i-1,j} + c^k_{i,j+1} + c^{k+1}_{i,j-1})
                        + (1-ω) * c^k_{i,j}

    The method converges for 0 < ω < 2.
    - ω < 1: under-relaxation
    - ω = 1: Gauss-Seidel
    - ω > 1: over-relaxation (faster convergence for optimal ω)

    Args:
        c: Current concentration field (N+1 x N+1 array), modified in-place
        dx: Grid spacing (not used in this method, kept for interface consistency)
        omega: Relaxation parameter (0 < omega < 2)

    Returns:
        delta: Maximum change in any grid point (convergence measure)
    """
    N = c.shape[0] - 1
    delta = 0.0

    for j in range(1, N):
        for i in range(N + 1):
            i_plus = (i + 1) % (N + 1)
            i_minus = (i - 1) % (N + 1)

            c_old = c[i, j]
            c_gs = 0.25 * (c[i_plus, j] + c[i_minus, j] + c[i, j + 1] + c[i, j - 1])
            c_new = omega * c_gs + (1 - omega) * c_old
            c[i, j] = c_new

            delta = max(delta, abs(c_new - c_old))

    return delta


# Method registry
METHODS = {
    "jacobi": jacobi_step,
    "gauss_seidel": gauss_seidel_step,
    "sor": sor_step,
}
