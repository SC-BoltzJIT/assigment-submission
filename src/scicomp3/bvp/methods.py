"""Iterative methods for BVP (steady-state) solvers.

Methods:
- jacobi: Jacobi iteration (requires two arrays, cannot update in place)
- gauss_seidel: Gauss-Seidel iteration (updates in place)
- sor: Successive Over-Relaxation (Gauss-Seidel + over-correction, omega param)

All methods have signature:
    step(y, **kwargs) -> y_new

All methods support an optional ``mask`` keyword argument (integer array,
same shape as y).  Where mask[i,j] == 1 the point is an object (sink)
and its concentration is forced to 0 instead of being updated by the
stencil.  This is used for Assignment 1.6.K and for DLA in Set 2.
"""

import numpy as np


def jacobi_step(y, mask=None, **kwargs):
    """One Jacobi iteration step (Eq. 12).

    c^{k+1}_{i,j} = (1/4)(c^k_{i+1,j} + c^k_{i-1,j} + c^k_{i,j+1} + c^k_{i,j-1})

    Uses np.roll for the five-point stencil (periodic in x via wrap-around).
    Requires two arrays — returns a new array, does not modify y.

    Args:
        y: Current solution array (N+1 x N+1)
        mask: Optional integer array (1 = object/sink, 0 = free)

    Returns:
        y_new: Updated solution array
    """
    y_new = 0.25 * (
        np.roll(y, -1, axis=0) + np.roll(y, 1, axis=0) +
        np.roll(y, -1, axis=1) + np.roll(y, 1, axis=1)
    )
    if mask is not None:
        y_new[mask == 1] = 0.0
    return y_new


def gauss_seidel_step(y, mask=None, **kwargs):
    """One Gauss-Seidel iteration step (Sec. 1.5).

    c^{k+1}_{i,j} = (1/4)(c^k_{i+1,j} + c^{k+1}_{i-1,j}
                         + c^k_{i,j+1} + c^{k+1}_{i,j-1})

    Uses already-updated values as soon as they are available.
    Sweeps rows: incrementing i for fixed j (as stated in the assignment).
    Updates in place — returns the same (modified) array.

    Periodic x boundary is handled via modular indexing.

    Args:
        y: Current solution array (N+1 x N+1), modified in place
        mask: Optional integer array (1 = object/sink, 0 = free)

    Returns:
        y: The same array, updated in place
    """
    n_i, n_j = y.shape
    for j in range(1, n_j - 1):        # interior y-points
        for i in range(n_i):            # all x-points (periodic)
            if mask is not None and mask[i, j]:
                y[i, j] = 0.0
                continue
            i_plus = (i + 1) % n_i
            i_minus = (i - 1) % n_i
            y[i, j] = 0.25 * (y[i_plus, j] + y[i_minus, j] +
                               y[i, j + 1] + y[i, j - 1])
    return y


def sor_step(y, omega=1.5, mask=None, **kwargs):
    """One SOR (Successive Over-Relaxation) iteration step (Sec. 1.6).

    c^{k+1}_{i,j} = (omega/4)(c^k_{i+1,j} + c^{k+1}_{i-1,j}
                              + c^k_{i,j+1} + c^{k+1}_{i,j-1})
                   + (1 - omega) * c^k_{i,j}

    Gauss-Seidel sweep with over-correction.  omega=1 recovers Gauss-Seidel.
    Converges for 0 < omega < 2.  Optimal omega is in [1.7, 2) for the
    diffusion problem, depending on grid size N.

    Updates in place — returns the same (modified) array.
    Periodic x boundary is handled via modular indexing.

    Args:
        y: Current solution array (N+1 x N+1), modified in place
        omega: Relaxation parameter (default: 1.5)
        mask: Optional integer array (1 = object/sink, 0 = free)

    Returns:
        y: The same array, updated in place
    """
    n_i, n_j = y.shape
    w4 = omega / 4.0
    w1 = 1.0 - omega
    for j in range(1, n_j - 1):        # interior y-points
        for i in range(n_i):            # all x-points (periodic)
            if mask is not None and mask[i, j]:
                y[i, j] = 0.0
                continue
            i_plus = (i + 1) % n_i
            i_minus = (i - 1) % n_i
            y[i, j] = w4 * (y[i_plus, j] + y[i_minus, j] +
                             y[i, j + 1] + y[i, j - 1]) + w1 * y[i, j]
    return y


# Method registry (Strategy Pattern) — mirrors ode/methods.py METHODS
METHODS = {
    "jacobi": jacobi_step,
    "gauss_seidel": gauss_seidel_step,
    "sor": sor_step,
}
