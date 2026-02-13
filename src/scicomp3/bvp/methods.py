"""Iterative methods for BVP (steady-state) solvers.

Methods:
- jacobi: Jacobi iteration (requires two arrays, cannot update in place)

All methods have signature:
    step(y, bc_func, **kwargs) -> y_new
"""

import numpy as np


def jacobi_step(y, bc_func, **kwargs):
    """One Jacobi iteration step (Eq. 12).

    c^{k+1}_{i,j} = (1/4)(c^k_{i+1,j} + c^k_{i-1,j} + c^k_{i,j+1} + c^k_{i,j-1})

    Uses np.roll for the five-point stencil (periodic in x via wrap-around).
    Requires two arrays — returns a new array, does not modify y.

    Args:
        y: Current solution array (N+1 x N+1)
        bc_func: Callable that applies boundary conditions in-place, bc_func(y)

    Returns:
        y_new: Updated solution array
    """
    y_new = 0.25 * (
        np.roll(y, -1, axis=0) + np.roll(y, 1, axis=0) +
        np.roll(y, -1, axis=1) + np.roll(y, 1, axis=1)
    )
    bc_func(y_new)
    return y_new


# Method registry (Strategy Pattern) — mirrors ode/methods.py METHODS
METHODS = {
    "jacobi": jacobi_step,
}
