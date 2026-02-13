"""Main BVP (boundary value problem) solver interface."""

import numpy as np
from ..core.result import BVPResult
from .methods import METHODS


def solve_bvp(y0, method="jacobi", bc_func=None, tol=1e-5, max_iter=100_000,
              **kwargs):
    """Solve a steady-state BVP using iterative relaxation.

    Solves nabla^2 y = 0 by iterating until convergence.

    Args:
        y0: Initial guess array (e.g. N+1 x N+1)
        method: Iterative method name (see METHODS registry)
        bc_func: Callable that applies boundary conditions in-place, bc_func(y).
            Called on y0 and after every iteration step.
        tol: Convergence tolerance for max-norm criterion (default: 1e-5)
        max_iter: Maximum number of iterations (default: 100,000)
        **kwargs: Additional arguments passed to the step function
            (e.g. omega for SOR)

    Returns:
        BVPResult with y (solution array), convergence info, and delta history
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    step_func = METHODS[method]

    # Initialise from y0, enforce BCs
    y = y0.copy()
    if bc_func is not None:
        bc_func(y)

    delta_history = []

    for k in range(max_iter):
        y_old = y.copy()
        y = step_func(y, bc_func, **kwargs)

        # Convergence measure (Eq. 14): delta = max|y_new - y_old|
        delta = np.max(np.abs(y - y_old))
        delta_history.append(delta)

        if delta < tol:
            return BVPResult(
                y=y,
                converged=True,
                n_iter=k + 1,
                delta_history=np.array(delta_history),
            )

    return BVPResult(
        y=y,
        converged=False,
        n_iter=max_iter,
        delta_history=np.array(delta_history),
    )
