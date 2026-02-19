"""Main BVP (boundary value problem) solver interface."""

import numpy as np
from ..core.result import BVPResult
from .methods import METHODS
from ..objects.insulator import get_insulator_grid
from ..objects.sink import get_sink_grid


def solve_bvp(y0, method="jacobi", tol=1e-5, max_iter=100_000,
              post_step=None, insulator_coordinates=None,
              sink_coordinates=None, **kwargs):
    """Solve a steady-state BVP using iterative relaxation.

    Solves nabla^2 y = 0 by iterating until convergence.

    Args:
        y0: Initial guess array (e.g. N+1 x N+1)
        method: Iterative method name (see METHODS registry)
        tol: Convergence tolerance for max-norm criterion (default: 1e-5)
        max_iter: Maximum number of iterations (default: 100,000)
        post_step: Optional callback f(k, y) -> y applied after each step,
            e.g. to enforce boundary conditions. Must return the modified y.
        insulator_coordinates: An array of coordinates that signify which
            points are insulating
        **kwargs: Additional arguments passed to the step function
            (e.g. omega for SOR)

    Returns:
        BVPResult with y (solution array), convergence info, and delta history
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    # Initialise from y0, enforce BCs
    y = y0.copy()
    if post_step is not None:
        y = post_step(0, y)

    # Initialise insulator mask
    is_insulator = get_insulator_grid(len(y) - 1, insulator_coordinates)
    y[is_insulator] = 1

    # Initialise sink mask
    is_sink = get_sink_grid(len(y) - 1, sink_coordinates)
    y[is_sink] = 0


    # Construct step function
    make_step = METHODS[method]
    step_func = make_step(is_insulator, is_sink, **kwargs)

    delta_history = []

    for k in range(max_iter):
        y_old = y.copy()
        y = step_func(y, **kwargs)
        if post_step is not None:
            y = post_step(k + 1, y)

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
