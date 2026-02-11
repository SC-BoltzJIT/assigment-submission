"""Main ODE/IVP solver interface."""

import numpy as np
from ..core.result import ODEResult
from .methods import METHODS


def solve_ivp(fun, t_span, y0, method="symplectic_euler", dt=1e-3, args=(),
              post_step=None):
    """Solve an initial value problem using time-stepping.

    Solves dy/dt = f(t, y) from t_span[0] to t_span[1].

    Args:
        fun: RHS function f(t, y, *args) -> dy_dt
        t_span: Tuple (t_start, t_end)
        y0: Initial state array
        method: Time-stepping method name (see METHODS registry)
        dt: Time step size
        args: Additional arguments to pass to fun
        post_step: Optional callback f(t, y) -> y applied after each step,
            e.g. to enforce boundary conditions. Must return the modified y.

    Returns:
        ODEResult with t (time array) and y (solution array)
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    step_func = METHODS[method]

    # Create time array
    t_start, t_end = t_span
    t = np.arange(t_start, t_end, dt)
    n_steps = len(t)

    # Allocate solution array
    y = np.zeros((n_steps,) + y0.shape)
    y[0] = y0
    if post_step is not None:
        y[0] = post_step(t[0], y[0])

    # Time stepping loop
    nfev = 0
    for i in range(1, n_steps):
        y[i], n_evals = step_func(fun, t[i-1], y[i-1], dt, args)
        if post_step is not None:
            y[i] = post_step(t[i], y[i])
        nfev += n_evals

    return ODEResult(
        t=t,
        y=y,
        success=True,
        message=f"Integration completed with {n_steps} steps",
        nfev=nfev
    )
