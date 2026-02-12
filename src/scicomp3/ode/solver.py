"""Main ODE/IVP solver interface."""

import numpy as np
from ..core.result import ODEResult
from .methods import METHODS


def solve_ivp(fun, t_span, y0, method="symplectic_euler", dt=1e-3, args=(),
              post_step=None, save_interval=1):
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
        save_interval: Save solution every N steps (default: 1 = save all)

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

    # Time stepping with selective saving
    y = y0.copy()
    t_save = [t[0]]
    y_save = [y0.copy()]
    nfev = 0

    for i in range(1, n_steps):
        y, n_evals = step_func(fun, t[i-1], y, dt, args)
        if post_step is not None:
            y = post_step(t[i], y)
        nfev += n_evals

        if i % save_interval == 0:
            t_save.append(t[i])
            y_save.append(y.copy())

    return ODEResult(
        t=np.array(t_save),
        y=np.array(y_save),
        success=True,
        message=f"Integration completed with {n_steps} steps",
        nfev=nfev
    )
