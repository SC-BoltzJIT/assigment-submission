"""Time-stepping methods for ODE integration.

Methods:
- euler: Forward Euler (1st order, explicit)
- symplectic_euler: Symplectic/Semi-implicit Euler (1st order, symplectic)

All methods have signature:
    step(fun, t, y, dt, args) -> (y_new, n_evals)
"""

import numpy as np


def euler_step(fun, t, y, dt, args=()):
    """Forward Euler step (explicit, 1st order).

    y_{n+1} = y_n + dt * f(t_n, y_n)

    Args:
        fun: RHS function f(t, y, *args) -> dy_dt
        t: Current time
        y: Current state [psi, v] with shape (N, 2)
        dt: Time step
        args: Additional arguments for fun

    Returns:
        y_new: Updated state
        n_evals: Number of function evaluations (1)
    """
    dydt = fun(t, y, *args)
    y_new = y + dt * dydt
    return y_new, 1


def symplectic_euler_step(fun, t, y, dt, args=()):
    """Symplectic Euler step (semi-implicit, 1st order, energy-preserving).

    Also known as: Euler-Cromer, Semi-implicit Euler.

    For a system [psi, v]:
        v_{n+1} = v_n + dt * a(psi_n)
        psi_{n+1} = psi_n + dt * v_{n+1}

    This is achieved by computing:
        dpsi_dt = v + dt * dv_dt  (uses updated velocity)

    Args:
        fun: RHS function f(t, y, *args) -> dy_dt
        t: Current time
        y: Current state [psi, v] with shape (N, 2)
        dt: Time step
        args: Additional arguments for fun

    Returns:
        y_new: Updated state
        n_evals: Number of function evaluations (1)
    """
    # Get derivatives [dpsi_dt, dv_dt] = [v, acceleration]
    dydt = fun(t, y, *args)

    # Extract components
    psi = y[:, 0]
    v = y[:, 1]
    dv_dt = dydt[:, 1]  # acceleration

    # Symplectic Euler: update velocity first, then position with new velocity
    v_new = v + dt * dv_dt
    psi_new = psi + dt * v_new  # uses updated velocity!

    y_new = np.column_stack([psi_new, v_new])
    return y_new, 1


# Method registry (Strategy Pattern)
METHODS = {
    "euler": euler_step,
    "forward_euler": euler_step,
    "symplectic_euler": symplectic_euler_step,
    "euler_cromer": symplectic_euler_step,
    "semi_implicit_euler": symplectic_euler_step,
}
