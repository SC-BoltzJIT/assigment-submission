"""Time-dependent 2D diffusion equation solver.

Solves the diffusion equation:
    ∂c/∂t = D∇²c

Using explicit finite difference scheme:
    c^{k+1}_{i,j} = c^k_{i,j} + (δtD/δx²)(c^k_{i+1,j} + c^k_{i-1,j} + c^k_{i,j+1} + c^k_{i,j-1} - 4c^k_{i,j})

Stability condition: 4δtD/δx² ≤ 1

Boundary conditions:
    c(x, y=1, t) = 1  (top)
    c(x, y=0, t) = 0  (bottom)
    periodic in x direction: c(x=0, y, t) = c(x=1, y, t)

Initial condition:
    c(x, y, t=0) = 0 for 0 ≤ x ≤ 1, 0 ≤ y < 1
"""

import numpy as np
from scipy.special import erfc


def apply_diffusion_bc(c):
    """Apply boundary conditions for the diffusion equation.

    Boundary conditions:
        c(x, y=0) = 0  (bottom, j=0)
        c(x, y=1) = 1  (top, j=N)
        periodic in x: c[0, :] = c[N, :]  (handled by np.roll in step)

    Args:
        c: Concentration field array (N+1 x N+1), modified in-place
    """
    c[:, 0] = 0   # bottom boundary (y=0)
    c[:, -1] = 1  # top boundary (y=1)


def diffusion2d_rhs(t, c, D, dx):
    """Compute RHS of 2D diffusion equation: dc/dt = D∇²c.

    Uses the 5-point stencil for the discrete Laplacian with
    periodic boundary in x direction (via np.roll).

    Args:
        t: Current time (unused, for interface consistency with solve_ivp)
        c: Concentration field (N+1 x N+1 array)
        D: Diffusion coefficient
        dx: Grid spacing

    Returns:
        dcdt: Time derivative D∇²c with same shape as c
    """
    # Shifted arrays for 5-point stencil (periodic in x via roll)
    c_ip = np.roll(c, -1, axis=0)  # c[i+1, j]
    c_im = np.roll(c, 1, axis=0)   # c[i-1, j]
    c_jp = np.roll(c, -1, axis=1)  # c[i, j+1]
    c_jm = np.roll(c, 1, axis=1)   # c[i, j-1]

    # Discrete Laplacian: ∇²c ≈ (c_{i+1} + c_{i-1} + c_{j+1} + c_{j-1} - 4c) / dx²
    laplacian = (c_ip + c_im + c_jp + c_jm - 4 * c) / dx**2

    return D * laplacian


def diffusion_step(c, D, dx, dt):
    """Perform one explicit time step of the diffusion equation.

    Uses forward Euler: c^{k+1} = c^k + dt * D∇²c^k

    Args:
        c: Current concentration field (N+1 x N+1 array)
        D: Diffusion coefficient
        dx: Grid spacing
        dt: Time step

    Returns:
        c_new: Updated concentration field
    """
    alpha = dt * D / (dx ** 2)

    # Check stability
    if 4 * alpha > 1:
        raise ValueError(f"Unstable: 4*α = {4*alpha:.3f} > 1. Reduce dt or increase dx.")

    # Forward Euler step using diffusion RHS
    c_new = c + dt * diffusion2d_rhs(0, c, D, dx)

    # Apply boundary conditions
    apply_diffusion_bc(c_new)

    return c_new


def solve_diffusion(grid, D=1.0, dt=None, T_sim=1.0, c0=None, save_interval=1):
    """Solve the 2D time-dependent diffusion equation.

    Delegates to solve_ivp with forward Euler method. The diffusion equation
    dc/dt = D∇²c is an initial value problem solved by:
        - RHS function: diffusion2d_rhs (computes D∇²c)
        - Method: forward Euler (explicit)
        - post_step: apply_diffusion_bc (enforce boundary conditions)

    Args:
        grid: Grid2D object defining the spatial discretization
        D: Diffusion coefficient (default: 1.0)
        dt: Time step (default: computed from stability condition)
        T_sim: Total simulation time
        c0: Initial concentration field (default: zeros with BC applied)
        save_interval: Save solution every N steps (default: 1 = save all)

    Returns:
        t: Array of time points
        c_history: Array of concentration fields at each saved time point
    """
    from ..ode.solver import solve_ivp

    N = grid.N
    dx = grid.dx

    # Compute stable time step if not provided
    if dt is None:
        # Use 90% of maximum stable dt
        dt = 0.9 * dx ** 2 / (4 * D)

    # Check stability
    alpha = dt * D / (dx ** 2)
    if 4 * alpha > 1:
        raise ValueError(f"Unstable: 4*α = {4*alpha:.3f} > 1. Reduce dt.")

    # Initialize concentration field
    if c0 is not None:
        c = c0.copy()
    else:
        c = np.zeros((N + 1, N + 1))

    # Apply boundary conditions
    apply_diffusion_bc(c)

    # Boundary conditions as post_step callback
    def post_step(t, y):
        apply_diffusion_bc(y)
        return y

    # Solve as IVP: dc/dt = D∇²c
    result = solve_ivp(
        diffusion2d_rhs,
        t_span=(0, T_sim),
        y0=c,
        method="forward_euler",
        dt=dt,
        args=(D, dx),
        post_step=post_step,
        save_interval=save_interval,
    )

    return result.t, result.y


def analytical_solution(y, t, D=1.0, n_terms=50):
    """Compute the analytical solution of the 1D diffusion equation.

    For boundary conditions c(y=0)=0, c(y=1)=1 and initial c(y,0)=0,
    the analytical solution is:

        c(y, t) = Σ [erfc((1-y+2i)/(2√(Dt))) - erfc((1+y+2i)/(2√(Dt)))]

    where the sum is over i = 0, 1, 2, ...

    Args:
        y: Position(s) where to evaluate (scalar or array)
        t: Time (scalar)
        D: Diffusion coefficient
        n_terms: Number of terms in the series

    Returns:
        c: Concentration at position(s) y and time t
    """
    scalar_input = np.isscalar(y)
    y = np.atleast_1d(y)

    if t <= 0:
        # At t=0, return 0 (initial condition) except at y=1
        c = np.zeros_like(y, dtype=float)
        c[y >= 1.0] = 1.0
        return float(c[0]) if scalar_input else c

    sqrt_Dt = np.sqrt(D * t)
    c = np.zeros_like(y, dtype=float)

    for i in range(n_terms):
        term1 = erfc((1 - y + 2 * i) / (2 * sqrt_Dt))
        term2 = erfc((1 + y + 2 * i) / (2 * sqrt_Dt))
        c = c + term1 - term2

    return float(c[0]) if scalar_input else c
