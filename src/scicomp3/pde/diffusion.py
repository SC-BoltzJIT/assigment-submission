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


def diffusion_stable_dt(D, dx, safety=0.9):
    """Compute a stable time step for the explicit diffusion scheme.

    Stability condition: 4δtD/δx² ≤ 1  →  δt ≤ δx²/(4D)

    Args:
        D: Diffusion coefficient
        dx: Grid spacing
        safety: Safety factor (default: 0.9, i.e. 90% of max stable dt)

    Returns:
        dt: Stable time step
    """
    return safety * dx ** 2 / (4 * D)


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
