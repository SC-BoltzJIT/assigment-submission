"""1D Wave equation implementation.

The wave equation: ∂²Ψ/∂t² = c² ∂²Ψ/∂x²

Converted to first-order system with state y = [Ψ, v] where v = ∂Ψ/∂t:
    dΨ/dt = v
    dv/dt = c² ∂²Ψ/∂x²
"""

import numpy as np


def wave1d_rhs(t, y, c, L, N):
    """Compute RHS of 1D wave equation.

    Args:
        t: Current time (unused, for interface consistency)
        y: State array with shape (N, 2) containing [psi, v]
        c: Wave speed
        L: Domain length
        N: Number of grid points

    Returns:
        dydt: Time derivatives [dpsi_dt, dv_dt] with shape (N, 2)
    """
    psi = y[:, 0].copy()
    v = y[:, 1].copy()

    # Spatial discretization
    dx = L / (N+1)

    # Second spatial derivative using central differences
    # d²Ψ/dx² ≈ (Ψ_{i+1} - 2Ψ_i + Ψ_{i-1}) / dx²
    d2psi_dx2 = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2

    # Time derivatives
    dpsi_dt = v  # dΨ/dt = v
    dv_dt = c**2 * d2psi_dx2  # dv/dt = c² ∂²Ψ/∂x²

    # Apply boundary conditions (fixed ends)
    dpsi_dt[0] = 0
    dpsi_dt[-1] = 0
    dv_dt[0] = 0
    dpsi_dt[-1] = 0

    return np.column_stack([dpsi_dt, dv_dt])


def initial_condition_case_i(x):
    """Case i: Ψ₀(x) = sin(2πx)"""
    return np.sin(2 * np.pi * x)


def initial_condition_case_ii(x):
    """Case ii: Ψ₀(x) = sin(5πx)"""
    return np.sin(5 * np.pi * x)


def initial_condition_case_iii(x):
    """Case iii: Ψ₀(x) = sin(5πx) if 1/5 < x < 2/5, else 0"""
    psi = np.sin(5 * np.pi * x)
    psi = np.where((x < 1 / 5) | (x > 2 / 5), 0, psi)
    return psi


def analytical_sol():
    return
