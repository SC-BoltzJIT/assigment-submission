"""Boundary Value Problem (BVP) solvers for the Laplace equation.

Provides iterative methods: Jacobi, Gauss-Seidel, and SOR.
"""

from .solver import solve_laplace, apply_boundary_conditions
from .methods import METHODS, gauss_seidel_step, jacobi_step, sor_step

__all__ = [
    "solve_laplace",
    "apply_boundary_conditions",
    "METHODS",
    "gauss_seidel_step",
    "jacobi_step",
    "sor_step",
]
