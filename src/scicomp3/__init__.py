"""
scicomp3: Scientific Computing Package for Assignment Set 1

This package provides:
- Solvers for the 1D wave equation using time-stepping methods
- Iterative solvers for the 2D Laplace equation (Jacobi, Gauss-Seidel, SOR)
"""

__version__ = "0.1.0"

from .core.grid import Grid1D, Grid2D
from .core.result import ODEResult, BVPResult
from .ode.solver import solve_ivp
from .ode.methods import METHODS as ODE_METHODS
from .pde.wave import wave1d_rhs
from .bvp.solver import solve_laplace
from .bvp.methods import METHODS as BVP_METHODS

__all__ = [
    # Core
    "Grid1D",
    "Grid2D",
    "ODEResult",
    "BVPResult",
    # ODE/IVP
    "solve_ivp",
    "ODE_METHODS",
    "wave1d_rhs",
    # BVP
    "solve_laplace",
    "BVP_METHODS",
]
