"""
scicomp3: Scientific Computing Package for Assignment Set 1

This package provides solvers for the 1D wave equation using
time-stepping methods (Forward Euler, Symplectic Euler).
"""

__version__ = "0.1.0"

from .core.grid import Grid1D
from .core.result import ODEResult, BVPResult
from .ode.solver import solve_ivp
from .ode.methods import METHODS
from .bvp.solver import solve_bvp
from .pde.wave import wave1d_rhs
from .pde.diffusion import diffusion2d_rhs

__all__ = [
    "Grid1D",
    "ODEResult",
    "BVPResult",
    "solve_ivp",
    "METHODS",
    "solve_bvp",
    "wave1d_rhs",
    "diffusion2d_rhs",
]
