"""
scicomp3: Scientific Computing Package for Assignment Set 1

This package provides solvers for the 1D wave equation using
time-stepping methods (Forward Euler, Symplectic Euler).
"""

from .core.grid import Grid1D
from .core.result import ODEResult
from .ode.solver import solve_ivp
from .ode.methods import METHODS
from .pde.wave import wave1d_rhs

__all__ = [
    "Grid1D",
    "ODEResult",
    "solve_ivp",
    "METHODS",
    "wave1d_rhs",
]
