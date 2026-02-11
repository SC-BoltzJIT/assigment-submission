"""PDE right-hand-side implementations."""

from .wave import wave1d_rhs
from .diffusion import (
    solve_diffusion,
    diffusion_step,
    apply_diffusion_bc,
    analytical_solution,
)

__all__ = [
    "wave1d_rhs",
    "solve_diffusion",
    "diffusion_step",
    "apply_diffusion_bc",
    "analytical_solution",
]
