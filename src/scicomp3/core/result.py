"""Result containers for solvers."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ODEResult:
    """Container for ODE/IVP solver results.

    Follows scipy conventions for result objects.

    Attributes:
        t: Time points array
        y: Solution array at each time point
        success: Whether the integration completed
        message: Description of termination
        nfev: Number of function evaluations
    """
    t: np.ndarray
    y: np.ndarray
    success: bool = True
    message: str = ""
    nfev: int = 0

def find_y(res: ODEResult, t):
    """Compute the y-value corresponding to the given t value"""
    for i in range(len(res.t)):
        if res.t[i] == t:
            return res.y[i]
    raise ValueError


@dataclass
class BVPResult:
    """Container for BVP (boundary value problem) iterative solver results.

    Attributes:
        c: Solution array (concentration field)
        converged: Whether the iteration converged
        iterations: Number of iterations performed
        delta_history: History of convergence measure at each iteration
        message: Description of termination
    """
    c: np.ndarray
    converged: bool = True
    iterations: int = 0
    delta_history: np.ndarray = None
    message: str = ""
