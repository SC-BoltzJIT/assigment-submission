"""Result containers for solvers."""

from dataclasses import dataclass, field
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
