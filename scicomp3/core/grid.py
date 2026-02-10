"""Grid definitions for spatial discretization."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Grid1D:
    """1D uniform grid for spatial discretization.

    Attributes:
        N: Number of grid intervals
        L: Domain length
        dx: Grid spacing (computed as L/N)
        x: Array of grid point coordinates
    """
    N: int
    L: float = 1.0
    dx: float = field(init=False)
    x: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.dx = self.L / self.N
        self.x = np.arange(self.N) * self.dx

    @property
    def shape(self) -> tuple:
        """Return the shape of the grid."""
        return (self.N,)
