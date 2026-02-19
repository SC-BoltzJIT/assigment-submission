"""
Utilities for constructing insulating regions on a 2D grid.
"""

import numpy as np

def get_insulator_grid(N: int, coords: np.ndarray):
    """
    Construct a boolean mask marking insulating grid points.

    Args:
        N: Defines the grid size. The resulting grid has shape (N+1, N+1).
        coords: Array of shape (k, 2) containing integer grid coordinates
                (row, col) that should be marked as insulators.
                If None, no grid points are marked.

    Returns
        is_insulator: A boolean array of shape (N+1, N+1) where True indicates
                      an insulating grid point and False otherwise.
    """
    shape = (N+1, N+1)
    is_insulator = np.zeros(shape, dtype=bool)
    if coords is not None and len(coords) > 0:
        rows = coords[:, 0]
        cols = coords[:, 1]

        is_insulator[rows, cols] = True
    return is_insulator
