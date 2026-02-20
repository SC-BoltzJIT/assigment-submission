"""
Utilities for constructing sink regions on a 2D grid.
"""

import numpy as np


def get_sink_grid(N: int, coords: np.ndarray):
    """
    Construct a boolean mask marking sink grid points.

    Args:
        N: Defines the grid size. The resulting grid has shape (N+1, N+1).
        coords: Array of shape (k, 2) containing integer grid coordinates
                (row, col) that should be marked as sinks.
                If None, no grid points are marked.

    Returns
        is_sink: A boolean array of shape (N+1, N+1) where True indicates
                      an sink grid point and False otherwise.
    """
    # I noticed such N+1 associates to len(y) - 1 usually
    # This seems a unnecessary complexity.
    shape = (N + 1, N + 1)
    is_sink = np.zeros(shape, dtype=bool)
    if coords is not None and len(coords) > 0:
        rows = coords[:, 0]
        cols = coords[:, 1]

        is_sink[rows, cols] = True
    return is_sink
