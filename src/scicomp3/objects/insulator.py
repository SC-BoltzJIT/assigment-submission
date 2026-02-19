"""
Contains the implementation of insulating objects.
"""

import numpy as np

def get_insulator_grid(N: int, coords: np.ndarray):
    shape = (N+1, N+1)
    is_insulator = np.zeros(shape, dtype=bool)
    if coords is not None:
        rows = coords[:, 0]
        cols = coords[:, 1]

        is_insulator[rows, cols] = True
    return is_insulator

def construct_rectangle(xmin, xmax, ymin, ymax):
    """
    Returns an np array with the coordinates of the given rectangle
    e.g. for (3,5,4,5):
         [(3,4), (4,4), (5,4), (3,5), (4,5), (5,5)]
    """
    xs = np.arange(xmin, xmax + 1)
    ys = np.arange(ymin, ymax + 1)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])
