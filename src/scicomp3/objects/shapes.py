"""
Generating arrays of coordinates for geometric descriptions
"""

import numpy as np

def construct_rectangle(xmin: int, xmax: int, ymin: int, ymax: int):
    """
    Returns an np array with the coordinates of the given rectangle
    e.g. for (3,5,4,5):
         [(3,4), (4,4), (5,4), (3,5), (4,5), (5,5)]
    """
    xs = np.arange(xmin, xmax + 1)
    ys = np.arange(ymin, ymax + 1)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])

def construct_circle(x_center: int, y_center: int, radius: float):
    """
    Returns an np array with the integer coordinates that fall within
    the given circle (including the boundary).
    """
    r = int(np.ceil(radius))

    # bounding square around the circle
    xs = np.arange(x_center - r, x_center + r + 1)
    ys = np.arange(y_center - r, y_center + r + 1)

    X, Y = np.meshgrid(xs, ys)

    # distance condition
    mask = (X - x_center)**2 + (Y - y_center)**2 <= radius**2

    return np.column_stack([X[mask], Y[mask]])
