"""
Finding the optimal omega for SOR
"""
import numpy as np

def get_optimal_omega(N):
    """
    Returns the theoretical optimal omega based on Poisson's equation
    Only works for rectangular grid and Dirichlet Boundary conditions
    """
    omega = 2 / (1 + np.sin(np.pi / N))
    return omega
