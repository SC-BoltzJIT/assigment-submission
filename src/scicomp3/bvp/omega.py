"""
Finding the optimal omega for SOR
"""
import numpy as np

from .solver import solve_bvp

def get_optimal_omega(N):
    """
    Returns the theoretical optimal omega based on Poisson's equation
    Only works for rectangular grid and Dirichlet Boundary conditions
    """
    omega = 2 / (1 + np.sin(np.pi / N))
    return omega

def search_for_optimal_omega(y0, tol=1e-5, max_iter=100_000, post_step=None,
                             insulator_coordinates=None,
                             omega_min=1.0, omega_max=2.0,
                             omega_tol=0.001):
    """
    Use ternary search to find the omega that minimises the number of
    iterations needed. Assumes the iteration count is unimodal on
    [omega_min, omega_max].

    Args:
        y0:                     Initial solution array
        tol:                    Convergence tolerance passed to solve_bvp
        max_iter:               Maximum iterations passed to solve_bvp
        post_step:              Optional post-step function passed to solve_bvp
        insulator_coordinates:  Optional insulator coordinates passed to solve_bvp
        omega_min:              Left bound of search interval (default 1.0)
        omega_max:              Right bound of search interval (default 2.0)
        omega_tol:              Stop when the search interval is narrower than
                                this (default 0.01)

    Returns:
        omega_optimal:  The omega value with the fewest iterations
        n_iter_opimal:  The corresponding optimal iteration count
        omega_values:   Sorted list of all omega values evaluated
        n_iterations:   Corresponding iteration counts
    """
    omega_left = omega_min
    omega_right = omega_max

    # Dictionary that stores the number of iterations needed for specific values of omega
    results = {2: np.inf} # Since the method divirges fro omega > 2, we set results[2] to infinity

    def get_n_iter(omega):
        """Get the number of iterations needed for this value of omega"""
        if omega in results:
            return results[omega]
        print(f"Trying omega = {omega:.4f}")
        res = solve_bvp(y0, method="sor", tol=tol, max_iter=max_iter, post_step=post_step,
                  insulator_coordinates=insulator_coordinates, omega=omega)
        print(f"Found solution in {res.n_iter} iterations")
        results[omega] = res.n_iter
        return res.n_iter

    while omega_right - omega_left > omega_tol:
        m1 = omega_left + (omega_right - omega_left) / 3
        m2 = omega_right - (omega_right - omega_left) / 3

        if get_n_iter(m1) < get_n_iter(m2):
            omega_right = m2
        else:
            omega_left = m1

        print(f"Bracket: [{omega_left:.4f}, {omega_right:.4f}]")
        print("Current difference between omegas "\
              f"is {np.abs(get_n_iter(omega_left) - get_n_iter(omega_right))} iterations")
    omega_optimal = min(results, key=results.get)
    n_iter_opimal = get_n_iter(omega_optimal)
    omega_values, n_iterations = zip(*sorted(results.items()))
    return omega_optimal, n_iter_opimal, omega_values, n_iterations
