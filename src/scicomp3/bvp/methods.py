"""Iterative methods for BVP (steady-state) solvers.

Methods:
- jacobi: Jacobi iteration (requires two arrays, cannot update in place)
- gauss_seidel: Gauss-Seidel iteration (updates in place)
- sor: Successive Over Relaxation (includes an overcorrection and parameter omega)

All methods have signature:
    step(y, **kwargs) -> y_new
"""

import numpy as np

def _compute_jacobi_weights(is_insulator):
    """
    For each point, count non-insulating neighbours.
    Returns a float array; insulator points get weight 1 to avoid division.
    """
    neighbour_count = (
        np.roll(~is_insulator, -1, axis=0).astype(float) +
        np.roll(~is_insulator,  1, axis=0).astype(float) +
        np.roll(~is_insulator, -1, axis=1).astype(float) +
        np.roll(~is_insulator,  1, axis=1).astype(float)
    )
    # Avoid division by zero at insulator points (value won't be used)
    neighbour_count[is_insulator] = 1.0
    return neighbour_count


def make_jacobi_step(is_insulator, **kwargs):
    """Make function for one Jacobi iteration step (Eq. 12).

    c^{k+1}_{i,j} = (1/4)(c^k_{i+1,j} + c^k_{i-1,j} + c^k_{i,j+1} + c^k_{i,j-1})

    Uses np.roll for the five-point stencil (periodic in x via wrap-around).
    Requires two arrays — returns a new array, does not modify y.

    Args:
        y: Current solution array (N+1 x N+1)

    Returns:
        y_new: Updated solution array
    """
    if np.any(is_insulator):
        jacobi_weights = _compute_jacobi_weights(is_insulator)

        def jacobi_step_with_insulator(y, **kwargs):
            neighbour_sum = (
                np.roll(y, -1, axis=0) + np.roll(y, 1, axis=0) +
                np.roll(y, -1, axis=1) + np.roll(y, 1, axis=1)
            )
            # Zero out insulating neighbours' contributions
            insulator_zeroed = y * is_insulator.astype(float)
            neighbour_sum -= (
                np.roll(insulator_zeroed, -1, axis=0) + np.roll(insulator_zeroed,  1, axis=0) +
                np.roll(insulator_zeroed, -1, axis=1) + np.roll(insulator_zeroed,  1, axis=1)
            )
            y_new = neighbour_sum / jacobi_weights
            y_new[is_insulator] = y[is_insulator]   # leave insulator points unchanged
            return y_new
        return jacobi_step_with_insulator
    else:
        def jacobi_step(y, **kwargs):
            y_new = 0.25 * (
                np.roll(y, -1, axis=0) + np.roll(y, 1, axis=0) +
                np.roll(y, -1, axis=1) + np.roll(y, 1, axis=1)
            )
            return y_new
        return jacobi_step


def make_gauss_seidel_step(is_insulator, **kwargs):
    """Make function for one Gauss-Seidel iteration step (Sec. 1.5).

    c^{k+1}_{i,j} = (1/4)(c^k_{i+1,j} + c^{k+1}_{i-1,j}
                         + c^k_{i,j+1} + c^{k+1}_{i,j-1})

    Uses already-updated values as soon as they are available.
    Sweeps rows: incrementing i for fixed j (as stated in the assignment).
    Updates in place — returns the same (modified) array.

    Periodic x boundary is handled via modular indexing.

    Args:
        y: Current solution array (N+1 x N+1), modified in place

    Returns:
        y: The same array, updated in place
    """
    if np.any(is_insulator):
        def gauss_seidel_step_with_insulator(y, **kwargs):
            n_i, n_j = y.shape
            for j in range(1, n_j - 1):        # interior y-points
                for i in range(n_i):            # all x-points (periodic)
                    if is_insulator[i, j]:
                        continue
                    i_plus = (i + 1) % n_i
                    i_minus = (i - 1) % n_i
                    coords = [(i_plus, j), (i_minus, j), (i, j + 1), (i, j - 1)]
                    new_value = np.mean([y[coord] for coord in coords if not is_insulator[coord]])
                    if not np.isnan(new_value):
                        y[i,j] = new_value
            return y
        return gauss_seidel_step_with_insulator
    else:
        def gauss_seidel_step(y, **kwargs):
            n_i, n_j = y.shape
            for j in range(1, n_j - 1):        # interior y-points
                for i in range(n_i):            # all x-points (periodic)
                    i_plus = (i + 1) % n_i
                    i_minus = (i - 1) % n_i
                    y[i, j] = 0.25 * (y[i_plus, j] + y[i_minus, j] +
                                    y[i, j + 1] + y[i, j - 1])
            return y
        return gauss_seidel_step



def make_sor_step(is_insulator, omega: float, **kwargs):
    """
    Make function for one Successive Over Relaxation (SOR) step with parameter omega (ω).

    c^{k+1}_{i,j} = (ω/4)(c^k_{i+1,j} + c^{k+1}_{i-1,j}
                          + c^k_{i,j+1} + c^{k+1}_{i,j-1})
                    + (1-ω)c^{k}_{i,j}

    Parameter omega should be between 0 and 2.

    Args:
        y: Current solution array (N+1 x N+1), modified in place

    Returns:
        y: The same array, updated in place
    """
    if not (0 <= omega <= 2):
        raise ValueError(f"omega must be in [0, 2], got {omega:.2f}")

    # Check if there are any insulating objects
    if np.any(is_insulator):
        def sor_step_with_insulator(y, **kwargs):
            n_i, n_j = y.shape
            for j in range(1, n_j - 1):        # interior y-points
                for i in range(n_i):            # all x-points (periodic)
                    if is_insulator[i, j]:
                        continue
                    i_plus = (i + 1) % n_i
                    i_minus = (i - 1) % n_i
                    coords = [(i_plus, j), (i_minus, j), (i, j + 1), (i, j - 1)]
                    new_value = np.mean([y[coord] for coord in coords if is_insulator[coord] == 0])
                    if not np.isnan(new_value):
                        y[i,j] = omega * new_value + (1 - omega) * y[i, j]
            return y
        return sor_step_with_insulator
    else:
        def sor_step(y, **kwargs):
            n_i, n_j = y.shape
            for j in range(1, n_j - 1):        # interior y-points
                for i in range(n_i):            # all x-points (periodic)
                    i_plus = (i + 1) % n_i
                    i_minus = (i - 1) % n_i
                    y[i, j] = omega * 0.25 * (y[i_plus, j] + y[i_minus, j] +
                                            y[i, j + 1] + y[i, j - 1]) \
                            + (1 - omega) * y[i, j]
            return y
        return sor_step

# Method registry (Strategy Pattern) — mirrors ode/methods.py METHODS
METHODS = {
    "jacobi": make_jacobi_step,
    "gauss_seidel": make_gauss_seidel_step,
    "sor": make_sor_step
}
