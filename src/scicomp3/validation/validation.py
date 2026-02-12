import numpy as np
from ..core.result import ODEResult, find_y

def is_zero_at_the_x_ends(result: ODEResult, t):
    """
    Checks whether the result evaluates at zero for the given t at the x-boundaries"""
    simulated_array = find_y(result, t)
    zero_at_x_start = simulated_array[0][0] == 0
    zero_at_x_end = simulated_array[-1][0] == 0

    return zero_at_x_start and zero_at_x_end


def validate_boundary_conditions(
        result: ODEResult,
        validate_function = is_zero_at_the_x_ends,
        verbose = False
):
    """
    Validates the boundary conditions on the full t domain.
    The validate_function should be of the form func(result, t)
    """
    for t in result.t:
        assert validate_function(result, t), \
            f"The function does not evaluate to zero at t={t} at (one of) the x-boundaries"
    if verbose:
        print("The boundary conditions were succesfully validated")
