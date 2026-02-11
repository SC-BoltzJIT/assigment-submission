"""Test that old (assignment01) and new (scicomp3) wave equation solvers produce consistent results.

Verifies that both solvers produce identical interior-point results and
that the new solver maintains zero boundary conditions via post_step.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add assigment-submission root to path so we can import both packages
sys.path.insert(0, str(Path(__file__).parent.parent))

from assignment01 import wave_eq_deriv, integrate_euler
from scicomp3 import solve_ivp, wave1d_rhs


# --- Fixtures ---

@pytest.fixture
def wave_params():
    """Common parameters for wave equation tests."""
    return {
        "c": 1,
        "L": 1,
        "N": 90,
        "dt": 1e-3,
        "T_sim": 0.1,  # short run for comparison
    }


@pytest.fixture
def initial_condition(wave_params):
    """Initial condition with boundary conditions enforced."""
    N = wave_params["N"]
    L = wave_params["L"]
    x = np.linspace(0, L, N + 1)
    psi0 = np.sin(5 * np.pi * x)
    psi0[0] = 0
    psi0[-1] = 0
    return x, psi0


def fixed_ends(t, y):
    """Enforce fixed boundary conditions: psi=0 at both ends."""
    y[0, 0] = 0
    y[-1, 0] = 0
    return y


@pytest.fixture
def old_solver_result(wave_params, initial_condition):
    """Run the old solver (assignment01)."""
    x, psi0 = initial_condition
    N = wave_params["N"]

    psi_t0 = np.zeros(N + 1)
    state0 = np.transpose([psi0, psi_t0])

    time_old, states_old = integrate_euler(
        wave_eq_deriv,
        state0=state0,
        dt=wave_params["dt"],
        T_sim=wave_params["T_sim"],
        c=wave_params["c"],
        L=wave_params["L"],
        N=N + 1,
    )
    return states_old[:, :, 0]  # amplitudes only


@pytest.fixture
def new_solver_result(wave_params, initial_condition):
    """Run the new solver (scicomp3) with post_step boundary enforcement."""
    x, psi0 = initial_condition
    N = wave_params["N"]

    v0 = np.zeros(N + 1)
    y0 = np.column_stack([psi0.copy(), v0])

    result = solve_ivp(
        wave1d_rhs,
        t_span=(0, wave_params["T_sim"]),
        y0=y0,
        method="symplectic_euler",
        dt=wave_params["dt"],
        args=(wave_params["c"], wave_params["L"], N + 1),
        post_step=fixed_ends,
    )
    return result.y[:, :, 0]  # amplitudes only


# --- Tests ---

class TestSolverComparison:
    """Tests comparing old and new wave equation solvers."""

    def test_interior_points_match(self, old_solver_result, new_solver_result):
        """Interior points (indices 1:-1) should match exactly between solvers."""
        amps_old = old_solver_result
        amps_new = new_solver_result

        interior_diff = np.max(np.abs(amps_old[:, 1:-1] - amps_new[:, 1:-1]))

        assert interior_diff == 0, f"Interior points differ, max diff = {interior_diff}"

    def test_new_solver_boundary_left_is_zero(self, new_solver_result):
        """New solver should maintain zero boundary condition at x=0."""
        amps_new = new_solver_result

        left_boundary_max = np.max(np.abs(amps_new[:, 0]))

        assert left_boundary_max == 0, f"Left boundary nonzero, max = {left_boundary_max}"

    def test_new_solver_boundary_right_is_zero(self, new_solver_result):
        """New solver should maintain zero boundary condition at x=L."""
        amps_new = new_solver_result

        right_boundary_max = np.max(np.abs(amps_new[:, -1]))

        assert right_boundary_max == 0, f"Right boundary nonzero, max = {right_boundary_max}"
