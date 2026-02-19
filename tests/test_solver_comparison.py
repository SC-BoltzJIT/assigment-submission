"""Test that old (assignment01) and new (scicomp3) wave equation solvers produce consistent results.

Verifies that both solvers produce identical interior-point results and
that the new solver maintains zero boundary conditions via post_step.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# assignment01.py is at project root (not a package), so we need sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from assignment01 import wave_eq_deriv, integrate_euler

from scicomp3 import solve_ivp, wave1d_rhs
from scicomp3.pde.diffusion import diffusion2d_rhs


# --- Wave Equation Fixtures ---


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
    x = np.linspace(0, L, N)
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

    psi_t0 = np.zeros(N)
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

    v0 = np.zeros(N)
    y0 = np.column_stack([psi0.copy(), v0])

    result = solve_ivp(
        wave1d_rhs,
        t_span=(0, wave_params["T_sim"]),
        y0=y0,
        method="symplectic_euler",
        dt=wave_params["dt"],
        args=(wave_params["c"], wave_params["L"], N),
        post_step=fixed_ends,
    )
    return result.y[:, :, 0]  # amplitudes only


# --- Diffusion Equation Fixtures ---

# Common diffusion parameters — used by both old and new fixtures.
# Kept small for test speed and memory (old solver stores all time steps).
DIFF_N = 50  # grid size
DIFF_D = 0.05  # diffusion coefficient
DIFF_DX = 1 / DIFF_N  # spatial step size
DIFF_DT = 1e-4  # time step size
DIFF_T_SIM = 0.1  # simulation time


@pytest.fixture
def old_solver_diffusion_result():
    """Run the old diffusion solver (assignment01-02)."""

    def discrete_diffusion_equation(c, D, dx, dt):
        """Compute the next time step of the discretized, 2D diffusion equation."""
        c_next = np.copy(c)

        # add the horizontal and vertical diffusion terms
        c_next += (
            D * dt / dx**2 * (np.roll(c, -1, axis=0) - 2 * c + np.roll(c, 1, axis=0))
        )
        c_next += (
            D * dt / dx**2 * (np.roll(c, -1, axis=1) - 2 * c + np.roll(c, 1, axis=1))
        )

        # set boundary conditions (e.g., zero concentration at the bottom, 1 at the top)
        c_next[:, 0] = 0  # bottom boundary
        c_next[:, -1] = 1  # top boundary

        return c_next

    def simulate_diffusion(c0, D, dx, dt, T_sim):
        """Simulate the dicretized diffusion process over time."""
        c = np.copy(c0)
        time_steps = int(T_sim / dt)
        states = np.zeros((time_steps + 1, *c.shape))
        states[0] = c  # store initial state

        for t in range(1, time_steps + 1):
            c = discrete_diffusion_equation(c, D, dx, dt)
            states[t] = c

        return np.transpose(
            states, (0, 2, 1)
        )  # Transpose to (time, x, y) for easier plotting

    N = DIFF_N
    c0 = np.zeros((N, N))  # initial concentration field
    c0[:, N - 1] = 1  # set top boundary to 1
    D = DIFF_D  # diffusion coefficient
    dx = 1 / N  # spatial step size
    dt = DIFF_DT  # time step size
    T_sim = DIFF_T_SIM  # total simulation time

    print(f"Stability condition: 4 * dt * D / dx^2 = {4 * dt * D / dx**2:.4f}")

    assert (
        4 * dt * D / dx**2 <= 1
    ), f"Stability condition not satisfied: increase N or decrease dt"

    states = simulate_diffusion(c0, D, dx, dt, T_sim)

    return states


@pytest.fixture
def new_solver_diffusion_result():
    """Run the new diffusion solver (scicomp3) on the same (N, N) grid as old solver.

    Uses the same grid shape and parameters so that results are directly comparable.
    The only difference is the code path: solve_ivp + diffusion2d_rhs + post_step
    versus the old solver's manual loop.
    """
    N = DIFF_N
    D = DIFF_D
    dx = DIFF_DX
    dt = DIFF_DT
    T_sim = DIFF_T_SIM

    c0 = np.zeros((N, N))
    c0[:, -1] = 1  # top boundary

    def enforce_bc(t, y):
        y[:, 0] = 0  # bottom boundary
        y[:, -1] = 1  # top boundary
        return y

    result = solve_ivp(
        diffusion2d_rhs,
        t_span=(0, T_sim),
        y0=c0,
        method="forward_euler",
        dt=dt,
        args=(D, dx),
        post_step=enforce_bc,
        save_interval=1,
    )
    return result.y  # shape (n_saved, N, N), indexed as [t, i, j]


# --- Wave Equation Tests ---


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

        assert (
            left_boundary_max == 0
        ), f"Left boundary nonzero, max = {left_boundary_max}"

    def test_new_solver_boundary_right_is_zero(self, new_solver_result):
        """New solver should maintain zero boundary condition at x=L."""
        amps_new = new_solver_result

        right_boundary_max = np.max(np.abs(amps_new[:, -1]))

        assert (
            right_boundary_max == 0
        ), f"Right boundary nonzero, max = {right_boundary_max}"


# --- Diffusion Equation Tests ---


class TestSolverComparisonDiffusion:
    """Tests comparing old and new diffusion equation solvers.

    Both solvers use the same (N, N) grid with identical parameters.
    The old solver returns transposed output [t, j, i] (y-first),
    while the new solver returns [t, i, j] (x-first).

    Since the solution c depends only on y (x-periodic BCs, uniform IC),
    we compare concentration profiles c(y) extracted along x=0.
    """

    def test_final_y_profiles_match(
        self, old_solver_diffusion_result, new_solver_diffusion_result
    ):
        """Final c(y) profiles should match between solvers.

        Both use the same (N, N) grid, same dx, dt, D, BCs. The only
        difference is the code path, so results should match to near
        machine precision (small rounding from different operation order).

        Note: the old solver stores int(T/dt)+1 steps while solve_ivp stores
        len(arange(0, T, dt)) steps (off-by-one), so we compare at the last
        common time index rather than [-1].
        """
        n_common = min(
            old_solver_diffusion_result.shape[0], new_solver_diffusion_result.shape[0]
        )

        # old output [t, j, i]: c(y) at x=0 is states[:, :, 0]
        prof_old = old_solver_diffusion_result[n_common - 1, :, 0]

        # new output [t, i, j]: c(y) at x=0 is states[:, 0, :]
        prof_new = new_solver_diffusion_result[n_common - 1, 0, :]

        diff = np.max(np.abs(prof_old[1:-1] - prof_new[1:-1]))
        assert diff < 1e-10, f"Final y-profiles differ on interior, max diff = {diff}"

    def test_interior_points_match(
        self, old_solver_diffusion_result, new_solver_diffusion_result
    ):
        """Interior concentration fields should match at all common time steps.

        The old solver returns [t, j, i], so we transpose axes 1,2 to get [t, i, j]
        matching the new solver's convention before comparing.
        """
        # Transpose old from [t, j, i] → [t, i, j] to match new solver
        old_ij = old_solver_diffusion_result.transpose(0, 2, 1)
        new_ij = new_solver_diffusion_result

        # Old stores T_sim/dt + 1 steps, new stores T_sim/dt steps (off-by-one
        # from int() vs np.arange). Compare on the common range.
        n_steps = min(old_ij.shape[0], new_ij.shape[0])

        interior_diff = np.max(
            np.abs(old_ij[:n_steps, 1:-1, 1:-1] - new_ij[:n_steps, 1:-1, 1:-1])
        )
        assert (
            interior_diff < 1e-10
        ), f"Interior points differ, max diff = {interior_diff}"

    def test_old_boundary_conditions(self, old_solver_diffusion_result):
        """Old solver should maintain c=0 at bottom and c=1 at top throughout."""
        # old output: [t, j, i] — j=0 is bottom, j=N-1 is top
        states = old_solver_diffusion_result
        assert np.allclose(states[:, 0, :], 0), "Old solver: bottom boundary not zero"
        assert np.allclose(states[:, -1, :], 1), "Old solver: top boundary not one"

    def test_new_boundary_conditions(self, new_solver_diffusion_result):
        """New solver should maintain c=0 at bottom and c=1 at top throughout."""
        # new output: [t, i, j] — j=0 is bottom, j=N-1 is top
        states = new_solver_diffusion_result
        assert np.allclose(states[:, :, 0], 0), "New solver: bottom boundary not zero"
        assert np.allclose(states[:, :, -1], 1), "New solver: top boundary not one"

    def test_old_initial_condition(self, old_solver_diffusion_result):
        """Old solver should start with c=0 in interior."""
        states = old_solver_diffusion_result
        assert np.allclose(
            states[0, 1:-1, :], 0
        ), "Old solver: initial interior not zero"

    def test_new_initial_condition(self, new_solver_diffusion_result):
        """New solver should start with c=0 in interior."""
        states = new_solver_diffusion_result
        assert np.allclose(
            states[0, :, 1:-1], 0
        ), "New solver: initial interior not zero"

    def test_diffusion_progresses(
        self, old_solver_diffusion_result, new_solver_diffusion_result
    ):
        """Both solvers should show nonzero concentration in interior after simulation."""
        # Old: [t, j, i] — check mid-height
        mid_old = old_solver_diffusion_result[-1, DIFF_N // 2, 0]
        assert mid_old > 0, "Old solver: no diffusion occurred at mid-height"

        # New: [t, i, j] — check mid-height
        mid_new = new_solver_diffusion_result[-1, 0, DIFF_N // 2]
        assert mid_new > 0, "New solver: no diffusion occurred at mid-height"
