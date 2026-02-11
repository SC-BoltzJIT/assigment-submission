"""Test 2D time-dependent diffusion equation solver.

Validates that the explicit finite difference solver produces
results consistent with the analytical solution.
"""

import numpy as np
import pytest

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import (
    solve_diffusion,
    diffusion_step,
    apply_diffusion_bc,
    analytical_solution,
)


# Parameters for tests
N = 20  # smaller grid for faster tests
D = 1.0


@pytest.fixture
def grid():
    """Create computational grid."""
    return Grid2D(N=N, L=1.0)


class TestDiffusionSolver:
    """Tests for the diffusion equation solver."""

    def test_boundary_conditions_bottom(self, grid):
        """Bottom boundary should be c=0."""
        t, c_history = solve_diffusion(grid, D=D, T_sim=0.1)
        for c in c_history:
            assert np.allclose(c[:, 0], 0), "Bottom boundary not zero"

    def test_boundary_conditions_top(self, grid):
        """Top boundary should be c=1."""
        t, c_history = solve_diffusion(grid, D=D, T_sim=0.1)
        for c in c_history:
            assert np.allclose(c[:, -1], 1), "Top boundary not one"

    def test_initial_condition(self, grid):
        """Initial condition should be c=0 in interior, with BCs applied."""
        t, c_history = solve_diffusion(grid, D=D, T_sim=0.01)
        c0 = c_history[0]
        # Interior should be zero at t=0
        assert np.allclose(c0[:, 1:-1], 0), "Initial interior not zero"

    def test_convergence_to_steady_state(self, grid):
        """Long simulation should approach steady state c(y) = y."""
        t, c_history = solve_diffusion(grid, D=D, T_sim=1.0, save_interval=100)
        c_final = c_history[-1]

        # Steady state is c = y
        c_steady = grid.Y
        error = np.max(np.abs(c_final - c_steady))

        assert error < 0.01, f"Did not converge to steady state, error = {error}"

    def test_matches_analytical_solution(self, grid):
        """Numerical solution should match analytical solution."""
        t, c_history = solve_diffusion(grid, D=D, T_sim=0.1, save_interval=10)

        # Check at a few time points
        for i in [len(t) // 4, len(t) // 2, -1]:
            ti = t[i]
            if ti > 0:
                # Compare profile at x=0.5 (mid-grid)
                mid_i = N // 2
                c_numerical = c_history[i][mid_i, :]
                c_analytical = analytical_solution(grid.y, ti, D=D)

                error = np.max(np.abs(c_numerical - c_analytical))
                assert error < 0.01, f"Error at t={ti}: {error}"

    def test_stability_check(self, grid):
        """Solver should raise error for unstable time step."""
        # Very large dt should be unstable
        with pytest.raises(ValueError, match="Unstable"):
            solve_diffusion(grid, D=D, T_sim=0.01, dt=1.0)


class TestAnalyticalSolution:
    """Tests for the analytical solution function."""

    def test_boundary_at_y_zero(self):
        """Analytical solution at y=0 should be 0."""
        c = analytical_solution(0.0, t=0.1, D=1.0)
        assert np.isclose(c, 0, atol=1e-6), f"c(y=0) = {c}, expected 0"

    def test_boundary_at_y_one(self):
        """Analytical solution at y=1 should be 1."""
        c = analytical_solution(1.0, t=0.1, D=1.0)
        assert np.isclose(c, 1, atol=1e-3), f"c(y=1) = {c}, expected 1"

    def test_steady_state(self):
        """At large t, analytical solution should approach c=y."""
        y = np.linspace(0, 1, 11)
        c = analytical_solution(y, t=10.0, D=1.0)
        error = np.max(np.abs(c - y))
        assert error < 1e-6, f"Steady state error: {error}"
