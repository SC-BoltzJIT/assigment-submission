"""Test Jacobi iteration for the 2D Laplace equation.

Verifies that the Jacobi iteration converges to the analytical
steady-state solution c(y) = y for the standard diffusion BCs
(c=0 at bottom, c=1 at top, periodic in x).
"""

import numpy as np
import pytest

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.solver import solve_bvp


N = 50


@pytest.fixture
def grid():
    return Grid2D(N=N, L=1.0)


@pytest.fixture
def jacobi_result(grid):
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)
    return solve_bvp(c0, method="jacobi", bc_func=apply_diffusion_bc,
                     tol=1e-5, max_iter=100_000)


class TestJacobiIteration:
    """Tests for Jacobi iteration solver (Assignment 1.6.H)."""

    def test_converges(self, jacobi_result):
        """Jacobi iteration should converge within max_iter."""
        assert jacobi_result.converged, (
            f"Did not converge in {jacobi_result.n_iter} iterations, "
            f"final delta = {jacobi_result.delta_history[-1]:.2e}"
        )

    def test_steady_state_profile(self, jacobi_result, grid):
        """Solution should match analytical steady state c(y) = y.

        With tol=1e-5 on the iteration change (delta), the absolute error
        is O(delta * N^2 / pi^2) ~ 1e-2 for N=50, due to Jacobi's spectral
        radius rho = cos(pi/N) ~ 0.998.
        """
        c_analytical = grid.Y
        error = np.max(np.abs(jacobi_result.y - c_analytical))
        assert error < 0.02, f"Steady-state error: {error:.2e}"

    def test_boundary_bottom(self, jacobi_result):
        """Bottom boundary should be c=0 throughout."""
        assert np.allclose(jacobi_result.y[:, 0], 0), "Bottom boundary not zero"

    def test_boundary_top(self, jacobi_result):
        """Top boundary should be c=1 throughout."""
        assert np.allclose(jacobi_result.y[:, -1], 1), "Top boundary not one"

    def test_delta_decreases(self, jacobi_result):
        """Convergence measure should decrease from first to last iteration."""
        delta = jacobi_result.delta_history
        assert delta[-1] < delta[0], "Delta did not decrease overall"

    def test_x_uniformity(self, jacobi_result):
        """Solution should be uniform in x (periodic BCs, x-uniform IC)."""
        y = jacobi_result.y
        for j in range(1, y.shape[1] - 1):
            col_values = y[:, j]
            assert np.allclose(col_values, col_values[0], atol=1e-10), \
                f"Solution varies in x at j={j}"

    def test_monotonic_in_y(self, jacobi_result, grid):
        """Solution should be monotonically increasing in y."""
        # Check at x=0 (any x column works due to x-uniformity)
        profile = jacobi_result.y[0, :]
        diffs = np.diff(profile)
        assert np.all(diffs >= 0), "Solution not monotonically increasing in y"
