"""Test Gauss-Seidel iteration for the 2D Laplace equation.

Verifies that the Gauss-Seidel iteration converges to the analytical
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


def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y


@pytest.fixture
def gs_result(grid):
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)
    return solve_bvp(c0, method="gauss_seidel", post_step=fixed_bc,
                     tol=1e-5, max_iter=100_000)


class TestGaussSeidelIteration:
    """Tests for Gauss-Seidel iteration solver (Assignment 1.6.H)."""

    def test_converges(self, gs_result):
        """Gauss-Seidel iteration should converge within max_iter."""
        assert gs_result.converged, (
            f"Did not converge in {gs_result.n_iter} iterations, "
            f"final delta = {gs_result.delta_history[-1]:.2e}"
        )

    def test_fewer_iterations_than_jacobi(self, gs_result, grid):
        """Gauss-Seidel should converge faster than Jacobi."""
        c0 = np.zeros(grid.shape)
        apply_diffusion_bc(c0)
        jacobi = solve_bvp(c0, method="jacobi", post_step=fixed_bc,
                           tol=1e-5, max_iter=100_000)
        assert gs_result.n_iter < jacobi.n_iter, (
            f"GS ({gs_result.n_iter}) not faster than Jacobi ({jacobi.n_iter})"
        )

    def test_steady_state_profile(self, gs_result, grid):
        """Solution should match analytical steady state c(y) = y."""
        c_analytical = grid.Y
        error = np.max(np.abs(gs_result.y - c_analytical))
        assert error < 0.02, f"Steady-state error: {error:.2e}"

    def test_boundary_bottom(self, gs_result):
        """Bottom boundary should be c=0 throughout."""
        assert np.allclose(gs_result.y[:, 0], 0), "Bottom boundary not zero"

    def test_boundary_top(self, gs_result):
        """Top boundary should be c=1 throughout."""
        assert np.allclose(gs_result.y[:, -1], 1), "Top boundary not one"

    def test_delta_decreases(self, gs_result):
        """Convergence measure should decrease from first to last iteration."""
        delta = gs_result.delta_history
        assert delta[-1] < delta[0], "Delta did not decrease overall"

    def test_x_uniformity(self, gs_result):
        """Solution should be approximately uniform in x.

        Gauss-Seidel's sequential sweep (incrementing i for fixed j)
        introduces a tiny directional bias, unlike Jacobi which is
        perfectly symmetric via np.roll. The x-variation is O(tol).
        """
        y = gs_result.y
        for j in range(1, y.shape[1] - 1):
            col_values = y[:, j]
            assert np.allclose(col_values, col_values[0], atol=1e-5), \
                f"Solution varies in x at j={j}"

    def test_monotonic_in_y(self, gs_result, grid):
        """Solution should be monotonically increasing in y."""
        profile = gs_result.y[0, :]
        diffs = np.diff(profile)
        assert np.all(diffs >= 0), "Solution not monotonically increasing in y"
