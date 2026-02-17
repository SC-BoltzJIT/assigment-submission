"""Test SOR iteration for the 2D Laplace equation with omega = 1.9.

Verifies that the SOR iteration converges to the analytical
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
def sor_result(grid):
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)
    return solve_bvp(c0, method="sor", post_step=fixed_bc,
                     tol=1e-5, max_iter=100_000, omega=1.9)


class TestSORIteration:
    """Tests for SOR iteration solver (Assignment 1.6.H)."""

    def test_converges(self, sor_result):
        """SOR iteration should converge within max_iter."""
        assert sor_result.converged, (
            f"Did not converge in {sor_result.n_iter} iterations, "
            f"final delta = {sor_result.delta_history[-1]:.2e}"
        )

    def test_fewer_iterations_than_gauss_seidel(self, sor_result, grid):
        """
        SOR should converge faster than Gauss-Seidel
        """
        c0 = np.zeros(grid.shape)
        apply_diffusion_bc(c0)
        gauss_seidel = solve_bvp(c0, method="gauss_seidel", post_step=fixed_bc,
                           tol=1e-5, max_iter=100_000)
        ratio = gauss_seidel.n_iter / sor_result.n_iter
        expected_ratio = 1 + np.cos(np.pi / N)  # ≈ 1.998 for N=50
        assert sor_result.n_iter < gauss_seidel.n_iter, (
            f"SOR ({sor_result.n_iter}) not faster than Gauss-Seidel ({gauss_seidel.n_iter})"
        )
        assert 1.5 < ratio < expected_ratio + 0.1, (
            f"Ratio {ratio:.3f} outside expected range [1.5, {expected_ratio + 0.1:.3f}] "
            f"(GS={gauss_seidel.n_iter}, SOR={sor_result.n_iter})"
        )

    def test_steady_state_profile(self, sor_result, grid):
        """Solution should match analytical steady state c(y) = y."""
        c_analytical = grid.Y
        error = np.max(np.abs(sor_result.y - c_analytical))
        assert error < 0.02, f"Steady-state error: {error:.2e}"

    def test_boundary_bottom(self, sor_result):
        """Bottom boundary should be c=0 throughout."""
        assert np.allclose(sor_result.y[:, 0], 0), "Bottom boundary not zero"

    def test_boundary_top(self, sor_result):
        """Top boundary should be c=1 throughout."""
        assert np.allclose(sor_result.y[:, -1], 1), "Top boundary not one"

    def test_delta_decreases(self, sor_result):
        """Convergence measure should decrease from first to last iteration."""
        delta = sor_result.delta_history
        assert delta[-1] < delta[0], "Delta did not decrease overall"

    def test_x_uniformity(self, sor_result):
        """Solution should be approximately uniform in x.
        """
        y = sor_result.y
        for j in range(1, y.shape[1] - 1):
            col_values = y[:, j]
            assert np.allclose(col_values, col_values[0], atol=1e-5), \
                f"Solution varies in x at j={j}"

    def test_monotonic_in_y(self, sor_result, grid):
        """Solution should be monotonically increasing in y."""
        profile = sor_result.y[0, :]
        diffs = np.diff(profile)
        assert np.all(diffs >= 0), "Solution not monotonically increasing in y"
