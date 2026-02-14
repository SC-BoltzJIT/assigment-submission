"""Test SOR (Successive Over-Relaxation) for the 2D Laplace equation.

Verifies that SOR converges to the analytical steady-state solution
c(y) = y and that omega=1 recovers Gauss-Seidel behaviour.
"""

import numpy as np
import pytest

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.solver import solve_bvp


N = 50


def fixed_bc(k, y):
    apply_diffusion_bc(y)
    return y


@pytest.fixture
def grid():
    return Grid2D(N=N, L=1.0)


@pytest.fixture
def sor_result(grid):
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)
    return solve_bvp(c0, method="sor", post_step=fixed_bc,
                     tol=1e-5, max_iter=100_000, omega=1.85)


class TestSORIteration:
    """Tests for SOR solver (Assignment 1.6.H)."""

    def test_converges(self, sor_result):
        """SOR should converge within max_iter."""
        assert sor_result.converged, (
            f"Did not converge in {sor_result.n_iter} iterations, "
            f"final delta = {sor_result.delta_history[-1]:.2e}"
        )

    def test_steady_state_profile(self, sor_result, grid):
        """Solution should match analytical steady state c(y) = y."""
        c_analytical = grid.Y
        error = np.max(np.abs(sor_result.y - c_analytical))
        assert error < 0.02, f"Steady-state error: {error:.2e}"

    def test_boundary_bottom(self, sor_result):
        """Bottom boundary should be c=0."""
        assert np.allclose(sor_result.y[:, 0], 0), "Bottom boundary not zero"

    def test_boundary_top(self, sor_result):
        """Top boundary should be c=1."""
        assert np.allclose(sor_result.y[:, -1], 1), "Top boundary not one"

    def test_delta_decreases(self, sor_result):
        """Convergence measure should decrease from first to last iteration."""
        delta = sor_result.delta_history
        assert delta[-1] < delta[0], "Delta did not decrease overall"

    def test_faster_than_gauss_seidel(self, grid):
        """SOR with a good omega should converge faster than Gauss-Seidel."""
        c0 = np.zeros(grid.shape)
        apply_diffusion_bc(c0)
        gs = solve_bvp(c0, method="gauss_seidel", post_step=fixed_bc,
                       tol=1e-5, max_iter=100_000)
        sor = solve_bvp(c0, method="sor", post_step=fixed_bc,
                        tol=1e-5, max_iter=100_000, omega=1.85)
        assert sor.n_iter < gs.n_iter, (
            f"SOR ({sor.n_iter}) not faster than GS ({gs.n_iter})"
        )

    def test_omega_1_matches_gauss_seidel(self, grid):
        """SOR with omega=1 should produce identical results to Gauss-Seidel."""
        c0 = np.zeros(grid.shape)
        apply_diffusion_bc(c0)
        gs = solve_bvp(c0, method="gauss_seidel", post_step=fixed_bc,
                       tol=1e-5, max_iter=100_000)
        sor1 = solve_bvp(c0, method="sor", post_step=fixed_bc,
                         tol=1e-5, max_iter=100_000, omega=1.0)
        assert sor1.n_iter == gs.n_iter, (
            f"SOR(omega=1) iterations ({sor1.n_iter}) != GS ({gs.n_iter})"
        )
        assert np.allclose(sor1.y, gs.y, atol=1e-12), "SOR(omega=1) != GS solution"

    def test_diverges_outside_range(self, grid):
        """SOR with omega >= 2 should fail to converge."""
        c0 = np.zeros(grid.shape)
        apply_diffusion_bc(c0)
        result = solve_bvp(c0, method="sor", post_step=fixed_bc,
                           tol=1e-5, max_iter=1000, omega=2.0)
        assert not result.converged, "SOR with omega=2 should not converge"

    def test_monotonic_in_y(self, sor_result):
        """Solution should be monotonically increasing in y."""
        profile = sor_result.y[0, :]
        diffs = np.diff(profile)
        assert np.all(diffs >= 0), "Solution not monotonically increasing in y"
