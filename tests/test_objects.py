"""Test BVP solvers with object sinks in the domain (Assignment 1.6.K).

Verifies that object points remain at c=0, that the solution converges,
and that the mask does not break existing behaviour when empty.
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
def rect_mask(grid):
    """A single rectangle in the centre of the domain."""
    mask = np.zeros(grid.shape, dtype=int)
    mask[20:31, 20:31] = 1
    return mask


@pytest.fixture
def sor_result_with_object(grid, rect_mask):
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)
    return solve_bvp(c0, method="sor", post_step=fixed_bc,
                     tol=1e-5, max_iter=100_000, omega=1.85,
                     mask=rect_mask)


class TestObjectSinks:
    """Tests for domains with sink objects (Assignment 1.6.K)."""

    def test_converges_with_object(self, sor_result_with_object):
        """SOR should still converge when an object is present."""
        assert sor_result_with_object.converged, (
            f"Did not converge in {sor_result_with_object.n_iter} iterations"
        )

    def test_object_points_are_zero(self, sor_result_with_object, rect_mask):
        """All points inside the object should be exactly zero."""
        vals = sor_result_with_object.y[rect_mask == 1]
        assert np.all(vals == 0), f"Object points not zero, max = {np.max(np.abs(vals))}"

    def test_boundary_conditions_held(self, sor_result_with_object):
        """Standard BCs (bottom=0, top=1) should still hold."""
        y = sor_result_with_object.y
        assert np.allclose(y[:, 0], 0), "Bottom boundary not zero"
        assert np.allclose(y[:, -1], 1), "Top boundary not one"

    def test_concentration_below_object_is_low(self, sor_result_with_object, grid):
        """Below the object the concentration should be lower than c=y."""
        y_sol = sor_result_with_object.y
        # Just below the object (j=19) at x-centre
        mid_i = N // 2
        assert y_sol[mid_i, 19] < grid.Y[mid_i, 19], \
            "Concentration below object should be reduced by the sink"

    def test_empty_mask_matches_no_mask(self, grid):
        """An all-zero mask should produce the same result as no mask."""
        c0 = np.zeros(grid.shape)
        apply_diffusion_bc(c0)

        no_mask = solve_bvp(c0, method="jacobi", post_step=fixed_bc,
                            tol=1e-5, max_iter=100_000)
        empty_mask = np.zeros(grid.shape, dtype=int)
        with_mask = solve_bvp(c0, method="jacobi", post_step=fixed_bc,
                              tol=1e-5, max_iter=100_000, mask=empty_mask)

        assert no_mask.n_iter == with_mask.n_iter
        assert np.allclose(no_mask.y, with_mask.y, atol=1e-12)

    def test_all_methods_respect_mask(self, grid, rect_mask):
        """Jacobi, Gauss-Seidel, and SOR should all enforce mask=0."""
        c0 = np.zeros(grid.shape)
        apply_diffusion_bc(c0)

        for method, kw in [("jacobi", {}),
                           ("gauss_seidel", {}),
                           ("sor", {"omega": 1.85})]:
            res = solve_bvp(c0, method=method, post_step=fixed_bc,
                            tol=1e-5, max_iter=100_000, mask=rect_mask, **kw)
            vals = res.y[rect_mask == 1]
            assert np.all(vals == 0), f"{method}: object points not zero"
            assert res.converged, f"{method}: did not converge with mask"
