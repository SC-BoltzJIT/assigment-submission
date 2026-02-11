"""Test iterative solvers for the 2D Laplace equation.

Validates that Jacobi, Gauss-Seidel, and SOR methods converge
to the analytical solution c(y) = y.
"""

import numpy as np
import pytest

from scicomp3 import Grid2D, solve_laplace


# Parameters
N = 20  # smaller grid for faster tests
tol = 1e-4


@pytest.fixture
def grid():
    """Create computational grid."""
    return Grid2D(N=N, L=1.0)


@pytest.fixture
def analytical_solution(grid):
    """Analytical solution: c(y) = y"""
    return grid.Y


class TestLaplaceSolvers:
    """Tests for Laplace equation iterative solvers."""

    def test_gauss_seidel_converges(self, grid):
        """Gauss-Seidel should converge."""
        result = solve_laplace(grid, method="gauss_seidel", tol=tol)
        assert result.converged, f"Did not converge: {result.message}"

    def test_gauss_seidel_solution_accuracy(self, grid, analytical_solution):
        """Gauss-Seidel solution should match analytical c(y) = y."""
        result = solve_laplace(grid, method="gauss_seidel", tol=tol)
        error = np.max(np.abs(result.c - analytical_solution))
        # Allow some numerical error (depends on grid size and tolerance)
        assert error < 0.05, f"Solution error too large: {error}"

    def test_gauss_seidel_boundary_conditions(self, grid):
        """Boundary conditions should be satisfied."""
        result = solve_laplace(grid, method="gauss_seidel", tol=tol)
        # Bottom boundary: c(y=0) = 0
        assert np.allclose(result.c[:, 0], 0), "Bottom boundary not zero"
        # Top boundary: c(y=1) = 1
        assert np.allclose(result.c[:, -1], 1), "Top boundary not one"

    def test_jacobi_converges(self, grid):
        """Jacobi should converge (slower than Gauss-Seidel)."""
        result = solve_laplace(grid, method="jacobi", tol=tol, max_iter=10000)
        assert result.converged, f"Did not converge: {result.message}"

    def test_jacobi_slower_than_gauss_seidel(self, grid):
        """Jacobi should require more iterations than Gauss-Seidel."""
        result_jacobi = solve_laplace(grid, method="jacobi", tol=tol, max_iter=10000)
        result_gs = solve_laplace(grid, method="gauss_seidel", tol=tol)

        assert result_jacobi.iterations > result_gs.iterations, \
            f"Jacobi ({result_jacobi.iterations}) should be slower than Gauss-Seidel ({result_gs.iterations})"

    def test_sor_with_omega_1_equals_gauss_seidel(self, grid):
        """SOR with ω=1 should behave like Gauss-Seidel."""
        result_gs = solve_laplace(grid, method="gauss_seidel", tol=tol)
        result_sor = solve_laplace(grid, method="sor", tol=tol, omega=1.0)

        # Should have same number of iterations (within 1 due to possible floating point)
        assert abs(result_sor.iterations - result_gs.iterations) <= 1, \
            f"SOR(ω=1) iterations ({result_sor.iterations}) should equal Gauss-Seidel ({result_gs.iterations})"

    def test_sor_optimal_omega_faster(self, grid):
        """SOR with optimal ω should converge faster than Gauss-Seidel."""
        result_gs = solve_laplace(grid, method="gauss_seidel", tol=tol)
        # For this problem, optimal ω is typically around 1.7-1.9
        result_sor = solve_laplace(grid, method="sor", tol=tol, omega=1.8)

        assert result_sor.iterations < result_gs.iterations, \
            f"SOR(ω=1.8) ({result_sor.iterations}) should be faster than Gauss-Seidel ({result_gs.iterations})"
