"""Convergence comparison: Jacobi, Gauss-Seidel, SOR (Assignment 1.6.I).

Shows how the convergence measure delta (Eq. 14) depends on the number
of iterations k for each method. Log-lin plot (semilogy).
For SOR, a few representative values of omega are shown.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.solver import solve_bvp


def fixed_bc(k, y):
    apply_diffusion_bc(y)
    return y


# Parameters
N = 50
tol = 1e-5
grid = Grid2D(N=N, L=1.0)

c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# Solve with all methods
jacobi = solve_bvp(c0, method="jacobi", post_step=fixed_bc, tol=tol)
gs = solve_bvp(c0, method="gauss_seidel", post_step=fixed_bc, tol=tol)

sor_omegas = [1.5, 1.7, 1.85, 1.95]
sor_results = {}
for omega in sor_omegas:
    sor_results[omega] = solve_bvp(c0, method="sor", post_step=fixed_bc,
                                   tol=tol, omega=omega)

# Print summary
print(f"{'Method':<20} {'Iterations':>10}")
print("-" * 32)
print(f"{'Jacobi':<20} {jacobi.n_iter:>10}")
print(f"{'Gauss-Seidel':<20} {gs.n_iter:>10}")
for omega, res in sor_results.items():
    print(f"{'SOR (w=' + f'{omega})' :<20} {res.n_iter:>10}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(range(1, jacobi.n_iter + 1), jacobi.delta_history,
            label=f"Jacobi ({jacobi.n_iter} iter)", linewidth=1.5)
ax.semilogy(range(1, gs.n_iter + 1), gs.delta_history,
            label=f"Gauss-Seidel ({gs.n_iter} iter)", linewidth=1.5)

for omega, res in sor_results.items():
    ax.semilogy(range(1, res.n_iter + 1), res.delta_history,
                label=rf"SOR $\omega$={omega} ({res.n_iter} iter)", linewidth=1.5)

ax.axhline(tol, color="gray", linestyle=":", linewidth=0.8, label=rf"$\epsilon = {tol}$")
ax.set_xlabel("Iteration k")
ax.set_ylabel(r"$\delta_k = \max_{i,j} |c^{k+1}_{i,j} - c^{k}_{i,j}|$")
ax.set_title(f"Convergence comparison (N = {N})")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "a1_6_convergence.png", dpi=150)
print(f"Saved to {out_dir / 'a1_6_convergence.png'}")

plt.show()
