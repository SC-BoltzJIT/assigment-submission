"""Convergence comparison: Jacobi vs Gauss-Seidel vs SOR (Assignment 1.6.I).

Shows how the convergence measure delta (Eq. 14) depends on the number
of iterations k for each method. Log-lin plot (semilogy).
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

# Solve with both methods
jacobi = solve_bvp(c0, method="jacobi", post_step=fixed_bc, tol=tol)
gs = solve_bvp(c0, method="gauss_seidel", post_step=fixed_bc, tol=tol)
omega_list = np.linspace(1.3, 1.9, 4)
sor_list = [solve_bvp(c0, method="sor", post_step=fixed_bc, tol=tol, omega = om)
            for om in omega_list]


print(f"Jacobi:       {jacobi.n_iter} iterations, final delta={jacobi.delta_history[-1]:.2e}")
print(f"Gauss-Seidel: {gs.n_iter} iterations, final delta={gs.delta_history[-1]:.2e}")
print(f"Ratio (Jacobi / GS): {jacobi.n_iter / gs.n_iter:.2f}")
for sor, omega in zip(sor_list, omega_list):
    print(f"SOR (ω={omega:.1f}):  {sor.n_iter} iterations, final delta={sor.delta_history[-1]:.2e}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(range(1, jacobi.n_iter + 1), jacobi.delta_history,
            label=f"Jacobi ({jacobi.n_iter} iter)")
ax.semilogy(range(1, gs.n_iter + 1), gs.delta_history,
            label=f"Gauss-Seidel ({gs.n_iter} iter)")
for sor, omega in zip(sor_list, omega_list):
    ax.semilogy(range(1, sor.n_iter + 1), sor.delta_history,
            label=f"SOR, $\\omega={omega:.1f}$ ({sor.n_iter} iter)")

ax.axhline(tol, color="gray", linestyle=":", linewidth=0.8, label=rf"$\epsilon = {tol}$")
ax.set_xlabel("Iteration k")
ax.set_ylabel(r"$\delta_k = \max_{i,j} |c^{k+1}_{i,j} - c^{k}_{i,j}|$")
ax.set_title(f"Convergence comparison (N = {N})")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "a1_6_convergence.png", dpi=150)
print(f"Saved to {out_dir / 'a1_6_convergence.png'}")

plt.show()
