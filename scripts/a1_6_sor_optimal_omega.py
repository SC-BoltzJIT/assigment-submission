"""Find the optimal SOR omega and its dependence on N (Assignment 1.6.J).

Sweeps omega in [1.5, 1.99] for several grid sizes N and records the
number of iterations to convergence. Plots iterations vs omega for
each N, and the optimal omega vs N with the theoretical prediction.

Theoretical optimal omega for the 2D Laplace equation on an NxN grid
with Dirichlet + periodic BCs:
    omega_opt = 2 / (1 + sin(pi / N))
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


tol = 1e-5
max_iter = 100_000

# --- Part 1: iterations vs omega for several N values ---
grid_sizes = [25, 50, 75, 100]
omegas = np.arange(1.50, 1.99, 0.02)

results = {}  # {N: [(omega, n_iter), ...]}

for N in grid_sizes:
    grid = Grid2D(N=N, L=1.0)
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)

    results[N] = []
    for omega in omegas:
        res = solve_bvp(c0, method="sor", post_step=fixed_bc,
                        tol=tol, max_iter=max_iter, omega=omega)
        results[N].append((omega, res.n_iter))

    best_omega, best_iter = min(results[N], key=lambda x: x[1])
    theory_omega = 2.0 / (1.0 + np.sin(np.pi / N))
    print(f"N={N:3d}: best omega={best_omega:.2f} ({best_iter} iter), "
          f"theoretical={theory_omega:.4f}")

# --- Plot 1: iterations vs omega ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for N in grid_sizes:
    ws = [r[0] for r in results[N]]
    iters = [r[1] for r in results[N]]
    ax.plot(ws, iters, "o-", markersize=3, label=f"N = {N}")

ax.set_xlabel(r"$\omega$")
ax.set_ylabel("Iterations to convergence")
ax.set_title(r"SOR iterations vs $\omega$")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 2: optimal omega vs N ---
N_range = np.arange(10, 151, 5)
optimal_omegas_measured = []
optimal_omegas_theory = []

for N in N_range:
    grid = Grid2D(N=N, L=1.0)
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)

    # Coarse sweep to find approximate minimum
    best_omega = 1.5
    best_iter = max_iter
    for omega in np.arange(1.50, 1.99, 0.02):
        res = solve_bvp(c0, method="sor", post_step=fixed_bc,
                        tol=tol, max_iter=max_iter, omega=omega)
        if res.n_iter < best_iter:
            best_iter = res.n_iter
            best_omega = omega

    optimal_omegas_measured.append(best_omega)
    optimal_omegas_theory.append(2.0 / (1.0 + np.sin(np.pi / N)))
    print(f"N={N:3d}: optimal omega={best_omega:.2f}, "
          f"theory={optimal_omegas_theory[-1]:.4f}, iter={best_iter}")

ax = axes[1]
ax.plot(N_range, optimal_omegas_measured, "o", markersize=4,
        label=r"Measured $\omega_{opt}$")
ax.plot(N_range, optimal_omegas_theory, "-", linewidth=1.5,
        label=r"Theory: $2 / (1 + \sin(\pi/N))$")
ax.set_xlabel("N")
ax.set_ylabel(r"Optimal $\omega$")
ax.set_title(r"Optimal $\omega$ vs grid size N")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "a1_6_sor_optimal_omega.png", dpi=150)
print(f"Saved to {out_dir / 'a1_6_sor_optimal_omega.png'}")

plt.show()
