"""Find the optimal SOR omega and its dependence on N (Assignment 1.6.J).

Sweeps omega for several grid sizes N and records iterations to convergence.
Plots: (1) iterations vs omega for selected N, (2) optimal omega vs N
with the theoretical prediction omega_opt = 2 / (1 + sin(pi/N)).

Note: SOR uses pure-Python loops so large N is slow.  The sweep is kept
compact (small N values, coarse omega grid) to finish in reasonable time.
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


def sweep_omega(N, omegas, tol=1e-5, max_iter=100_000):
    """Run SOR for each omega on an NxN grid. Returns list of (omega, n_iter)."""
    grid = Grid2D(N=N, L=1.0)
    c0 = np.zeros(grid.shape)
    apply_diffusion_bc(c0)

    results = []
    for omega in omegas:
        res = solve_bvp(c0, method="sor", post_step=fixed_bc,
                        tol=tol, max_iter=max_iter, omega=omega)
        results.append((omega, res.n_iter))
    return results


tol = 1e-5
max_iter = 100_000

# --- Part 1: iterations vs omega for a few N values ---
plot_grid_sizes = [15, 25, 50]
omegas = np.arange(1.40, 1.98, 0.04)

print("Part 1: iterations vs omega")
print(f"{'N':>4}  {'best omega':>10}  {'iterations':>10}  {'theory':>10}")
print("-" * 40)

plot_data = {}
for N in plot_grid_sizes:
    pairs = sweep_omega(N, omegas, tol=tol, max_iter=max_iter)
    plot_data[N] = pairs
    best_omega, best_iter = min(pairs, key=lambda x: x[1])
    theory = 2.0 / (1.0 + np.sin(np.pi / N))
    print(f"{N:4d}  {best_omega:10.2f}  {best_iter:10d}  {theory:10.4f}")

# --- Part 2: optimal omega vs N ---
print("\nPart 2: optimal omega vs N")
N_range = [10, 15, 20, 25, 30, 35, 40, 50]
optimal_measured = []
optimal_theory = []

for N in N_range:
    pairs = sweep_omega(N, omegas, tol=tol, max_iter=max_iter)
    best_omega, best_iter = min(pairs, key=lambda x: x[1])
    theory = 2.0 / (1.0 + np.sin(np.pi / N))
    optimal_measured.append(best_omega)
    optimal_theory.append(theory)
    print(f"N={N:3d}: measured={best_omega:.2f}, theory={theory:.4f}, iter={best_iter}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: iterations vs omega
ax = axes[0]
for N in plot_grid_sizes:
    ws = [p[0] for p in plot_data[N]]
    iters = [p[1] for p in plot_data[N]]
    ax.plot(ws, iters, "o-", markersize=4, label=f"N = {N}")
    theory = 2.0 / (1.0 + np.sin(np.pi / N))
    ax.axvline(theory, color="gray", linestyle=":", linewidth=0.5)

ax.set_xlabel(r"$\omega$")
ax.set_ylabel("Iterations to convergence")
ax.set_title(r"SOR iterations vs $\omega$")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: optimal omega vs N
ax = axes[1]
ax.plot(N_range, optimal_measured, "o", markersize=5,
        label=r"Measured $\omega_{opt}$")
N_fine = np.linspace(8, 55, 200)
ax.plot(N_fine, 2.0 / (1.0 + np.sin(np.pi / N_fine)), "-", linewidth=1.5,
        label=r"Theory: $\frac{2}{1 + \sin(\pi/N)}$")
ax.set_xlabel("N")
ax.set_ylabel(r"Optimal $\omega$")
ax.set_title(r"Optimal $\omega$ vs grid size N")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "a1_6_sor_optimal_omega.png", dpi=150)
print(f"\nSaved to {out_dir / 'a1_6_sor_optimal_omega.png'}")

plt.show()
