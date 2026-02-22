"""Impact of insulator objects on iteration count (Assignment 1.6).

Compares iteration counts for Jacobi, Gauss-Seidel, and SOR
with and without a rectangular insulator object.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scienceplots  # noqa: F401

styles = ["science"] if shutil.which("latex") else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.solver import solve_bvp
from scicomp3.objects.shapes import construct_rectangle


def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y


# Parameters
N = 50
tol = 1e-5
grid = Grid2D(N=N, L=1.0)
omega = 1.9

# Initial guess
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# Insulator coordinates (same as a1_6_insulators_*.py)
insulator_coords = construct_rectangle(21, 38, 34, 43)

# --- Solve without insulator ---
res_jacobi = solve_bvp(c0, method="jacobi", post_step=fixed_bc, tol=tol)
res_gs = solve_bvp(c0, method="gauss_seidel", post_step=fixed_bc, tol=tol)
res_sor = solve_bvp(c0, method="sor", post_step=fixed_bc, tol=tol, omega=omega)

# --- Solve with insulator ---
res_jacobi_ins = solve_bvp(
    c0,
    method="jacobi",
    post_step=fixed_bc,
    tol=tol,
    insulator_coordinates=insulator_coords,
)
res_gs_ins = solve_bvp(
    c0,
    method="gauss_seidel",
    post_step=fixed_bc,
    tol=tol,
    insulator_coordinates=insulator_coords,
)
res_sor_ins = solve_bvp(
    c0,
    method="sor",
    post_step=fixed_bc,
    tol=tol,
    omega=omega,
    insulator_coordinates=insulator_coords,
)

# Collect results
methods = ["Jacobi", "Gauss-Seidel", f"SOR ($\\omega={omega}$)"]
iters_no_ins = [res_jacobi.n_iter, res_gs.n_iter, res_sor.n_iter]
iters_ins = [res_jacobi_ins.n_iter, res_gs_ins.n_iter, res_sor_ins.n_iter]

# Print summary
print(f"{'Method':<20} {'No insulator':>14} {'With insulator':>16} {'Change':>10}")
print("-" * 65)
for m, n, s in zip(methods, iters_no_ins, iters_ins):
    label = m.replace("$\\omega=", "w=").replace("$", "")
    print(f"{label:<20} {n:>14d} {s:>16d} {s - n:>+10d}")

# --- Bar chart ---
x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))
bars1 = ax.bar(x - width / 2, iters_no_ins, width, label="Without insulator")
bars2 = ax.bar(x + width / 2, iters_ins, width, label="With insulator")

# Value labels on bars
for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(bar.get_height())}",
        ha="center",
        va="bottom",
        fontsize=7,
    )
for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(bar.get_height())}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax.set_ylabel("Iterations to convergence")
ax.set_title(
    f"Impact of insulator object on iteration count ($N={N}$, $\\varepsilon={tol:.0e}$)"
)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
filename = "a1_6_insulators_k_impact.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()
