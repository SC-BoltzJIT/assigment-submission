"""Impact of objects on iteration count (Assignment 1.6).

Compares iteration counts for Jacobi, Gauss-Seidel, and SOR
with no object, a sink, and an insulator — all in one grouped bar chart.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scienceplots  # noqa: F401

plt.style.use("science")

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

# Object coordinates
sink_coords = construct_rectangle(21, 25, 24, 26)
insulator_coords = construct_rectangle(21, 38, 34, 43)

# --- Solve: no object ---
res_j = solve_bvp(c0, method="jacobi", post_step=fixed_bc, tol=tol)
res_gs = solve_bvp(c0, method="gauss_seidel", post_step=fixed_bc, tol=tol)
res_sor = solve_bvp(c0, method="sor", post_step=fixed_bc, tol=tol, omega=omega)

# --- Solve: with sink ---
res_j_sink = solve_bvp(
    c0, method="jacobi", post_step=fixed_bc, tol=tol, sink_coordinates=sink_coords
)
res_gs_sink = solve_bvp(
    c0, method="gauss_seidel", post_step=fixed_bc, tol=tol, sink_coordinates=sink_coords
)
res_sor_sink = solve_bvp(
    c0, method="sor", post_step=fixed_bc, tol=tol, omega=omega,
    sink_coordinates=sink_coords,
)

# --- Solve: with insulator ---
res_j_ins = solve_bvp(
    c0, method="jacobi", post_step=fixed_bc, tol=tol,
    insulator_coordinates=insulator_coords,
)
res_gs_ins = solve_bvp(
    c0, method="gauss_seidel", post_step=fixed_bc, tol=tol,
    insulator_coordinates=insulator_coords,
)
res_sor_ins = solve_bvp(
    c0, method="sor", post_step=fixed_bc, tol=tol, omega=omega,
    insulator_coordinates=insulator_coords,
)

# Collect results
methods = ["Jacobi", "Gauss-Seidel", f"SOR ($\\omega={omega}$)"]
iters_none = [res_j.n_iter, res_gs.n_iter, res_sor.n_iter]
iters_sink = [res_j_sink.n_iter, res_gs_sink.n_iter, res_sor_sink.n_iter]
iters_ins = [res_j_ins.n_iter, res_gs_ins.n_iter, res_sor_ins.n_iter]

# Print summary
print(f"{'Method':<20} {'No object':>12} {'Sink':>10} {'Insulator':>12}")
print("-" * 58)
for m, a, b, c in zip(methods, iters_none, iters_sink, iters_ins):
    label = m.replace("$\\omega=", "w=").replace("$", "")
    print(f"{label:<20} {a:>12d} {b:>10d} {c:>12d}")

# --- Grouped bar chart (3 bars per method) ---
x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(6, 4))
bars1 = ax.bar(x - width, iters_none, width, label="No object")
bars2 = ax.bar(x, iters_sink, width, label="Sink")
bars3 = ax.bar(x + width, iters_ins, width, label="Insulator")

# Value labels
for bars in (bars1, bars2, bars3):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

ax.set_ylabel("Iterations to convergence")
ax.set_title(
    f"Impact of objects on iteration count ($N={N}$, $\\varepsilon={tol:.0e}$)"
)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
filename = "a1_6_objects_k_impact.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()
