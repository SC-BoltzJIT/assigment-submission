"""
Comparison between various omega values to find the optimal omega value
that minimises the number of iterations needed — with and without sink/insulator objects.
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
from scicomp3.bvp.omega import search_for_optimal_omega
from scicomp3.objects.shapes import construct_rectangle


def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y


# Parameters
N = 50
grid = Grid2D(N=N, L=1.0)

# Initial guess: zero everywhere, then apply BCs
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# Sink coordinates (same as a1_6_sinks_sor.py)
sink_coords = construct_rectangle(21, 25, 24, 26)

# Insulator coordinates (same as a1_6_insulators_sor.py)
insulator_coords = construct_rectangle(21, 38, 34, 43)

# --- Search without sink ---
omega_optimal, n_iter_optimal, omega_values, n_iterations = search_for_optimal_omega(
    c0, post_step=fixed_bc, tol=1e-5
)

# --- Search with sink ---
omega_optimal_sink, n_iter_optimal_sink, omega_values_sink, n_iterations_sink = (
    search_for_optimal_omega(
        c0, post_step=fixed_bc, tol=1e-5, sink_coordinates=sink_coords
    )
)

# --- Search with insulator ---
omega_optimal_ins, n_iter_optimal_ins, omega_values_ins, n_iterations_ins = (
    search_for_optimal_omega(
        c0, post_step=fixed_bc, tol=1e-5, insulator_coordinates=insulator_coords
    )
)

# Print summary
print(
    f"Without objects: optimal omega = {omega_optimal:.4f}, iterations = {n_iter_optimal}"
)
print(
    f"With sink:       optimal omega = {omega_optimal_sink:.4f}, iterations = {n_iter_optimal_sink}"
)
print(
    f"With insulator:  optimal omega = {omega_optimal_ins:.4f}, iterations = {n_iter_optimal_ins}"
)

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.semilogy(omega_values, n_iterations, marker="o", markersize=3, label="No object")
ax.semilogy(
    omega_values_sink, n_iterations_sink, marker="s", markersize=3, label="With sink"
)
ax.semilogy(
    omega_values_ins, n_iterations_ins, marker="^", markersize=3, label="With insulator"
)
ax.plot(
    omega_optimal,
    n_iter_optimal,
    marker="*",
    markersize=10,
    color="C0",
    zorder=5,
    label=f"Optimal $\\omega = {omega_optimal:.3f}$, $k = {n_iter_optimal}$",
)
ax.plot(
    omega_optimal_sink,
    n_iter_optimal_sink,
    marker="*",
    markersize=10,
    color="C1",
    zorder=5,
    label=f"Optimal $\\omega = {omega_optimal_sink:.3f}$, $k = {n_iter_optimal_sink}$ (sink)",
)
ax.plot(
    omega_optimal_ins,
    n_iter_optimal_ins,
    marker="*",
    markersize=10,
    color="C2",
    zorder=5,
    label=f"Optimal $\\omega = {omega_optimal_ins:.3f}$, $k = {n_iter_optimal_ins}$ (insulator)",
)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel("Iterations to convergence")
ax.set_title(r"Optimal $\omega$ search: effect of objects")

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
filename = "a1_6_seeking_optimal_omega.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()
