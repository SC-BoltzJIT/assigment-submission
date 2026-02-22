"""SOR iteration for the 2D Laplace equation (Assignment 1.6.H).

Solves the steady-state diffusion equation with SOR iteration with omega = 1.9
on an N=50 grid and compares to the analytical solution c(y) = y.
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
grid = Grid2D(N=N, L=1.0)
omega = 1.9

# Initial guess: zero everywhere, then apply BCs
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# Create insulating objects
coords = construct_rectangle(21, 38, 34, 43)

# Solve
result = solve_bvp(
    c0,
    method="sor",
    post_step=fixed_bc,
    tol=1e-5,
    omega=omega,
    insulator_coordinates=coords,
)

print(
    f"SOR iteration: converged={result.converged}, "
    f"iterations={result.n_iter}, "
    f"final delta={result.delta_history[-1]:.2e}"
)

# Compare to analytical c(y) = y
c_analytical = grid.Y
error = np.max(np.abs(result.y - c_analytical))
print(f"Max error vs c(y)=y: {error:.2e}")

# Save directory
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

mid_i = N // 2

# --- Figure 1: 3-panel overview ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. Concentration field
ax = axes[0]
im = ax.pcolormesh(
    grid.X, grid.Y, result.y, shading="auto", cmap="gist_heat", vmin=0, vmax=1
)
fig.colorbar(
    im,
    ax=ax,
    label=r"$c(x,y)$",
    location="top",
    orientation="horizontal",
    fraction=0.05,
    pad=0.06,
)
ax.tick_params(axis="both", which="minor", direction="out", length=1)
ax.tick_params(axis="both", which="major", direction="out", length=2.5)
ax.set_title("SOR solution $c(x, y)$")
ax.set_xlabel(r"$x$ [m]")
ax.set_ylabel(r"$y$ [m]")
ax.set_aspect("equal")

# 2. Profile c(y) at x=0.5 vs analytical
ax = axes[1]
ax.plot(
    grid.y, result.y[mid_i, :], "o-", markersize=3, label=f"SOR, $\\omega={omega:.1f}$"
)
ax.plot(grid.y, grid.y, "--", color="black", linewidth=1.2, label="Analytical $c=y$")
ax.set_xlabel(r"$y$ [m]")
ax.set_ylabel(r"$c(x{=}0.5,\,y)$")
ax.set_title("Profile at $x = 0.5$")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Convergence history
ax = axes[2]
ax.semilogy(result.delta_history)
ax.set_xlabel("Iteration $k$")
ax.set_ylabel(r"$\delta_k$ (max-norm)")
ax.set_title(f"Convergence ({result.n_iter} iterations)")
ax.grid(True, which="both", alpha=0.3)

fig.suptitle(f"SOR method with an insulating object ($\\omega = {omega:.1f}$)")

plt.tight_layout()

filename = "a1_6_insulators_sor.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()

# --- Figure 2: 2-panel (2D field top, profile bottom) ---
fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(4, 7), constrained_layout=True)

# Top: concentration field
im2 = ax_top.pcolormesh(
    grid.X, grid.Y, result.y, shading="auto", cmap="gist_heat", vmin=0, vmax=1
)
fig2.colorbar(
    im2,
    ax=ax_top,
    label=r"$c(x,y)$",
    location="top",
    orientation="horizontal",
    fraction=0.05,
    pad=0.06,
)
ax_top.tick_params(axis="both", which="minor", direction="out", length=1)
ax_top.tick_params(axis="both", which="major", direction="out", length=2.5)
ax_top.set_xlabel(r"$x$ [m]")
ax_top.set_ylabel(r"$y$ [m]")
ax_top.set_title("SOR solution $c(x, y)$")
ax_top.set_aspect("equal")

# Bottom: profile c(y) at x=0.5
ax_bot.plot(
    grid.y, result.y[mid_i, :], "o-", markersize=3, label=f"SOR, $\\omega={omega:.1f}$"
)
ax_bot.plot(
    grid.y, grid.y, "--", color="black", linewidth=1.2, label="Analytical $c=y$"
)
ax_bot.set_xlabel(r"$y$ [m]")
ax_bot.set_ylabel(r"$c(x{=}0.5,\,y)$")
ax_bot.set_title("Profile at $x = 0.5$")
ax_bot.legend()
ax_bot.grid(True, alpha=0.3)

filename2 = "a1_6_insulators_sor_2panel.png"
fig2.savefig(out_dir / filename2, dpi=150)
print(f"Saved to {out_dir / filename2}")

plt.show()
