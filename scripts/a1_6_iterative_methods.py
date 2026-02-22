"""Compare iterative methods for the 2D Laplace equation (Assignment 1.6).

Solves the steady-state diffusion equation with Jacobi, Gauss-Seidel, and SOR,
then plots all three profiles c(y) at x=0.5 against the analytical solution.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scienceplots  # noqa: F401 (registers styles on import)

styles = ["science"] if shutil.which("latex") else ["science", "no-latex"]
plt.style.use(styles)

from scicomp3.core.grid import Grid2D
from scicomp3.pde.diffusion import apply_diffusion_bc
from scicomp3.bvp.solver import solve_bvp
from scicomp3.bvp.omega import get_optimal_omega


def fixed_bc(k, y):
    """Enforce diffusion BCs after each iteration."""
    apply_diffusion_bc(y)
    return y


# Parameters
N = 50
tol = 1e-5
grid = Grid2D(N=N, L=1.0)
omega = get_optimal_omega(N)
mid_i = N // 2

# Initial guess: zero everywhere, then apply BCs
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# Solve with each method
res_jacobi = solve_bvp(c0, method="jacobi", post_step=fixed_bc, tol=tol)
res_gs = solve_bvp(c0, method="gauss_seidel", post_step=fixed_bc, tol=tol)
res_sor = solve_bvp(c0, method="sor", post_step=fixed_bc, tol=tol, omega=omega)

print(f"Jacobi:        {res_jacobi.n_iter:5d} iterations")
print(f"Gauss-Seidel:  {res_gs.n_iter:5d} iterations")
print(f"SOR (w={omega:.3f}): {res_sor.n_iter:5d} iterations")

# --- Theoretical spectral radii ---
# For 2D Laplace on N-interval grid (h = 1/N):
#   rho_J    = cos(pi/N)
#   rho_GS   = rho_J^2           (GS converges twice as fast as Jacobi)
#   rho_SOR  = omega_opt - 1     (optimal SOR)
# Convergence rate R = -log10(rho): digits of accuracy gained per iteration.
# Iterations to convergence ~ log10(1/tol) / R  (rough estimate).

rho_J = np.cos(np.pi / N)
rho_GS = rho_J**2
rho_SOR = omega - 1

R_J = -np.log10(rho_J)
R_GS = -np.log10(rho_GS)
R_SOR = -np.log10(rho_SOR)

digits_needed = -np.log10(tol)

print()
print(
    f"{'Method':<20} {'rho (spectral radius)':>22} {'R = -log10(rho)':>18} {'Est. iters to tol':>20}"
)
print("-" * 84)
print(f"{'Jacobi':<20} {rho_J:>22.6f} {R_J:>18.6f} {digits_needed / R_J:>20.1f}")
print(
    f"{'Gauss-Seidel':<20} {rho_GS:>22.6f} {R_GS:>18.6f} {digits_needed / R_GS:>20.1f}"
)
print(
    f"{'SOR (optimal)':<20} {rho_SOR:>22.6f} {R_SOR:>18.6f} {digits_needed / R_SOR:>20.1f}"
)
print()
print(f"Note: rho_GS / rho_J^2 = {rho_GS / rho_J**2:.6f}  (should be 1.0)")
print(f"Note: rho_SOR = omega - 1 = {omega:.6f} - 1 = {rho_SOR:.6f}")

# Analytical solution and deviations
c_analytical = grid.y

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: profile c(y) at x = 0.5
ax = axes[0]
ax.plot(
    grid.y,
    res_jacobi.y[mid_i, :],
    "o-",
    markersize=3,
    label=f"Jacobi ({res_jacobi.n_iter} iter)",
)
ax.plot(
    grid.y,
    res_gs.y[mid_i, :],
    "s-",
    markersize=3,
    label=f"Gauss-Seidel ({res_gs.n_iter} iter)",
)
ax.plot(
    grid.y,
    res_sor.y[mid_i, :],
    "^-",
    markersize=3,
    label=f"SOR, $\\omega={omega:.3f}$ ({res_sor.n_iter} iter)",
)
ax.plot(
    grid.y, c_analytical, "--", color="black", linewidth=1.5, label="Analytical $c = y$"
)
ax.set_xlabel("y")
ax.set_ylabel("c(x = 0.5, y)")
ax.set_title("Profile comparison at x = 0.5")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: deviation from analytical
ax = axes[1]
ax.plot(
    grid.y,
    res_jacobi.y[mid_i, :] - c_analytical,
    "o-",
    markersize=3,
    label=f"Jacobi ({res_jacobi.n_iter} iter)",
)
ax.plot(
    grid.y,
    res_gs.y[mid_i, :] - c_analytical,
    "s-",
    markersize=3,
    label=f"Gauss-Seidel ({res_gs.n_iter} iter)",
)
ax.plot(
    grid.y,
    res_sor.y[mid_i, :] - c_analytical,
    "^-",
    markersize=3,
    label=f"SOR, $\\omega={omega:.3f}$ ({res_sor.n_iter} iter)",
)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("y")
ax.set_ylabel("$c_{\\mathrm{numerical}} - c_{\\mathrm{analytical}}$")
ax.set_title("Deviation from analytical at x = 0.5")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
filename = "a1_6_iterative_methods.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()

# Combined twin-axis figure: profile + deviation on one axes
fig3, ax_left = plt.subplots(figsize=(7, 5))
ax_right = ax_left.twinx()

colors = ["tab:blue", "tab:orange", "tab:green"]
labels = [
    f"Jacobi ({res_jacobi.n_iter} iter)",
    f"Gauss-Seidel ({res_gs.n_iter} iter)",
    f"SOR, $\\omega={omega:.3f}$ ({res_sor.n_iter} iter)",
]
profiles = [res_jacobi.y[mid_i, :], res_gs.y[mid_i, :], res_sor.y[mid_i, :]]

for profile, label, color in zip(profiles, labels, colors):
    ax_left.plot(grid.y, profile, "-", color=color, linewidth=1.5, label=label)
    ax_right.plot(
        grid.y, profile - c_analytical, "--", color=color, linewidth=1.0, alpha=0.7
    )

ax_left.plot(grid.y, c_analytical, "k-", linewidth=1.2, label="Analytical $c = y$")
ax_right.axhline(0, color="black", linewidth=0.6, linestyle=":")

ax_left.set_xlabel("y")
ax_left.set_ylabel("$c(x=0.5,\\ y)$")
ax_right.set_ylabel("$c_{\\mathrm{num}} - c_{\\mathrm{ana}}$ (dashed)", color="gray")
ax_right.tick_params(axis="y", labelcolor="gray")
ax_left.set_title("Profile and deviation from analytical at $x = 0.5$")
ax_left.legend(loc="upper left", fontsize=8)
ax_left.grid(True, alpha=0.3)

plt.tight_layout()

filename3 = "a1_6_iterative_methods_combined.png"
plt.savefig(out_dir / filename3, dpi=150)
print(f"Saved to {out_dir / filename3}")

plt.show()

# Combined twin-axis figure: profile (left) + ratio of deviation to Jacobi (right)
fig4, ax_left4 = plt.subplots(figsize=(7, 5))
ax_right4 = ax_left4.twinx()

dev_jacobi = res_jacobi.y[mid_i, :] - c_analytical
dev_gs = res_gs.y[mid_i, :] - c_analytical
dev_sor = res_sor.y[mid_i, :] - c_analytical

# Avoid division by zero at boundary points where dev_jacobi == 0
safe_denom = np.where(np.abs(dev_jacobi) > 1e-15, dev_jacobi, np.nan)

ratio_jacobi = dev_jacobi / safe_denom  # = 1 everywhere (except boundary NaNs)
ratio_gs = dev_gs / safe_denom
ratio_sor = dev_sor / safe_denom

# Print ratio values at y = 0.5
mid_j = N // 2
y_mid = grid.y[mid_j]
print()
print(f"Deviation ratio at y = {y_mid:.3f} (j={mid_j}):")
print(f"  dev_jacobi         = {dev_jacobi[mid_j]:.6e}")
print(f"  dev_gs             = {dev_gs[mid_j]:.6e}")
print(f"  dev_sor            = {dev_sor[mid_j]:.6e}")
print(f"  ratio Jacobi/Jacobi = {ratio_jacobi[mid_j]:.6f}")
print(f"  ratio GS/Jacobi     = {ratio_gs[mid_j]:.6f}")
print(f"  ratio SOR/Jacobi    = {ratio_sor[mid_j]:.6f}")
print(f"  theoretical GS/J    = (1-rho_J)/(1-rho_GS) = {(1-rho_J)/(1-rho_GS):.6f}")
print(f"  theoretical SOR/J   = (1-rho_J)/(1-rho_SOR) = {(1-rho_J)/(1-rho_SOR):.6f}")

for profile, label, color in zip(profiles, labels, colors):
    ax_left4.plot(grid.y, profile, "-", color=color, linewidth=1.5, label=label)

ax_left4.plot(grid.y, c_analytical, "k-", linewidth=1.2, label="Analytical $c = y$")

ax_right4.plot(
    grid.y,
    ratio_jacobi,
    "--",
    color=colors[0],
    linewidth=1.0,
    alpha=0.8,
    label=f"Jacobi / Jacobi",
)
ax_right4.plot(
    grid.y,
    ratio_gs,
    "--",
    color=colors[1],
    linewidth=1.0,
    alpha=0.8,
    label=f"GS / Jacobi",
)
ax_right4.plot(
    grid.y,
    ratio_sor,
    "--",
    color=colors[2],
    linewidth=1.0,
    alpha=0.8,
    label=f"SOR / Jacobi",
)
ax_right4.axhline(1.0, color="black", linewidth=0.6, linestyle=":")

ax_left4.set_xlabel("y")
ax_left4.set_ylabel("$c(x=0.5,\\ y)$")
ax_right4.set_ylabel(
    r"$(c_{\mathrm{num}} - c_{\mathrm{ana}})\ /\ (\mathrm{Jacobi\ deviation})$ (dashed)",
    color="gray",
)
ax_right4.tick_params(axis="y", labelcolor="gray")
ax_left4.set_title("Profile and deviation ratio relative to Jacobi at $x = 0.5$")
ax_left4.legend(loc="upper left", fontsize=8)
ax_right4.legend(loc="lower right", fontsize=8)
ax_left4.grid(True, alpha=0.3)

plt.tight_layout()

filename4 = "a1_6_iterative_methods_ratio.png"
plt.savefig(out_dir / filename4, dpi=150)
print(f"Saved to {out_dir / filename4}")

plt.show()

# Convergence history comparison (standalone figure)
fig2, ax2 = plt.subplots(figsize=(6, 5))

ax2.semilogy(
    range(1, res_jacobi.n_iter + 1),
    res_jacobi.delta_history,
    label=f"Jacobi ({res_jacobi.n_iter} iter)",
)
ax2.semilogy(
    range(1, res_gs.n_iter + 1),
    res_gs.delta_history,
    label=f"Gauss-Seidel ({res_gs.n_iter} iter)",
)
ax2.semilogy(
    range(1, res_sor.n_iter + 1),
    res_sor.delta_history,
    label=f"SOR, $\\omega={omega:.3f}$ ({res_sor.n_iter} iter)",
)
ax2.axhline(
    tol, color="black", linewidth=0.8, linestyle="--", label=f"Tolerance ({tol:.0e})"
)

ax2.set_xlabel("Iteration k")
ax2.set_ylabel(r"$\delta_k$ (max-norm)")
ax2.set_title("Convergence comparison of iterative methods")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()

filename2 = "a1_6_iterative_methods_convergence.png"
plt.savefig(out_dir / filename2, dpi=150)
print(f"Saved to {out_dir / filename2}")

plt.show()
