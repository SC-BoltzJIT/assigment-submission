"""Insulating objects in the computational domain (Assignment 1.6.L).

Compares sink objects (c=0, Dirichlet) vs insulating objects (dc/dn=0,
Neumann) to understand how each type changes the concentration field
and convergence behaviour.

Key physical differences:
- Sink: absorbs concentration → creates a "shadow" below with lower c
- Insulator: blocks diffusive flux → concentration accumulates above the
  object (higher c) and is depleted below (lower c) along the centreline
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


def make_rect_mask(grid, rectangles, value=1):
    """Create a mask with rectangular objects.

    Args:
        grid: Grid2D instance
        rectangles: list of (i_start, i_end, j_start, j_end) tuples
        value: mask value (1 = sink, 2 = insulator)
    """
    mask = np.zeros(grid.shape, dtype=int)
    for i0, i1, j0, j1 in rectangles:
        mask[i0:i1, j0:j1] = value
    return mask


# --- Parameters ---
N = 50
tol = 1e-5
max_iter = 100_000
omega = 1.85

grid = Grid2D(N=N, L=1.0)
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

rect = [(20, 31, 20, 31)]  # centred rectangle

# --- Solve: no object, sink, insulator ---
print("=" * 60)
print("Comparison: no object vs sink vs insulator")
print("=" * 60)

res_none = solve_bvp(c0, method="sor", post_step=fixed_bc,
                     tol=tol, max_iter=max_iter, omega=omega)
mask_sink = make_rect_mask(grid, rect, value=1)
res_sink = solve_bvp(c0, method="sor", post_step=fixed_bc,
                     tol=tol, max_iter=max_iter, omega=omega,
                     mask=mask_sink)
mask_ins = make_rect_mask(grid, rect, value=2)
res_ins = solve_bvp(c0, method="sor", post_step=fixed_bc,
                    tol=tol, max_iter=max_iter, omega=omega,
                    mask=mask_ins)

print(f"No object:   {res_none.n_iter:5d} iterations")
print(f"Sink:        {res_sink.n_iter:5d} iterations")
print(f"Insulator:   {res_ins.n_iter:5d} iterations")

# --- Vertical profile through the centre ---
mid_i = N // 2
y_coords = grid.y
c_none_profile = res_none.y[mid_i, :]
c_sink_profile = res_sink.y[mid_i, :]
c_ins_profile = res_ins.y[mid_i, :]

# --- Plotting ---
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# Row 1: concentration fields
titles = ["No object", "Sink (c=0)", "Insulator (dc/dn=0)"]
results = [res_none, res_sink, res_ins]
masks = [None, mask_sink, mask_ins]

for col, (title, res, mask) in enumerate(zip(titles, results, masks)):
    ax = axes[0, col]
    y_plot = res.y.copy()
    if mask is not None:
        y_plot[mask > 0] = np.nan
    im = ax.contourf(grid.X, grid.Y, y_plot, levels=30, cmap="viridis")
    if mask is not None:
        i0, i1, j0, j1 = rect[0]
        x0, y0_ = grid.X[i0, j0], grid.Y[i0, j0]
        x1, y1_ = grid.X[i1 - 1, j1 - 1], grid.Y[i1 - 1, j1 - 1]
        ax.add_patch(plt.Rectangle(
            (x0, y0_), x1 - x0, y1_ - y0_,
            fill=True, facecolor="white", edgecolor="black", linewidth=1.5))
    fig.colorbar(im, ax=ax, label="c")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} ({res.n_iter} iter)")
    ax.set_aspect("equal")

# Row 2, left: vertical profile comparison
ax = axes[1, 0]
ax.plot(y_coords, c_none_profile, "-", label="No object", linewidth=1.5)
ax.plot(y_coords, c_sink_profile, "--", label="Sink", linewidth=1.5)
ax.plot(y_coords, c_ins_profile, ":", label="Insulator", linewidth=2)
ax.plot(y_coords, y_coords, ":", color="gray", linewidth=0.8, label="c = y")
# Shade the object region
ax.axvspan(grid.y[20], grid.y[30], alpha=0.15, color="gray", label="Object y-range")
ax.set_xlabel("y")
ax.set_ylabel("c(x=0.5, y)")
ax.set_title("Vertical profile at x = 0.5")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Row 2, centre: convergence comparison
ax = axes[1, 1]
ax.semilogy(range(1, res_none.n_iter + 1), res_none.delta_history,
            label=f"No object ({res_none.n_iter})")
ax.semilogy(range(1, res_sink.n_iter + 1), res_sink.delta_history,
            label=f"Sink ({res_sink.n_iter})")
ax.semilogy(range(1, res_ins.n_iter + 1), res_ins.delta_history,
            label=f"Insulator ({res_ins.n_iter})")
ax.axhline(tol, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("Iteration k")
ax.set_ylabel(r"$\delta_k$")
ax.set_title("Convergence comparison")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# Row 2, right: difference field (insulator − no object)
ax = axes[1, 2]
diff = res_ins.y - res_none.y
diff[mask_ins > 0] = np.nan
vmax = np.nanmax(np.abs(diff))
im = ax.contourf(grid.X, grid.Y, diff, levels=30,
                 cmap="RdBu_r", vmin=-vmax, vmax=vmax)
i0, i1, j0, j1 = rect[0]
x0, y0_ = grid.X[i0, j0], grid.Y[i0, j0]
x1, y1_ = grid.X[i1 - 1, j1 - 1], grid.Y[i1 - 1, j1 - 1]
ax.add_patch(plt.Rectangle(
    (x0, y0_), x1 - x0, y1_ - y0_,
    fill=True, facecolor="white", edgecolor="black", linewidth=1.5))
fig.colorbar(im, ax=ax, label=r"$\Delta c$ = insulator $-$ no object")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Concentration difference (insulator vs none)")
ax.set_aspect("equal")

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "a1_6_insulating.png", dpi=150)
print(f"\nSaved to {out_dir / 'a1_6_insulating.png'}")

# --- Summary ---
print("\n" + "=" * 60)
print("Physical interpretation")
print("=" * 60)
print("""
Sink (c=0):
  - Object absorbs concentration → creates a "shadow" below with lower c.
  - Concentration below the sink is reduced; above is slightly increased.
  - Fewer iterations than no object: the Dirichlet constraint pins c=0
    inside the object, reducing the number of unknowns. The solver has
    a smaller "search space" and converges faster.

Insulator (dc/dn=0):
  - Object blocks diffusive flux → concentration must go around it.
  - BELOW the insulator (at x=0.5): c is LOWER than c=y. The upward flux
    from y=0 is blocked, so concentration cannot freely diffuse upward
    and stays closer to the bottom boundary value (like a traffic jam).
  - ABOVE the insulator (at x=0.5): c is HIGHER than c=y. Flux that went
    around the sides converges and accumulates above the object.
  - In the SIDE GAPS: the effect is smaller; c remains close to c=y.
  - More iterations than no object: the Neumann BC adds coupling between
    neighbours without fixing values, making the system less constrained.
  - Total flux from bottom to top is conserved, but redistributed around
    the obstacle.
""")

plt.show()
