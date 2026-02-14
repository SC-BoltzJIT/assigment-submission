"""Objects as sinks in the computational domain (Assignment 1.6.K).

Experiments:
1. Concentration field with a single centred rectangle
2. Concentration field with multiple rectangles
3. Influence of objects on iteration count
4. Influence of objects on optimal omega

All objects are sinks: c = 0 on every object point.
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


def make_rect_mask(grid, rectangles):
    """Create a mask with rectangular objects.

    Args:
        grid: Grid2D instance
        rectangles: list of (i_start, i_end, j_start, j_end) tuples
            (slicing convention, end is exclusive)

    Returns:
        mask: integer array (1 = object, 0 = free)
    """
    mask = np.zeros(grid.shape, dtype=int)
    for i0, i1, j0, j1 in rectangles:
        mask[i0:i1, j0:j1] = 1
    return mask


# --- Parameters ---
N = 50
tol = 1e-5
max_iter = 100_000
omega_default = 1.85

grid = Grid2D(N=N, L=1.0)
c0 = np.zeros(grid.shape)
apply_diffusion_bc(c0)

# --- Experiment 1: single centred rectangle ---
print("=" * 60)
print("Experiment 1: Single centred rectangle")
print("=" * 60)

mask_single = make_rect_mask(grid, [(20, 31, 20, 31)])

res_no_obj = solve_bvp(c0, method="sor", post_step=fixed_bc,
                        tol=tol, max_iter=max_iter, omega=omega_default)
res_single = solve_bvp(c0, method="sor", post_step=fixed_bc,
                        tol=tol, max_iter=max_iter, omega=omega_default,
                        mask=mask_single)

print(f"Without object:  {res_no_obj.n_iter} iterations")
print(f"With rectangle:  {res_single.n_iter} iterations")

# --- Experiment 2: multiple rectangles ---
print(f"\n{'=' * 60}")
print("Experiment 2: Multiple rectangles")
print("=" * 60)

mask_multi = make_rect_mask(grid, [
    (8, 15, 10, 18),    # bottom-left rectangle
    (36, 43, 10, 18),   # bottom-right rectangle
    (20, 31, 25, 35),   # centre rectangle
    (8, 15, 35, 43),    # top-left rectangle
    (36, 43, 35, 43),   # top-right rectangle
])

res_multi = solve_bvp(c0, method="sor", post_step=fixed_bc,
                       tol=tol, max_iter=max_iter, omega=omega_default,
                       mask=mask_multi)
print(f"With 5 rectangles: {res_multi.n_iter} iterations")

# --- Experiment 3: influence on iteration count ---
print(f"\n{'=' * 60}")
print("Experiment 3: Object size vs iteration count")
print("=" * 60)

sizes = [5, 8, 11, 15, 20, 25]
iter_counts = []
for s in sizes:
    half = s // 2
    c_i, c_j = N // 2, N // 2
    mask_s = make_rect_mask(grid, [(c_i - half, c_i - half + s,
                                    c_j - half, c_j - half + s)])
    res_s = solve_bvp(c0, method="sor", post_step=fixed_bc,
                       tol=tol, max_iter=max_iter, omega=omega_default,
                       mask=mask_s)
    iter_counts.append(res_s.n_iter)
    print(f"  size {s:2d}x{s:2d}: {res_s.n_iter} iterations")

print(f"  no object:  {res_no_obj.n_iter} iterations (baseline)")

# --- Experiment 4: influence on optimal omega ---
print(f"\n{'=' * 60}")
print("Experiment 4: Optimal omega with and without object")
print("=" * 60)

omegas = np.arange(1.50, 1.96, 0.03)

print(f"Sweeping omega from {omegas[0]:.2f} to {omegas[-1]:.2f} ...")

iters_no_obj = []
iters_with_obj = []
for omega in omegas:
    r1 = solve_bvp(c0, method="sor", post_step=fixed_bc,
                    tol=tol, max_iter=max_iter, omega=omega)
    r2 = solve_bvp(c0, method="sor", post_step=fixed_bc,
                    tol=tol, max_iter=max_iter, omega=omega,
                    mask=mask_single)
    iters_no_obj.append(r1.n_iter)
    iters_with_obj.append(r2.n_iter)

best_no = omegas[np.argmin(iters_no_obj)]
best_with = omegas[np.argmin(iters_with_obj)]
print(f"Optimal omega (no object):    {best_no:.2f}  ({min(iters_no_obj)} iter)")
print(f"Optimal omega (with object):  {best_with:.2f}  ({min(iters_with_obj)} iter)")
theory = 2.0 / (1.0 + np.sin(np.pi / N))
print(f"Theory (no object):           {theory:.4f}")

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) Concentration field — single rectangle
ax = axes[0, 0]
y_plot = res_single.y.copy()
y_plot[mask_single == 1] = np.nan  # hide object interior
im = ax.contourf(grid.X, grid.Y, y_plot, levels=30, cmap="viridis")
# Draw object outline
obj_x = grid.X[20, 20], grid.X[30, 20]
obj_y = grid.Y[20, 20], grid.Y[20, 30]
ax.add_patch(plt.Rectangle(
    (obj_x[0], obj_y[0]), obj_x[1] - obj_x[0], obj_y[1] - obj_y[0],
    fill=True, facecolor="white", edgecolor="black", linewidth=1.5))
fig.colorbar(im, ax=ax, label="c")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Single rectangle (SOR, {res_single.n_iter} iter)")
ax.set_aspect("equal")

# (b) Concentration field — multiple rectangles
ax = axes[0, 1]
y_plot2 = res_multi.y.copy()
y_plot2[mask_multi == 1] = np.nan
im2 = ax.contourf(grid.X, grid.Y, y_plot2, levels=30, cmap="viridis")
# Draw object outlines
rects = [(8, 15, 10, 18), (36, 43, 10, 18), (20, 31, 25, 35),
         (8, 15, 35, 43), (36, 43, 35, 43)]
for i0, i1, j0, j1 in rects:
    x0, y0_ = grid.X[i0, j0], grid.Y[i0, j0]
    x1, y1_ = grid.X[i1 - 1, j1 - 1], grid.Y[i1 - 1, j1 - 1]
    ax.add_patch(plt.Rectangle(
        (x0, y0_), x1 - x0, y1_ - y0_,
        fill=True, facecolor="white", edgecolor="black", linewidth=1.0))
fig.colorbar(im2, ax=ax, label="c")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"5 rectangles (SOR, {res_multi.n_iter} iter)")
ax.set_aspect("equal")

# (c) Object size vs iteration count
ax = axes[1, 0]
ax.plot(sizes, iter_counts, "o-", markersize=5, label="With centred square")
ax.axhline(res_no_obj.n_iter, color="gray", linestyle=":", linewidth=1,
           label=f"No object ({res_no_obj.n_iter})")
ax.set_xlabel("Object side length (grid points)")
ax.set_ylabel("Iterations to convergence")
ax.set_title("Object size vs iterations")
ax.legend()
ax.grid(True, alpha=0.3)

# (d) Omega sweep: with vs without object
ax = axes[1, 1]
ax.plot(omegas, iters_no_obj, "o-", markersize=4, label="No object")
ax.plot(omegas, iters_with_obj, "s-", markersize=4, label="With rectangle")
ax.axvline(theory, color="gray", linestyle=":", linewidth=0.8,
           label=rf"Theory $\omega_{{opt}}$ = {theory:.3f}")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel("Iterations to convergence")
ax.set_title(r"Optimal $\omega$: effect of objects")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "a1_6_objects.png", dpi=150)
print(f"\nSaved to {out_dir / 'a1_6_objects.png'}")

plt.show()
