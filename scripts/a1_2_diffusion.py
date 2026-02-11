"""
Assignment 1.2 - Time Dependent Diffusion Equation

Solves the 2D diffusion equation:
    ∂c/∂t = D∇²c

Boundary conditions:
    c(x, y=1, t) = 1  (top)
    c(x, y=0, t) = 0  (bottom)
    periodic in x direction

Initial condition:
    c(x, y, t=0) = 0 for 0 ≤ x ≤ 1, 0 ≤ y < 1

Tasks E, F: Test correctness and plot at several times.
"""

import numpy as np
import matplotlib.pyplot as plt

from scicomp3.core.grid import Grid2D
from scicomp3.ode.solver import solve_ivp
from scicomp3.pde.diffusion import (
    diffusion2d_rhs, apply_diffusion_bc, diffusion_stable_dt, analytical_solution,
)


# Parameters
N = 50  # grid intervals
D = 1.0  # diffusion coefficient
T_sim = 1.0  # total simulation time

# Target times for plotting
target_times = [0, 0.001, 0.01, 0.1, 1.0]

# Create grid
grid = Grid2D(N=N, L=1.0)

# Setup diffusion IVP
dt = diffusion_stable_dt(D, grid.dx)
c0 = np.zeros((N + 1, N + 1))
apply_diffusion_bc(c0)

def enforce_bc(t, y):
    apply_diffusion_bc(y)
    return y

# Solve diffusion equation
print(f"Solving diffusion equation (N={N}, D={D}, T_sim={T_sim})...")
result = solve_ivp(
    diffusion2d_rhs, t_span=(0, T_sim), y0=c0,
    method="forward_euler", dt=dt, args=(D, grid.dx),
    post_step=enforce_bc, save_interval=10,
)
t, c_history = result.t, result.y
print(f"Completed: {len(t)} saved time points")

# --- Figure 1: 2D concentration at several times (Task F) ---
fig1, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, t_target in enumerate(target_times):
    # Find closest time in saved data
    i_closest = np.argmin(np.abs(t - t_target))
    t_actual = t[i_closest]
    c = c_history[i_closest]

    ax = axes[idx]
    im = ax.pcolormesh(grid.X, grid.Y, c, shading='auto', cmap='hot', vmin=0, vmax=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f't = {t_actual:.4f}')
    ax.set_aspect('equal')
    fig1.colorbar(im, ax=ax, label='c')

# Hide the 6th subplot (we only have 5 times)
axes[5].axis('off')

fig1.suptitle('2D Diffusion: Concentration field at different times', fontsize=14)
plt.tight_layout()
plt.savefig('images/a1_2_diffusion_2d.png', dpi=150)
print("Saved: images/a1_2_diffusion_2d.png")

# --- Figure 2: Compare with analytical solution (Task E) ---
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Plot c(y) profiles at different times
ax1 = axes2[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(target_times[1:])))  # skip t=0

for idx, (t_target, color) in enumerate(zip(target_times[1:], colors)):
    i_closest = np.argmin(np.abs(t - t_target))
    t_actual = t[i_closest]
    c = c_history[i_closest]

    # Take profile at x = 0.5 (mid-grid)
    mid_i = N // 2
    c_numerical = c[mid_i, :]

    # Analytical solution
    c_analytical = analytical_solution(grid.y, t_actual, D=D)

    ax1.plot(c_numerical, grid.y, '-', color=color, linewidth=2,
             label=f't={t_actual:.3f} (num)')
    ax1.plot(c_analytical, grid.y, '--', color=color, linewidth=1.5,
             label=f't={t_actual:.3f} (ana)')

ax1.set_xlabel('Concentration c')
ax1.set_ylabel('y')
ax1.set_title('Concentration profile c(y) at x=0.5')
ax1.legend(loc='upper left', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Plot error vs analytical solution
ax2 = axes2[1]
errors = []
times_for_error = []

for i, ti in enumerate(t):
    if ti > 0:  # skip t=0
        c_numerical = c_history[i][N // 2, :]
        c_analytical = analytical_solution(grid.y, ti, D=D)
        error = np.max(np.abs(c_numerical - c_analytical))
        errors.append(error)
        times_for_error.append(ti)

ax2.loglog(times_for_error, errors, 'b-', linewidth=2)
ax2.set_xlabel('Time t')
ax2.set_ylabel('Max error |c_num - c_ana|')
ax2.set_title('Error vs analytical solution')
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('images/a1_2_diffusion_validation.png', dpi=150)
print("Saved: images/a1_2_diffusion_validation.png")

plt.show()

# Print summary
print("\nSummary:")
print(f"  Grid: {N}x{N} intervals ({N+1}x{N+1} points)")
print(f"  Diffusion coefficient D = {D}")
print(f"  Simulation time: {T_sim}")
print(f"  Time steps saved: {len(t)}")

# Final error at t=1
c_final_numerical = c_history[-1][N // 2, :]
c_final_analytical = analytical_solution(grid.y, t[-1], D=D)
final_error = np.max(np.abs(c_final_numerical - c_final_analytical))
print(f"  Final error at t={t[-1]:.2f}: {final_error:.2e}")
