"""
Assignment 1.2 - Task G: Animated plot of time dependent diffusion

Creates an animation of the concentration field evolving until equilibrium.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from scicomp3.core.grid import Grid2D
from scicomp3.ode.solver import solve_ivp
from scicomp3.pde.diffusion import (
    diffusion2d_rhs, apply_diffusion_bc, diffusion_stable_dt,
)


# Parameters
N = 50  # grid intervals
D = 1.0  # diffusion coefficient
T_sim = 0.5  # simulation time (equilibrium is approached by t~0.5)

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
    post_step=enforce_bc, save_interval=50,
)
t, c_history = result.t, result.y
print(f"Completed: {len(t)} frames for animation")

# Create animation
fig, ax = plt.subplots(figsize=(6, 5))

# Initial plot
im = ax.pcolormesh(grid.X, grid.Y, c_history[0], shading='auto', cmap='hot', vmin=0, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
title = ax.set_title(f't = {t[0]:.4f}')
ax.set_aspect('equal')
fig.colorbar(im, ax=ax, label='Concentration c')


def update(frame):
    """Update function for animation."""
    im.set_array(c_history[frame].ravel())
    title.set_text(f't = {t[frame]:.4f}')
    return [im, title]


# Create animation
anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

# Save animation
output_dir = Path('images/gifs')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'a1_2_diffusion.gif'

print(f"Saving animation to {output_path}...")
anim.save(output_path, writer='pillow', fps=20)
print(f"Animation saved: {output_path}")

plt.show()
