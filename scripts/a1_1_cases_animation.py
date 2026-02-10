"""
Assignment 1.1 - Animated Plots

Creates GIF animations for all three test cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scicomp3 import Grid1D, solve_ivp, wave1d_rhs
from scicomp3.pde.wave import (
    initial_condition_case_i,
    initial_condition_case_ii,
    initial_condition_case_iii,
)


def animate_wave(grid, amplitudes, time, case, dt, T_sim, c, L, N, output_dir, dpi=100):
    """Create and save an animated GIF showing the temporal evolution."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid.x, amplitudes[0], color="ForestGreen")

    ax.set_xlabel('Position along string (x)')
    ax.set_ylabel('String amplitude (Ψ)')
    ax.set_title('Vibrating string at time 0', fontsize=14)

    ylim = np.max(np.abs(amplitudes)) * 1.5
    ax.set_ylim(-ylim, ylim)

    def animate(frame):
        i, plot = frame
        ax.clear()
        ax.plot(grid.x, plot, color="ForestGreen")
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel('Position along string (x)')
        ax.set_ylabel('String amplitude (Ψ)')
        ax.set_title(f'Vibrating string at time {i*10*dt:.2f}', fontsize=14)
        return []

    def progress_callback(current_frame, total_frames):
        if current_frame % 10 == 0:
            print(f"  Saving frame {current_frame}/{total_frames} ...")

    # Sample every 10th frame for animation
    amplitudes_to_animate = list(enumerate(amplitudes[::10]))

    anim = FuncAnimation(fig, animate, amplitudes_to_animate, interval=10)

    print(f"Saving animation for Case {case}...")
    filename = output_dir / f"vibrating_string_case={case}_dt={dt}_Tsim={T_sim}_c={c}_L={L}_N={N}.gif"

    anim.save(
        filename,
        writer=PillowWriter(fps=20),
        dpi=dpi,
        progress_callback=progress_callback,
    )
    print(f"Saved: {filename}")
    plt.close(fig)


# Parameters
c = 1
L = 1
N = 90
dt = 1e-3
T_sim = 10

# Setup grid
grid = Grid1D(N=N, L=L)

# Output directory
output_dir = Path(__file__).parent.parent / "images" / "gifs"
output_dir.mkdir(parents=True, exist_ok=True)

# Define test cases
test_cases = [
    ("Case i", initial_condition_case_i),
    ("Case ii", initial_condition_case_ii),
    ("Case iii", initial_condition_case_iii),
]

# Run each case and create animation
for i, (name, ic_func) in enumerate(test_cases):
    print(f"\nProcessing {name}...")

    psi0 = ic_func(grid.x)
    v0 = np.zeros(N)
    y0 = np.column_stack([psi0, v0])

    result = solve_ivp(
        wave1d_rhs,
        t_span=(0, T_sim),
        y0=y0,
        method="symplectic_euler",
        dt=dt,
        args=(c, L, N)
    )

    amplitudes = result.y[:, :, 0]
    animate_wave(grid, amplitudes, result.t, i+1, dt, T_sim, c, L, N, output_dir)

print("\nAll animations complete!")
