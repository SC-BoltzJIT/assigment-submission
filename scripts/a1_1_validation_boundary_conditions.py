import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scicomp3.validation.validation import validate_boundary_conditions

import numpy as np

from scicomp3 import Grid1D, solve_ivp, wave1d_rhs
from scicomp3.pde.wave import (
    initial_condition_case_i,
    initial_condition_case_ii,
    initial_condition_case_iii,
)


def fixed_ends(t, y):
    """Enforce fixed boundary conditions: psi=0 at both ends."""
    y[0, 0] = 0
    y[-1, 0] = 0
    return y

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
        args=(c, L, N),
        post_step=fixed_ends,
    )

    validate_boundary_conditions(result, verbose=True)

    # amplitudes = result.y[:, :, 0]
    # animate_wave(grid, amplitudes, result.t, i+1, dt, T_sim, c, L, N, output_dir)

print("\nAll animations complete!")
