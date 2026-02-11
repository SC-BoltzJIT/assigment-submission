"""Compare old (assignment01) and new (scicomp3) wave equation solvers.

Verifies that both solvers produce identical interior-point results and
that the new solver maintains zero boundary conditions via post_step.

Exit code 0 = pass, 1 = fail (suitable for CI).
"""

import numpy as np
import sys
from pathlib import Path

# Add assigment-submission root to path so we can import both packages
sys.path.insert(0, str(Path(__file__).parent.parent))

from assignment01 import wave_eq_deriv, integrate_euler
from scicomp3 import solve_ivp, wave1d_rhs

# Shared parameters
c = 1
L = 1
N = 90
dt = 1e-3
T_sim = 0.1  # short run for comparison

x = np.linspace(0, L, N + 1)
psi0 = np.sin(5 * np.pi * x)
psi0[0] = 0
psi0[-1] = 0

# --- Old solver (assignment01) ---
psi_t0 = np.zeros(N + 1)
state0 = np.transpose([psi0, psi_t0])
time_old, states_old = integrate_euler(
    wave_eq_deriv, state0=state0, dt=dt, T_sim=T_sim, c=c, L=L, N=N + 1
)
amps_old = states_old[:, :, 0]

# --- New solver (scicomp3) with post_step boundary enforcement ---
def fixed_ends(t, y):
    """Enforce fixed boundary conditions: psi=0 at both ends."""
    y[0, 0] = 0
    y[-1, 0] = 0
    return y

v0 = np.zeros(N + 1)
y0 = np.column_stack([psi0.copy(), v0])
result = solve_ivp(
    wave1d_rhs,
    t_span=(0, T_sim),
    y0=y0,
    method="symplectic_euler",
    dt=dt,
    args=(c, L, N + 1),
    post_step=fixed_ends,
)
amps_new = result.y[:, :, 0]

# --- Assertions ---
failed = False

# Interior points (indices 1:-1) should match exactly
interior_diff = np.max(np.abs(amps_old[:, 1:-1] - amps_new[:, 1:-1]))
if interior_diff != 0:
    print(f"FAIL: interior points differ, max diff = {interior_diff}")
    failed = True
else:
    print("PASS: interior points match exactly")

# New solver boundary values should be exactly 0
new_boundary_max = max(
    np.max(np.abs(amps_new[:, 0])),
    np.max(np.abs(amps_new[:, -1])),
)
if new_boundary_max != 0:
    print(f"FAIL: new solver boundary values nonzero, max = {new_boundary_max}")
    failed = True
else:
    print("PASS: new solver boundary values are exactly 0")

if failed:
    sys.exit(1)
else:
    print("All checks passed.")
