# scicomp3 — Scientific Computing Assignment Package

Numerical solvers for the 1D wave equation and 2D diffusion equation, built for the Scientific Computing course (Assignment Set 1).

## Quick Start

```bash
# Clone and enter the project
cd assigment-submission

# Create/activate venv and install (editable mode with dev deps)
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run a script
python scripts/a1_1_smoke_test.py
```

## Project Structure

```
.
├── src/scicomp3/              # Main package
│   ├── core/
│   │   ├── grid.py            # Grid1D, Grid2D — spatial discretization
│   │   └── result.py          # ODEResult — solver output container
│   ├── ode/
│   │   ├── methods.py         # Time-stepping methods (Euler, symplectic Euler)
│   │   └── solver.py          # solve_ivp() — main solver entry point
│   ├── pde/
│   │   ├── wave.py            # wave1d_rhs, initial conditions (cases i–iii)
│   │   └── diffusion.py       # diffusion2d_rhs, BCs, stable dt, analytical solution
│   └── validation/
│       └── validation.py      # Boundary condition validation utilities
│
├── tests/                     # Pytest test suite
│   ├── test_boundary_conditions.py  # Wave BC enforcement (3 cases)
│   ├── test_diffusion.py           # Diffusion solver + analytical comparison
│   └── test_solver_comparison.py   # Legacy vs scicomp3 solver parity
│
├── scripts/                   # Runnable plotting/animation scripts
│   ├── a1_1_smoke_test.py          # Quick wave equation smoke test
│   ├── a1_1_cases_plot.py          # Wave plots for all 3 initial conditions
│   ├── a1_1_cases_animation.py     # Wave animated GIFs
│   ├── a1_2_diffusion.py           # 2D diffusion: fields + analytical validation
│   └── a1_2_diffusion_animation.py # Diffusion animated GIF
│
├── assignment01.py            # Legacy wave solver (kept for comparison tests)
├── run_assignment01_*.py      # Legacy plotting scripts using assignment01.py
├── pyproject.toml             # Build config (hatchling), deps, pytest settings
└── images/                    # Generated figures and GIFs (.gitignored)
```

## How It Works

### Solving a PDE

The package separates concerns into three layers:

1. **PDE right-hand side** (`pde/`) — defines the physics (spatial derivatives)
2. **ODE solver** (`ode/`) — advances the solution in time
3. **Post-step callback** — enforces boundary conditions after each step

Example: solving the wave equation end-to-end:

```python
import numpy as np
from scicomp3 import Grid1D, solve_ivp, wave1d_rhs
from scicomp3.pde.wave import initial_condition_case_ii

# 1. Grid and parameters
grid = Grid1D(N=90, L=1.0)
c, dt, T_sim = 1.0, 1e-3, 2.0

# 2. Initial condition: Ψ(x,0) = sin(5πx), Ψ_t(x,0) = 0
psi0 = initial_condition_case_ii(grid.x)
psi0[0] = psi0[-1] = 0          # enforce BCs on IC
v0 = np.zeros(grid.N)
y0 = np.column_stack([psi0, v0]) # state = [Ψ, v], shape (N, 2)

# 3. Boundary condition callback
def fixed_ends(t, y):
    y[0, 0] = y[-1, 0] = 0
    return y

# 4. Solve
result = solve_ivp(
    wave1d_rhs,
    t_span=(0, T_sim),
    y0=y0,
    method="symplectic_euler",    # energy-preserving for wave eq
    dt=dt,
    args=(c, grid.L, grid.N),
    post_step=fixed_ends,
)

# 5. Extract results
amplitudes = result.y[:, :, 0]   # Ψ at each saved time step
times = result.t
```

Example: 2D diffusion equation:

```python
import numpy as np
from scicomp3.core.grid import Grid2D
from scicomp3.ode.solver import solve_ivp
from scicomp3.pde.diffusion import (
    diffusion2d_rhs, apply_diffusion_bc, diffusion_stable_dt, analytical_solution,
)

grid = Grid2D(N=50, L=1.0)
D = 1.0
dt = diffusion_stable_dt(D, grid.dx)  # auto-compute safe dt

c0 = np.zeros((grid.N + 1, grid.N + 1))
apply_diffusion_bc(c0)                # set top=1, bottom=0

result = solve_ivp(
    diffusion2d_rhs,
    t_span=(0, 1.0),
    y0=c0,
    method="forward_euler",
    dt=dt,
    args=(D, grid.dx),
    post_step=lambda t, y: (apply_diffusion_bc(y), y)[1],
    save_interval=100,
)
```

### Available Time-Stepping Methods

| Name | Aliases | Order | Properties |
|------|---------|-------|------------|
| `"symplectic_euler"` | `"euler_cromer"`, `"semi_implicit_euler"` | 1st | Symplectic (energy-preserving). Default. Use for wave equation. |
| `"forward_euler"` | `"euler"` | 1st | Explicit. Use for diffusion equation. |

Methods are registered in `scicomp3.ode.methods.METHODS` and looked up by name in `solve_ivp()`.

### Key Design Decisions

**Post-step callback for boundary conditions.** The PDE RHS functions use `np.roll` for the spatial stencil, which wraps boundary values incorrectly. The `post_step` callback in `solve_ivp` corrects this after every time step. This separates the physics from the constraints cleanly.

**Symplectic Euler for wave equation.** Forward Euler is dissipative — it loses energy over time. The symplectic Euler method preserves the phase-space structure of Hamiltonian systems, keeping energy bounded over long simulations. This is the default method.

**Explicit scheme for diffusion.** The FTCS (Forward Time, Centered Space) scheme is simple but conditionally stable. Use `diffusion_stable_dt()` to compute a safe time step (90% of the theoretical maximum $\delta x^2 / 4D$).

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_diffusion.py -v

# With print output
pytest tests/ -v -s
```

The test suite covers:
- **test_boundary_conditions.py** — Wave equation BCs hold for all 3 initial conditions; documents np.roll boundary pollution without post_step.
- **test_diffusion.py** — Diffusion BCs (top=1, bottom=0), convergence to steady state $c=y$, match against analytical erfc series solution, stability check.
- **test_solver_comparison.py** — Legacy `assignment01.py` and `scicomp3` solvers produce identical interior-point results.

## Running Scripts

Scripts live in `scripts/` and produce plots or animations. They require `matplotlib`:

```bash
# Quick smoke test (interactive plot)
python scripts/a1_1_smoke_test.py

# Generate static plots for all wave cases → images/figures/
python scripts/a1_1_cases_plot.py

# Generate animated GIFs for all wave cases → images/gifs/
python scripts/a1_1_cases_animation.py

# 2D diffusion: concentration fields + analytical validation → images/
python scripts/a1_2_diffusion.py

# 2D diffusion animation → images/gifs/
python scripts/a1_2_diffusion_animation.py
```

## Adding a New Time-Stepping Method

1. Define a step function in `src/scicomp3/ode/methods.py`:

```python
def my_step(fun, t, y, dt, args=()):
    """One step of my method. Returns (y_new, n_function_evals)."""
    dydt = fun(t, y, *args)
    y_new = ...  # your update rule
    return y_new, 1
```

2. Register it in `METHODS`:

```python
METHODS["my_method"] = my_step
```

3. Use it:

```python
result = solve_ivp(..., method="my_method")
```

## Adding a New PDE

1. Create a RHS function in `src/scicomp3/pde/`:

```python
def my_pde_rhs(t, y, *params):
    """Compute dy/dt for my PDE. Returns array same shape as y."""
    ...
```

2. Write a post_step if boundary conditions need enforcement.
3. Call `solve_ivp(my_pde_rhs, ...)` with appropriate method and parameters.

## Legacy Code

`assignment01.py` at the project root is the original wave equation implementation. It uses a different API (`integrate_euler` with `**kwargs`) and has known boundary pollution from `np.roll` without post-step correction. It is kept for backward compatibility and tested against `scicomp3` in `test_solver_comparison.py`.

## Dependencies

- **Required**: `numpy`, `scipy`
- **Plotting**: `matplotlib` (optional, needed for scripts)
- **Testing**: `pytest` (optional)

Python >= 3.10.
