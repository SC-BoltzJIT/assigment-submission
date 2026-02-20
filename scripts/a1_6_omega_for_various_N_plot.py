"""
Comparison between various omega values to find the optimal omega value
that minimises the number of iterations needed

This file generates the plot with data from a1_6_omega_for_various_N_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load

from scicomp3.bvp.omega import get_optimal_omega

# Load data
data_dir = Path(__file__).parent.parent / "data"
filename = "n_vs_omega.pkl"
results = load(data_dir / filename)

N_values = results["N_values"]
omega_values = results["omega_values"]
n_iterations = results["n_iterations"]
theoretical_omega_values = np.array([get_optimal_omega(N) for N in N_values])

# Create the plot
COLOUR_OMEGA = "tab:green"
COLOUR_ITER  = "tab:blue"
COLOUR_THEORY = "tab:orange"

fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

# Plot 2 minus omega to get a nice log plot
distance_to_two = 2 - np.array(omega_values)
ax.semilogy(N_values, distance_to_two, marker="o", color=COLOUR_OMEGA,
            label = "Optimal $\\omega$ by ternary search")
ax.semilogy(N_values, 2 - theoretical_omega_values, marker="x", 
            color=COLOUR_THEORY, label = "Theoretical optimal $\\omega$")

# Replace tick labels with corresponding omega values
ax.set_ylim(1e-3, 1)
yticks = np.logspace(-2, 0, 3)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{2 - t:.3f}" for t in yticks])
ax.invert_yaxis()

ax.set_xlabel("$N$ (grid size)")
ax.set_ylabel("Optimal omega ($\\omega$)", color=COLOUR_OMEGA)
ax.tick_params(axis="y", colors=COLOUR_OMEGA)
ax.grid(True)
ax.legend()

ax2 = plt.twinx(ax)
ax2.plot(N_values, n_iterations, marker="o", color=COLOUR_ITER)
ax2.set_ylabel("Iterations at optimal omega ($\\omega$)", color=COLOUR_ITER)
ax2.tick_params(axis="y", colors=COLOUR_ITER)

ax2.spines["left"].set_color(COLOUR_OMEGA)
ax2.spines["right"].set_color(COLOUR_ITER)
ax.set_title("Optimal omega ($\\omega$) and iterations needed for various $N$")

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
filename = "a1_6_N_vs_omega.png"
plt.savefig(out_dir / filename, dpi=150)
print(f"Saved to {out_dir / filename}")

plt.show()
