from assignment01 import *

c=1; L=1; N=90
dx = L/N
dt = 1e-3
T_sim = 10

# Use N+1 points to include both boundaries at x=0 and x=L
x = np.linspace(0, L, N+1)  # [0, dx, 2dx, ..., L]
psi0 = np.sin(5*np.pi*x) # initial amplitudes Ψ
psi0[0] = 0   # enforce boundary condition at x=0
psi0[-1] = 0  # enforce boundary condition at x=L
psi_t0 = np.zeros(N+1) # initial velocities Ψ_t are set to 0
state0 = np.transpose([psi0, psi_t0]) # initial state vector [Ψ, Ψ_t]

time, states = integrate_euler(wave_eq_deriv, state0=state0, dt=dt, T_sim=T_sim, c=c, L=L, N=N+1)
amplitudes = states[:, :, 0]  # Extract the amplitudes Ψ over time
fig, ax = plt.subplots(figsize=(6, 4))
for i in range(0, len(time), len(time)//11):
    ax.plot(x, amplitudes[i], color=plt.cm.cividis(i/len(time)))
ax.set_xlabel('Position along string (x)')
ax.set_ylabel('String amplitude (Ψ)')
ax.set_title('Vibrating string over time')
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cividis'), ax=ax, label='Time', ticks=np.linspace(0, 1, 3))
cbar.ax.set_yticklabels([f"{t:.1f}" for t in np.linspace(0, T_sim, 3)])  # Set colorbar ticks to actual time values
plt.show()
