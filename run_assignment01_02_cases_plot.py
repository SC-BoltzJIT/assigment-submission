from assignment01 import *

c=1; L=1; N=90
dx = L/N
dt = 1e-3
T_sim = 10

psi0_i = np.sin(2*np.pi*np.arange(N)*dx) # initial amplitudes of case i
psi0_ii = np.sin(5*np.pi*np.arange(N)*dx) # initial amplitudes of case ii
psi0_iii = np.sin(5*np.pi*np.arange(N)*dx) 
psi0_iii = np.where((np.arange(N)*dx < 1/5) | (np.arange(N)*dx > 2/5), 0, psi0_iii) # initial amplitudes of case iii
initial_conditions = [psi0_i, psi0_ii, psi0_iii]

psi_t0 = np.zeros(N) # initial velocities Ψ_t are set to 0

amplitude_list = []

for i, psi0 in enumerate(initial_conditions):
    state0 = np.transpose([psi0, psi_t0]) # initial state vector [Ψ, Ψ_t]
    time, states = integrate_euler(wave_eq_deriv, state0=state0, dt=dt, T_sim=T_sim, c=c, L=L, N=N)
    amplitude_list.append(states[:, :, 0])  # Extract the amplitudes Ψ over time for each case

for i, amplitudes in enumerate(amplitude_list):
    fig, ax = plt.subplots(figsize=(6, 4))
    for j in range(0, len(time), len(time)//51):
        ax.plot(np.arange(N)*dx, amplitudes[j], color=plt.cm.cividis(j/len(time)))
    ax.set_xlabel('Position along string (x)')
    ax.set_ylabel('String amplitude (Ψ)')
    ax.set_title(f'Vibrating string over time - Case {i+1}')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cividis'), ax=ax, label='Time', ticks=np.linspace(0, 1, 3))
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in np.linspace(0, T_sim, 3)])  # Set colorbar ticks to actual time values
    plt.savefig(f"../images/figures/vibrating_string_over_time_case={i}_dt={dt}_Tsim={T_sim}_c={c}_L={L}_N={N}.png", dpi=300)
    plt.show()

