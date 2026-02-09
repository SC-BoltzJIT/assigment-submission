import numpy as np
import matplotlib.pyplot as plt

def wave_eq_deriv(state, t, c=1, L=1, N=100):
    """
    Compute the derivatives for the 1D wave equation, to use for a forward Euler integration scheme.
    Takes as arguments: 
     - current state vector [Ψ, Ψ_t];
     - current time t; 
     - wave speed c; 
     - string length L; 
     - number of spatial points N.
    Returns the derivatives [dΨ/dt, d^2 Ψ/dt^2] as a numpy array.
    """
    # psi, psi_t = state # unpack the state vector [Ψ, Ψ_t]
    psi = state[:, 0]
    psi_t = state[:, 1]
    psi[0] = 0 # boundary condition at x=0
    psi[-1] = 0 # boundary condition at x=L

    dx = L / N # spatial step size
    dpsi_dt = psi_t # first deriv in time to update Ψ
    d2psi_dx2 = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2 # second deriv in space
    dpsi_t_dt = c**2 * d2psi_dx2 # second deriv in time to update first deriv
    
    # Check that the boundary conditions remain satisfied
    assert psi[0] == 0, f"Expected 0 at the boundary, got {d2psi_dx2[0]}"
    assert psi[-1] == 0, f"Expected 0 at the boundary, got {d2psi_dx2[-1]}"
    
    return np.array([dpsi_dt, dpsi_t_dt])

def integrate_euler(deriv_func, state0, dt=1e-3, T_sim=10, **kwargs):
    """
    Integrate a system of ODEs using the forward Euler method.
    Takes as arguments:
     - deriv_func: function that computes the derivatives in time;
     - state0: initial state vector at time t=0;
     - dt: time step size;
     - T_sim: total simulation time;
     - kwargs: additional arguments to pass to deriv_func.
    Returns an array of states at each time step.
    """
    # Create the time array based on simulation time and time step size
    time = np.arange(0, T_sim, dt)
    # Create an array to hold the states at each time step
    states = np.zeros((len(time),) + state0.shape)
    # Fill it with the initial state
    states[0] = state0 
    print(np.shape(states), np.shape(state0))

    # Perform the forward Euler integration
    for i in range(1, len(time)):
        derivs = deriv_func(states[i-1], time[i-1], **kwargs)
        # Update according to: new_state = old_state + dt * derivs
        state = states[i-1] + dt * np.transpose(derivs)
        # Save the updated state
        states[i] = state
    
    return time, states

c=1; L=1; N=100
dx = L/N
dt = 1e-4
T_sim = 2

psi0 = np.sin(5*np.pi*np.arange(N)*dx) # initial amplitudes Ψ
psi_t0 = np.zeros(N) # initial velocities Ψ_t are set to 0
state0 = np.transpose([psi0, psi_t0]) # initial state vector [Ψ, Ψ_t]

time, states = integrate_euler(wave_eq_deriv, state0=state0, dt=dt, T_sim=T_sim, c=c, L=L, N=N)


psis = states[:, :, 0]  # Extract the amplitudes Ψ over time
fig, ax = plt.subplots(figsize=(6, 4))
for i in range(0, len(time), len(time)//100):
    ax.plot(np.arange(N)*dx, psis[i], color=plt.cm.viridis(i/len(time)))
ax.set_xlabel('Position along string (x)')
ax.set_ylabel('String Amplitude (Ψ)')
ax.set_title('Vibrating String over Time')
# label=f't={time[i]:.2f}s', 
fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Time (s)')
plt.show()
