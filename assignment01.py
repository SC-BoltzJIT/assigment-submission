import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def wave_eq_deriv(state, t, dt=1e-3, c=1, L=1, N=100):
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
    d2psi_dx2 = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2 # second deriv in space
    dpsi_t_dt = c**2 * d2psi_dx2 # second deriv in time to update first deriv

    dpsi_dt = psi_t + dt * dpsi_t_dt # first deriv in time to update Ψ
    
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
        derivs = deriv_func(states[i-1], time[i-1], dt, **kwargs)
        # Update according to: new_state = old_state + dt * derivs
        state = states[i-1] + dt * np.transpose(derivs)
        # Save the updated state
        states[i] = state
    
    return time, states


def animate_wave(amplitudes: list, dpi=100, case="i", dt=1e-3, T_sim=10,**kwargs):
    """Create and save an animated GIF showing the temporal evolution of CA grids."""

    c,L,N = kwargs['c'], kwargs['L'], kwargs['N']
    dx = L/N

    # Set up the figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(N)*dx, amplitudes[0], color="ForestGreen")

    ax.set_xlabel('Position along string (x)')
    ax.set_ylabel('String amplitude (Ψ)')
    ax.set_title('Vibrating string over at time 0', fontsize=14)
    ylim = np.max(np.abs(amplitudes)) * 1.5
    ax.set_ylim(-ylim, ylim)

    # Define animation function
    def animate(frame):
        # i, grid = frame
        i, plot = frame
        ax.clear()
        ax.plot(np.arange(N)*dx, plot, color="ForestGreen")
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel('Position along string (x)')
        ax.set_ylabel('String amplitude (Ψ)')
        ax.set_title(f'Vibrating string at time {i*10*dt:.2f}', fontsize=14)
        return []

    # Set up helper function to display progress
    def progress_callback(current_frame, total_frames):
        """Shows saving progress"""
        if current_frame % 10 == 0:
            print(f"Saving frame {current_frame}/{total_frames} ...")

    amplitudes_to_animate = list(enumerate(amplitudes[::10]))  # Sample every 10th frame for animation

    # Create animation
    anim = FuncAnimation(
        fig, animate, amplitudes_to_animate, interval=10
    )
    print("Saving animation ... (This can take a while depending on the dpi.)")

    filename = f"../images/gifs/vibrating_string_case={case}_dt={dt}_Tsim={T_sim}_c={c}_L={L}_N={N}.gif"
    anim.save(
        filename,
        writer=PillowWriter(fps=20),
        dpi=dpi,
        progress_callback=progress_callback,
    )
    print(f"Saved successfully as '{filename}'")