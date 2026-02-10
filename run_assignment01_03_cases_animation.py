from run_assignment01_02_cases_plot import *

for i, amplitudes in enumerate(amplitude_list):
    animate_wave(amplitudes, case=i+1, dt=dt, T_sim=T_sim, c=c, L=L, N=N)
