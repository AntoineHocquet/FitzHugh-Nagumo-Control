import numpy as np

# Simulation parameters
T = 20.0              # Final time
dt = 0.01             # Time step
N = int(T / dt)       # Number of time steps
time = np.linspace(0, T, N+1)

# Model parameters
epsilon = 0.05
beta = 1.0
gamma = 1.0
a = 0.1
b = 0.5
c = 1.0

# Control parameters
alpha = 0.01

# Optimization parameters
max_iter = 100
lr = 0.1

# Misc
noise_strength = 0.1

config = {
    "T": T,
    "dt": dt,
    "N": N,
    "time": time,
    "epsilon": epsilon,
    "beta": beta,
    "gamma": gamma,
    "a": a,
    "b": b,
    "c": c,
    "alpha": alpha,
    "max_iter": max_iter,
    "lr": lr,
    "noise_strength": noise_strength
}
