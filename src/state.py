import numpy as np

def solve_state_equation(control, config):
    N = config["N"]
    dt = config["dt"]
    epsilon = config["epsilon"]
    beta = config["beta"]
    gamma = config["gamma"]
    a = config["a"]
    b = config["b"]
    c = config["c"]

    u = np.zeros(N+1)
    v = np.zeros(N+1)

    # Initial conditions
    u[0] = 0.0
    v[0] = 0.0

    for n in range(N):
        u[n+1] = u[n] + dt * (u[n] - (u[n]**3)/3 - v[n] + control[n]) / epsilon
        v[n+1] = v[n] + dt * (u[n] + a - b * v[n]) / c

    return {"u": u, "v": v}
