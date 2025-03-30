import numpy as np

def solve_adjoint_equation(state, config):
    u = state["u"]
    v = state["v"]
    N = config["N"]
    dt = config["dt"]
    epsilon = config["epsilon"]
    beta = config["beta"]
    gamma = config["gamma"]
    a = config["a"]
    b = config["b"]
    c = config["c"]

    p = np.zeros(N+1)
    q = np.zeros(N+1)

    # Terminal conditions
    p[N] = 0.0
    q[N] = 0.0

    for n in reversed(range(N)):
        p[n] = p[n+1] + dt * ( (1 - u[n]**2) * p[n+1] / epsilon + q[n+1] / c - (u[n] - 1)**2 / 2 )
        q[n] = q[n+1] + dt * ( p[n+1] / epsilon * (-1) + q[n+1] * b / c )

    return {"p": p, "q": q}
