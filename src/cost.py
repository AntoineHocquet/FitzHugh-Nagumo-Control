import numpy as np

def compute_cost(state, control, config):
    u = state["u"]
    alpha = config["alpha"]
    dt = config["dt"]

    # LQ cost: penalize distance from u=1 and control effort
    J = 0.5 * np.sum((u - 1)**2 + alpha * control**2) * dt
    return J

def compute_gradient(state, adjoint, control, config):
    p = adjoint["p"]
    alpha = config["alpha"]
    dt = config["dt"]

    # Gradient of J w.r.t. control
    grad = (p + alpha * control) * dt
    return grad
