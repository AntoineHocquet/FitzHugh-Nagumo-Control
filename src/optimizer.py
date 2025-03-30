import numpy as np
from src.state import solve_state_equation
from src.adjoint import solve_adjoint_equation
from src.cost import compute_cost, compute_gradient


def gradient_descent(config):
    N = config["N"]
    alpha = config["alpha"]
    max_iter = config["max_iter"]
    lr = config["lr"]

    # Initialize control (zero control)
    control = np.zeros(N+1)
    cost_evolution = []

    for iteration in range(max_iter):
        # Forward solve
        state = solve_state_equation(control, config)

        # Adjoint solve
        adjoint = solve_adjoint_equation(state, config)

        # Compute cost and gradient
        cost = compute_cost(state, control, config)
        grad = compute_gradient(state, adjoint, control, config)

        # Gradient descent update
        control -= lr * grad

        cost_evolution.append(cost)
        print(f"Iteration {iteration+1}, Cost = {cost:.6f}")

    return control, cost_evolution
