import matplotlib.pyplot as plt
import numpy as np
import os

def plot_controls(control):
    plt.figure()
    plt.plot(control)
    plt.title("Optimal Control")
    plt.xlabel("Time step")
    plt.ylabel("Control")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/optimal_control.png")
    plt.close()

def save_results(control, costs):
    os.makedirs("results", exist_ok=True)
    np.save("results/control.npy", control)
    np.save("results/costs.npy", costs)

    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(costs)
    plt.title("Cost Function over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/cost_evolution.png")
    plt.close()
