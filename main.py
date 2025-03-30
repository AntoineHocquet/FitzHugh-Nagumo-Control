from src.config import config
from src.optimizer import gradient_descent
from src.utils import plot_controls, save_results

def main():
    # Run gradient descent optimization
    controls, costs = gradient_descent(config)

    # Save and plot results
    save_results(controls, costs)
    plot_controls(controls)

if __name__ == "__main__":
    main()
