import time

import numpy as np
import matplotlib.pyplot as plt

from dual_autodiff.dual import Dual


def main(plot: bool = True, verbose: bool = False):
    """
    Demonstration file for the differentiation of the function:

    f(x) = log(sin(x)) + x^2 cos(x)

    Compute the derivative at x = 1.5

    Analytical Solution: df/dx = cot(x) + 2 * x * cos(x) - x**2 * sin(x), df(1.5)/dx = -1.9123
    """

    # Define the function
    def func(x):
        return np.log(np.sin(x)) + x**2 * np.cos(x)

    # Analytical Solution
    analytical_sol = (1 / np.tan(1.5)) + 2 * 1.5 * np.cos(1.5) - 1.5**2 * np.sin(1.5)

    # Calculate using Dual Numbers
    dual = Dual(1.5, 1)
    dual_sol = dual.derivative(func=func, x=1.5)

    # Calculate numerical solution using central difference
    def numerical_derivative(func, x, h=1e-5):
        return (func(x + h) - func(x - h)) / (2 * h)

    dx = np.linspace(1e-6, 1, 100)
    numerical_sol = []
    for d in dx:
        numerical_sol.append(numerical_derivative(func, 1.5, d))

    # Print results
    if verbose:
        print("Solutions:")
        print(f"Analytical Solution = {analytical_sol}")
        print(f"Dual Solution = {dual_sol}")
        for i, d in enumerate(dx):
            print(f"Numerical Solution: dx: {d}, value: {numerical_sol[i]}")

    # Plot Results
    if plot:
        _, ax = plt.subplots()
        ax.plot(dx, numerical_sol, color="g", label="Numerical")
        ax.axhline(analytical_sol, color="r", label="Analytical")
        ax.axhline(dual_sol, color="b", linestyle="--", label="Dual")
        ax.set_title("Comparison of Differentiation Methods", fontsize=18)
        ax.set_ylabel("Value", fontsize=14)
        ax.set_xlabel("Numerical Differentiation Step Size", fontsize=14)
        ax.set_ylim([-2.55, -1.9])
        ax.set_xlim([0, 1])

        plt.gca().invert_xaxis()
        plt.legend()
        plt.grid()
        plt.show()
        
    # Perform a speedup timing:
    # Perform 1000 numerical differentiations and time (for stepsize = 1E-6)
    start_t = time.time()
    for _ in range(1000):
        dx = numerical_derivative(func, 1.5, 1E-6)
    end_t = time.time()
    num_time = end_t - start_t
    
    # Perform AD
    start_t = time.time()
    for _ in range(1000):
        dual = Dual(1.5, 1)
        dual_sol = dual.derivative(func=func, x=1.5)
    end_t = time.time()
    ad_time = end_t - start_t
    
    print(f"\nAverage Time Numerical Differentiations: {num_time} ms")
    print(f"\nAverage Time for Automatic Differentiations: {ad_time} ms")


if __name__ == "__main__":
    main()

