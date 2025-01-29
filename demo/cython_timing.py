import operator
import time
import tracemalloc
import os

import matplotlib.pyplot as plt
import numpy as np


from dual_autodiff.dual import Dual
from dual_autodiff_x.dual import Dual as Dualx

def main(save_plots=False):
    """
    A demo file to compare the run times of the Normal python package compared to the Cythonized version
    """
    # --- Inputs ---
    input_length = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    
    # --- Operators ---
    operators = [
        (operator.add, "Addition"),
        (operator.mul, "Multiplication"),
        (operator.truediv, "Division"),
        (lambda x: Dual.derivative(lambda z: z**2, x.real), "Derivative")
    ]
    
    # --- Single Operation Comparison ---
    print("\nSingle Operation Comparison:")
    print("-" * 80)
    print(f"{'Operation':<15} | {'Python Time (ms)':<15} | {'Cython Time (ms)':<15} | {'Python Memory (B)':<15} | {'Cython Memory (B)':<15}")
    print("-" * 80)
    
    # Generate random numbers for single operation
    first_real = np.random.randint(1, 1000, 1)[0]
    first_dual = np.random.randint(1, 1000, 1)[0]
    second_real = np.random.randint(1, 1000, 1)[0]
    second_dual = np.random.randint(1, 1000, 1)[0]
    
    # Create dual numbers
    first = Dual(first_real, first_dual)
    second = Dual(second_real, second_dual)
    firstx = Dualx(first_real, first_dual)
    secondx = Dualx(second_real, second_dual)
    
    for op, name in operators:
        # Python timing and memory
        tracemalloc.start()
        start_t = time.perf_counter()
        if name == "Derivative":
            _ = op(first)
        else:
            _ = op(first, second)
        end_t = time.perf_counter()
        python_memory = tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        python_time = (end_t - start_t) * 1000  # Convert to milliseconds
        
        # Cython timing and memory
        tracemalloc.start()
        start_t = time.perf_counter()
        if name == "Derivative":
            _ = op(firstx)
        else:
            _ = op(firstx, secondx)
        end_t = time.perf_counter()
        cython_memory = tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        cython_time = (end_t - start_t) * 1000  # Convert to milliseconds
        
        print(f"{name:<15} | {python_time:<15.6f} | {cython_time:<15.6f} | {python_memory:<15d} | {cython_memory:<15d}")
    
    print("-" * 80)
    print("\n")
    
    
    # --- Results ---
    # Create an empty dictionary to store the results
    results_time = {
        "Addition": {"Python": [], "Cython": []},
        "Multiplication": {"Python": [], "Cython": []},
        "Division": {"Python": [], "Cython": []},
        "Derivative": {"Python": [], "Cython": []}
    }   
    
    results_memory = {
        "Addition": {"Python": [], "Cython": []},
        "Multiplication": {"Python": [], "Cython": []},
        "Division": {"Python": [], "Cython": []},
        "Derivative": {"Python": [], "Cython": []}
    }    
    
    # --- Perform operations and time ---
    
    # Loop through operators
    np.random.seed(81001)
    for op, name in operators:
        # Loop through input lengths
        for length in input_length:
            # Generate random numbers
            first_real = np.random.randint(1, 1000, length)
            first_dual = np.random.randint(1, 1000, length)
            second_real = np.random.randint(1, 1000, length)
            second_dual = np.random.randint(1, 1000, length)
            
            # Create a two list of dual numbers
            # Python
            first = [Dual(first_real[i], first_dual[i]) for i in range(length)]
            second = [Dual(second_real[i], second_dual[i]) for i in range(length)]
            
            # Cython
            firstx = [Dualx(first_real[i], first_dual[i]) for i in range(length)]
            secondx = [Dualx(second_real[i], second_dual[i]) for i in range(length)]
            
            # Perform and time operation with Python
            tracemalloc.start()
            start_t = time.process_time()
            if name == "Derivative":
                _ = [op(first[i]) for i in range(length)]
            else:
                _ = [op(first[i], second[i]) for i in range(length)]
            end_t = time.process_time()
            python_memory = tracemalloc.get_tracemalloc_memory()
            tracemalloc.stop()
            results_time[name]["Python"].append(end_t - start_t)
            results_memory[name]["Python"].append(python_memory)
            
            
            # Perform and time operation with Cython
            tracemalloc.start()
            start_t = time.process_time()
            if name == "Derivative":
                _ = [op(firstx[i]) for i in range(length)]
            else:
                _ = [op(firstx[i], secondx[i]) for i in range(length)]
            end_t = time.process_time()
            cython_memory = tracemalloc.get_tracemalloc_memory()
            tracemalloc.stop()
            results_time[name]["Cython"].append(end_t - start_t)
            results_memory[name]["Cython"].append(cython_memory)
            
    
    # --- Plotting the Results ---
    _, axes = plt.subplots(2, 2)
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(input_length, results_time[operators[i][1]]["Python"], "b", label="Python")
        ax.plot(input_length, results_time[operators[i][1]]["Cython"], "r", label="Cython")
        ax.set_ylabel("Process Time [s]", fontsize=14)
        ax.set_xlabel("Number of Operations", fontsize=14)
        ax.set_title(f"{operators[i][1]}", fontsize=18)
        ax.grid(True)
        ax.legend()
        ax.set_xscale("log")
        
                
    plt.suptitle("Comparison of Process Time between Python and Cython for Dual Operations", fontsize=18)
    plt.tight_layout()
    plt.show()
    
    
    
    _, axes = plt.subplots(2, 2)
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(input_length, results_memory[operators[i][1]]["Python"], "b", label="Python")
        ax.plot(input_length, results_memory[operators[i][1]]["Cython"], "r", label="Cython")
        ax.set_ylabel("Memory Usage [bytes]", fontsize=14)
        ax.set_xlabel("Number of Operations", fontsize=14)
        ax.set_title(f"{operators[i][1]}", fontsize=18)
        ax.grid(True)
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        
                
    plt.suptitle("Comparison of Memory Usage between Python and Cython for Dual Operations", fontsize=18)
    plt.tight_layout()
    plt.show()
    if save_plots: 
        fp = os.path.join("demo", "performance_plots", "cython_time.png")
        plt.savefig(fp)
        fp = os.path.join("demo", "performance_plots", "cython_memory.png")
        plt.savefig(fp)
            
            
            
if __name__ == "__main__":
    main()     

