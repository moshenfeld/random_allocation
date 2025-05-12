import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary components
from random_allocation.comparisons.structs import PrivacyParams, SchemeConfig, Direction
from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType, BoundType
from random_allocation.random_allocation_scheme.decomposition import (
    allocation_epsilon_decomposition_add,
    allocation_delta_decomposition_add_from_PLD,
    allocation_epsilon_decomposition_remove,
    Poisson_PLD
)

def test_search_callbacks():
    """Test the search function with callbacks to analyze its behavior."""
    print("Testing search function optimization performance...")
    
    # Parameters
    sigma = 1.0
    delta = 1e-10
    num_steps = 10000
    num_selected = 1
    num_epochs = 1
    discretization = 1e-3
    epsilon_tolerance = 1e-6
    
    # Create params
    params = PrivacyParams(
        sigma=sigma,
        delta=delta,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        epsilon=None
    )
    
    # Create config
    config = SchemeConfig(
        discretization=discretization,
        epsilon_tolerance=epsilon_tolerance
    )
    
    # Setup variables for tracking function evaluations
    num_steps_per_round = int(np.ceil(params.num_steps/params.num_selected))
    num_rounds = int(np.ceil(params.num_steps/num_steps_per_round))
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    
    # Create the PLD object (this is expensive and done once in the original code)
    Poisson_PLD_obj = Poisson_PLD(
        sigma=params.sigma, 
        num_steps=num_steps_per_round, 
        num_epochs=num_rounds*params.num_epochs, 
        sampling_prob=1.0/num_steps_per_round, 
        discretization=config.discretization, 
        direction='add'
    )
    
    # First optimization function - the one that takes long
    evals_first = []
    x_vals_first = []
    y_vals_first = []
    
    def optimization_func_first(eps):
        result = float(Poisson_PLD_obj.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps)))))
        evals_first.append(len(evals_first) + 1)
        x_vals_first.append(eps)
        y_vals_first.append(result)
        return result
    
    print("Running first search...")
    start_time = time.time()
    epsilon_first = search_function_with_bounds(
        func=optimization_func_first, 
        y_target=params.delta, 
        bounds=(0, 100),  # Adjusted for better visibility
        tolerance=config.epsilon_tolerance, 
        function_type=FunctionType.DECREASING
    )
    time_first = time.time() - start_time
    
    print(f"First search completed in {time_first:.4f} seconds with {len(evals_first)} evaluations")
    print(f"Result: {epsilon_first}")
    
    # Second optimization function - the fine-tuning one
    evals_second = []
    x_vals_second = []
    y_vals_second = []
    
    # Skip this if first search failed
    if epsilon_first is not None:
        lower_bound = max(0, (epsilon_first-config.epsilon_tolerance)/2)
        upper_bound = min((epsilon_first + config.epsilon_tolerance)*2, 100)
        
        def optimization_func_second(eps):
            result = allocation_delta_decomposition_add_from_PLD(
                epsilon=eps, 
                num_steps=num_steps_per_round,
                Poisson_PLD_obj=Poisson_PLD_obj
            )
            evals_second.append(len(evals_second) + 1)
            x_vals_second.append(eps)
            y_vals_second.append(result)
            return result
        
        print("\nRunning second search...")
        start_time = time.time()
        epsilon_second = search_function_with_bounds(
            func=optimization_func_second, 
            y_target=params.delta, 
            bounds=(lower_bound, upper_bound),
            tolerance=config.epsilon_tolerance, 
            function_type=FunctionType.DECREASING
        )
        time_second = time.time() - start_time
        
        print(f"Second search completed in {time_second:.4f} seconds with {len(evals_second)} evaluations")
        print(f"Result: {epsilon_second}")
    
    # Plot the search progression
    plt.figure(figsize=(12, 8))
    
    # First search
    plt.subplot(2, 1, 1)
    plt.semilogy(evals_first, y_vals_first, 'o-', label='Function evaluations')
    plt.axhline(y=params.delta, color='r', linestyle='--', label=f'Target delta = {params.delta}')
    plt.title('First search progression')
    plt.xlabel('Evaluation number')
    plt.ylabel('Function value (delta)')
    plt.legend()
    plt.grid(True)
    
    # Second search (if applicable)
    if epsilon_first is not None:
        plt.subplot(2, 1, 2)
        plt.semilogy(evals_second, y_vals_second, 'o-', label='Function evaluations')
        plt.axhline(y=params.delta, color='r', linestyle='--', label=f'Target delta = {params.delta}')
        plt.title('Second search progression')
        plt.xlabel('Evaluation number')
        plt.ylabel('Function value (delta)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, 'search_function_analysis.png'))
    print(f"Plot saved to {os.path.join(plot_dir, 'search_function_analysis.png')}")
    
    # Create a comparison of the two function's behavior
    if epsilon_first is not None:
        # Generate points for visualization
        x_range = np.linspace(lower_bound, upper_bound, 100)
        y_first = [optimization_func_first(x) for x in x_range]
        y_second = [optimization_func_second(x) for x in x_range]
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(x_range, y_first, 'b-', label='First function')
        plt.semilogy(x_range, y_second, 'g-', label='Second function')
        plt.axhline(y=params.delta, color='r', linestyle='--', label=f'Target delta = {params.delta}')
        
        plt.title('Comparison of the two search functions')
        plt.xlabel('Epsilon value')
        plt.ylabel('Delta value (log scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'search_functions_comparison.png'))
        print(f"Comparison plot saved to {os.path.join(plot_dir, 'search_functions_comparison.png')}")
    
if __name__ == "__main__":
    test_search_callbacks()
