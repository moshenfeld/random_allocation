import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (random_allocation)
parent_dir = os.path.dirname(current_dir)

# Get the parent of parent directory (the project root)
project_root = os.path.dirname(parent_dir)

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to sys.path")

from random_allocation.comparisons.utils import *
from random_allocation.comparisons.structs import *
from random_allocation.other_schemes.poisson import *
from random_allocation.random_allocation_scheme.recursive import *
from random_allocation.random_allocation_scheme.decomposition import *


def Poisson_mean_estimation_vectorized(data, num_steps, sigma):
    """
    Vectorized implementation of Poisson mean estimation.
    
    Args:
        data: Array of shape (num_experiments, sample_size) containing the data
        num_steps: Number of steps in the privacy mechanism
        sigma: Noise parameter
        
    Returns:
        Array of mean estimates for each experiment
    """
    num_experiments, sample_size = data.shape
    sampling_probability = 1.0/num_steps
    # Generate participation counts for all experiments at once
    num_participations = np.random.binomial(num_steps, sampling_probability, size=(num_experiments, sample_size))
    # Calculate mean for each experiment
    means = np.mean(num_participations * data, axis=1)
    # Add noise to each mean
    noise = np.random.normal(0, sigma/np.sqrt(num_steps), size=num_experiments)
    return means + noise


def Poisson_accuracy(sampling_func, sample_size, num_experiments, num_steps, sigma, true_mean, low_quantile=0.25, high_quantile=0.75):
    """
    Calculates the accuracy of Poisson sampling method, returning mean and quantiles of errors.
    Uses vectorized operations for improved performance.
    
    Args:
        sampling_func: Function to generate sample data
        sample_size: Size of each sample
        num_experiments: Number of experiments to run
        num_steps: Number of steps in the privacy mechanism
        sigma: Noise parameter
        true_mean: True mean to compare against
        low_quantile: Lower quantile (default: 0.25 for 25th percentile)
        high_quantile: Upper quantile (default: 0.75 for 75th percentile)
        
    Returns:
        tuple: (mean_error, low_quantile_value, high_quantile_value)
    """
    data = sampling_func(sample_size, num_experiments)
    
    # Get estimates for all experiments at once
    estimates = Poisson_mean_estimation_vectorized(data, num_steps, sigma)
    
    # Calculate squared errors
    errors = (estimates - true_mean) ** 2
    
    return np.mean(errors), np.quantile(errors, low_quantile), np.quantile(errors, high_quantile)


def Poisson_epsilon(num_steps, sigma, delta):
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=1,
        num_epochs=1,
        delta=delta,
    )
    config = SchemeConfig()
    return Poisson_epsilon_PLD(params, config, direction=Direction.BOTH)


def Poisson_sigma(num_steps, epsilon, delta, lower = 0.1, upper = 10):
    optimization_func = lambda sig: Poisson_epsilon(num_steps=num_steps, sigma=sig, delta=delta) 
    
    sigma = search_function_with_bounds(
        func=optimization_func, 
        y_target=epsilon,
        bounds=(lower, upper),
        tolerance=0.05,
        function_type=FunctionType.DECREASING
    )
    if sigma is None:
        lower_epsilon = Poisson_epsilon(num_steps=num_steps, sigma=upper, delta=delta)
        upper_epsilon = Poisson_epsilon(num_steps=num_steps, sigma=lower, delta=delta)
        print(f"Poisson_sigma: lower_epsilon={lower_epsilon}, upper_epsilon={upper_epsilon}, target_epsilon={epsilon}")
        return np.inf
    return sigma


def allocation_mean_estimation_vectorized(data, num_steps, sigma):
    """
    Vectorized implementation of Random Allocation mean estimation.
    
    Args:
        data: Array of shape (num_experiments, sample_size) containing the data
        num_steps: Number of steps in the privacy mechanism
        sigma: Noise parameter
        
    Returns:
        Array of mean estimates for each experiment
    """
    # Calculate means for each experiment
    means = np.mean(data, axis=1)
    # Add noise to each mean
    noise = np.random.normal(0, sigma/np.sqrt(num_steps), size=means.shape)
    return means + noise


def allocation_mean_estimation(data, num_steps, sigma):
    """Legacy non-vectorized function (kept for compatibility)"""
    return np.mean(data) + np.random.normal(0, sigma/np.sqrt(num_steps))


def allocation_accuracy(sampling_func, sample_size, num_experiments, num_steps, sigma, true_mean, low_quantile=0.25, high_quantile=0.75):
    """
    Calculates the accuracy of Random Allocation method, returning mean and quantiles of errors.
    Uses vectorized operations for improved performance.
    
    Args:
        sampling_func: Function to generate sample data
        sample_size: Size of each sample
        num_experiments: Number of experiments to run
        num_steps: Number of steps in the privacy mechanism
        sigma: Noise parameter
        true_mean: True mean to compare against
        low_quantile: Lower quantile (default: 0.25 for 25th percentile)
        high_quantile: Upper quantile (default: 0.75 for 75th percentile)
        
    Returns:
        tuple: (mean_error, low_quantile_value, high_quantile_value)
    """
    data = sampling_func(sample_size, num_experiments)
    
    # Get estimates for all experiments at once using vectorized implementation
    estimates = allocation_mean_estimation_vectorized(data, num_steps, sigma)
    
    # Calculate squared errors
    errors = (estimates - true_mean) ** 2
    
    return np.mean(errors), np.quantile(errors, low_quantile), np.quantile(errors, high_quantile)


def allocation_epsilon(num_steps, sigma, delta):
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=1,
        num_epochs=1,
        delta=delta,
    )
    config = SchemeConfig()
    return allocation_epsilon_recursive(params, config, direction=Direction.BOTH)


def allocation_sigma(num_steps, epsilon, delta, lower = 0.1, upper = 10):
    optimization_func = lambda sig: allocation_epsilon(num_steps=num_steps, sigma=sig, delta=delta)
    sigma = search_function_with_bounds(
        func=optimization_func, 
        y_target=epsilon,
        bounds=(lower, upper),
        tolerance=0.05,
        function_type=FunctionType.DECREASING
    )
    if sigma is None:
        lower_epsilon = allocation_epsilon(num_steps=num_steps, sigma=upper, delta=delta)
        upper_epsilon = allocation_epsilon(num_steps=num_steps, sigma=lower, delta=delta)
        print(f"Allocation_sigma: lower_epsilon={lower_epsilon}, upper_epsilon={upper_epsilon}, target_epsilon={epsilon}")
        return np.inf    
    return sigma


# Comprehensive plot function that creates the complete visualization
def create_comparison_plot(sample_size_arr, experiment_data_list, titles, num_steps, low_quantile=0.25, high_quantile=0.75, show_quantiles=True):
    """
    Creates a comprehensive plot with subplots comparing Poisson and Random Allocation
    schemes using quantile-based visualization.
    
    Args:
        sample_size_arr: Array of sample sizes 
        experiment_data_list: List of dictionaries containing results from experiments
        titles: List of titles for each subplot
        num_steps: Number of steps used in the experiment
        low_quantile: Lower quantile boundary (default: 0.25 for 25th percentile)
        high_quantile: Upper quantile boundary (default: 0.75 for 75th percentile)
        show_quantiles: Whether to display quantile bands (default: True)
    """
    # Create figure and subplots
    fig, axs = plt.subplots(1, len(experiment_data_list), figsize=(20, 6))
    
    # Plot each experiment in its own subplot
    for i, (data, title) in enumerate(zip(experiment_data_list, titles)):
        plot_subplot_with_quantiles(
            axs[i], sample_size_arr, data, 
            title, "Sample Size", "Square Error", 
            low_quantile=low_quantile, high_quantile=high_quantile,
            show_quantiles=show_quantiles
        )
    
    # Add overall title
    fig.suptitle(f"Square Error of Poisson vs. Random Allocation Schemes with Number of Steps = {num_steps}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make space for the suptitle


# Improved subplot function for quantile-based visualization
def plot_subplot_with_quantiles(ax, x_data, data, title, xlabel, ylabel, low_quantile=0.25, high_quantile=0.75, show_quantiles=True):
    """
    Create a subplot with quantile-based visualization of error distributions.
    This version prioritizes pre-calculated quantiles over raw error data.
    
    Args:
        ax: Matplotlib axis to plot on
        x_data: Array of x values (sample sizes)
        data: Dictionary containing results data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        low_quantile: Lower quantile boundary (default: 0.25 for 25th percentile)
        high_quantile: Upper quantile boundary (default: 0.75 for 75th percentile)
        show_quantiles: Whether to display quantile bands (default: True)
    """
    # Colors for consistency
    Poisson_color = 'tab:blue'
    allocation_color = 'tab:orange'
    
    # Plot lines for experimental data (means)
    ax.plot(x_data, data['Poisson accuracy'], 'o', color=Poisson_color,
            label=f"Poisson (σ_scaled = {data['Poisson sigma scaled']:.2f})")
    ax.plot(x_data, data['Allocation accuracy'], 's', color=allocation_color,
            label=f"Random allocation (σ_scaled = {data['Allocation sigma scaled']:.2f})")
    
    # Only display quantiles if requested
    if show_quantiles:
        # Format quantiles for legend text
        quantile_text = f"{int(low_quantile*100)}-{int(high_quantile*100)}th percentile"
        
        # Always use pre-calculated quantiles if available - these come directly from our accuracy functions
        if 'Poisson low quantile' in data and 'Poisson high quantile' in data:
            ax.fill_between(
                x_data, 
                data['Poisson low quantile'], 
                data['Poisson high quantile'],
                alpha=0.3, color=Poisson_color, 
                label=f"Poisson {quantile_text}"
            )
        else:
            print(f"Warning: No pre-calculated Poisson quantiles available for {title}.")
            
        if 'Allocation low quantile' in data and 'Allocation high quantile' in data:
            ax.fill_between(
                x_data, 
                data['Allocation low quantile'], 
                data['Allocation high quantile'],
                alpha=0.3, color=allocation_color, 
                label=f"Random allocation {quantile_text}"
            )
        else:
            print(f"Warning: No pre-calculated Allocation quantiles available for {title}.")
    
    # Plot analytic approximations
    ax.plot(x_data, data['Poisson accuracy (analytic)'], '--', color=Poisson_color, 
            label="Poisson (analytic)")
    ax.plot(x_data, data['Allocation accuracy (analytic)'], '--', color=allocation_color, 
            label="Random allocation (analytic)")
    
    # Finalize plot formatting
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)


def run_experiments(epsilon, delta, num_steps, dimension, true_mean, num_experiments, sample_size_arr, low_quantile=0.25, high_quantile=0.75, use_hardcoded_sigma=False, hardcoded_poisson_sigma=None, hardcoded_allocation_sigma=None):
    """
    Run a single experiment with given parameters and measure execution time for different stages.
    
    Args:
        epsilon: Privacy parameter epsilon
        delta: Privacy parameter delta
        num_steps: Number of steps in the privacy mechanism
        dimension: Dimension factor for noise scaling
        true_mean: True mean to compare against
        num_experiments: Number of experiments to run
        sample_size_arr: Array of sample sizes to test
        low_quantile: Lower quantile boundary (default: 0.25 for 25th percentile)
        high_quantile: Upper quantile boundary (default: 0.75 for 75th percentile)
        use_hardcoded_sigma: Whether to use hardcoded sigma values instead of calculating them
        hardcoded_poisson_sigma: Hardcoded sigma value for Poisson scheme (already scaled by sqrt(dimension))
        hardcoded_allocation_sigma: Hardcoded sigma value for allocation scheme (already scaled by sqrt(dimension))
    
    Returns:
        tuple: (experiment_data, sigma_calc_time, simulation_time)
    """
    # Start timing sigma calculation
    sigma_start_time = time.time()
    
    # Calculate sigma values (preliminary step) or use hardcoded values
    if use_hardcoded_sigma and hardcoded_poisson_sigma is not None and hardcoded_allocation_sigma is not None:
        # The hardcoded values are already the final scaled sigma values - use them directly
        Poisson_sigma_scaled = hardcoded_poisson_sigma
        allocation_sigma_scaled = hardcoded_allocation_sigma
        
        # We don't need the base sigma values for this code path, but we'll calculate them
        # for informational purposes only
        if dimension > 0:
            Poisson_sigma_val = hardcoded_poisson_sigma / np.sqrt(dimension)
            allocation_sigma_val = hardcoded_allocation_sigma / np.sqrt(dimension)
        else:
            Poisson_sigma_val = hardcoded_poisson_sigma
            allocation_sigma_val = hardcoded_allocation_sigma
            
        print(f"Using hardcoded sigma values (already scaled): Poisson={Poisson_sigma_scaled:.4f}, Allocation={allocation_sigma_scaled:.4f}")
    else:
        # Calculate base sigma values
        Poisson_sigma_val = Poisson_sigma(num_steps, epsilon, delta)
        allocation_sigma_val = allocation_sigma(num_steps, epsilon, delta)
        print(f"Calculated base sigma values: Poisson={Poisson_sigma_val:.4f}, Allocation={allocation_sigma_val:.4f}")
        
        # Scale sigma values by sqrt(dimension)
        Poisson_sigma_scaled = Poisson_sigma_val * np.sqrt(dimension)
        allocation_sigma_scaled = allocation_sigma_val * np.sqrt(dimension)
        print(f"Scaled sigma values (√d = {np.sqrt(dimension):.2f}): Poisson={Poisson_sigma_scaled:.4f}, Allocation={allocation_sigma_scaled:.4f}")
    
    sigma_end_time = time.time()
    sigma_calc_time = sigma_end_time - sigma_start_time
    
    # Start timing simulation stage
    simulation_start_time = time.time()
    
    # Run the actual experiment
    sampling_func = lambda sample_size, num_experiments: np.random.binomial(1, true_mean, size=(num_experiments, sample_size))
    
    Poisson_accuracy_means = []
    Poisson_low_quantiles = []
    Poisson_high_quantiles = []
    allocation_accuracy_means = []
    allocation_low_quantiles = []
    allocation_high_quantiles = []
    
    # Progress bar replaced with simple print statement
    print(f"Testing sample sizes (ε={epsilon}, d={dimension})")
    
    # Iterate through sample sizes
    for sample_size in sample_size_arr:
        # Calculate Poisson accuracy metrics using the updated function
        p_mean, p_low, p_high = Poisson_accuracy(
            sampling_func, sample_size, num_experiments, num_steps, Poisson_sigma_scaled, 
            true_mean, low_quantile, high_quantile
        )
        
        # Calculate allocation accuracy metrics using the updated function
        a_mean, a_low, a_high = allocation_accuracy(
            sampling_func, sample_size, num_experiments, num_steps, allocation_sigma_scaled, 
            true_mean, low_quantile, high_quantile
        )
        
        # Store calculated metrics
        Poisson_accuracy_means.append(p_mean)
        Poisson_low_quantiles.append(p_low)
        Poisson_high_quantiles.append(p_high)
        allocation_accuracy_means.append(a_mean)
        allocation_low_quantiles.append(a_low)
        allocation_high_quantiles.append(a_high)
    
    # Analytic approximation
    Poisson_accuracy_analytic = true_mean * (1 - true_mean) / sample_size_arr + true_mean / sample_size_arr + Poisson_sigma_scaled**2 / num_steps
    allocation_accuracy_analytic = true_mean * (1 - true_mean) / sample_size_arr + allocation_sigma_scaled**2 / num_steps
    
    simulation_end_time = time.time()
    simulation_time = simulation_end_time - simulation_start_time
    
    experiment_data = {
        'Poisson accuracy': np.array(Poisson_accuracy_means),
        'Poisson low quantile': np.array(Poisson_low_quantiles),
        'Poisson high quantile': np.array(Poisson_high_quantiles),
        'Allocation accuracy': np.array(allocation_accuracy_means),
        'Allocation low quantile': np.array(allocation_low_quantiles),
        'Allocation high quantile': np.array(allocation_high_quantiles),
        'Poisson accuracy (analytic)': Poisson_accuracy_analytic, 
        'Allocation accuracy (analytic)': allocation_accuracy_analytic, 
        'Poisson sigma': Poisson_sigma_val,  # Store base sigma
        'Allocation sigma': allocation_sigma_val,  # Store base sigma
        'Poisson sigma scaled': Poisson_sigma_scaled,  # Store scaled sigma for plots
        'Allocation sigma scaled': allocation_sigma_scaled  # Store scaled sigma for plots
    }
    
    return experiment_data, sigma_calc_time, simulation_time


def run_comparison_experiments(small_eps, large_eps, small_dim, large_dim, num_steps, num_experiments, true_mean, delta, 
                             low_quantile, high_quantile, sample_size_arr, use_hardcoded_sigma, 
                             small_eps_small_dim_poisson_sigma, small_eps_small_dim_allocation_sigma,
                             large_eps_small_dim_poisson_sigma, large_eps_small_dim_allocation_sigma,
                             large_eps_large_dim_poisson_sigma, large_eps_large_dim_allocation_sigma,
                             show_quantiles=True):
    """
    Run the utility comparison experiments with the given parameters.
    
    This function executes three experiments with different combinations of epsilon and dimension values,
    and plots the results. It performs all the execution logic previously in the main function.
    
    Args:
        small_eps: Small epsilon value for privacy parameter
        large_eps: Large epsilon value for privacy parameter
        small_dim: Small dimension value
        large_dim: Large dimension value
        num_steps: Number of steps in the privacy mechanism
        num_experiments: Number of experiments to run
        true_mean: True mean to compare against
        delta: Privacy parameter delta
        low_quantile: Lower quantile boundary
        high_quantile: Upper quantile boundary
        sample_size_arr: Array of sample sizes to test
        use_hardcoded_sigma: Whether to use hardcoded sigma values
        small_eps_small_dim_poisson_sigma: Hardcoded sigma for small epsilon, small dimension, Poisson scheme
        small_eps_small_dim_allocation_sigma: Hardcoded sigma for small epsilon, small dimension, allocation scheme
        large_eps_small_dim_poisson_sigma: Hardcoded sigma for large epsilon, small dimension, Poisson scheme
        large_eps_small_dim_allocation_sigma: Hardcoded sigma for large epsilon, small dimension, allocation scheme
        large_eps_large_dim_poisson_sigma: Hardcoded sigma for large epsilon, large dimension, Poisson scheme
        large_eps_large_dim_allocation_sigma: Hardcoded sigma for large epsilon, large dimension, allocation scheme
        show_quantiles: Whether to display quantile bands in the plot (default: True)
    """
    # Start tracking total execution time
    total_start_time = time.time()
    total_sigma_calc_time = 0
    total_simulation_time = 0
    
    print("=" * 80)
    print("UTILITY COMPARISON: POISSON SAMPLING VS RANDOM ALLOCATION")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  - Number of steps: {num_steps}")
    print(f"  - Number of experiments: {num_experiments}")
    print(f"  - Sample sizes: {min(sample_size_arr)} to {max(sample_size_arr)}")
    print(f"  - True mean: {true_mean}")
    print(f"  - Delta: {delta}")
    print(f"  - Using hardcoded sigma values: {use_hardcoded_sigma}")
    print("=" * 80)
    
    # Run experiments with detailed time measurement
    print("Running experiment 1: small epsilon, small dimension...")
    small_eps_small_dim_data, sigma_time1, sim_time1 = run_experiments(
        small_eps, delta, num_steps, small_dim, true_mean, num_experiments, 
        sample_size_arr, low_quantile, high_quantile, 
        use_hardcoded_sigma, small_eps_small_dim_poisson_sigma, small_eps_small_dim_allocation_sigma
    )
    print(f"Sigma calculation time: {sigma_time1:.2f} seconds")
    print(f"Simulation time: {sim_time1:.2f} seconds")
    print(f"Total time: {sigma_time1 + sim_time1:.2f} seconds")
    total_sigma_calc_time += sigma_time1
    total_simulation_time += sim_time1
    
    print("\nRunning experiment 2: large epsilon, small dimension...")
    large_eps_small_dim_data, sigma_time2, sim_time2 = run_experiments(
        large_eps, delta, num_steps, small_dim, true_mean, num_experiments, 
        sample_size_arr, low_quantile, high_quantile,
        use_hardcoded_sigma, large_eps_small_dim_poisson_sigma, large_eps_small_dim_allocation_sigma
    )
    print(f"Sigma calculation time: {sigma_time2:.2f} seconds")
    print(f"Simulation time: {sim_time2:.2f} seconds")
    print(f"Total time: {sigma_time2 + sim_time2:.2f} seconds")
    total_sigma_calc_time += sigma_time2
    total_simulation_time += sim_time2
    
    print("\nRunning experiment 3: large epsilon, large dimension...")
    large_eps_large_dim_data, sigma_time3, sim_time3 = run_experiments(
        large_eps, delta, num_steps, large_dim, true_mean, num_experiments, 
        sample_size_arr, low_quantile, high_quantile,
        use_hardcoded_sigma, large_eps_large_dim_poisson_sigma, large_eps_large_dim_allocation_sigma
    )
    print(f"Sigma calculation time: {sigma_time3:.2f} seconds")
    print(f"Simulation time: {sim_time3:.2f} seconds")
    print(f"Total time: {sigma_time3 + sim_time3:.2f} seconds")
    total_sigma_calc_time += sigma_time3
    total_simulation_time += sim_time3
    
    # Create subplot titles
    titles = [
        f"ε = {small_eps:.1f}, d = {small_dim}", 
        f"ε = {large_eps:.1f}, d = {small_dim}", 
        f"ε = {large_eps:.1f}, d = {large_dim}"
    ]
    
    # Create visualization with the new function
    print("\nCreating visualization...")
    experiment_data_list = [small_eps_small_dim_data, large_eps_small_dim_data, large_eps_large_dim_data]
    create_comparison_plot(sample_size_arr, experiment_data_list, titles, num_steps, low_quantile, high_quantile, show_quantiles)
    
    # Show the plot
    plt.show()
    
    # Calculate and print timing information
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    other_time = total_execution_time - (total_sigma_calc_time + total_simulation_time)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Sigma calculation time: {total_sigma_calc_time:.2f} seconds ({total_sigma_calc_time/total_execution_time*100:.1f}%)")
    print(f"Simulation time:        {total_simulation_time:.2f} seconds ({total_simulation_time/total_execution_time*100:.1f}%)")
    print(f"Other operations:       {other_time:.2f} seconds ({other_time/total_execution_time*100:.1f}%)")
    print(f"Total execution time:   {total_execution_time:.2f} seconds")
    print("=" * 80)
    if use_hardcoded_sigma:
        print("Note: Used hardcoded sigma values to avoid expensive recomputation.")
    else:
        print("Note: Used calculated sigma values (can be hardcoded for future runs).")
    print("      Using vectorized operations for improved performance.")
    print("=" * 80)
    
    # Print sigma values for future hardcoding
    if not use_hardcoded_sigma:
        print("\n" + "=" * 80)
        print("SIGMA VALUES (for hardcoding in future runs)")
        print("=" * 80)
        print(f"small_eps_small_dim_poisson_sigma = {small_eps_small_dim_data['Poisson sigma']:.4f}")
        print(f"small_eps_small_dim_allocation_sigma = {small_eps_small_dim_data['Allocation sigma']:.4f}")
        print(f"large_eps_small_dim_poisson_sigma = {large_eps_small_dim_data['Poisson sigma']:.4f}")
        print(f"large_eps_small_dim_allocation_sigma = {large_eps_small_dim_data['Allocation sigma']:.4f}")
        print(f"large_eps_large_dim_poisson_sigma = {large_eps_large_dim_data['Poisson sigma']:.4f}")
        print(f"large_eps_large_dim_allocation_sigma = {large_eps_large_dim_data['Allocation sigma']:.4f}")
        print("=" * 80)
    
    # Return data for potential further analysis
    return {
        'small_eps_small_dim': small_eps_small_dim_data,
        'large_eps_small_dim': large_eps_small_dim_data,
        'large_eps_large_dim': large_eps_large_dim_data
    }


def main(show_quantiles=True):
    """
    Main function to run the utility comparison experiment.
    
    This script compares the accuracy of two privacy mechanisms:
    1. Poisson sampling
    2. Random Allocation
    
    It runs three experiments with different combinations of epsilon and dimension values,
    and plots the results with error quantiles. The script uses vectorized operations
    for improved performance and provides detailed timing information.
    
    Args:
        show_quantiles: Whether to display quantile bands in the plot (default: True)
    """
    # Default experiment parameters
    small_eps = 0.1
    large_eps = 1.0
    small_dim = 1
    large_dim = 1000

    num_steps = 10_000
    num_experiments = 1_000  # Reduced from 20_000 to make testing faster
    true_mean = 0.9
    delta = 1e-10
    
    # Quantile parameters
    low_quantile = 0.25  # 25th percentile
    high_quantile = 0.75  # 75th percentile

    # Create sample size array with 20 elements
    sample_size_arr = np.logspace(2, 4, num=20, dtype=int)
    
    # Option to use hardcoded sigma values (set to False to calculate values, then True after getting the values)
    use_hardcoded_sigma = True
    
    # Hardcoded sigma values will be determined by running with use_hardcoded_sigma = False
    # Then copying the printed values from the "SIGMA VALUES (for hardcoding in future runs)" section
    # These are now the SCALED sigma values (already multiplied by sqrt(dimension))
    small_eps_small_dim_poisson_sigma = 0.9637
    small_eps_small_dim_allocation_sigma = 1.0737
    large_eps_small_dim_poisson_sigma = 0.7031
    large_eps_small_dim_allocation_sigma = 0.7479
    large_eps_large_dim_poisson_sigma = 22.2348
    large_eps_large_dim_allocation_sigma = 23.6492
    
    # Run the experiments with all the parameters
    run_comparison_experiments(
        small_eps, large_eps, small_dim, large_dim,
        num_steps, num_experiments, true_mean, delta,
        low_quantile, high_quantile, sample_size_arr,
        use_hardcoded_sigma,
        small_eps_small_dim_poisson_sigma, small_eps_small_dim_allocation_sigma,
        large_eps_small_dim_poisson_sigma, large_eps_small_dim_allocation_sigma,
        large_eps_large_dim_poisson_sigma, large_eps_large_dim_allocation_sigma,
        show_quantiles
    )


if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Utility comparison between Poisson and Random Allocation schemes')
    parser.add_argument('--hide-quantiles', action='store_true', help='Hide quantile bands in the plot')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function with specified options
    main(show_quantiles=not args.hide_quantiles)
