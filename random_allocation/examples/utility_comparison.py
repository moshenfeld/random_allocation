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
        num_steps: Number of steps in the scheme
        sigma: Noise parameter
    """
    num_experiments, sample_size = data.shape
    sampling_probability = 1.0/num_steps
    # Generate participation counts for all experiments at once
    num_participations = np.random.binomial(num_steps, sampling_probability, size=(num_experiments, sample_size))
    # Calculate mean for each experiment
    sums = np.sum(num_participations * data, axis=1)
    # Add noise to each mean
    noise = np.random.normal(0, sigma*np.sqrt(num_steps), size=num_experiments)
    return (sums + noise)/sample_size


def Poisson_accuracy(sampling_func, sample_size, num_experiments, num_steps, sigma, true_mean):
    """
    Calculates the accuracy of Poisson scheme, returning mean and standard deviation of squared errors.
    
    Args:
        sampling_func: Function to generate sample data
        sample_size: Size of each sample
        num_experiments: Number of experiments to run
        num_steps: Number of steps in the scheme
        sigma: Noise parameter
        true_mean: True mean to compare against
        
    Returns:
        tuple: (mean_error, std_error)
    """
    data = sampling_func(sample_size, num_experiments)
    # Get estimates for all experiments at once
    estimates = Poisson_mean_estimation_vectorized(data, num_steps, sigma)
    # Calculate squared errors
    errors = (estimates - true_mean) ** 2
    return np.mean(errors), np.std(errors)

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
        num_steps: Number of steps in the scheme
        sigma: Noise parameter
    """
    # Calculate means for each experiment
    sums = np.sum(data, axis=1)
    # Add noise to each mean
    noise = np.random.normal(0, sigma*np.sqrt(num_steps), size=sums.shape)
    return (sums + noise) / data.shape[1]

def allocation_accuracy(sampling_func, sample_size, num_experiments, num_steps, sigma, true_mean):
    """
    Calculates the accuracy of Random Allocation scheme, returning mean and standard deviation of errors.
    
    Args:
        sampling_func: Function to generate sample data
        sample_size: Size of each sample
        num_experiments: Number of experiments to run
        num_steps: Number of steps in the scheme
        sigma: Noise parameter
        true_mean: True mean to compare against
        
    Returns:
        tuple: (mean_error, std_error)
    """
    data = sampling_func(sample_size, num_experiments)
    # Get estimates for all experiments at once using vectorized implementation
    estimates = allocation_mean_estimation_vectorized(data, num_steps, sigma)
    # Calculate squared errors
    errors = (estimates - true_mean) ** 2
    return np.mean(errors), np.std(errors)

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

def run_experiments(epsilon, delta, num_steps, dimension, true_mean, num_experiments, sample_size_arr):
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
        use_hardcoded_sigma: Whether to use hardcoded sigma values instead of calculating them
        hardcoded_poisson_sigma: Hardcoded sigma value for Poisson scheme (already scaled by sqrt(dimension))
        hardcoded_allocation_sigma: Hardcoded sigma value for allocation scheme (already scaled by sqrt(dimension))
    
    Returns:
        tuple: (experiment_data, sigma_calc_time, simulation_time)
    """
    # Calculate sigma values
    Poisson_sigma_val = Poisson_sigma(num_steps, epsilon, delta) * np.sqrt(dimension)
    allocation_sigma_val = allocation_sigma(num_steps, epsilon, delta) * np.sqrt(dimension)
    
    # Create sampling function
    sampling_func = lambda sample_size, num_experiments: np.random.binomial(1, true_mean, size=(num_experiments, sample_size))
    
    Poisson_accuracy_means = []
    Poisson_stds = []
    allocation_accuracy_means = []
    allocation_stds = []       
    for sample_size in sample_size_arr:
        # Calculate Poisson accuracy metrics using the updated function
        p_mean, p_std = Poisson_accuracy(
            sampling_func, sample_size, num_experiments, num_steps, Poisson_sigma_val, 
            true_mean
        )
        Poisson_accuracy_means.append(p_mean)
        Poisson_stds.append(p_std)
        
        # Calculate allocation accuracy metrics using the updated function
        a_mean, a_std = allocation_accuracy(
            sampling_func, sample_size, num_experiments, num_steps, allocation_sigma_val, 
            true_mean
        )
        allocation_accuracy_means.append(a_mean)
        allocation_stds.append(a_std)
    
    # Analytic approximation
    Poisson_accuracy_analytic = true_mean * (1 - true_mean) / sample_size_arr + true_mean / sample_size_arr + Poisson_sigma_val**2 * num_steps / sample_size_arr**2
    allocation_accuracy_analytic = true_mean * (1 - true_mean) / sample_size_arr + allocation_sigma_val**2 * num_steps / sample_size_arr**2

    return {
        'Poisson accuracy': np.array(Poisson_accuracy_means),
        'Poisson std': np.array(Poisson_stds),
        'Allocation accuracy': np.array(allocation_accuracy_means),
        'Allocation std': np.array(allocation_stds),
        'Poisson accuracy (analytic)': Poisson_accuracy_analytic,
        'Allocation accuracy (analytic)': allocation_accuracy_analytic,
        'Poisson sigma': Poisson_sigma_val,
        'Allocation sigma': allocation_sigma_val
    }

def plot_subplot_with_ci(ax, x_data, data, title, xlabel, ylabel, num_experiments, C=3, show_ci=True):
    """
    Create a subplot with confidence interval-based visualization of error distributions.
    Uses standard deviation and standard error for confidence intervals.
    
    Args:
        ax: Matplotlib axis to plot on
        x_data: Array of x values (sample sizes)
        data: Dictionary containing results data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        num_experiments: Number of experiments run
        C: Multiplier for the confidence interval (standard error = std/sqrt(n))
        show_ci: Whether to display confidence interval bands (default: True)
    """
    # Colors for consistency
    Poisson_color = 'tab:blue'
    allocation_color = 'tab:orange'
    
    # Plot lines for experimental data (means)
    ax.plot(x_data, data['Poisson accuracy'], 'o', color=Poisson_color,
            label=f"Poisson (σ_scaled = {data['Poisson sigma']:.2f})")
    ax.plot(x_data, data['Allocation accuracy'], 's', color=allocation_color,
            label=f"Random allocation (σ_scaled = {data['Allocation sigma']:.2f})")
    
    # Calculate standard error from standard deviation
    std_error = lambda std: std / np.sqrt(num_experiments)
    
    # Only display confidence intervals if requested
    if show_ci:
        # Format CI for legend text
        ci_text = f"{C}σ confidence interval"
        
        # Always use pre-calculated standard deviations if available
        if 'Poisson std' in data:
            poisson_se = std_error(data['Poisson std'])
            ax.fill_between(
                x_data, 
                data['Poisson accuracy'] - C * poisson_se, 
                data['Poisson accuracy'] + C * poisson_se,
                alpha=0.3, color=Poisson_color, 
                label=f"Poisson {ci_text}"
            )
        else:
            print(f"Warning: No pre-calculated Poisson std available for {title}.")
            
        if 'Allocation std' in data:
            allocation_se = std_error(data['Allocation std'])
            ax.fill_between(
                x_data, 
                data['Allocation accuracy'] - C * allocation_se, 
                data['Allocation accuracy'] + C * allocation_se,
                alpha=0.3, color=allocation_color, 
                label=f"Random allocation {ci_text}"
            )
        else:
            print(f"Warning: No pre-calculated Allocation std available for {title}.")
    
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


# Comprehensive plot function that creates the complete visualization
def create_comparison_plot(sample_size_arr, experiment_data_list, titles, num_steps, num_experiments, C=3, show_ci=True):
    """
    Creates a comprehensive plot with subplots comparing Poisson and Random Allocation
    schemes using standard deviation-based confidence intervals.
    
    Args:
        sample_size_arr: Array of sample sizes 
        experiment_data_list: List of dictionaries containing results from experiments
        titles: List of titles for each subplot
        num_steps: Number of steps used in the experiment
        num_experiments: Number of experiments run
        C: Multiplier for the confidence interval (standard error = std/sqrt(n))
        show_ci: Whether to display confidence interval bands (default: True)
    """
    # Create figure and subplots
    fig, axs = plt.subplots(1, len(experiment_data_list), figsize=(20, 6))
    
    # Plot each experiment in its own subplot
    for i, (data, title) in enumerate(zip(experiment_data_list, titles)):
        plot_subplot_with_ci(
            axs[i], sample_size_arr, data, 
            title, "Sample Size", "Square Error", 
            num_experiments=num_experiments,
            C=C,
            show_ci=show_ci
        )
    
    # Add overall title
    fig.suptitle(f"Square Error of Poisson vs. Random Allocation Schemes with Number of Steps = {num_steps}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make space for the suptitle