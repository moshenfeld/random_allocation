from typing import Dict, Any, Callable, List, Tuple
import inspect
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from enum import Enum

from random_allocation.comparisons.definitions import *
from random_allocation.comparisons.visualization import plot_combined_data, plot_comparison, plot_as_table
from random_allocation.other_schemes import poisson, shuffle, local
from random_allocation.random_allocation_scheme import RDP_DCO, analytic, RDP, decomposition
import random_allocation.comparisons.definitions as definitions

class PlotType(Enum):
    COMPARISON = 1
    COMBINED = 2

def match_function_args(params_dict: Dict[str, Any],
                        config_dict: Dict[str, Any],
                        func: Callable,
                        x_var: str,
                        ) -> List[Dict[str, Any]]:
    """
    Match the function arguments with the parameters and configuration dictionaries.
    """
    params = inspect.signature(func).parameters
    args = {}
    for key in params_dict.keys():
        if key in params and key != x_var:
            args[key] = params_dict[key]
    for key in config_dict.keys():
        if key in params:
            args[key] = config_dict[key]
    args_arr = []
    for x in params_dict[x_var]:
        args_arr.append(args.copy())
        if x_var in params:
            args_arr[-1][x_var] = x
    return args_arr

def get_x_y_vars(params_dict: Dict[str, Any]) -> Tuple[str, str]:
    """
    Get the x and y variables from the parameters dictionary.
    """
    x_var = params_dict['x_var']
    if x_var not in params_dict.keys():
        raise ValueError(f"{x_var} was defined as the x-axis variable but does not appear in the params_dict.")
    y_var = params_dict['y_var']
    if y_var == x_var:
        raise ValueError(f"{x_var} was chosen as both the x-axis and y-axis variable.")
    return x_var, y_var

def get_main_var(params_dict: Dict[str, Any]) -> str:
    """
    Get the main variable from the parameters dictionary.
    """
    if 'main_var' in params_dict:
        return params_dict['main_var']
    return params_dict['x_var']

def get_func_dict(methods: list[str],
                  y_var: str
                  ) -> Dict[str, Any]:
    """
    Get the function dictionary for the given methods and y variable.
    """
    if y_var == EPSILON:
        return get_features_for_methods(methods, 'epsilon_calculator')
    return get_features_for_methods(methods, 'delta_calculator')

def clear_all_caches():
    """
    Clear all caches for all modules.
    """
    for module in [analytic, RDP_DCO, RDP, decomposition, poisson, shuffle, local, definitions]:
        for name, obj in module.__dict__.items():
            if callable(obj) and hasattr(obj, 'cache_clear'):
                obj.cache_clear()

def calc_experiment_data(params_dict: Dict[str, Any],
                         config_dict: Dict[str, Any],
                         methods: list[str],
                         )-> Dict[str, Any]:
    x_var, y_var = get_x_y_vars(params_dict)
    data = {'y data': {}}
    func_dict = get_func_dict(methods, y_var)
    for method in methods:
        start_time = time.time()
        func = func_dict[method]
        if func is None:
            raise ValueError(f"Method {method} does not have a valid function for {y_var}")
        args_arr = match_function_args(params_dict, config_dict, func, x_var)
        data['y data'][method] = np.array([func(**args) for args in args_arr])
        if data['y data'][method].ndim > 1:
            data['y data'][method + '- std'] = data['y data'][method][:,1]
            data['y data'][method] = data['y data'][method][:,0]
        end_time = time.time()
        print(f"Calculating {method} took {end_time - start_time:.3f} seconds")

    data['x name'] = names_dict[x_var]
    data['y name'] = names_dict[y_var]
    data['x data'] = params_dict[x_var]
    data['title'] = f"{names_dict[y_var]} as a function of {names_dict[x_var]} \n"
    for var in VARIABLES:
        if var != x_var and var != y_var:
            data[var] = params_dict[var]
            data['title'] += f"{names_dict[var]} = {params_dict[var]}, "
    return data

def save_experiment_data(data: Dict[str, Any], methods: List[str], experiment_name: str) -> None:
    """
    Save experiment data as a CSV file.
    
    Args:
        data: The experiment data dictionary
        methods: List of methods used in the experiment
        experiment_name: Name of the experiment for the output file (full path)
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(experiment_name), exist_ok=True)
    
    # Create DataFrame
    df_data = {'x': data['x data']}
    
    # Save y data for each method
    for method in methods:
        df_data[method] = data['y data'][method]
        if method + '- std' in data['y data']:
            df_data[method + '_std'] = data['y data'][method + '- std']
    
    # Include additional relevant data
    df_data['title'] = data.get('title', '')
    df_data['x name'] = data.get('x name', '')
    df_data['y name'] = data.get('y name', '')
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(df_data)
    df.to_csv(experiment_name, index=False)

def save_experiment_plot(data: Dict[str, Any], methods: List[str], experiment_name: str) -> None:
    """
    Save the experiment plot to a file.
    
    Args:
        data: The experiment data dictionary
        methods: List of methods used in the experiment
        experiment_name: Name of the experiment for the output file (full path)
    """
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(experiment_name), exist_ok=True)
    
    # Create and save the plot using plot_comparison
    plot_comparison(data)
    plt.savefig(f'{experiment_name}_plot.png')
    plt.close()

def run_experiment(params_dict: Dict[str, Any], config_dict: Dict[str, Any],
                  methods: List[str], visualization_config: Dict[str, Any],
                  experiment_name: str, plot_type: PlotType,
                  save_data: bool = True, save_plots: bool = True) -> None:
    """
    Run an experiment and handle its results.
    
    Args:
        params_dict: Dictionary of experiment parameters
        config_dict: Dictionary of configuration parameters
        methods: List of methods to use in the experiment
        visualization_config: Additional keyword arguments for the plot function
        experiment_name: Name of the experiment for the output file
        plot_type: Type of plot to create (COMPARISON or COMBINED)
        save_data: Whether to save data to CSV files
        save_plots: Whether to save plots to files
    """
    # Clear all caches before running the experiment
    clear_all_caches()
    
    # Get the examples directory path
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    data_file = os.path.join(examples_dir, 'data', f'{experiment_name}.csv')
    # Data logic:
    # If save_data is True: always recalculate and save
    # If save_data is False: try to read existing data, if not exists - recalculate but don't save
    if save_data:
        print(f"Computing data for {experiment_name}")
        data = calc_experiment_data(params_dict, config_dict, methods)
        save_experiment_data(data, methods, data_file)
    else:
        if os.path.exists(data_file):
            print(f"Reading data from {data_file}")
            data = pd.read_csv(data_file)
        else:
            print(f"Computing data for {experiment_name}")
            data = calc_experiment_data(params_dict, config_dict, methods)
    
    # Plot logic:
    # If save_plots is True: only save the plot, don't display it
    # If save_plots is False: only display the plot, don't save it
    if visualization_config is None:
        visualization_config = {}
    
    # Create the appropriate plot based on plot_type
    if plot_type == PlotType.COMPARISON:
        fig = plot_comparison(data, **visualization_config)
    else:  # PlotType.COMBINED
        fig = plot_combined_data(data, **visualization_config)
    
    if save_plots:
        # Save the plot
        os.makedirs(os.path.join(examples_dir, 'plots'), exist_ok=True)
        fig.savefig(os.path.join(examples_dir, 'plots', f'{experiment_name}_plot.png'))
        plt.close(fig)
    else:
        # Display the plot and table
        plt.show()
        plot_as_table(data)
    return data