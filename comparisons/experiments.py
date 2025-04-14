from typing import Dict, Any, Callable, List, Tuple, Optional
import inspect
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import cache, lru_cache
from dataclasses import dataclass

from comparisons.definitions import *
from comparisons.visualization import plot_combined_data, plot_comparison, plot_as_table
import other_schemes.poisson as poisson
import other_schemes.shuffle as shuffle
import other_schemes.local as local
import random_allocation_scheme.analytic as analytic
import random_allocation_scheme.loose_RDP as loose_RDP
import random_allocation_scheme.RDP as RDP
import random_allocation_scheme.decomposition as decomposition
import comparisons.definitions as definitions

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
    for module in [analytic, loose_RDP, RDP, decomposition, poisson, shuffle, local, definitions]:
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
    for method in methods:
        df_data[method] = data['y data'][method]
        if method + '- std' in data['y data']:
            df_data[method + '_std'] = data['y data'][method + '- std']
    
    df = pd.DataFrame(df_data)
    df.to_csv(f'{experiment_name}_data.csv', index=False)

def save_experiment_plot(data: Dict[str, Any], methods: List[str], experiment_name: str) -> None:
    """
    Save experiment plot as a PNG file.
    
    Args:
        data: The experiment data dictionary
        methods: List of methods used in the experiment
        experiment_name: Name of the experiment for the output file (full path)
    """
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(experiment_name), exist_ok=True)
    
    # Create and save the plot
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.plot(data['x data'], data['y data'][method], label=method)
        if method + '- std' in data['y data']:
            plt.fill_between(data['x data'],
                           data['y data'][method] - data['y data'][method + '- std'],
                           data['y data'][method] + data['y data'][method + '- std'],
                           alpha=0.2)
    
    plt.xlabel(data['x name'])
    plt.ylabel(data['y name'])
    plt.title(data['title'])
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{experiment_name}_plot.png')
    plt.close()

def run_experiment(params_dict: Dict[str, Any], config_dict: Dict[str, Any],
                  methods: List[str], visualization_config: Dict[str, Any],
                  experiment_name: str, plot_func: Callable,
                  save_data: bool = True, save_plots: bool = True) -> None:
    """
    Run an experiment and handle its results.
    
    Args:
        params_dict: Dictionary of experiment parameters
        config_dict: Dictionary of configuration parameters
        methods: List of methods to use in the experiment
        visualization_config: Additional keyword arguments for the plot function
        experiment_name: Name of the experiment for the output file
        plot_func: Function to use for plotting (plot_combined_data or plot_comparison)
        save_data: Whether to save data to CSV files
        save_plots: Whether to save plots to files
    """
    # Clear all caches before running the experiment
    clear_all_caches()
    
    # Get the examples directory path
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    data_file = os.path.join(examples_dir, 'data', f'{experiment_name}.csv')
    
    # If save_data is True and the data file exists, read it
    if save_data and os.path.exists(data_file):
        print(f"Reading data from {data_file}")
        data = pd.read_csv(data_file)
    else:
        # Execute the experiment
        print(f"Computing data for {experiment_name}")
        data = calc_experiment_data(params_dict, config_dict, methods)
        
        if save_data:
            save_experiment_data(data, methods, os.path.join(examples_dir, 'data', experiment_name))
    
    if save_plots:
        save_experiment_plot(data, methods, os.path.join(examples_dir, 'plots', experiment_name))
    else:
        if visualization_config is None:
            visualization_config = {}
        plot_func(data, **visualization_config)
        plot_as_table(data)