from typing import Dict, Any, Callable, List, Tuple, Union
import inspect
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from enum import Enum
import copy

from random_allocation.comparisons.definitions import *
from random_allocation.comparisons.visualization import plot_combined_data, plot_comparison, plot_as_table

class PlotType(Enum):
    COMPARISON = 1
    COMBINED = 2

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

def calc_experiment_data(params: PrivacyParams,
                         config: SchemeConfig,
                         methods: list[str],
                         x_var: str,
                         x_values: list[Union[float, int]],
                         y_var: str,
                         ) -> Dict[str, Any]:
    """
    Calculate experiment data using PrivacyParams and SchemeConfig objects directly.
    
    Args:
        params: Base privacy parameters object
        config: Scheme configuration object
        methods: List of methods to use in the experiment
        x_var: Name of parameter to vary (x-axis)
        x_values: List of values for the x-axis parameter
        y_var: Name of result to compute (y-axis: 'epsilon' or 'delta')
        
    Returns:
        Dictionary with the experiment data
    """
    data = {'y data': {}}
    func_dict = get_func_dict(methods, y_var)
    
    for method in methods:
        start_time = time.time()
        func = func_dict[method]
        if func is None:
            raise ValueError(f"Method {method} does not have a valid function for {y_var}")
        
        # Calculate results for each x value
        results = []
        for x_value in x_values:
            # Create a copy of params and set the x_var to the current value
            param_copy = copy.deepcopy(params)
            setattr(param_copy, x_var, x_value)
            
            # Reset any computed parameter if we're changing one of the input parameters
            if x_var == 'epsilon' and param_copy.delta is not None:
                param_copy.delta = None
            elif x_var == 'delta' and param_copy.epsilon is not None:
                param_copy.epsilon = None
            
            # Call the function with the modified params
            results.append(func(params=param_copy, config=config))
        
        data['y data'][method] = np.array(results)
        
        if data['y data'][method].ndim > 1:
            data['y data'][method + '- std'] = data['y data'][method][:,1]
            data['y data'][method] = data['y data'][method][:,0]
        
        end_time = time.time()
        print(f"Calculating {method} took {end_time - start_time:.3f} seconds")

    data['x name'] = names_dict[x_var]
    data['y name'] = names_dict[y_var]
    data['x data'] = x_values
    
    # Build title for the plot
    data['title'] = f"{names_dict[y_var]} as a function of {names_dict[x_var]} \n"
    
    for var in VARIABLES:
        if var != x_var and var != y_var:
            value = getattr(params, var, None)
            if value is not None:
                data[var] = value
                data['title'] += f"{names_dict[var]} = {value}, "
    
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

def run_experiment(
    params_dict_or_obj: Union[Dict[str, Any], PrivacyParams],
    config: SchemeConfig,
    methods: List[str], 
    visualization_config: Dict[str, Any] = None,
    experiment_name: str = '',
    plot_type: PlotType = PlotType.COMPARISON,
    save_data: bool = True, 
    save_plots: bool = True
) -> None:
    """
    Run an experiment and handle its results.
    
    Args:
        params_dict_or_obj: Either a dictionary of parameters or a PrivacyParams object
            If dictionary, must contain 'x_var', 'y_var', and x_values
        config_dict_or_obj: Either a dictionary of configuration values or a SchemeConfig object
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
    
    # Convert params_dict to object if they're dictionaries
    if isinstance(params_dict_or_obj, dict):
        # Extract x_var, y_var and x_values from the dictionary
        params_dict = params_dict_or_obj
        x_var = params_dict.pop('x_var', None)
        y_var = params_dict.pop('y_var', None)
        
        if x_var is None or y_var is None:
            raise ValueError("params_dict must contain 'x_var' and 'y_var' keys")
        
        if x_var not in params_dict:
            raise ValueError(f"params_dict must contain values for '{x_var}'")
            
        x_values = params_dict[x_var]
        
        # Create a PrivacyParams object with base values (excluding x_values)
        base_params = {k: v for k, v in params_dict.items() if not isinstance(v, (list, np.ndarray))}
        
        # Use epsilon/delta from params_dict if present
        epsilon = base_params.get(EPSILON)
        delta = base_params.get(DELTA)
        
        # Create PrivacyParams object with initial values
        params = PrivacyParams(
            sigma=base_params.get(SIGMA, 0),
            num_steps=base_params.get(NUM_STEPS, 0),
            num_selected=base_params.get(NUM_SELECTED, 0),
            num_epochs=base_params.get(NUM_EPOCHS, 0),
            epsilon=epsilon,
            delta=delta
        )
    else:
        # params_dict_or_obj is already a PrivacyParams object
        params = params_dict_or_obj
        
        # In this case, x_var, y_var, and x_values must be provided separately
        if not hasattr(params, 'x_var') or not hasattr(params, 'x_values'):
            raise ValueError("When providing a PrivacyParams object, x_var and x_values must be attributes")
        
        x_var = params.x_var
        y_var = params.y_var
        x_values = params.x_values

    # Data logic:
    # If save_data is True: always recalculate and save
    # If save_data is False: try to read existing data, if not exists - recalculate but don't save
    if save_data:
        print(f"Computing data for {experiment_name}")
        data = calc_experiment_data(params, config, methods, x_var, x_values, y_var)
        save_experiment_data(data, methods, data_file)
    else:
        if os.path.exists(data_file):
            print(f"Reading data from {data_file}")
            data = pd.read_csv(data_file)
        else:
            print(f"Computing data for {experiment_name}")
            data = calc_experiment_data(params, config, methods, x_var, x_values, y_var)
    
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