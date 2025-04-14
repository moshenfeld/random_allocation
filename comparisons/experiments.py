from typing import Dict, Any, Callable, List, Tuple
import inspect
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from comparisons.definitions import EPSILON, DELTA, VARIABLES, methods_dict, names_dict

def get_features_for_methods(method_keys: List[str], feature_name: str) -> Dict[str, Any]:
    """
    Extract a specific feature for a list of methods using the global methods_dict.
    """
    if not all(key in methods_dict for key in method_keys):
        invalid_keys = [key for key in method_keys if key not in methods_dict]
        raise KeyError(f"Invalid method keys: {invalid_keys}")
    if not hasattr(methods_dict[method_keys[0]], feature_name):
        raise AttributeError(f"Invalid feature name: {feature_name}")
    return {key: getattr(methods_dict[key], feature_name) for key in method_keys}

def match_function_args(params_dict: Dict[str, Any],
                        config_dict: Dict[str, Any],
                        func: Callable,
                        x_var: str,
                        ) -> List[Dict[str, Any]]:
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
    x_var = params_dict['x_var']
    if x_var not in params_dict.keys():
        raise ValueError(f"{x_var} was defined as the x-axis variable but does not appear in the params_dict.")
    y_var = params_dict['y_var']
    if y_var == x_var:
        raise ValueError(f"{x_var} was chosen as both the x-axis and y-axis variable.")
    return x_var, y_var

def get_main_var(params_dict: Dict[str, Any]) -> str:
    main_var = params_dict['main_var']
    if main_var not in params_dict.keys():
        raise ValueError(f"{main_var} was defined as the main variable but does not appear in the params_dict.")
    x_var = params_dict['x_var']
    if main_var == x_var:
        raise ValueError(f"{main_var} was chosen as both the main and x-axis variable.")
    y_var = params_dict['y_var']
    if main_var == y_var:
        raise ValueError(f"{main_var} was chosen as both the main and y-axis variable.")
    return main_var

def get_func_dict(methods: list[str],
                  y_var: str
                  ) -> Dict[str, Any]:
    if y_var == EPSILON:
        return get_features_for_methods(methods, 'epsilon_calculator')
    if y_var == DELTA:
        return get_features_for_methods(methods, 'delta_calculator')
    raise ValueError(f"Invalid y_var: {y_var}")

def calc_params_inner(params_dict: Dict[str, Any],
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
    return data

def save_experiment_data(data: Dict[str, Any], methods: List[str], experiment_name: str) -> None:
    """
    Save experiment data as a CSV file.
    
    Args:
        data: The experiment data dictionary
        methods: List of methods used in the experiment
        experiment_name: Name of the experiment for the output file
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create DataFrame
    df_data = {'x': data['x data']}
    for method in methods:
        df_data[method] = data['y data'][method]
        if method + '- std' in data['y data']:
            df_data[method + '_std'] = data['y data'][method + '- std']
    
    df = pd.DataFrame(df_data)
    df.to_csv(f'data/{experiment_name}_data.csv', index=False)

def save_experiment_plot(data: Dict[str, Any], methods: List[str], experiment_name: str) -> None:
    """
    Save experiment plot as a PNG file.
    
    Args:
        data: The experiment data dictionary
        methods: List of methods used in the experiment
        experiment_name: Name of the experiment for the output file
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
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
    plt.savefig(f'plots/{experiment_name}_plot.png')
    plt.close()

def calc_params(params_dict: Dict[str, Any],
                config_dict: Dict[str, Any],
                methods: list[str],
                save_data: bool = False,
                save_plots: bool = False,
                experiment_name: str = None,
                )-> Dict[str, Any]:
    x_var, y_var = get_x_y_vars(params_dict)
    data = calc_params_inner(params_dict, config_dict, methods)
    data['x name'] = names_dict[x_var]
    data['y name'] = names_dict[y_var]
    data['x data'] = params_dict[x_var]
    data['title'] = f"{names_dict[y_var]} as a function of {names_dict[x_var]} \n"
    for var in VARIABLES:
        if var != x_var and var != y_var:
            data[var] = params_dict[var]
            data['title'] += f"{names_dict[var]} = {params_dict[var]}, "
    
    # Save data and plots if requested
    if experiment_name:
        if save_data:
            save_experiment_data(data, methods, experiment_name)
        if save_plots:
            save_experiment_plot(data, methods, experiment_name)
    
    return data