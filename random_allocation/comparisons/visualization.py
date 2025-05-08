# Standard library imports
from typing import Any, Dict, List, Callable, Union, Optional, Tuple, TypeVar, cast, Literal

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib.axes import Axes

# Local application imports
from random_allocation.comparisons.definitions import *

# Type aliases
DataDict = Dict[str, Any]
FormatterFunc = Callable[[float, int], str]
AxisScale = Literal['linear', 'log']

def plot_comparison(data: DataDict, 
                log_x_axis: bool = False, 
                log_y_axis: bool = False, 
                format_x: FormatterFunc = lambda x, _: f'{x:.2f}', 
                format_y: FormatterFunc = lambda x, _: f'{x:.2f}', 
                figsize: Tuple[int, int] = (16, 9)) -> Figure:
    """
    Create a comparison plot and return the figure.
    
    Args:
        data: Dictionary containing the data to plot
        log_x_axis: Whether to use logarithmic scale for x-axis
        log_y_axis: Whether to use logarithmic scale for y-axis
        format_x: Function to format x-axis labels
        format_y: Function to format y-axis labels
        figsize: Size of the figure
        
    Returns:
        The created matplotlib figure
    """
    methods: List[str] = list(data['y data'].keys())
    #remove keys that end with '- std'
    filtered_methods: List[str] = [method for method in methods if not method.endswith('- std')]
    methods_data = data['y data']
    legend_map = get_features_for_methods(filtered_methods, 'legend')
    markers_map = get_features_for_methods(filtered_methods, 'marker')
    colors_map = get_features_for_methods(filtered_methods, 'color')
    legend_prefix: str = '$\\varepsilon' if data['y name'] == names_dict[EPSILON] else '$\\delta'
    fig: Figure = plt.figure(figsize=figsize)
    for method in filtered_methods:
        plt.plot(data['x data'], methods_data[method], label=legend_prefix+legend_map[method], marker=markers_map[method], color=colors_map[method], linewidth=2.5, markersize=12, alpha=0.8)
        if method + '- std' in methods:
            plt.fill_between(data['x data'], np.clip(methods_data[method] - methods_data[method + '- std'], 0, 1),  np.clip(methods_data[method] + methods_data[method + '- std'], 0, 1), color=colors_map[method], alpha=0.1)
    plt.xlabel(data['x name'], fontsize=20)
    plt.ylabel(data['y name'], fontsize=20)
    #compute the maximum y value over all methods
    none_inf_min = lambda arr: np.min(arr[np.isfinite(arr)])
    min_y_val: float = np.min([none_inf_min(methods_data[method]) for method in filtered_methods if method in methods_data], axis=0)
    none_inf_max = lambda arr: np.max(arr[np.isfinite(arr)])
    max_y_val: float = np.max([none_inf_max(methods_data[method]) for method in filtered_methods if method in methods_data], axis=0)
    if data['y name'] == names_dict[EPSILON]:
        plt.ylim(max(0, min_y_val * 0.9), min(max_y_val * 1.1, 100))
    elif data['y name'] == names_dict[DELTA]:
        plt.ylim(max(0, min_y_val * 0.9), min(max_y_val * 1.1, 1))
    if log_x_axis:
        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    else:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_y))
    if log_y_axis:
        plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(data['x data'])
    plt.legend(fontsize=20)
    return fig

def plot_as_table(data: DataDict) -> DataFrame:
    """
    Create a pandas DataFrame table from plot data.
    
    Args:
        data: Dictionary containing the data to tabulate
        
    Returns:
        DataFrame containing the tabulated data
    """
    methods: List[str] = list(data['y data'].keys())
    methods_data = data['y data']
    table: DataFrame = pd.DataFrame(methods_data, index=data['x data'])
    table.index.name = data['x name']
    table.columns = [method for method in methods]
    return table

def plot_combined_data(data: DataDict, 
                      log_x_axis: bool = False, 
                      log_y_axis: bool = False, 
                      format_x: FormatterFunc = lambda x, _: f'{x:.2f}', 
                      format_y: FormatterFunc = lambda x, _: f'{x:.2f}', 
                      figsize: Tuple[int, int] = (16, 9)) -> Figure:
    """
    Create a combined data plot and return the figure.
    
    Args:
        data: Dictionary containing the data to plot
        log_x_axis: Whether to use logarithmic scale for x-axis
        log_y_axis: Whether to use logarithmic scale for y-axis
        format_x: Function to format x-axis labels
        format_y: Function to format x-axis labels
        figsize: Size of the figure
        
    Returns:
        The created matplotlib figure
    """
    methods: List[str] = list(data['y data'].keys())
    min_allocation: np.ndarray = np.ones_like(data['x data'])*10000
    if ALLOCATION_ANALYTIC in methods:
        min_allocation = np.min([min_allocation, data['y data'][ALLOCATION_ANALYTIC]], axis=0)
    if ALLOCATION_DIRECT in methods:
        min_allocation = np.min([min_allocation, data['y data'][ALLOCATION_DIRECT]], axis=0)
    if ALLOCATION_DECOMPOSITION in methods:
        min_allocation = np.min([min_allocation, data['y data'][ALLOCATION_DECOMPOSITION]], axis=0)
    if ALLOCATION_RECURSIVE in methods:
        min_allocation = np.min([min_allocation, data['y data'][ALLOCATION_RECURSIVE]], axis=0)
    methods_data = data['y data']
    legend_map = get_features_for_methods(methods, 'legend')
    markers_map = get_features_for_methods(methods, 'marker')
    colors_map = get_features_for_methods(methods, 'color')
    legend_prefix: str = '$\\varepsilon' if data['y name'] == names_dict[EPSILON] else '$\\delta'
    fig: Figure = plt.figure(figsize=figsize)
    ax: Axes = fig.add_subplot(111)
    
    for method in methods:
        linewidth: float = 1 if (method == ALLOCATION_DECOMPOSITION or method == ALLOCATION_DIRECT or method == ALLOCATION_ANALYTIC
                             or method == ALLOCATION_RECURSIVE) else 2
        linestyle: str = 'dotted' if (method == ALLOCATION_DECOMPOSITION or method == ALLOCATION_DIRECT or method == ALLOCATION_ANALYTIC
                             or method == ALLOCATION_RECURSIVE) else 'solid'
        ax.plot(data['x data'], methods_data[method], label=legend_prefix+legend_map[method], marker=markers_map[method], 
               color=colors_map[method], linewidth=linewidth, linestyle=linestyle, markersize=10, alpha=0.8)
    
    ax.plot(data['x data'], min_allocation, label='_{\\mathcal{A}}$ - (Our - Combined)', color=colors_dict[ALLOCATION], linewidth=2, alpha=1)
    ax.set_xlabel(data['x name'], fontsize=20)
    ax.set_ylabel(data['y name'], fontsize=20)
    
    # Compute the max of arr where arr is not inf
    none_inf_min = lambda arr: np.min(arr[np.isfinite(arr)])
    min_y_val: float = np.min([none_inf_min(methods_data[method]) for method in methods if method in methods_data], axis=0)
    none_inf_max = lambda arr: np.max(arr[np.isfinite(arr)])
    max_y_val: float = np.max([none_inf_max(methods_data[method]) for method in methods if method in methods_data], axis=0)
    
    if data['y name'] == names_dict[EPSILON]:
        ax.set_ylim(max(0, min_y_val * 0.9), min(max_y_val * 1.1, 100))
    elif data['y name'] == names_dict[DELTA]:
        ax.set_ylim(max(0, min_y_val * 0.9), min(max_y_val * 1.1, 1))
    
    if log_x_axis:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    
    if log_y_axis:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks(data['x data'])
    ax.legend(fontsize=20, loc='lower left', framealpha=0.)
    
    return fig