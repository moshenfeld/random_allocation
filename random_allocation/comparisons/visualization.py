# Standard library imports
from typing import Any, Dict, List, Callable, Union, Optional, Tuple, TypeVar, cast, Literal
import math

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

def clean_log_axis_ticks(
    ax: Axes, 
    data_values: np.ndarray, 
    formatter: FormatterFunc, 
    axis: str = 'x'
) -> None:
    """
    Clean up logarithmic axis to show only data point ticks.
    
    Args:
        ax: The matplotlib axes to modify
        data_values: The data values to use for tick positions
        formatter: Function to format tick labels
        axis: Which axis to clean ('x' or 'y')
    """
    # Get the axis object
    axis_obj = ax.xaxis if axis == 'x' else ax.yaxis
    
    # Turn off minor ticks
    ax.minorticks_off()
    
    # Disable automatic formatters and locators for the specified axis
    axis_obj.set_major_formatter(plt.NullFormatter())
    axis_obj.set_major_locator(plt.NullLocator())
    
    # Add only our desired ticks
    if axis == 'x':
        ax.set_xticks(data_values)
        ax.set_xticklabels([formatter(x, 0) for x in data_values])
    else:  # axis == 'y'
        ax.set_yticks(data_values)
        ax.set_yticklabels([formatter(y, 0) for y in data_values])

def find_optimal_legend_position(
    ax: Axes,
    data: DataDict,
    filtered_methods: List[str],
    methods_data: Dict[str, Any],
    user_position: Optional[str] = None
) -> str:
    """
    Find the optimal position for the legend based on data distribution.
    
    Args:
        ax: The matplotlib axes
        data: Data dictionary containing plot data
        filtered_methods: List of methods to include
        methods_data: Dictionary mapping methods to their data
        user_position: Optional user-specified position
        
    Returns:
        Best legend position as a string ('best', 'upper right', etc.)
    """
    # If user specified a position, use that
    if user_position:
        return user_position
    
    # Get x and y data
    x_data = data['x data']
    
    # Calculate data density in different regions
    # We'll divide the plot into four quadrants and count points in each
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    
    x_mid = (x_lim[0] + x_lim[1]) / 2
    y_mid = (y_lim[0] + y_lim[1]) / 2
    
    # Count points in each quadrant
    lower_left_count = 0
    lower_right_count = 0
    upper_left_count = 0
    upper_right_count = 0
    
    for method in filtered_methods:
        if method not in methods_data:
            continue
            
        y_data = methods_data[method]
        
        # Skip methods with std suffix
        if method.endswith('- std'):
            continue
            
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            # Skip non-finite values
            if not np.isfinite(y):
                continue
                
            if x <= x_mid and y <= y_mid:
                lower_left_count += 1
            elif x > x_mid and y <= y_mid:
                lower_right_count += 1
            elif x <= x_mid and y > y_mid:
                upper_left_count += 1
            else:
                upper_right_count += 1
    
    # Find quadrant with least points
    counts = {
        'lower left': lower_left_count,
        'lower right': lower_right_count,
        'upper left': upper_left_count,
        'upper right': upper_right_count
    }
    
    min_count = min(counts.values())
    
    # Special case: If the counts are all close, prefer upper right
    if all(count <= min_count + 2 for count in counts.values()):
        return 'upper right'
    
    # Map quadrants to matplotlib legend positions
    position_map = {
        'lower left': 'lower left',
        'lower right': 'lower right',
        'upper left': 'upper left',
        'upper right': 'upper right'
    }
    
    # Find the quadrant with minimal data points
    for position, count in counts.items():
        if count == min_count:
            return position_map[position]
    
    # Default fallback
    return 'best'

def setup_plot_axes(
    ax: Axes,
    data: DataDict,
    filtered_methods: List[str],
    methods_data: Dict[str, Any],
    log_x_axis: bool = False,
    log_y_axis: bool = False,
    format_x: FormatterFunc = lambda x, _: f'{x:.2f}',
    format_y: FormatterFunc = lambda x, _: f'{x:.2f}',
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    title: Optional[str] = None,
    title_fontsize: int = 16
) -> None:
    """
    Set up common axis properties for plots.
    
    Args:
        ax: The matplotlib axes to configure
        data: Data dictionary containing plot data
        filtered_methods: List of methods to include (without '- std' entries)
        methods_data: Dictionary mapping methods to their data
        log_x_axis: Whether to use logarithmic scale for x-axis
        log_y_axis: Whether to use logarithmic scale for y-axis
        format_x: Function to format x-axis labels
        format_y: Function to format x-axis labels
        xlabel_fontsize: Font size for x-axis label
        ylabel_fontsize: Font size for y-axis label
        title: Optional title for the plot
        title_fontsize: Font size for plot title
    """
    # Set axis labels
    ax.set_xlabel(data['x name'], fontsize=xlabel_fontsize)
    ax.set_ylabel(data['y name'], fontsize=ylabel_fontsize)
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    
    # Compute min/max y values for setting limits
    none_inf_min = lambda arr: np.min(arr[np.isfinite(arr)])
    min_y_val: float = np.min([none_inf_min(methods_data[method]) 
                             for method in filtered_methods if method in methods_data], axis=0)
                             
    none_inf_max = lambda arr: np.max(arr[np.isfinite(arr)])
    max_y_val: float = np.max([none_inf_max(methods_data[method]) 
                             for method in filtered_methods if method in methods_data], axis=0)
    
    # Set y-axis limits based on data type
    if data['y name'] == names_dict[EPSILON]:
        ax.set_ylim(max(0, min_y_val * 0.9), min(max_y_val * 1.1, 100))
    elif data['y name'] == names_dict[DELTA]:
        ax.set_ylim(max(0, min_y_val * 0.9), min(max_y_val * 1.1, 1))
    
    # Set axis scales and formatters
    if log_x_axis:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
        # Clean up x-axis log scale ticks
        clean_log_axis_ticks(ax, data['x data'], format_x, 'x')
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    
    if log_y_axis:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
        # Clean up y-axis log scale ticks if needed
        if 'y data' in data and filtered_methods:
            # For y-axis, we need to extract unique y values across all methods
            y_values = []
            for method in filtered_methods:
                if method in methods_data:
                    y_values.extend(methods_data[method])
            y_values = np.unique(np.array(y_values)[np.isfinite(y_values)])
            clean_log_axis_ticks(ax, y_values, format_y, 'y')
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xticks(data['x data'])

def plot_comparison(data: DataDict, 
                log_x_axis: bool = False, 
                log_y_axis: bool = False, 
                format_x: FormatterFunc = lambda x, _: f'{x:.2f}', 
                format_y: FormatterFunc = lambda x, _: f'{x:.2f}', 
                figsize: Tuple[int, int] = (16, 9),
                legend_position: Optional[str] = None) -> Figure:
    """
    Create a comparison plot and return the figure.
    
    Args:
        data: Dictionary containing the data to plot
        log_x_axis: Whether to use logarithmic scale for x-axis
        log_y_axis: Whether to use logarithmic scale for y-axis
        format_x: Function to format x-axis labels
        format_y: Function to format x-axis labels
        figsize: Size of the figure
        legend_position: Optional custom position for the legend
        
    Returns:
        The created matplotlib figure
    """
    methods: List[str] = list(data['y data'].keys())
    # Remove keys that end with '- std'
    filtered_methods: List[str] = [method for method in methods if not method.endswith('- std')]
    methods_data = data['y data']
    legend_map = get_features_for_methods(filtered_methods, 'legend')
    markers_map = get_features_for_methods(filtered_methods, 'marker')
    colors_map = get_features_for_methods(filtered_methods, 'color')
    legend_prefix: str = '$\\varepsilon' if data['y name'] == names_dict[EPSILON] else '$\\delta'
    
    fig: Figure = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Plot each method
    for method in filtered_methods:
        legend_value = legend_map.get(method, "")
        if legend_value is None:
            legend_value = ""
        plt.plot(data['x data'], methods_data[method], 
               label=legend_prefix + str(legend_value), 
               marker=markers_map[method], 
               color=colors_map[method], 
               linewidth=2.5, 
               markersize=12, 
               alpha=0.8)
        
        # Add error bars if available
        if method + '- std' in methods:
            plt.fill_between(
                data['x data'], 
                np.clip(methods_data[method] - methods_data[method + '- std'], 0, 1),  
                np.clip(methods_data[method] + methods_data[method + '- std'], 0, 1), 
                color=colors_map[method], 
                alpha=0.1
            )
    
    # Use the common axis setup function
    setup_plot_axes(
        ax=ax,
        data=data,
        filtered_methods=filtered_methods,
        methods_data=methods_data,
        log_x_axis=log_x_axis,
        log_y_axis=log_y_axis,
        format_x=format_x,
        format_y=format_y,
        xlabel_fontsize=20,
        ylabel_fontsize=20
    )
    
    # Find optimal legend position if not specified
    legend_pos = find_optimal_legend_position(ax, data, filtered_methods, methods_data, legend_position)
    
    # Set legend with optimal position and slightly transparent background
    plt.legend(fontsize=20, loc=legend_pos, framealpha=0.7)
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
                      figsize: Tuple[int, int] = (16, 9),
                      legend_position: Optional[str] = None) -> Figure:
    """
    Create a combined data plot and return the figure.
    
    Args:
        data: Dictionary containing the data to plot
        log_x_axis: Whether to use logarithmic scale for x-axis
        log_y_axis: Whether to use logarithmic scale for y-axis
        format_x: Function to format x-axis labels
        format_y: Function to format x-axis labels
        figsize: Size of the figure
        legend_position: Optional custom position for the legend
        
    Returns:
        The created matplotlib figure
    """
    methods: List[str] = list(data['y data'].keys())
    # Remove keys that end with '- std'
    filtered_methods: List[str] = [method for method in methods if not method.endswith('- std')]
    methods_data = data['y data']
    
    # Calculate min allocation
    min_allocation: np.ndarray = np.ones_like(data['x data'])*10000
    if ALLOCATION_ANALYTIC in filtered_methods:
        min_allocation = np.min([min_allocation, data['y data'][ALLOCATION_ANALYTIC]], axis=0)
    if ALLOCATION_DIRECT in filtered_methods:
        min_allocation = np.min([min_allocation, data['y data'][ALLOCATION_DIRECT]], axis=0)
    if ALLOCATION_DECOMPOSITION in filtered_methods:
        min_allocation = np.min([min_allocation, data['y data'][ALLOCATION_DECOMPOSITION]], axis=0)
    if ALLOCATION_RECURSIVE in filtered_methods:
        min_allocation = np.min([min_allocation, data['y data'][ALLOCATION_RECURSIVE]], axis=0)
    
    legend_map = get_features_for_methods(filtered_methods, 'legend')
    markers_map = get_features_for_methods(filtered_methods, 'marker')
    colors_map = get_features_for_methods(filtered_methods, 'color')
    legend_prefix: str = '$\\varepsilon' if data['y name'] == names_dict[EPSILON] else '$\\delta'
    
    # Create the figure and axis
    fig: Figure = plt.figure(figsize=figsize)
    ax: Axes = fig.add_subplot(111)
    
    # Plot each method
    for method in filtered_methods:
        legend_value = legend_map.get(method, "")
        if legend_value is None:
            legend_value = ""
            
        linewidth: float = 1 if (method == ALLOCATION_DECOMPOSITION or 
                              method == ALLOCATION_DIRECT or 
                              method == ALLOCATION_ANALYTIC or 
                              method == ALLOCATION_RECURSIVE) else 2
                              
        linestyle: str = 'dotted' if (method == ALLOCATION_DECOMPOSITION or 
                                   method == ALLOCATION_DIRECT or 
                                   method == ALLOCATION_ANALYTIC or 
                                   method == ALLOCATION_RECURSIVE) else 'solid'
        
        ax.plot(data['x data'], methods_data[method], 
               label=legend_prefix + str(legend_value), 
               marker=markers_map[method], 
               color=colors_map[method], 
               linewidth=linewidth, 
               linestyle=linestyle, 
               markersize=10, 
               alpha=0.8)
    
    # Plot combined allocation
    ax.plot(data['x data'], min_allocation, 
           label='_{\\mathcal{A}}$ - (Our - Combined)', 
           color=colors_dict[ALLOCATION], 
           linewidth=2, 
           alpha=1)
    
    # Use the common axis setup function
    setup_plot_axes(
        ax=ax,
        data=data,
        filtered_methods=filtered_methods,
        methods_data=methods_data,
        log_x_axis=log_x_axis,
        log_y_axis=log_y_axis,
        format_x=format_x,
        format_y=format_y,
        xlabel_fontsize=20,
        ylabel_fontsize=20
    )
    
    # Find optimal legend position if not specified
    legend_pos = find_optimal_legend_position(ax, data, filtered_methods, methods_data, legend_position)
    
    # Set legend with optimal position and slightly transparent background
    ax.legend(fontsize=20, loc=legend_pos, framealpha=0.7)
    return fig

def plot_multiple_data(data_list: List[DataDict],
                      titles: Optional[List[str]] = None,
                      log_x_axis: bool = False,
                      log_y_axis: bool = False,
                      format_x: FormatterFunc = lambda x, _: f'{x:.2f}',
                      format_y: FormatterFunc = lambda x, _: f'{x:.2f}',
                      figsize: Tuple[int, int] = (20, 16),
                      plot_type: str = 'comparison',
                      grid_layout: Optional[Tuple[int, int]] = None,
                      legend_position: Optional[str] = None) -> Figure:
    """
    Create a grid of subplots for multiple data dictionaries.
    
    Args:
        data_list: List of dictionaries containing data to plot
        titles: Optional list of titles for each subplot (defaults to data['title'] if available)
        log_x_axis: Whether to use logarithmic scale for x-axis
        log_y_axis: Whether to use logarithmic scale for y-axis
        format_x: Function to format x-axis labels
        format_y: Function to format x-axis labels
        figsize: Size of the figure
        plot_type: Type of plot to create ('comparison' or 'combined')
        grid_layout: Optional tuple specifying grid dimensions (rows, cols)
                    If not provided, it will be automatically determined
        legend_position: Optional custom position for the legend
        
    Returns:
        The created matplotlib figure with a grid of subplots
    """
    n_plots = len(data_list)
    
    # Determine grid layout if not provided
    if grid_layout is None:
        n_cols = min(3, n_plots)  # Maximum 3 columns
        n_rows = math.ceil(n_plots / n_cols)
        grid_layout = (n_rows, n_cols)
    else:
        n_rows, n_cols = grid_layout
    
    # Create figure and a grid of subplots
    fig: Figure = plt.figure(figsize=figsize)
    
    # For each data dict in the list
    for idx, data in enumerate(data_list):
        if idx >= n_rows * n_cols:
            print(f"Warning: Only displaying {n_rows * n_cols} of {n_plots} plots due to grid layout limitations.")
            break
        
        # Get methods and data
        methods: List[str] = list(data['y data'].keys())
        # Remove keys that end with '- std'
        filtered_methods: List[str] = [method for method in methods if not method.endswith('- std')]
        methods_data = data['y data']
        
        # Create subplot
        ax: Axes = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        # Get necessary method features
        legend_map = get_features_for_methods(filtered_methods, 'legend')
        markers_map = get_features_for_methods(filtered_methods, 'marker')
        colors_map = get_features_for_methods(filtered_methods, 'color')
        legend_prefix: str = '$\\varepsilon' if data['y name'] == names_dict[EPSILON] else '$\\delta'
        
        # Plot based on type
        if plot_type == 'combined':
            # Special handling for combined plot type
            min_allocation: np.ndarray = np.ones_like(data['x data']) * 10000
            if ALLOCATION_ANALYTIC in filtered_methods:
                min_allocation = np.min([min_allocation, methods_data[ALLOCATION_ANALYTIC]], axis=0)
            if ALLOCATION_DIRECT in filtered_methods:
                min_allocation = np.min([min_allocation, methods_data[ALLOCATION_DIRECT]], axis=0)
            if ALLOCATION_DECOMPOSITION in filtered_methods:
                min_allocation = np.min([min_allocation, methods_data[ALLOCATION_DECOMPOSITION]], axis=0)
            if ALLOCATION_RECURSIVE in filtered_methods:
                min_allocation = np.min([min_allocation, methods_data[ALLOCATION_RECURSIVE]], axis=0)
            
            # Plot each method
            for method in filtered_methods:
                legend_value = legend_map.get(method, "")
                if legend_value is None:
                    legend_value = ""
                    
                linewidth: float = 1 if (method == ALLOCATION_DECOMPOSITION or 
                                        method == ALLOCATION_DIRECT or 
                                        method == ALLOCATION_ANALYTIC or 
                                        method == ALLOCATION_RECURSIVE) else 2
                                        
                linestyle: str = 'dotted' if (method == ALLOCATION_DECOMPOSITION or 
                                            method == ALLOCATION_DIRECT or 
                                            method == ALLOCATION_ANALYTIC or 
                                            method == ALLOCATION_RECURSIVE) else 'solid'
                                            
                ax.plot(data['x data'], methods_data[method], 
                      label=legend_prefix + str(legend_value), 
                      marker=markers_map[method], 
                      color=colors_map[method], 
                      linewidth=linewidth, 
                      linestyle=linestyle, 
                      markersize=6, 
                      alpha=0.8)
            
            # Plot combined allocation
            ax.plot(data['x data'], min_allocation, 
                  label='_{\\mathcal{A}}$ - (Our - Combined)', 
                  color=colors_dict[ALLOCATION], 
                  linewidth=2, 
                  alpha=1)
        else:
            # Standard comparison plot
            for method in filtered_methods:
                legend_value = legend_map.get(method, "")
                if legend_value is None:
                    legend_value = ""
                    
                ax.plot(data['x data'], methods_data[method], 
                      label=legend_prefix + str(legend_value), 
                      marker=markers_map[method], 
                      color=colors_map[method], 
                      linewidth=2, 
                      markersize=6, 
                      alpha=0.8)
                      
                # Add std deviation if available
                if method + '- std' in methods:
                    ax.fill_between(
                        data['x data'], 
                        np.clip(methods_data[method] - methods_data[method + '- std'], 0, 1),
                        np.clip(methods_data[method] + methods_data[method + '- std'], 0, 1), 
                        color=colors_map[method], 
                        alpha=0.1
                    )
        
        # Set axis labels and scales using the common function
        setup_plot_axes(
            ax=ax,
            data=data,
            filtered_methods=filtered_methods,
            methods_data=methods_data,
            log_x_axis=log_x_axis,
            log_y_axis=log_y_axis,
            format_x=format_x,
            format_y=format_y,
            xlabel_fontsize=14,
            ylabel_fontsize=14,
            title=titles[idx] if titles and idx < len(titles) else data.get('title'),
            title_fontsize=16
        )
        
        # Set legend
        optimal_legend_position = find_optimal_legend_position(ax, data, filtered_methods, methods_data, legend_position)
        ax.legend(fontsize=12, loc=optimal_legend_position, framealpha=0.7)
    
    # Adjust layout to prevent clipping and overlap
    fig.tight_layout()
    
    return fig