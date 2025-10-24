# Standard library imports
from typing import Any, Dict, List, Callable, Optional, Tuple, Literal
import math

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib.axes import Axes
import matplotlib as mpl
from matplotlib.ticker import NullFormatter, NullLocator, FuncFormatter

# Local application imports
from random_allocation.comparisons.definitions import *

# Ensure LaTeX is not required for mathtext rendering
mpl.rcParams['text.usetex'] = False

# Type aliases
DataDict = Dict[str, Any]
FormatterFunc = Callable[[float, int], str]
AxisScale = Literal['linear', 'log']
PlotResult = Tuple[Figure, Axes]

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
    axis_obj.set_major_formatter(NullFormatter())
    axis_obj.set_major_locator(NullLocator())
    
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
    
    # Check if axes are using logarithmic scales
    x_scale = ax.get_xscale()
    y_scale = ax.get_yscale()
    
    # Determine midpoints based on scale type
    if x_scale == 'log':
        # For log scale, use geometric mean if bounds are positive
        if x_lim[0] > 0 and x_lim[1] > 0:
            x_mid = np.sqrt(x_lim[0] * x_lim[1])
        else:
            # Fall back to arithmetic mean if we have non-positive values
            x_mid = (x_lim[0] + x_lim[1]) / 2
    else:
        # For linear scale, use arithmetic mean
        x_mid = (x_lim[0] + x_lim[1]) / 2
    
    if y_scale == 'log':
        # For log scale, use geometric mean if bounds are positive
        if y_lim[0] > 0 and y_lim[1] > 0:
            y_mid = np.sqrt(y_lim[0] * y_lim[1])
        else:
            # Fall back to arithmetic mean if we have non-positive values
            y_mid = (y_lim[0] + y_lim[1]) / 2
    else:
        # For linear scale, use arithmetic mean
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
    title_fontsize: int = 16,
    num_y_ticks: Optional[int] = None,
    tick_labelsize: int = 14
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
    ax.set_xlabel(f"${data['x name']}$", fontsize=xlabel_fontsize)
    ax.set_ylabel(f"${data['y name']}$", fontsize=ylabel_fontsize, rotation=0)
    
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
        ax.xaxis.set_major_formatter(FuncFormatter(format_x))
        # Clean up x-axis log scale ticks
        clean_log_axis_ticks(ax, data['x data'], format_x, 'x')
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(format_x))
    
    if log_y_axis:
        ax.set_yscale('log')
        # Use a concise formatter for log scale y-axis
        if data['y name'] == names_dict[EPSILON] or data['y name'] == names_dict[DELTA]:
            # For epsilon and delta, use scientific notation for small values
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda y, _: f'{y:.1e}' if y < 0.01 else f'{y:.2f}'
            ))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(format_y))
        
        # Optionally limit number of ticks on y-axis
        if num_y_ticks is not None:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=num_y_ticks))
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    ax.set_xticks(data['x data'])

def plot_data_lines(
    ax: Axes,
    data: DataDict,
    filtered_methods: List[str],
    methods_data: Dict[str, Any],
    legend_map: Dict[str, Any],
    markers_map: Dict[str, Any],
    colors_map: Dict[str, Any],
    legend_prefix: str,
    is_allocation_method: bool = False
) -> None:
    """
    Plot data lines for each method on the given axes.
    
    Args:
        ax: The matplotlib axes to plot on
        data: Data dictionary containing plot data
        filtered_methods: List of methods to include (without '- std' entries)
        methods_data: Dictionary mapping methods to their data
        legend_map: Dictionary mapping methods to their legend text
        markers_map: Dictionary mapping methods to their marker styles
        colors_map: Dictionary mapping methods to their colors
        legend_prefix: Prefix for the legend labels
        is_allocation_method: Whether to apply special styling for allocation methods
    """
    for method in filtered_methods:
        legend_value = legend_map.get(method, "")
        if legend_value is None:
            legend_value = ""
            
        # Determine line style based on method type
        if is_allocation_method:
            linewidth: float = 1 if (method == ALLOCATION_DECOMPOSITION or 
                                  method == ALLOCATION_DIRECT or 
                                  method == ALLOCATION_ANALYTIC or 
                                  method == ALLOCATION_RECURSIVE) else 2
                                  
            linestyle: str = 'dotted' if (method == ALLOCATION_DECOMPOSITION or 
                                       method == ALLOCATION_DIRECT or 
                                       method == ALLOCATION_ANALYTIC or 
                                       method == ALLOCATION_RECURSIVE) else 'solid'
        else:
            linewidth: float = 2.5
            linestyle: str = 'solid'
            
        # Adjust marker size based on plot type
        markersize = 10 if is_allocation_method else 12
            
        legend_label = str(legend_value)
        # Only prepend legend_prefix if legend_value does not already start with epsilon
        if not legend_label.strip().startswith('$\\varepsilon'):
            legend_label = legend_prefix + legend_label
        ax.plot(data['x data'], methods_data[method], 
               label=legend_label, 
               marker=markers_map[method], 
               color=colors_map[method], 
               linewidth=linewidth, 
               linestyle=linestyle, 
               markersize=markersize, 
               alpha=0.8)

def plot_error_bars(
    ax: Axes,
    data: DataDict,
    filtered_methods: List[str],
    methods_data: Dict[str, Any],
    colors_map: Dict[str, Any]
) -> None:
    """
    Plot error bars (standard deviation) for each method if available.
    
    Args:
        ax: The matplotlib axes to plot on
        data: Data dictionary containing plot data
        filtered_methods: List of methods to include (without '- std' entries)
        methods_data: Dictionary mapping methods to their data
        colors_map: Dictionary mapping methods to their colors
    """
    for method in filtered_methods:
        # Add error bars if available (method + '- std' exists)
        if method + '- std' in methods_data:
            ax.fill_between(
                data['x data'], 
                np.clip(methods_data[method] - methods_data[method + '- std'], 0, 1),  
                np.clip(methods_data[method] + methods_data[method + '- std'], 0, 1), 
                color=colors_map[method], 
                alpha=0.1
            )

def calculate_min_allocation(
    data: DataDict,
    filtered_methods: List[str],
    methods_data: Dict[str, Any]
) -> np.ndarray:
    """
    Calculate the minimum allocation across specified allocation methods.
    
    Args:
        data: Data dictionary containing plot data
        filtered_methods: List of methods to include
        methods_data: Dictionary mapping methods to their data
        
    Returns:
        NumPy array containing the min allocation values
    """
    min_allocation: np.ndarray = np.ones_like(data['x data']) * 10000
    
    # Check each allocation method
    allocation_methods = [
        ALLOCATION_ANALYTIC,
        ALLOCATION_DIRECT,
        ALLOCATION_DECOMPOSITION,
        ALLOCATION_RECURSIVE
    ]
    
    for method in allocation_methods:
        if method in filtered_methods:
            min_allocation = np.min([min_allocation, methods_data[method]], axis=0)
            
    return min_allocation

def plot_min_allocation(
    ax: Axes,
    data: DataDict,
    min_allocation: np.ndarray
) -> None:
    """
    Plot the minimum allocation line.
    
    Args:
        ax: The matplotlib axes to plot on
        data: Data dictionary containing plot data
        min_allocation: NumPy array containing the min allocation values
    """
    ax.plot(data['x data'], min_allocation, 
           label=r'_{\mathcal{A}}$ - (Our - Combined)', 
           color=colors_dict[ALLOCATION], 
           linewidth=2, 
           alpha=1)

def prepare_plot_data(
    data: DataDict
) -> Tuple[List[str], List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    """
    Prepare common data needed for plotting.
    
    Args:
        data: Data dictionary containing plot data
        
    Returns:
        Tuple containing:
        - List of all methods
        - List of filtered methods (without '- std' entries)
        - Dictionary of methods data
        - Dictionary of legend mappings
        - Dictionary of marker mappings
        - Dictionary of color mappings
        - Legend prefix string
    """
    methods: List[str] = list(data['y data'].keys())
    # Remove keys that end with '- std'
    filtered_methods: List[str] = [method for method in methods if not method.endswith('- std')]
    methods_data = data['y data']
    
    # Get method features
    legend_map = get_features_for_methods(filtered_methods, 'legend')
    markers_map = get_features_for_methods(filtered_methods, 'marker')
    colors_map = get_features_for_methods(filtered_methods, 'color')
    
    # Determine legend prefix based on y-axis data
    legend_prefix: str = f"${names_dict[EPSILON]}$" if data['y name'] == names_dict[EPSILON] else f"${names_dict[DELTA]}$"
    
    return methods, filtered_methods, methods_data, legend_map, markers_map, colors_map, legend_prefix

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
    methods, filtered_methods, methods_data, legend_map, markers_map, colors_map, legend_prefix = prepare_plot_data(data)
    
    fig: Figure = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Plot data lines
    plot_data_lines(ax, data, filtered_methods, methods_data, legend_map, markers_map, colors_map, legend_prefix)
    
    # Plot error bars
    plot_error_bars(ax, data, filtered_methods, methods_data, colors_map)
    
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
        ylabel_fontsize=20,
        num_y_ticks=None
    )
    
    # Find optimal legend position if not specified
    legend_pos = find_optimal_legend_position(ax, data, filtered_methods, methods_data, legend_position)
    
    # Set legend with optimal position and no frame or background
    plt.legend(fontsize=20, loc=legend_pos, frameon=False)
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
    methods, filtered_methods, methods_data, legend_map, markers_map, colors_map, legend_prefix = prepare_plot_data(data)
    
    # Calculate min allocation
    min_allocation = calculate_min_allocation(data, filtered_methods, methods_data)
    
    # Create the figure and axis
    fig: Figure = plt.figure(figsize=figsize)
    ax: Axes = fig.add_subplot(111)
    
    # Plot data lines
    plot_data_lines(ax, data, filtered_methods, methods_data, legend_map, markers_map, colors_map, legend_prefix, is_allocation_method=True)
    
    # Plot combined allocation
    plot_min_allocation(ax, data, min_allocation)
    
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
        ylabel_fontsize=20,
        num_y_ticks=None
    )
    
    # Find optimal legend position if not specified
    legend_pos = find_optimal_legend_position(ax, data, filtered_methods, methods_data, legend_position)
    
    # Set legend with optimal position and no frame or background
    ax.legend(fontsize=20, loc=legend_pos, frameon=False)
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

    # Store handles and labels from the first subplot for shared legend
    handles, labels = None, None

    # For each data dict in the list
    for idx, data in enumerate(data_list):
        if idx >= n_rows * n_cols:
            print(f"Warning: Only displaying {n_rows * n_cols} of {n_plots} plots due to grid layout limitations.")
            break

        methods, filtered_methods, methods_data, legend_map, markers_map, colors_map, legend_prefix = prepare_plot_data(data)

        # Create subplot
        ax: Axes = fig.add_subplot(n_rows, n_cols, idx + 1)

        if plot_type == 'combined':
            # Special handling for combined plot type
            min_allocation = calculate_min_allocation(data, filtered_methods, methods_data)

            # Plot data lines
            plot_data_lines(ax, data, filtered_methods, methods_data, legend_map, markers_map, colors_map, legend_prefix, is_allocation_method=True)

            # Plot combined allocation
            plot_min_allocation(ax, data, min_allocation)
        else:
            # Standard comparison plot
            plot_data_lines(ax, data, filtered_methods, methods_data, legend_map, markers_map, colors_map, legend_prefix)
            plot_error_bars(ax, data, filtered_methods, methods_data, colors_map)

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
            xlabel_fontsize=21,
            ylabel_fontsize=21,
            title=titles[idx] if titles and idx < len(titles) else data.get('title'),
            title_fontsize=24,
            num_y_ticks=None
        )

        # Get legend handles from the first subplot
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # Add shared legend below all subplots
    if handles and labels:
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            fontsize=18,
            frameon=False
        )

    # Adjust layout to prevent clipping and overlap, leaving room for legend
    fig.tight_layout(rect=(0, 0.12, 1, 0.98))

    return fig

def plot_privacy_curves(
    deltas_dict_arr: List[Dict[str, np.ndarray]],
    epsilon_mat: List[np.ndarray],
    subplot_titles: List[str]
) -> Figure:
    num_plots = len(deltas_dict_arr)
    n_rows = (num_plots + 1) // 2
    n_cols = 2
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))

    # Store handles and labels from the first subplot that has data
    handles, labels = None, None

    for i, (deltas_dict, epsilon_arr) in enumerate(zip(deltas_dict_arr, epsilon_mat)):
        plt.subplot(n_rows, n_cols, i + 1)
        for method, deltas in deltas_dict.items():
            plt.plot(epsilon_arr, deltas, label=method)
        plt.title(subplot_titles[i])
        plt.xlabel(f"${names_dict[EPSILON]}$")
        plt.ylabel(f"${names_dict[DELTA]}$")
        # plt.xscale("log")
        plt.yscale("log")

        # Get legend handles from the first subplot with data
        if handles is None:
            current_ax = plt.gca()
            handles, labels = current_ax.get_legend_handles_labels()

    # Add the legend below all the subplots if we have handles
    if handles and labels:
        # First adjust the spacing to make room for the legend
        plt.subplots_adjust(bottom=0.18, hspace=0.5)

        # Now add the legend in the space we created - use a positive y value
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.04),
                   ncol=3, fontsize=24, frameon=True, framealpha=0.9)

        # Final tight layout to ensure proper spacing, leaving room at bottom
        plt.tight_layout(rect=(0, 0.12, 1, 0.95))
    else:
        # If no legend data, just use standard tight layout
        plt.tight_layout()

    return fig

def plot_chau_et_al_epsilon_comparison(
    data_list: List[DataDict],
    titles: List[str],
    visualization_config: Dict[str, Any],
    save_plots: bool = False,
    show_plots: bool = False,
    plots_dir: Optional[str] = None,
    filename: str = 'Chau_et_al_epsilon_plot.png'
) -> Figure:
    """
    Create a single figure with 4 subplots (2x2 grid) for Chau et al. epsilon comparison
    Uses standard colors and markers from methods_dict for consistency
    Matches main plot styling with dotted lines for allocation methods

    Args:
        data_list: List of data dictionaries to plot
        titles: List of subplot titles
        visualization_config: Dictionary with visualization settings
        save_plots: Whether to save the plot
        show_plots: Whether to display the plot
        plots_dir: Directory to save plots (required if save_plots=True)
        filename: Filename for the saved plot

    Returns:
        The created matplotlib figure
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    # Store handles and labels from the first subplot
    handles, labels = None, None

    for idx, (data, title) in enumerate(zip(data_list, titles)):
        ax = axs[idx]

        # Use prepare_plot_data to get the correct color/marker mappings
        methods, filtered_methods, methods_data, legend_map, markers_map, colors_map, legend_prefix = prepare_plot_data(data)

        # Extract plot data
        x_data = data['x data']

        # Plot each method using the same approach as plot_data_lines
        for method in filtered_methods:
            y_data = methods_data[method]

            # Get color and marker from the maps (same as main plot)
            color = colors_map[method]
            marker = markers_map[method]
            legend_label = legend_map.get(method, method)

            # Apply line styling matching main plot behavior
            # Allocation methods (DIRECT, DECOMPOSITION, RECURSIVE, ANALYTIC) get dotted lines and thinner width
            if method in [ALLOCATION_DECOMPOSITION, ALLOCATION_DIRECT, ALLOCATION_ANALYTIC, ALLOCATION_RECURSIVE]:
                linewidth = 1
                linestyle = 'dotted'
                markersize = 10
            else:
                # Non-allocation methods (LOCAL, POISSON, LOWER_BOUND) get solid lines and thicker width
                linewidth = 2
                linestyle = 'solid'
                markersize = 10

            ax.plot(
                x_data,
                y_data,
                label=legend_label,
                color=color,
                marker=marker,
                linewidth=linewidth,
                linestyle=linestyle,
                markersize=markersize,
                alpha=0.8
            )

        # Axis formatting
        ax.set_xlabel(f"${data['x name']}$", fontsize=21)
        ax.set_ylabel(f"${data['y name']}$", fontsize=21, rotation=0, labelpad=20)
        ax.set_title(title, fontsize=24, pad=10)

        # Apply log scale if specified
        if visualization_config.get('log_y_axis', False):
            ax.set_yscale('log')
        if visualization_config.get('log_x_axis', False):
            ax.set_xscale('log')

        # Format x-axis
        format_x = visualization_config.get('format_x', lambda x, _: f'{x:.2f}')
        ax.set_xticks(x_data)
        ax.set_xticklabels([format_x(x, 0) for x in x_data])

        ax.tick_params(axis='both', which='major', labelsize=17)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Get legend handles from the first subplot
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # Add shared legend below all subplots
    if handles and labels:
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            fontsize=20,
            frameon=False
        )

    plt.tight_layout(rect=(0, 0.05, 1, 0.98))

    # Save plot if requested
    if save_plots and plots_dir:
        import os
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_mc_comparison(
    deltas_dict_arr,
    epsilon_mat,
    subplot_titles: List[str],
    save_plots: bool = False,
    show_plots: bool = False,
    plots_dir: Optional[str] = None,
    filename: str = 'MC_comparison_plot.png'
) -> Figure:
    """
    Create Monte Carlo comparison plot with distinct colors for each method
    Uses standard colors and markers for consistency with other plots

    Args:
        deltas_dict_arr: List of dictionaries mapping method names to delta arrays
        epsilon_mat: List of epsilon arrays
        subplot_titles: List of subplot titles
        save_plots: Whether to save the plot
        show_plots: Whether to display the plot
        plots_dir: Directory to save plots (required if save_plots=True)
        filename: Filename for the saved plot

    Returns:
        The created matplotlib figure
    """
    # Distinct colors for each method in experiment 2
    method_colors = {
        'Poisson': '#27AE60',           # Green (same as POISSON_PLD)
        'allocation - Our': '#157DED',  # Blue (same as allocation methods)
        'allocation - MC HP': '#E74C3C',    # Red
        'allocation - MC mean': '#9B59B6'   # Purple
    }

    # Distinct markers for each method
    method_markers = {
        'Poisson': 'x',      # x marker (same as POISSON_PLD)
        'allocation - Our': 's',  # Square (same as ALLOCATION_COMBINED)
        'allocation - MC HP': 'D',    # Diamond
        'allocation - MC mean': 'o'   # Circle
    }

    # Custom labels
    custom_labels = {
        'Poisson': r'$\varepsilon_{\mathcal{P}}$ - Poisson (PLD)',
        'allocation - Our': r'$\varepsilon_{\mathcal{A}}$ - Allocation (Our - Combined)',
        'allocation - MC HP': r'$\varepsilon_{\mathcal{A}}$ - Allocation (CGHKKLMSZ24 - Monte Carlo HP)',
        'allocation - MC mean': r'$\varepsilon_{\mathcal{A}}$ - Allocation (CGHKKLMSZ24 - Monte Carlo mean)'
    }

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    # Store handles and labels from the first subplot that has data
    handles, labels = None, None

    for i, (deltas_dict, epsilon_arr) in enumerate(zip(deltas_dict_arr, epsilon_mat)):
        ax = axs[i]

        for method, deltas in deltas_dict.items():
            # Get distinct color and marker for this method
            color = method_colors.get(method, '#95A5A6')  # Fallback to gray
            marker = method_markers.get(method, 'o')  # Fallback to circle
            legend_label = custom_labels.get(method, method)  # Fallback to method name

            ax.plot(
                epsilon_arr,
                deltas,
                label=legend_label,
                color=color,
                marker=marker,
                linewidth=2.5,
                markersize=8,
                alpha=0.9
            )

        ax.set_title(subplot_titles[i], fontsize=24, pad=10)
        ax.set_xlabel(f"${names_dict[EPSILON]}$", fontsize=21)
        ax.set_ylabel(f"${names_dict[DELTA]}$", fontsize=21, rotation=0, labelpad=20)
        ax.set_yscale("log")
        ax.tick_params(axis='both', which='major', labelsize=17)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Get legend handles from the first subplot with data
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # Add the legend below all the subplots if we have handles
    if handles and labels:
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=20,
            frameon=False
        )

        plt.tight_layout(rect=(0, 0.05, 1, 0.98))
    else:
        plt.tight_layout()

    # Save plot if requested
    if save_plots and plots_dir:
        import os
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return fig