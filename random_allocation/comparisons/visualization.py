from typing import Dict, Any, List, Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from random_allocation.comparisons.definitions import ALLOCATION, ALLOCATION_ANALYTIC, ALLOCATION_RDP, ALLOCATION_DECOMPOSITION, EPSILON, names_dict, colors_dict, get_features_for_methods

def plot_comparison(data, log_x_axis = False, log_y_axis = False, format_x=lambda x, _: f'{x:.2f}', format_y=lambda x, _: f'{x:.2f}', figsize=(16, 9)):
    """
    Create a comparison plot and return the figure.
    """
    methods = list(data['y data'].keys())
    #remove keys that end with '- std'
    filtered_methods = [method for method in methods if not method.endswith('- std')]
    methods_data = data['y data']
    legend_map = get_features_for_methods(filtered_methods, 'legend')
    markers_map = get_features_for_methods(filtered_methods, 'marker')
    colors_map = get_features_for_methods(filtered_methods, 'color')
    legend_prefix = '$\\varepsilon' if data['y name'] == names_dict[EPSILON] else '$\\delta'
    fig = plt.figure(figsize=figsize)
    for method in filtered_methods:
        plt.plot(data['x data'], methods_data[method], label=legend_prefix+legend_map[method], marker=markers_map[method], color=colors_map[method], linewidth=2.5, markersize=12, alpha=0.8)
        if method + '- std' in methods:
            plt.fill_between(data['x data'], np.clip(methods_data[method] - methods_data[method + '- std'], 0, 1),  np.clip(methods_data[method] + methods_data[method + '- std'], 0, 1), color=colors_map[method], alpha=0.1)
    plt.xlabel(data['x name'], fontsize=20)
    plt.ylabel(data['y name'], fontsize=20)
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
    plt.grid(True)
    return fig

def plot_as_table(data):
    methods = list(data['y data'].keys())
    methods_data = data['y data']
    table = pd.DataFrame(methods_data, index=data['x data'])
    table.index.name = data['x name']
    table.columns = [method for method in methods]
    return table

def plot_combined_data(data, log_x_axis = False, log_y_axis = False, format_x=lambda x, _: f'{x:.2f}', format_y=lambda x, _: f'{x:.2f}', figsize=(16, 9)):
    """
    Create a combined data plot and return the figure.
    """
    methods = list(data['y data'].keys())
    if ALLOCATION_ANALYTIC in methods and ALLOCATION_RDP in methods and ALLOCATION_DECOMPOSITION in methods:
        min_allocation = np.min(np.array([data['y data'][ALLOCATION_ANALYTIC], data['y data'][ALLOCATION_RDP], data['y data'][ALLOCATION_DECOMPOSITION]]), axis=0)
    methods_data = data['y data']
    legend_map = get_features_for_methods(methods, 'legend')
    markers_map = get_features_for_methods(methods, 'marker')
    colors_map = get_features_for_methods(methods, 'color')
    legend_prefix = '$\\varepsilon' if data['y name'] == names_dict[EPSILON] else '$\\delta'
    fig = plt.figure(figsize=figsize)
    for method in methods:
        linewidth = 1        if (method == ALLOCATION_DECOMPOSITION or method ==  ALLOCATION_RDP or method ==  ALLOCATION_ANALYTIC) else 2
        linestyle = 'dotted' if (method == ALLOCATION_DECOMPOSITION or method ==  ALLOCATION_RDP or method ==  ALLOCATION_ANALYTIC) else 'solid'
        plt.plot(data['x data'], methods_data[method], label=legend_prefix+legend_map[method], marker=markers_map[method], color=colors_map[method], linewidth=linewidth, linestyle=linestyle, markersize=10, alpha=0.8)
    if ALLOCATION_ANALYTIC in methods and ALLOCATION_RDP in methods and ALLOCATION_DECOMPOSITION in methods:
        plt.plot(data['x data'], min_allocation, label='_{\\mathcal{A}}$ - (Our - Combined)', color=colors_dict[ALLOCATION], linewidth=2, alpha=1)
    plt.xlabel(data['x name'], fontsize=20)
    plt.ylabel(data['y name'], fontsize=20)
    if log_x_axis:
        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    else:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    if log_y_axis:
        plt.yscale('log')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(data['x data'])
    plt.legend(fontsize=20, loc='lower left', framealpha=0.)
    plt.grid(True)
    return fig