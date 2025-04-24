"""
Random Allocation for Differential Privacy

This package provides tools for analyzing and comparing different random allocation schemes
in the context of differential privacy.
"""

from random_allocation.comparisons.experiments import run_experiment, PlotType
from random_allocation.comparisons.visualization import plot_comparison, plot_combined_data, plot_as_table
from random_allocation.comparisons.definitions import (
    ALLOCATION, ALLOCATION_ANALYTIC, ALLOCATION_RDP, ALLOCATION_DECOMPOSITION,
    EPSILON, DELTA, VARIABLES, methods_dict, names_dict, colors_dict
)
from random_allocation.random_allocation_scheme import (
    allocation_epsilon_analytic, allocation_delta_analytic,
    allocation_epsilon_rdp, allocation_delta_rdp,
    allocation_epsilon_rdp_DCO, allocation_delta_rdp_DCO,
    allocation_epsilon_decomposition, allocation_delta_decomposition
)

__all__ = [
    # Experiment functions
    'run_experiment',
    'PlotType',
    
    # Plotting functions
    'plot_comparison',
    'plot_combined_data',
    'plot_as_table',
    
    # Constants and configurations
    'ALLOCATION',
    'ALLOCATION_ANALYTIC',
    'ALLOCATION_RDP',
    'ALLOCATION_DECOMPOSITION',
    'EPSILON',
    'DELTA',
    'VARIABLES',
    'methods_dict',
    'names_dict',
    'colors_dict',
    
    # Core allocation functions
    'allocation_epsilon_analytic',
    'allocation_delta_analytic',
    'allocation_epsilon_rdp',
    'allocation_delta_rdp',
    'allocation_epsilon_rdp_DCO',
    'allocation_delta_rdp_DCO',
    'allocation_epsilon_decomposition',
    'allocation_delta_decomposition'
] 