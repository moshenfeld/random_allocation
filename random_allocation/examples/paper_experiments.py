# Standard library imports
import os
import sys
from typing import Dict, List, Union, Any, Callable

# Add the correct project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from random_allocation.comparisons.definitions import *
from random_allocation.comparisons.experiments import *
from random_allocation.comparisons.visualization import *

# Configuration
READ_DATA: bool = True  # Set to True to try reading data from existing files first
SAVE_DATA: bool = True  # Set to True to save computed data to CSV files
SAVE_PLOTS: bool = True  # Set to True to save plots to files
SHOW_PLOTS: bool = False  # Set to True to display plots interactively

# First experiment - Compare different schemes for varying sigma
params_dict_1: Dict[str, Any] = {
    'x_var': SIGMA,
    'y_var': EPSILON,
    SIGMA: np.exp(np.linspace(np.log(0.2), np.log(2), 20)),
    DELTA: 1e-10,
    NUM_STEPS: 100_000,
    NUM_SELECTED: 1,
    NUM_EPOCHS: 1
}

config_1: SchemeConfig = SchemeConfig(allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)])

methods_list_1: List[str] = [LOCAL, SHUFFLE, POISSON_PLD, ALLOCATION_DIRECT, ALLOCATION_RECURSIVE, ALLOCATION_DECOMPOSITION]

visualization_config_1: Dict[str, Union[bool, Callable[[float, int], str]]] = {
    'log_x_axis': True, 
    'log_y_axis': True, 
    'format_x': lambda x, _: f'{x:.2f}'
}

methods_list_1_add_rem: List[str] = [LOCAL, POISSON_PLD, ALLOCATION_DIRECT, ALLOCATION_RECURSIVE, ALLOCATION_DECOMPOSITION]

data_1: Dict[str, Any] = run_experiment(
    params_dict=params_dict_1, 
    config=config_1, 
    methods=methods_list_1, 
    visualization_config=visualization_config_1, 
    experiment_name='main', 
    plot_type=PlotType.COMBINED,
    read_data=READ_DATA,
    save_data=SAVE_DATA,
    save_plots=SAVE_PLOTS,
    show_plots=SHOW_PLOTS,
    direction=Direction.BOTH
)

data_1_add: Dict[str, Any] = run_experiment(
    params_dict=params_dict_1, 
    config=config_1, 
    methods=methods_list_1_add_rem, 
    visualization_config=visualization_config_1, 
    experiment_name='main_add', 
    plot_type=PlotType.COMBINED,
    read_data=READ_DATA,
    save_data=SAVE_DATA,
    save_plots=False,
    show_plots=False,
    direction=Direction.ADD
)

data_1_rem: Dict[str, Any] = run_experiment(
    params_dict=params_dict_1, 
    config=config_1, 
    methods=methods_list_1_add_rem, 
    visualization_config=visualization_config_1, 
    experiment_name='main_remove',     
    plot_type=PlotType.COMBINED,
    read_data=READ_DATA,
    save_data=SAVE_DATA,
    save_plots=False,
    show_plots=False,
    direction=Direction.REMOVE
)

data_list_add_remove = [data_1_add, data_1_rem]

titles = [
    f"Add direction",
    f"Remove direction"
]

# Use the plot_multiple_data function to create a multi-subplot figure
fig = plot_multiple_data(
    data_list=data_list_add_remove,
    titles=titles,
    log_x_axis=True,
    log_y_axis=True,
    format_x=lambda x, _: f'{x:.2f}',
    plot_type='main_add_remove', 
    figsize=(15, 6),  # Width, height in inches
    grid_layout=(1, 2)  # 1 row, 2 columns
)

if SAVE_PLOTS:
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'main_add_remove_plot.png'))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)


# Second experiment - Compare different schemes for varying number of epochs
params_dict_2: Dict[str, Any] = {
    'x_var': NUM_EPOCHS,
    'y_var': EPSILON,
    SIGMA: 1,
    DELTA: 1e-8,
    NUM_STEPS: 10_000,
    NUM_SELECTED: 1,
    NUM_EPOCHS: np.exp(np.linspace(np.log(1), np.log(1001), 10)).astype(int)
}

config_2: SchemeConfig = SchemeConfig(allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)])

methods_list_2: List[str] = [POISSON_PLD, POISSON_RDP, ALLOCATION_DIRECT]

visualization_config_2: Dict[str, Union[bool, Callable[[float, int], str]]] = {
    'log_x_axis': True, 
    'log_y_axis': False, 
    'format_x': lambda x, _: str(int(x))
}

data_2: Dict[str, Any] = run_experiment(
    params_dict=params_dict_2, 
    config=config_2, 
    methods=methods_list_2, 
    visualization_config=visualization_config_2, 
    experiment_name='multi_epoch', 
    plot_type=PlotType.COMPARISON,
    read_data=READ_DATA,
    save_data=SAVE_DATA,
    save_plots=SAVE_PLOTS,
    show_plots=SHOW_PLOTS,
    direction=Direction.BOTH
)

# Third experiment - Compare different schemes for varying number of steps
params_dict_3: Dict[str, Any] = {
    'x_var': NUM_STEPS,
    'y_var': EPSILON,
    SIGMA: 0.3,
    DELTA: 1e-4,
    NUM_STEPS: np.arange(25, 551, 50),
    NUM_SELECTED: 1,
    NUM_EPOCHS: 1
}

config_3: SchemeConfig = SchemeConfig(allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)])

methods_list_3: List[str] = [POISSON_RDP, POISSON_PLD, ALLOCATION_DIRECT, ALLOCATION_DECOMPOSITION]

visualization_config_3: Dict[str, Union[bool, Callable[[float, int], str]]] = {
    'log_x_axis': False, 
    'log_y_axis': False, 
    'format_x': lambda x, _: str(int(x))
}

data_3: Dict[str, Any] = run_experiment(
    params_dict=params_dict_3, 
    config=config_3, 
    methods=methods_list_3, 
    visualization_config=visualization_config_3, 
    experiment_name='RDP_domination', 
    plot_type=PlotType.COMPARISON,
    read_data=READ_DATA,
    save_data=SAVE_DATA,
    save_plots=SAVE_PLOTS,
    show_plots=SHOW_PLOTS,
    direction=Direction.BOTH
)

# Fourth experiment - Compare different schemes for varying number of selected items
params_dict_4: Dict[str, Any] = {
    'x_var': NUM_SELECTED,
    'y_var': EPSILON,
    SIGMA: 0.6,
    DELTA: 1e-6,
    NUM_STEPS: 2**10,
    NUM_SELECTED: 2**np.arange(0, 7),
    NUM_EPOCHS: 1,
}

config_4: SchemeConfig = SchemeConfig(allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)])

methods_list_4: List[str] = [POISSON_RDP, ALLOCATION_DIRECT, ALLOCATION_RDP_DCO]

visualization_config_4: Dict[str, Union[bool, Callable[[float, int], str]]] = {
    'log_x_axis': True, 
    'log_y_axis': True, 
    'format_x': lambda x, _: str(int(x))
}

data_4: Dict[str, Any] = run_experiment(
    params_dict=params_dict_4, 
    config=config_4, 
    methods=methods_list_4, 
    visualization_config=visualization_config_4, 
    experiment_name='DCO_comp', 
    plot_type=PlotType.COMPARISON,
    read_data=READ_DATA,
    save_data=SAVE_DATA,
    save_plots=SAVE_PLOTS,
    show_plots=SHOW_PLOTS,
    direction=Direction.BOTH
)

config_5: SchemeConfig = SchemeConfig(allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)])

methods_list_5: List[str] = [POISSON_PLD, ALLOCATION_RDP_DCO, ALLOCATION_COMBINED]

visualization_config_5: Dict[str, Union[bool, Callable[[float, int], str]]] = {
    'log_x_axis': True, 
    'log_y_axis': False, 
    'format_x': lambda x, _: f'{x:.2f}'
}

params_dict_5_1: Dict[str, Any] = {
    'x_var': SIGMA,
    'y_var': EPSILON,
    SIGMA: np.exp(np.linspace(np.log(0.8), np.log(2.5), 10)),
    DELTA: 1e-10,
    NUM_STEPS: 100,
    NUM_SELECTED: 1,
    NUM_EPOCHS: 1
}

data_5_1: Dict[str, Any] = run_experiment(
    params_dict=params_dict_5_1,
    config=config_5,
    methods=methods_list_5,
    visualization_config=visualization_config_5,
    experiment_name='epsilon_vs_sigma_small_t',
    plot_type=PlotType.COMBINED,
    read_data=READ_DATA,
    save_data=SAVE_DATA,
    save_plots=False,
    show_plots=False,
    direction=Direction.BOTH
)

params_dict_5_2: Dict[str, Any] = {
    'x_var': SIGMA,
    'y_var': EPSILON,
    SIGMA: np.exp(np.linspace(np.log(0.65), np.log(2), 10)),
    DELTA: 1e-10,
    NUM_STEPS: 1_000,
    NUM_SELECTED: 1,
    NUM_EPOCHS: 1
}

data_5_2: Dict[str, Any] = run_experiment(
    params_dict=params_dict_5_2,
    config=config_5,
    methods=methods_list_5,
    visualization_config=visualization_config_5,
    experiment_name='epsilon_vs_sigma_mid_t',
    plot_type=PlotType.COMBINED,
    read_data=READ_DATA,
    save_data=SAVE_DATA,
    save_plots=False,
    show_plots=False,
    direction=Direction.BOTH
)

params_dict_5_3: Dict[str, Any] = {
    'x_var': SIGMA,
    'y_var': EPSILON,
    SIGMA: np.exp(np.linspace(np.log(0.6), np.log(0.9), 10)),
    DELTA: 1e-10,
    NUM_STEPS: 10_000,
    NUM_SELECTED: 1,
    NUM_EPOCHS: 1
}

data_5_3: Dict[str, Any] = run_experiment(
    params_dict=params_dict_5_3,
    config=config_5,
    methods=methods_list_5,
    visualization_config=visualization_config_5,
    experiment_name='epsilon_vs_sigma_large_t',
    plot_type=PlotType.COMBINED,
    read_data=READ_DATA,
    save_data=SAVE_DATA,
    save_plots=False,
    show_plots=False,
    direction=Direction.BOTH
)

data_list_5 = [data_5_1, data_5_2, data_5_3]

titles = [
    f"{params_dict_5_1[NUM_STEPS]} steps",
    f"{params_dict_5_2[NUM_STEPS]} steps",
    f"{params_dict_5_3[NUM_STEPS]} steps"
]

# Use the plot_multiple_data function to create a multi-subplot figure
fig = plot_multiple_data(
    data_list=data_list_5,
    titles=titles,
    log_x_axis=True,
    log_y_axis=False,
    format_x=lambda x, _: f'{x:.2f}',
    plot_type='epsilon_vs_sigma_combined', 
    figsize=(20, 6),  # Width, height in inches
    grid_layout=(1, 3)  # 1 row, 3 columns
)

# Adjust the layout to prevent clipping and add a super title
plt.suptitle("Effect of σ on Privacy Guarantee (ε) Across Different Number of Steps", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the super title

if SAVE_PLOTS:
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'multi_range_plot.png'))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

config_6 = SchemeConfig(
    discretization=1e-4,
    allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)],
    delta_tolerance=1e-15,
    epsilon_tolerance=1e-3,
    MC_use_order_stats=True,
    MC_use_mean=False,
    MC_conf_level=0.99,
    MC_sample_size=500_000,
    verbosity=Verbosity.NONE,
)

num_steps_arr_6 = [100, 1000, 10_000, 100_000]
sigma_arr_6 = [1.0, 0.7, 0.5, 0.4]
epsilon_arr_6 = [np.linspace(0.05, 0.8, 10),
                 np.linspace(0.1, 1.1, 10),
                 np.linspace(0.3, 2.0, 10),
                 np.linspace(0.5, 3.0, 10)]
num_selected_6 = 1
num_epochs_6 = 1
params_mat_6 = [[PrivacyParams(num_steps=num_steps, sigma=sigma, epsilon=epsilon, num_selected=num_selected_6, num_epochs=num_epochs_6) for epsilon in epsilon_arr] for num_steps, sigma, epsilon_arr in zip(num_steps_arr_6, sigma_arr_6, epsilon_arr_6)]
deltas_dict_arr = [calc_all_methods_delta(params_arr, config_6) for params_arr in params_mat_6]
subplot_titles = [f"Steps: {num_steps}, Sigma: {sigma}" for num_steps, sigma in zip(num_steps_arr_6, sigma_arr_6)]
fig = plot_privacy_curves(deltas_dict_arr, epsilon_arr_6, subplot_titles)

if SAVE_PLOTS:
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'MC_comparison_plot.png'))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)