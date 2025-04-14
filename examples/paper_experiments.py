import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from comparisons.definitions import *
from comparisons.experiments import calc_params
from comparisons.visualization import plot_combined_data, plot_as_table, plot_comparison

# First experiment
params_dict_1 = {'x_var': SIGMA,
                 'y_var': EPSILON,
                 SIGMA: np.exp(np.linspace(np.log(0.2), np.log(5), 20)),
                 DELTA: 1e-10,
                 NUM_STEPS: 100_000,
                 NUM_SELECTED: 1,
                 NUM_EPOCHS: 1}

config_dict_1 = {DISCRETIZATION: 1e-4,
                 MIN_ALPHA: 2,
                 MAX_ALPHA: 60}

methods_list_1 = [LOCAL, POISSON_PLD, SHUFFLE, ALLOCATION_RDP, ALLOCATION_ANALYTIC, ALLOCATION_DECOMPOSITION]

experiment_data_1 = calc_params(params_dict_1, config_dict_1, methods_list_1)

plot_combined_data(experiment_data_1, log_x_axis=True, log_y_axis=True)
plot_as_table(experiment_data_1)

# Second experiment
params_dict_2 = {'x_var': NUM_EPOCHS,
                 'y_var': EPSILON,
                 SIGMA: 1,
                 DELTA: 1e-8,
                 NUM_STEPS: 10_000,
                 NUM_SELECTED: 1,
                 NUM_EPOCHS: np.exp(np.linspace(np.log(1), np.log(1001), 10)).astype(int)}

config_dict_2 = {DISCRETIZATION: 1e-4,
                 MIN_ALPHA: 2,
                 MAX_ALPHA: 60}

methods_list_2 = [POISSON_RDP, ALLOCATION_RDP, POISSON_PLD, ALLOCATION_DECOMPOSITION]

experiment_data_2 = calc_params(params_dict_2, config_dict_2, methods_list_2)

plot_comparison(experiment_data_2, log_x_axis=True, log_y_axis=False, format_x=lambda x, _: x)
plot_as_table(experiment_data_2)
