import sys
import os

# Add the correct project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from random_allocation.other_schemes.poisson import poisson_epsilon_pld, poisson_pld
from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition_remove

sigma = 1.2893288035581452
delta = 6.321223982317534e-11
num_steps = 100000
num_selected = 1
num_epochs = 1
discretization = 0.0001
sampling_prob = num_selected/num_steps
# pld_1 =  poisson_pld(sigma=sigma, num_steps=num_steps, num_epochs=num_epochs, sampling_prob=sampling_prob,
#                       discretization=discretization)

epsilon = poisson_epsilon_pld(sigma=sigma, delta=delta, num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs, discretization=discretization)
print(f'Epsilon: {epsilon}')

sigma_2 = 1.2893288035581452
delta_2 = 1e-10
num_steps_2 = 100000
num_selected_2 = 1
num_epochs_2 = 1
discretization_2 = 0.0001

# num_steps_per_round = np.ceil(num_steps/num_selected).astype(int)
# num_rounds = np.ceil(num_steps/num_steps_per_round).astype(int)
# lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
# delta_new = delta * lambda_val


epsilon = allocation_epsilon_decomposition_remove(sigma=sigma_2, delta=delta_2, num_steps=num_steps_2,
                                                  num_selected=num_selected_2, num_epochs=num_epochs_2, discretization=discretization_2)
print(f'Epsilon: {epsilon}')