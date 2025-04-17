import sys
import os

# Add the correct project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from random_allocation.other_schemes.poisson import poisson_epsilon_pld

sigma = 1.2893288035581452
delta = 6.321223982317534e-11
num_steps = 100000
num_selected = 1
num_epochs = 1
sampling_prob = 0.0
discretization = 0.0001

epsilon = poisson_epsilon_pld(sigma=sigma, delta=delta, num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs, discretization=discretization)
print(f'Epsilon: {epsilon}')

from random_allocation.random_allocation_scheme.decomposition import allocation_epsilon_decomposition_remove
sigma = 1.2893288035581452
delta = 1e-10
num_steps = 100000
num_selected = 1
num_epochs = 1
discretization = 0.0001

epsilon = allocation_epsilon_decomposition_remove(sigma=sigma, delta=delta, num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs, discretization=discretization)
print(f'Epsilon: {epsilon}')